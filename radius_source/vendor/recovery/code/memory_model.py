"""Memory-bounded execution of the frozen FIMS site MPNN mathematics.

The caller must put the verified ORIGINAL package's ``code`` directory on
sys.path before importing this module. No original file is modified. All node
and edge selections, phi/psi layers, parameter names, initialization order,
and output logic remain those of graph_models_v1.py.

The execution change is edge-chunked messages and a one-dimensional index_add
sum, with nonreentrant activation checkpointing while gradients are enabled.
It avoids materializing an E-by-hidden expanded scatter index and retaining
all E-by-hidden MLP activations. No neighbor is sampled, dropped, or truncated.
Floating-point reduction grouping can differ from the original scatter;
bitwise continuation across the implementation change is not claimed.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from graph_models_v1 import (
    GeometricMessageLayer as OriginalGeometricMessageLayer,
    GeometricMPNN as OriginalGeometricMPNN,
    HIDDEN_CHANNELS,
    VARIANTS,
)

DEFAULT_EDGE_CHUNK = 16384


def _chunk_size(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("edge_chunk_size must be a positive integer")
    return value


class MemoryBoundedMessageLayer(OriginalGeometricMessageLayer):
    """The same message/sum/update layer with bounded edge activations."""

    def __init__(self, input_channels, hidden_channels=HIDDEN_CHANNELS,
                 *, edge_chunk_size=DEFAULT_EDGE_CHUNK):
        super().__init__(input_channels, hidden_channels)
        self.edge_chunk_size = _chunk_size(edge_chunk_size)

    def _sum_chunk(self, h, source, target, edge_attr):
        # Gathering is inside the checkpoint, so its E_chunk-by-input-channel
        # result is recomputed in backward rather than retained for every edge.
        messages = self.phi(torch.cat((h.index_select(0, source), edge_attr), dim=-1))
        # index_add's autograd implementation saves its source tensor. Keep
        # this operation INSIDE the checkpoint, otherwise the complete E-by-64
        # collection of chunk message outputs would still survive to backward.
        partial = h.new_zeros((h.shape[0], self.phi[-2].out_features))
        return partial.index_add_(0, target, messages)

    def forward(self, h, edge_index, edge_attr):
        if (edge_index.ndim != 2 or edge_index.shape[0] != 2
                or edge_index.dtype != torch.long
                or edge_attr.ndim != 2 or edge_attr.shape != (edge_index.shape[1], 3)):
            raise ValueError("Expected edge_index [2,E] int64 and edge_attr [E,3]")
        if h.ndim != 2:
            raise ValueError("Expected a node-by-channel feature matrix")
        hidden = self.phi[-2].out_features
        aggregated = h.new_zeros((h.shape[0], hidden))
        n_edges = edge_index.shape[1]
        # Even zero-edge graphs evaluate phi on an empty input, as the original
        # PyG propagate does. This preserves zero (rather than missing) phi
        # gradients, which matters for Adam's per-parameter state updates.
        for begin in range(0, max(1, n_edges), self.edge_chunk_size):
            end = min(n_edges, begin + self.edge_chunk_size)
            source = edge_index[0, begin:end]
            target = edge_index[1, begin:end]
            attrs = edge_attr[begin:end]
            if torch.is_grad_enabled():
                # Nonreentrant checkpoint supports first-layer constant node
                # inputs: the learned phi parameters still receive gradients.
                # phi contains Linear/ReLU only; it has no stochastic draws.
                partial = checkpoint(self._sum_chunk, h, source, target, attrs,
                                     use_reentrant=False, preserve_rng_state=False)
            else:
                partial = self._sum_chunk(h, source, target, attrs)
            # AddBackward saves neither value. Only a current N-by-64 partial
            # and accumulator are needed; edge activations are recomputed one
            # chunk at a time. No E-by-hidden expanded scatter index is used.
            aggregated = aggregated + partial
        return self.psi(torch.cat((h, aggregated), dim=-1))


class MemoryBoundedMPNN(OriginalGeometricMPNN):
    """Drop-in model with exactly the original state_dict and parameters."""

    def __init__(self, variant="full2", hidden_channels=HIDDEN_CHANNELS,
                 *, edge_chunk_size=DEFAULT_EDGE_CHUNK):
        # Construct exactly once, in original order, to preserve seeded model
        # initialization and the RNG state consumed by construction.
        nn.Module.__init__(self)
        if variant not in VARIANTS:
            raise ValueError("Unknown variant: " + str(variant))
        if hidden_channels != HIDDEN_CHANNELS:
            raise ValueError("This fixed protocol uses exactly 64 hidden channels")
        self.variant = variant
        self.edge_chunk_size = _chunk_size(edge_chunk_size)
        self.layers = nn.ModuleList([
            MemoryBoundedMessageLayer(1 if i == 0 else HIDDEN_CHANNELS,
                                     edge_chunk_size=self.edge_chunk_size)
            for i in range(VARIANTS[variant]["layers"])
        ])
        self.output = nn.Linear(HIDDEN_CHANNELS, 1)


# The recovery adapter can import the original name without modifying the
# original package or replacing graph_models_v1 globally.
GeometricMPNN = MemoryBoundedMPNN
