#!/usr/bin/env python3
"""Query-bounded, causal site MPNNs for the FIMS CS224W study.

Only x/y readout position, integer 25 ns time bins, and endcap enter this
encoder. A query graph is rebuilt from its own fixed observation window;
message-passing depth never expands that window. Whole-current-bin access
means a prediction is made after that bin closes, not at the first arrival.

The four fixed variants are full2, full1, no_xy2, and current2. The no_xy2
variant removes explicit spatial edge attributes but retains radius-based
connectivity; it is not a geometry-free control. Network outputs are logits.
"""
from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
# Required for deterministic CUDA matrix products. A caller's explicit value
# is preserved and the allocated-GPU self-test checks its actual support.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing

RADIUS_MM = 2.0
HISTORY_BINS = 8
HIDDEN_CHANNELS = 64
VARIANTS = {
    "full2": {"layers": 2, "history_bins": 8, "spatial_attributes": True},
    "full1": {"layers": 1, "history_bins": 8, "spatial_attributes": True},
    "no_xy2": {"layers": 2, "history_bins": 8, "spatial_attributes": False},
    "current2": {"layers": 2, "history_bins": 1, "spatial_attributes": True},
}
GEOMETRY_KEYS = frozenset({"x_mm", "y_mm", "time_bin", "side"})


class GraphLimitError(RuntimeError):
    """An explicit resource ceiling was exceeded; no neighborhood was clipped."""


class QuerySiteData(Data):
    """One query graph; query_index is a node offset when graphs are batched."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "query_index":
            return int(self.num_nodes)
        if key == "observation_indices":
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("query_index", "observation_indices"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def _geometry(geometry: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(geometry) != GEOMETRY_KEYS:
        raise ValueError("Geometry must contain only x_mm, y_mm, time_bin, side; no truth fields")
    raw = {k: np.asarray(geometry[k]) for k in GEOMETRY_KEYS}
    n = len(raw["x_mm"]) if raw["x_mm"].ndim == 1 else -1
    if n < 0 or any(a.ndim != 1 or len(a) != n for a in raw.values()):
        raise ValueError("All geometry arrays must be aligned one-dimensional vectors")
    if not all(np.issubdtype(raw[k].dtype, np.number) for k in ("x_mm", "y_mm")):
        raise ValueError("Readout coordinates must be numeric")
    if any(not np.isfinite(raw[k]).all() for k in ("x_mm", "y_mm")):
        raise ValueError("Readout coordinates must be finite")
    if not np.issubdtype(raw["time_bin"].dtype, np.integer):
        raise ValueError("time_bin must have an integer dtype; no inferred or rounded times")
    if np.issubdtype(raw["time_bin"].dtype, np.unsignedinteger) and n:
        if int(raw["time_bin"].max()) > np.iinfo(np.int64).max:
            raise ValueError("time_bin cannot be represented by int64")
    if not np.issubdtype(raw["side"].dtype, np.integer) or not np.isin(raw["side"], [0, 1]).all():
        raise ValueError("side must contain integer endcap labels 0 or 1")
    out = {
        "x_mm": np.array(raw["x_mm"], dtype=np.float64, copy=True),
        "y_mm": np.array(raw["y_mm"], dtype=np.float64, copy=True),
        "time_bin": np.array(raw["time_bin"], dtype=np.int64, copy=True),
        "side": np.array(raw["side"], dtype=np.int8, copy=True),
    }
    # A site is one occupied address/time/endcap. Repeated electrons must not
    # silently be presented as distinct binary sites.
    if n:
        order = np.lexsort((out["y_mm"], out["x_mm"], out["time_bin"], out["side"]))
        equal = np.ones(max(0, n - 1), dtype=bool)
        for k in GEOMETRY_KEYS:
            equal &= out[k][order[1:]] == out[k][order[:-1]]
        if equal.any():
            raise ValueError("Duplicate binary site address/time/endcap")
    for a in out.values():
        a.setflags(write=False)
    return out


class QueryGraphBuilder:
    """Reusable endcap trees for one event; build() never consults labels.

    max_nodes and max_edges are optional *failure* ceilings. Exceeding either
    raises GraphLimitError. They never change node selection, sample neighbors,
    or silently remove a dense event from training or evaluation.
    """

    def __init__(self, geometry, variant="full2", *, max_nodes=None, max_edges=None):
        if variant not in VARIANTS:
            raise ValueError("Unknown variant: " + str(variant))
        for name, value in (("max_nodes", max_nodes), ("max_edges", max_edges)):
            if value is not None and (isinstance(value, bool) or int(value) != value or value < 0):
                raise ValueError(name + " must be a nonnegative integer or None")
        self.geometry = _geometry(geometry)
        self.variant = variant
        self.config = dict(VARIANTS[variant])
        self.max_nodes = None if max_nodes is None else int(max_nodes)
        self.max_edges = None if max_edges is None else int(max_edges)
        self.xy = np.column_stack((self.geometry["x_mm"], self.geometry["y_mm"]))
        self.side_indices = {side: np.flatnonzero(self.geometry["side"] == side) for side in (0, 1)}
        self.trees = {
            side: cKDTree(self.xy[indices]) if len(indices) else None
            for side, indices in self.side_indices.items()
        }

    def __len__(self):
        return len(self.xy)

    def build(self, q: int) -> QuerySiteData:
        if isinstance(q, bool) or int(q) != q or not 0 <= int(q) < len(self):
            raise IndexError("Query must name an existing site in this event")
        q = int(q)
        g = self.geometry
        tq = int(g["time_bin"][q])
        side = int(g["side"][q])
        local = self.trees[side].query_ball_point(self.xy[q], RADIUS_MM, p=2, eps=0)
        indices = self.side_indices[side][np.asarray(local, dtype=np.int64)]
        delta = self.xy[indices] - self.xy[q]
        lower = max(np.iinfo(np.int64).min, tq - self.config["history_bins"] + 1)
        keep = ((np.einsum("ij,ij->i", delta, delta) <= RADIUS_MM**2)
                & (g["time_bin"][indices] >= lower)
                & (g["time_bin"][indices] <= tq))
        indices = indices[keep]
        # Canonical observed-coordinate order makes graph construction independent
        # of source row order without consulting identity or truth.
        order = np.lexsort((g["y_mm"][indices], g["x_mm"][indices], g["time_bin"][indices]))
        indices = indices[order]
        n = len(indices)
        if self.max_nodes is not None and n > self.max_nodes:
            raise GraphLimitError(f"query {q}: {n} nodes exceed max_nodes={self.max_nodes}; no truncation")
        query_matches = np.flatnonzero(indices == q)
        if len(query_matches) != 1:
            raise AssertionError("Query graph must contain its query exactly once")
        xy = self.xy[indices]
        times = g["time_bin"][indices]
        tree = cKDTree(xy)
        if self.max_edges is not None:
            # Every unordered spatial pair produces at least one causal edge.
            pairs_lower_bound = (int(tree.count_neighbors(tree, RADIUS_MM)) - n) // 2
            if pairs_lower_bound > self.max_edges:
                raise GraphLimitError(f"query {q}: at least {pairs_lower_bound} edges exceed "
                                      f"max_edges={self.max_edges}; no truncation")
        pairs = tree.query_pairs(RADIUS_MM, p=2, eps=0, output_type="ndarray")
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            d = xy[a] - xy[b]
            exact = np.einsum("ij,ij->i", d, d) <= RADIUS_MM**2
            a, b = a[exact], b[exact]
            forward = times[a] <= times[b]
            backward = times[b] <= times[a]
            source = np.concatenate((a[forward], b[backward]))
            target = np.concatenate((b[forward], a[backward]))
            order = np.lexsort((source, target))
            source, target = source[order], target[order]
            edges = np.stack((source, target)).astype(np.int64, copy=False)
            attrs = np.column_stack(((xy[source] - xy[target]) / RADIUS_MM,
                                     (times[source] - times[target]) / 7.0)).astype(np.float32)
            if not self.config["spatial_attributes"]:
                attrs[:, :2] = 0.0
        else:
            edges = np.empty((2, 0), dtype=np.int64)
            attrs = np.empty((0, 3), dtype=np.float32)
        if self.max_edges is not None and edges.shape[1] > self.max_edges:
            raise GraphLimitError(f"query {q}: {edges.shape[1]} edges exceed "
                                  f"max_edges={self.max_edges}; no truncation")
        return QuerySiteData(
            x=torch.ones((n, 1), dtype=torch.float32),
            edge_index=torch.from_numpy(edges),
            edge_attr=torch.from_numpy(attrs),
            query_index=torch.tensor([int(query_matches[0])], dtype=torch.long),
            # Audit mapping only: this field is never read by the model.
            observation_indices=torch.from_numpy(np.array(indices, dtype=np.int64, copy=True)),
        )


def build_query_graph(geometry, q, variant="full2", *, max_nodes=None, max_edges=None):
    return QueryGraphBuilder(geometry, variant, max_nodes=max_nodes, max_edges=max_edges).build(q)


def batch_graphs(graphs: Sequence[QuerySiteData], device="cpu") -> Batch:
    if len(graphs):
        return Batch.from_data_list(list(graphs)).to(device)
    # PyG's from_data_list([]) is undefined; represent the empty query batch
    # explicitly. There are zero predictions, not an invented empty-event node.
    return Batch(
        x=torch.empty((0, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 3), dtype=torch.float32),
        query_index=torch.empty((0,), dtype=torch.long),
        observation_indices=torch.empty((0,), dtype=torch.long),
        batch=torch.empty((0,), dtype=torch.long),
        ptr=torch.zeros((1,), dtype=torch.long),
    ).to(device)


class GeometricMessageLayer(MessagePassing):
    def __init__(self, input_channels: int, hidden_channels: int = HIDDEN_CHANNELS):
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.phi = nn.Sequential(
            nn.Linear(input_channels + 3, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
        )
        self.psi = nn.Sequential(
            nn.Linear(input_channels + hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
        )

    def forward(self, h, edge_index, edge_attr):
        aggregated = self.propagate(edge_index, h=h, edge_attr=edge_attr,
                                    size=(h.shape[0], h.shape[0]))
        return self.psi(torch.cat((h, aggregated), dim=-1))

    def message(self, h_j, edge_attr):
        return self.phi(torch.cat((h_j, edge_attr), dim=-1))


class GeometricMPNN(nn.Module):
    def __init__(self, variant="full2", hidden_channels=HIDDEN_CHANNELS):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("Unknown variant: " + str(variant))
        if hidden_channels != HIDDEN_CHANNELS:
            raise ValueError("This fixed protocol uses exactly 64 hidden channels")
        self.variant = variant
        self.layers = nn.ModuleList([
            GeometricMessageLayer(1 if i == 0 else HIDDEN_CHANNELS)
            for i in range(VARIANTS[variant]["layers"])
        ])
        self.output = nn.Linear(HIDDEN_CHANNELS, 1)

    def forward(self, batch):
        if batch.x.ndim != 2 or batch.x.shape[1] != 1 or not torch.all(batch.x == 1):
            raise ValueError("Node inputs must be the single constant occupancy feature")
        if batch.edge_attr.ndim != 2 or batch.edge_attr.shape[1] != 3:
            raise ValueError("Expected three normalized relative edge attributes")
        if batch.query_index.numel() == 0:
            return self.output.weight.new_empty((0,))
        h = batch.x
        # Mask in forward as well as the encoder: no_xy2 cannot accidentally
        # consume spatial attributes if a caller reuses a full graph batch.
        edge_attr = batch.edge_attr
        if self.variant == "no_xy2":
            edge_attr = torch.cat((torch.zeros_like(edge_attr[:, :2]), edge_attr[:, 2:]), dim=1)
        for layer in self.layers:
            h = layer(h, batch.edge_index, edge_attr)
        return self.output(h[batch.query_index]).reshape(-1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
