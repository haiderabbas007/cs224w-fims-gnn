"""Exact one-layer query execution for the FIMS radius sensitivity study.

The scientific graph has all same-endcap sites within r mm and bins t-7..t,
and causal edges of length <= r. With one message layer, only edges ending at
the query can affect its output. We retain every node and every such edge,
without constructing irrelevant neighbor-to-neighbor edges. This equivalence
does not extend to two-layer models. Spatial attributes always use /2 mm.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from graph_models_v1 import _geometry, QuerySiteData, GraphLimitError

RADII = (1.0, 2.0, 4.0)
NORMALIZATION_MM = 2.0
HISTORY_BINS = 8


class RadiusProfile:
    def __init__(self):
        self.builder_count = 0
        self.graph_queries = 0
        self.total_nodes = 0
        self.total_executed_edges = 0
        self.max_nodes = 0
        self.max_executed_edges = 0
        self.builder_seconds = 0.0
        self.graph_seconds = 0.0
        self.training_batch_seconds = 0.0
        self.training_batches = 0
        self.evaluation_seconds = 0.0
        self.evaluations = 0

    def as_dict(self):
        return dict(self.__dict__,
            interpretation='Counters include repeated training/validation visits; not distinct physical sites. '
                           'Edges are executed query-incoming edges, not the full induced graph. '
                           'Batch/evaluation totals include graph work; timing categories overlap.')


class RadiusQueryBuilder:
    def __init__(self, geometry, radius_mm, *, variant='full1', max_nodes=None,
                 max_edges=2000000, profile=None):
        started = time.perf_counter()
        if variant != 'full1':
            raise ValueError('Query-only edge execution is valid only for the fixed one-layer full1 model')
        if isinstance(radius_mm, bool) or float(radius_mm) not in RADII:
            raise ValueError('Radius must be 1, 2, or 4 mm')
        for name, value in (('max_nodes', max_nodes), ('max_edges', max_edges)):
            if value is not None and (isinstance(value, bool) or int(value) != value or value < 0):
                raise ValueError(name + ' must be a nonnegative integer or None')
        self.radius_mm = float(radius_mm)
        self.geometry = _geometry(geometry)
        self.xy = np.column_stack((self.geometry['x_mm'], self.geometry['y_mm']))
        self.side_indices = {s: np.flatnonzero(self.geometry['side'] == s) for s in (0, 1)}
        self.trees = {s: cKDTree(self.xy[ids]) if len(ids) else None
                      for s, ids in self.side_indices.items()}
        self.max_nodes, self.max_edges = max_nodes, max_edges
        self.profile = profile
        if profile is not None:
            profile.builder_count += 1
            profile.builder_seconds += time.perf_counter() - started

    def __len__(self):
        return len(self.xy)

    def build(self, q):
        started = time.perf_counter()
        if isinstance(q, bool) or int(q) != q or not 0 <= int(q) < len(self):
            raise IndexError('Query must identify an existing site')
        q = int(q)
        g, r = self.geometry, self.radius_mm
        tq, side = int(g['time_bin'][q]), int(g['side'][q])
        local = self.trees[side].query_ball_point(self.xy[q], r, p=2, eps=0)
        indices = self.side_indices[side][np.asarray(local, np.int64)]
        delta = self.xy[indices] - self.xy[q]
        lower = max(np.iinfo(np.int64).min, tq - HISTORY_BINS + 1)
        keep = ((np.einsum('ij,ij->i', delta, delta) <= r*r)
                & (g['time_bin'][indices] >= lower) & (g['time_bin'][indices] <= tq))
        indices = indices[keep]
        indices = indices[np.lexsort((g['y_mm'][indices], g['x_mm'][indices],
                                     g['time_bin'][indices]))]
        matches = np.flatnonzero(indices == q)
        if len(matches) != 1:
            raise AssertionError('The query must occur exactly once')
        query = int(matches[0])
        n = len(indices)
        source = np.flatnonzero(np.arange(n) != query).astype(np.int64)
        target = np.full(len(source), query, np.int64)
        if self.max_nodes is not None and n > self.max_nodes:
            raise GraphLimitError(f'{n} nodes exceed max_nodes={self.max_nodes}; no truncation')
        if self.max_edges is not None and len(source) > self.max_edges:
            raise GraphLimitError(f'{len(source)} incoming edges exceed max_edges={self.max_edges}; no truncation')
        if len(source):
            # Canonical source order equals the query's incoming-edge order in
            # the full graph. Every source is already <= the query time.
            attrs = np.column_stack(((self.xy[indices[source]] - self.xy[q])/NORMALIZATION_MM,
                                     (g['time_bin'][indices[source]] - tq)/7.0)).astype(np.float32)
            edges = np.stack((source, target))
        else:
            attrs = np.empty((0, 3), np.float32)
            edges = np.empty((2, 0), np.int64)
        result = QuerySiteData(x=torch.ones((n, 1), dtype=torch.float32),
            edge_index=torch.from_numpy(edges), edge_attr=torch.from_numpy(attrs),
            query_index=torch.tensor([query], dtype=torch.long),
            observation_indices=torch.from_numpy(np.array(indices, np.int64, copy=True)))
        if self.profile is not None:
            p = self.profile
            p.graph_queries += 1; p.total_nodes += n; p.total_executed_edges += len(source)
            p.max_nodes = max(p.max_nodes, n)
            p.max_executed_edges = max(p.max_executed_edges, len(source))
            p.graph_seconds += time.perf_counter() - started
        return result


def graph_chunks(arrays, variant, radius_mm, indices=None, max_queries=64,
                 max_edges=250000, *, profile=None):
    from site_data_v1 import geometry_only
    if max_queries <= 0 or max_edges <= 0:
        raise ValueError('Chunk budgets must be positive')
    builder = RadiusQueryBuilder(geometry_only(arrays), radius_mm,
                                 variant=variant, profile=profile)
    if indices is None:
        indices = range(len(arrays['target']))
    pending, ids, edges = [], [], 0
    for q in indices:
        graph = builder.build(int(q)); n = graph.edge_index.shape[1]
        if pending and (len(pending) >= max_queries or edges + n > max_edges):
            yield ids, pending
            pending, ids, edges = [], [], 0
        pending.append(graph); ids.append(int(q)); edges += n
    if pending:
        yield ids, pending
