#!/usr/bin/env python3
"""Synthetic, independently enumerated checks for the fixed site GNN contract."""
from __future__ import annotations

import unittest
import os
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F

from graph_models_v1 import (
    GEOMETRY_KEYS, VARIANTS, GeometricMPNN, GraphLimitError,
    QueryGraphBuilder, batch_graphs, build_query_graph, parameter_count,
)


def geometry(x, y=None, time=None, side=None):
    n = len(x)
    return dict(x_mm=np.asarray(x, dtype=np.float64),
                y_mm=np.asarray(np.zeros(n) if y is None else y, dtype=np.float64),
                time_bin=np.asarray(np.zeros(n) if time is None else time, dtype=np.int64),
                side=np.asarray(np.zeros(n) if side is None else side, dtype=np.int8))


def scalar_reference(g, q, variant):
    """Explicit scalar membership/edge loops; no KD tree or encoder calls."""
    history = VARIANTS[variant]["history_bins"]
    nodes = []
    for i in range(len(g["x_mm"])):
        if int(g["side"][i]) != int(g["side"][q]):
            continue
        if not int(g["time_bin"][q])-history+1 <= int(g["time_bin"][i]) <= int(g["time_bin"][q]):
            continue
        dx = float(g["x_mm"][i])-float(g["x_mm"][q])
        dy = float(g["y_mm"][i])-float(g["y_mm"][q])
        if dx*dx+dy*dy <= 4.0:
            nodes.append(i)
    nodes.sort(key=lambda i: (int(g["time_bin"][i]), float(g["x_mm"][i]), float(g["y_mm"][i])))
    edges = []
    attrs = []
    for i, native_i in enumerate(nodes):
        for j, native_j in enumerate(nodes):
            if i == j or int(g["time_bin"][native_j]) > int(g["time_bin"][native_i]):
                continue
            dx = float(g["x_mm"][native_j])-float(g["x_mm"][native_i])
            dy = float(g["y_mm"][native_j])-float(g["y_mm"][native_i])
            if dx*dx+dy*dy > 4.0:
                continue
            edges.append((j, i))
            attrs.append((dx/2 if VARIANTS[variant]["spatial_attributes"] else 0,
                          dy/2 if VARIANTS[variant]["spatial_attributes"] else 0,
                          (int(g["time_bin"][native_j])-int(g["time_bin"][native_i]))/7))
    return np.asarray(nodes), np.asarray(edges, np.int64).reshape(-1, 2).T, np.asarray(attrs, np.float32).reshape(-1, 3)


class GraphContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        torch.manual_seed(177)
        self.g = geometry([0, 2, 2.000001, -1, 0.1, 0.2, 0.3, 0.4],
                          y=[0, 0, 0, 0.2, 0, 0.1, 0.1, 0.2],
                          time=[10, 10, 10, 3, 2, 11, 9, 10],
                          side=[0, 0, 0, 0, 0, 0, 1, 0])

    def test_scalar_graph_radius_time_and_side(self):
        for variant in VARIANTS:
            builder = QueryGraphBuilder(self.g, variant)
            for q in range(len(builder)):
                got = builder.build(q)
                nodes, edges, attrs = scalar_reference(self.g, q, variant)
                np.testing.assert_array_equal(got.observation_indices.numpy(), nodes)
                np.testing.assert_array_equal(got.edge_index.numpy(), edges)
                np.testing.assert_array_equal(got.edge_attr.numpy(), attrs)
                self.assertEqual(int(got.query_index[0]), list(nodes).index(q))
                self.assertTrue(torch.all(got.x == 1))
        full = build_query_graph(self.g, 0)
        self.assertEqual(set(full.observation_indices.tolist()), {0, 1, 3, 7})
        current = build_query_graph(self.g, 0, "current2")
        self.assertEqual(set(current.observation_indices.tolist()), {0, 1, 7})

    def test_input_permutation_and_translation(self):
        perm = np.array([5, 2, 7, 3, 4, 1, 6, 0])
        shuffled = {k: v[perm] for k, v in self.g.items()}
        translated = {k: v.copy() for k, v in self.g.items()}
        translated["x_mm"] += 16
        translated["y_mm"] -= 8
        translated["time_bin"] += 500
        for variant in VARIANTS:
            a = build_query_graph(self.g, 0, variant)
            b = build_query_graph(shuffled, int(np.flatnonzero(perm == 0)[0]), variant)
            c = build_query_graph(translated, 0, variant)
            for other in (b, c):
                torch.testing.assert_close(a.edge_index, other.edge_index, rtol=0, atol=0)
                torch.testing.assert_close(a.edge_attr, other.edge_attr, rtol=0, atol=1e-6)
            model = GeometricMPNN(variant)
            with torch.no_grad():
                expected = model(batch_graphs([a]))
                torch.testing.assert_close(expected, model(batch_graphs([b])), rtol=0, atol=0)
                torch.testing.assert_close(expected, model(batch_graphs([c])), rtol=1e-6, atol=1e-7)

    def test_query_window_is_bounded_at_two_layers(self):
        base = geometry([0.0, 1.5], time=[20, 19])
        # This point could reach q in two steps in a full-event graph but lies
        # outside q's own 2 mm radius, so it must not enter this computation.
        added = geometry([0.0, 1.5, 3.0, 0.2, 0.3], time=[20, 19, 18, 21, 18], side=[0, 0, 0, 0, 1])
        model = GeometricMPNN("full2")
        a, b = build_query_graph(base, 0), build_query_graph(added, 0)
        self.assertEqual(a.num_nodes, b.num_nodes)
        torch.testing.assert_close(model(batch_graphs([a])), model(batch_graphs([b])), rtol=0, atol=0)

    def test_forward_agrees_with_explicit_message_sum(self):
        graph = build_query_graph(self.g, 0)
        model = GeometricMPNN("full2")
        h = graph.x
        for layer in model.layers:
            sums = torch.zeros((len(h), 64), dtype=h.dtype)
            for e in range(graph.edge_index.shape[1]):
                j, i = graph.edge_index[:, e]
                sums[i] = sums[i] + layer.phi(torch.cat((h[j], graph.edge_attr[e])))
            h = layer.psi(torch.cat((h, sums), dim=1))
        expected = model.output(h[graph.query_index]).reshape(-1)
        torch.testing.assert_close(model(batch_graphs([graph])), expected, rtol=1e-5, atol=1e-6)

    def test_batch_query_offsets_and_gradient_parity(self):
        builder = QueryGraphBuilder(self.g)
        graphs = [builder.build(q) for q in (0, 1, 6)]
        model = GeometricMPNN("full2")
        batch = batch_graphs(graphs)
        offsets = np.cumsum([0] + [g.num_nodes for g in graphs[:-1]])
        self.assertEqual(batch.query_index.tolist(), [int(g.query_index)+int(off) for g, off in zip(graphs, offsets)])
        batched = model(batch)
        single = torch.cat([model(batch_graphs([g])) for g in graphs])
        torch.testing.assert_close(batched, single, rtol=1e-5, atol=1e-6)
        target = torch.tensor([1.0, 0.0, 1.0])
        F.binary_cross_entropy_with_logits(batched, target, reduction="sum").backward()
        full_grads = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        for graph, y in zip(graphs, target):
            F.binary_cross_entropy_with_logits(model(batch_graphs([graph])), y[None], reduction="sum").backward()
        for full, p in zip(full_grads, model.parameters()):
            torch.testing.assert_close(full, p.grad, rtol=1e-4, atol=1e-6)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(sum(float(g.abs().sum()) for g in full_grads), 0)

    def test_no_xy_forward_mask_and_same_parameter_count(self):
        graph = build_query_graph(self.g, 0)
        model = GeometricMPNN("no_xy2")
        torch.manual_seed(177)
        full = GeometricMPNN("full2")
        self.assertEqual(parameter_count(model), parameter_count(full))
        before = model(batch_graphs([graph]))
        graph.edge_attr[:, :2] *= 123
        after = model(batch_graphs([graph]))
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        # Audit metadata is not a model feature.
        graph.observation_indices += 1000
        torch.testing.assert_close(before, model(batch_graphs([graph])), rtol=0, atol=0)

    def test_all_variants_have_finite_isolated_predictions(self):
        for variant in VARIANTS:
            one = build_query_graph(geometry([0.0]), 0, variant)
            self.assertEqual(one.edge_index.shape, (2, 0))
            model = GeometricMPNN(variant)
            self.assertTrue(torch.isfinite(model(batch_graphs([one]))).all())
            self.assertEqual(tuple(model(batch_graphs([])).shape), (0,))
        with self.assertRaises(IndexError):
            QueryGraphBuilder(geometry([])).build(0)

    def test_no_truth_fields_duplicates_or_timestamp_rounding(self):
        with self.assertRaises(ValueError):
            QueryGraphBuilder({**self.g, "target": np.zeros(len(self.g["side"]))})
        with self.assertRaises(ValueError):
            QueryGraphBuilder(geometry([0.0, 0.0]))
        with self.assertRaises(ValueError):
            QueryGraphBuilder({**self.g, "time_bin": self.g["time_bin"].astype(float)})
        with self.assertRaises(ValueError):
            QueryGraphBuilder({**self.g, "side": np.full(len(self.g["side"]), 2)})

    def test_limits_fail_without_changing_graphs(self):
        with self.assertRaises(GraphLimitError):
            QueryGraphBuilder(self.g, max_nodes=2).build(0)
        with self.assertRaises(GraphLimitError):
            QueryGraphBuilder(self.g, max_edges=1).build(0)
        a = QueryGraphBuilder(self.g).build(0)
        b = QueryGraphBuilder(self.g, max_nodes=a.num_nodes, max_edges=a.num_edges).build(0)
        torch.testing.assert_close(a.edge_index, b.edge_index, rtol=0, atol=0)

    def test_builder_copies_input_and_stable_cpu_replay(self):
        builder = QueryGraphBuilder(self.g)
        expected = builder.build(0)
        self.g["x_mm"][:] = 900
        torch.testing.assert_close(expected.edge_attr, builder.build(0).edge_attr, rtol=0, atol=0)
        torch.manual_seed(91)
        a = GeometricMPNN("full2")
        torch.manual_seed(91)
        b = GeometricMPNN("full2")
        torch.testing.assert_close(a(batch_graphs([expected])), b(batch_graphs([expected])), rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable on this test host")
class CUDAContractTests(unittest.TestCase):
    def test_deterministic_cuda_forward_backward_all_variants(self):
        # Run this test on the allocated GPU before scientific training. It
        # explicitly checks the installed PyTorch/PyG deterministic sum path.
        torch.use_deterministic_algorithms(True)
        g = geometry([0.0, 0.5, 1.0, 1.5], y=[0, 0.2, 0.4, 0.1], time=[8, 7, 8, 1])
        for variant in VARIANTS:
            builder = QueryGraphBuilder(g, variant)
            batch = batch_graphs([builder.build(i) for i in range(4)], "cuda")
            torch.manual_seed(71)
            model = GeometricMPNN(variant).to("cuda")
            before = model(batch)
            repeated = model(batch)
            torch.testing.assert_close(before, repeated, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(before).all())
            F.binary_cross_entropy_with_logits(before, torch.tensor([1, 0, 1, 0],
                                               dtype=torch.float32, device="cuda")).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
