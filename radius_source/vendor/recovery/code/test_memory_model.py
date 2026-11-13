#!/usr/bin/env python3
"""Parity tests and optional allocated-GPU memory stress for the recovery.

Put the verified original code directory on PYTHONPATH. Run this file normally
for CPU parity, or add --cuda-stress inside a GPU allocation for CUDA tests.
The GPU stress uses all 2,000,810 directed edges of a dense synthetic graph.
It never loads detector observations or labels.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch_geometric.data import Batch

from graph_models_v1 import (GeometricMPNN as OriginalMPNN, QueryGraphBuilder,
                             QuerySiteData, VARIANTS, batch_graphs)
from memory_model import MemoryBoundedMPNN


def small_batch(variant, dtype=torch.float32, device="cpu"):
    # Distinct observed sites include current-bin ties, history, another endcap,
    # and isolated queries. No truth enters this geometric fixture.
    g = dict(x_mm=np.array([0., .1, .2, .4, .8, 1.4, 1.9, 3.8, 8., 0.]),
             y_mm=np.array([0., .1, -.1, .2, -.3, .6, 0., 0., 0., 0.]),
             time_bin=np.array([7, 7, 6, 5, 3, 7, 2, 7, 7, 7], np.int64),
             side=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], np.int8))
    builder = QueryGraphBuilder(g, variant)
    batch = batch_graphs([builder.build(q) for q in (0, 5, 7, 8, 9)], device)
    batch.x = batch.x.to(dtype=dtype)
    batch.edge_attr = batch.edge_attr.to(dtype=dtype)
    return batch


def assert_close(a, b, dtype, *, label="tensor"):
    rtol, atol = (3e-5, 3e-6) if dtype == torch.float32 else (2e-11, 2e-12)
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=label)


class MemoryModelParity(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_seed_state_keys_shapes_and_parameter_count(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(27031)
                old = OriginalMPNN(variant)
                old_rng = torch.get_rng_state().clone()
                torch.manual_seed(27031)
                new = MemoryBoundedMPNN(variant)
                self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
                self.assertEqual(list(old.state_dict()), list(new.state_dict()))
                self.assertEqual(sum(p.numel() for p in old.parameters()),
                                 sum(p.numel() for p in new.parameters()))
                for key, tensor in old.state_dict().items():
                    self.assertTrue(torch.equal(tensor, new.state_dict()[key]), key)
                old.load_state_dict(new.state_dict(), strict=True)
                new.load_state_dict(old.state_dict(), strict=True)

    def test_all_variants_float32_float64_logits_and_gradients(self):
        for variant in VARIANTS:
            for dtype in (torch.float32, torch.float64):
                for chunk in (1, 7, 16384):
                    with self.subTest(variant=variant, dtype=dtype, chunk=chunk):
                        torch.manual_seed(341)
                        old = OriginalMPNN(variant).to(dtype=dtype)
                        new = MemoryBoundedMPNN(variant, edge_chunk_size=chunk).to(dtype=dtype)
                        new.load_state_dict(old.state_dict(), strict=True)
                        a = small_batch(variant, dtype)
                        b = copy.deepcopy(a)
                        for batch in (a, b):
                            batch.x.requires_grad_(True)
                            batch.edge_attr.requires_grad_(True)
                        old_logits, new_logits = old(a), new(b)
                        assert_close(old_logits, new_logits, dtype, label="query logits")
                        coefficients = torch.tensor([.31, -.72, .9, .45, -.13], dtype=dtype)
                        (old_logits * coefficients).sum().backward()
                        (new_logits * coefficients).sum().backward()
                        assert_close(a.x.grad, b.x.grad, dtype, label="node gradients")
                        assert_close(a.edge_attr.grad, b.edge_attr.grad, dtype, label="edge gradients")
                        for (name, p), (name2, q) in zip(old.named_parameters(), new.named_parameters()):
                            self.assertEqual(name, name2)
                            if p.grad is None:
                                self.assertIsNone(q.grad, name)
                            else:
                                assert_close(p.grad, q.grad, dtype, label="parameter gradient " + name)

    def test_constant_input_first_layer_parameters_receive_gradient(self):
        torch.manual_seed(910)
        model = MemoryBoundedMPNN("full2", edge_chunk_size=3)
        a = small_batch("full2")
        self.assertFalse(a.x.requires_grad)
        self.assertFalse(a.edge_attr.requires_grad)
        model(a).sum().backward()
        self.assertIsNotNone(model.layers[0].phi[0].weight.grad)
        self.assertGreater(float(model.layers[0].phi[0].weight.grad.abs().sum()), 0.)

    def test_no_grad_and_empty_graphs(self):
        for variant in VARIANTS:
            torch.manual_seed(51)
            old = OriginalMPNN(variant)
            new = MemoryBoundedMPNN(variant, edge_chunk_size=2)
            new.load_state_dict(old.state_dict())
            a = small_batch(variant)
            with torch.no_grad():
                assert_close(old(a), new(a), torch.float32)
                self.assertEqual(tuple(new(batch_graphs([])).shape), (0,))
            # A lone node has no edges: phi has no role and its gradients stay
            # absent, matching the original message-passing layer.
            lone = batch_graphs([QuerySiteData(x=torch.ones((1, 1)),
                       edge_index=torch.empty((2, 0), dtype=torch.long),
                       edge_attr=torch.empty((0, 3)), query_index=torch.tensor([0]))])
            old.zero_grad(set_to_none=True); new.zero_grad(set_to_none=True)
            oa, na = old(lone), new(lone)
            assert_close(oa, na, torch.float32)
            oa.sum().backward(); na.sum().backward()
            for p, q in zip(old.parameters(), new.parameters()):
                if p.grad is None:
                    self.assertIsNone(q.grad)
                else:
                    assert_close(p.grad, q.grad, torch.float32)

    def test_chunk_validation(self):
        for value in (0, -1, True, 1.5, "12"):
            with self.assertRaises(ValueError):
                MemoryBoundedMPNN(edge_chunk_size=value)

    def test_checkpoint_does_not_retain_all_edge_activations(self):
        # Structural CPU memory check, independent of CUDA allocator behavior:
        # count unique saved intermediate storages, excluding immutable graph
        # inputs and learned parameters. index_add outside the checkpoint would
        # fail this test because its backward saves all chunk source messages.
        n = 100
        source = torch.arange(n).repeat_interleave(n)
        target = torch.arange(n).repeat(n)
        keep = source != target
        edges = torch.stack((source[keep], target[keep]))
        batch = Batch(x=torch.ones((n, 1)), edge_index=edges,
                      edge_attr=torch.zeros((edges.shape[1], 3)),
                      query_index=torch.tensor([n - 1]))

        def retained_bytes(model):
            excluded = {t.untyped_storage().data_ptr() for t in
                        [batch.x, batch.edge_index, batch.edge_attr, batch.query_index,
                         *model.parameters()]}
            saved = {}
            def pack(t):
                storage = t.untyped_storage()
                if storage.data_ptr() not in excluded:
                    saved[storage.data_ptr()] = storage.nbytes()
                return t
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                logits = model(batch)
            logits.sum().backward()
            return sum(saved.values())

        torch.manual_seed(251)
        old = OriginalMPNN("full2")
        new = MemoryBoundedMPNN("full2", edge_chunk_size=128)
        new.load_state_dict(old.state_dict())
        before, after = retained_bytes(old), retained_bytes(new)
        self.assertLess(after, before / 3,
                        f"Checkpoint retains too much edge activation storage: {before=} {after=}")
        print(f"Saved intermediate storage: original={before:,} bytes; chunked={after:,} bytes")


def cuda_stress():
    if not torch.cuda.is_available():
        raise RuntimeError("--cuda-stress requires an allocated CUDA GPU; no CPU pass substitutes for it")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reports = []
    # First verify actual CUDA forward/backward parity at practical tolerance.
    for variant in VARIANTS:
        torch.manual_seed(412)
        old = OriginalMPNN(variant).cuda()
        new = MemoryBoundedMPNN(variant, edge_chunk_size=7).cuda()
        new.load_state_dict(old.state_dict())
        a = small_batch(variant, device="cuda")
        old_score, new_score = old(a), new(a)
        assert_close(old_score, new_score, torch.float32, label="CUDA query logits")
        old_score.sum().backward(); new_score.sum().backward()
        for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
            if p.grad is None:
                if q.grad is not None: raise AssertionError("Unexpected CUDA gradient: " + name)
            else: assert_close(p.grad, q.grad, torch.float32, label="CUDA gradient " + name)
        del old, new, a, old_score, new_score
    torch.cuda.empty_cache()
    n = 1415
    source = torch.arange(n, dtype=torch.long).repeat_interleave(n)
    target = torch.arange(n, dtype=torch.long).repeat(n)
    keep = source != target
    edges = torch.stack((source[keep], target[keep])).to("cuda")
    edge_count = int(edges.shape[1])
    assert edge_count == 2_000_810
    # Equal-time, all-to-all synthetic message graph: no edge is discarded.
    dense = Batch(x=torch.ones((n, 1), device="cuda"), edge_index=edges,
                  edge_attr=torch.zeros((edge_count, 3), device="cuda"),
                  query_index=torch.tensor([0, n-1], device="cuda"))
    for variant in VARIANTS:
        torch.manual_seed(19)
        model = MemoryBoundedMPNN(variant).cuda()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        logits = model(dense)
        loss = logits.square().mean() / 1e6
        loss.backward()
        torch.cuda.synchronize()
        if not bool(torch.isfinite(logits).all()): raise AssertionError("Nonfinite dense CUDA scores")
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise AssertionError("Nonfinite dense CUDA gradients")
        peak = torch.cuda.max_memory_allocated()
        # This is an allocation test, not a device-power/throughput claim.
        # 2 GiB is comfortably below the affected 12-GiB hardware capacity.
        if peak - baseline >= 2 * 1024**3:
            raise AssertionError(f"Chunked dense graph exceeds 2 GiB activation budget: {peak-baseline}")
        reports.append(dict(variant=variant, nodes=n, edges=edge_count,
                            edge_chunk_size=model.edge_chunk_size,
                            peak_allocated_bytes=peak, baseline_allocated_bytes=baseline,
                            additional_peak_bytes=peak-baseline))
        del model, logits, loss
        torch.cuda.empty_cache()
    print("CUDA MEMORY STRESS: PASS " + json.dumps(reports), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-stress", action="store_true")
    args, remaining = parser.parse_known_args()
    program = unittest.main(argv=[sys.argv[0], *remaining], exit=False, verbosity=2)
    if not program.result.wasSuccessful(): raise SystemExit(1)
    if args.cuda_stress: cuda_stress()
    else: print("CUDA stress not run: invoke --cuda-stress inside an allocated GPU job.")
