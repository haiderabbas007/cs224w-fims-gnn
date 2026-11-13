#!/usr/bin/env python3
"""Nine fresh, resumable one-layer GNN fits on fixed development data only."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = ROOT / 'vendor/recovery/vendor/original/code'
RECOVERY_CODE = ROOT / 'vendor/recovery/code'
for directory in (ORIGINAL_CODE, RECOVERY_CODE):
    sys.path.insert(0, str(directory))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import train_site_gnn_v1 as train
import gnn_adapter_v1 as adapter
from site_data_v1 import SiteData
from memory_model import MemoryBoundedMPNN
from radius_graph import RADII, RadiusProfile, graph_chunks

BASE_IDENTITY = train.make_identity
BASE_TRAIN_BATCH = train.training_batch
BASE_EVALUATE = train.evaluate
MODEL = 'full1'


def radius_tag(radius):
    if isinstance(radius, bool) or float(radius) not in RADII:
        raise ValueError('Radius must be 1, 2, or 4 mm')
    return 'r' + str(int(radius))


def create_model(variant, seed, device='cpu', threads=2):
    if variant != MODEL:
        raise ValueError('This fixed experiment uses only full1')
    adapter.seed_all(seed, threads)
    model = MemoryBoundedMPNN(MODEL, edge_chunk_size=16384).to(device)
    train.require(sum(p.numel() for p in model.parameters()) == 12929,
                  'One-layer parameter count changed')
    return model


def make_identity(data, variant, seed, device, *, study_root, radius):
    train.require(variant == MODEL, 'Wrong architecture')
    protocol, _ = BASE_IDENTITY(data, variant, seed, device)
    own_files = ['code/radius_graph.py', 'code/train_radius.py',
                 'vendor/recovery/code/memory_model.py']
    protocol.update(
        schema='fims_radius_gnn_protocol_v1',
        parent_protocol_family='fims_site_gnn_protocol_v1; fresh fits, no checkpoint migration',
        experiment_sources={name: train.sha(ROOT/name) for name in own_files},
        radius_mm=float(radius), observation_radius_mm=float(radius), connection_radius_mm=float(radius),
        spatial_normalization_mm=2.0, temporal_normalization_bins=7.0,
        history_bins=8, hidden_channels=64, layers=1,
        variants={'full1': 'one message layer, 64 hidden channels, x/y/time relative attributes'},
        input='all same-endcap binary sites within radius_mm of query, bins t-7..t inclusive; occupancy=1',
        graph_definition='query-induced graph, causal distinct-node edges within connection_radius_mm',
        execution='retain all window nodes; execute only incoming query edges, exactly sufficient for one layer; '
                  '16384-edge checkpointed aggregation chunks; no neighbor truncation',
        execution_equivalence='verified against full induced graphs in logits, gradients, and Adam updates; '
                              'floating-point reduction grouping can differ; bitwise historical parity not claimed',
        individual_graph_edge_safety_limit=2000000,
        edge_safety_limit_interpretation='executed query-incoming edges, not the full induced edge inventory',
        comparison='fresh same-runtime 1/2/4 mm radius fits; historical 2 mm run is context only',
        thresholds='descriptive development-validation curves; never fresh confirmation',
    )
    plan = Path(study_root)/'EXPERIMENT_PLAN.json'
    if plan.is_file():
        protocol['experiment_plan_sha256'] = train.sha(plan)
    elif data.manifest.get('synthetic'):
        protocol['experiment_plan_sha256'] = 'synthetic-self-test'
    else:
        raise ValueError('Missing fixed experiment plan')
    digest = hashlib.sha256(train.canonical(protocol)).hexdigest()
    identity = dict(protocol_sha256=digest, model=MODEL, seed=seed,
                    parameters=12929, train_events=sum(e['split']=='train' for e in data.events.values()))
    return protocol, identity


@contextlib.contextmanager
def radius_runtime(study_root, radius, profile=None):
    radius_tag(radius)
    profile = profile if profile is not None else RadiusProfile()
    old = (train.make_identity, train.graph_chunks, adapter.create_model,
           train.training_batch, train.evaluate)
    train.make_identity = lambda d,v,s,dev: make_identity(d,v,s,dev,study_root=study_root,radius=radius)
    train.graph_chunks = lambda a,v,indices=None,max_queries=64,max_edges=250000: graph_chunks(
        a,v,radius,indices,max_queries,max_edges,profile=profile)
    adapter.create_model = create_model
    def training_batch(*args, **kwargs):
        started = time.perf_counter()
        try:
            return BASE_TRAIN_BATCH(*args, **kwargs)
        finally:
            profile.training_batch_seconds += time.perf_counter()-started
            profile.training_batches += 1
    def evaluate(*args, **kwargs):
        started = time.perf_counter()
        try:
            return BASE_EVALUATE(*args, **kwargs)
        finally:
            profile.evaluation_seconds += time.perf_counter()-started
            profile.evaluations += 1
    train.training_batch, train.evaluate = training_batch, evaluate
    try:
        yield profile
    finally:
        (train.make_identity, train.graph_chunks, adapter.create_model,
         train.training_batch, train.evaluate) = old


def append_profile(folder, profile, radius, status, error=None):
    path = folder/'execution_profile.json'
    prior = json.loads(path.read_text()) if path.is_file() else {
        'schema':'fims_radius_execution_profile_v1', 'radius_mm':float(radius), 'sessions':[]}
    train.require(prior['radius_mm'] == float(radius), 'Profile belongs to another radius')
    prior['sessions'].append(dict(utc=datetime.now(timezone.utc).isoformat(), status=status,
                                  error=error, counters=profile.as_dict()))
    train.write_json(path, prior)
    train.write_json(folder/'execution_profile_complete.json',dict(
        schema='fims_radius_execution_profile_receipt_v1',
        execution_profile_sha256=train.sha(path),sessions=len(prior['sessions']),
        latest_session_status=status))


def fit_radius(study_root, data, radius, seed, device):
    output = Path(study_root)/'results/fits'/radius_tag(radius)
    folder = output/MODEL/f'seed{seed}'
    profile = RadiusProfile()
    status, error = 'COMPLETE', None
    try:
        with radius_runtime(study_root, radius, profile):
            train.fit(data, output, MODEL, seed, device)
    except train.Paused as exc:
        status, error = 'PAUSED', str(exc)
        raise
    except BaseException as exc:
        status, error = 'INTERRUPTED_OR_FAILED', str(exc)
        if folder.is_dir() and not (folder/'complete.json').exists():
            train.write_json(folder/'radius_failure.json',dict(status=status,error=error,
                utc=datetime.now(timezone.utc).isoformat(),radius_mm=float(radius),seed=seed))
        raise
    finally:
        if folder.is_dir() and (profile.graph_queries or profile.training_batches or profile.evaluations):
            append_profile(folder, profile, radius, status, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=ROOT)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--radius', type=float, choices=RADII)
    parser.add_argument('--seed', type=int, choices=train.SEEDS)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        from test_radius_training import run_tests
        run_tests(device=args.device)
        return
    train.require(args.data is not None and args.radius is not None and args.seed is not None,
                  '--data, --radius and --seed are required')
    study = args.study_root.resolve()
    from radius_common import verify_package, verify_data_binding
    verify_package(study)
    train.require(study == ROOT, 'Use the installed study code from its own root')
    verify_data_binding(study, args.data)
    device = torch.device(args.device)
    train.require(device.type == 'cuda' and torch.cuda.is_available(),
                  'Real training requires an allocated CUDA GPU; no silent CPU fallback')
    from test_radius_training import runtime_parity_checks
    runtime_parity_checks(device)
    def request_stop(signum, frame):
        train.STOP = True
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    # SiteData verifies every input hash, the fixed split, orders and supports.
    data = SiteData(args.data)
    try:
        fit_radius(study, data, args.radius, args.seed, device)
    except train.Paused:
        raise SystemExit(75)


if __name__ == '__main__':
    main()
