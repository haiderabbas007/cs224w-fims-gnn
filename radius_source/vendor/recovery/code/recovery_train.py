#!/usr/bin/env python3
"""Explicit, versioned recovery of the frozen FIMS site-GNN optimizer state.

The original package, event data and fit directories are never overwritten.
Only the aggregation implementation changes; all graph edges, queries, optimizer
event batches, loss normalization and checkpoint-selection rules are retained.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MANIFEST_SHA = '6b4fc507d9df8340fc7c728f1576f6565073f4670dfe6e65f1c763261734045a'
CODE_NAMES = ('train_site_gnn_v1.py', 'graph_models_v1.py', 'gnn_adapter_v1.py',
              'site_data_v1.py', 'vendor/capacity_model_v2.py')
STATE_KEYS = {'identity', 'initialization_sha256', 'epoch', 'next_batch', 'history',
              'stats', 'best_bce', 'best_epoch', 'best_model', 'fit_seconds',
              'model', 'optimizer', 'rng'}
ENV_CHANGE_ALLOWED = {'gpu_name', 'gpu_capability'}
sys.path.insert(0, str(ROOT / 'vendor/original/code'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import train_site_gnn_v1 as train
import gnn_adapter_v1 as adapter
from memory_model import MemoryBoundedMPNN

ORIGINAL_IDENTITY = train.make_identity
ORIGINAL_CREATE_MODEL = adapter.create_model


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha(path):
    return train.sha(path)


def canonical(value):
    return train.canonical(value)


def safe_file(root, name):
    return train.safe_file(Path(root).resolve(), name)


def validate_package(root):
    root = Path(root).resolve()
    manifest = safe_file(root, 'PACKAGE_MANIFEST.json')
    require(sha(manifest) == ORIGINAL_MANIFEST_SHA, 'Original package manifest changed')
    inventory = json.loads(manifest.read_text())
    require(inventory.get('schema') == 'fims_parallel_package_v1', 'Wrong original package')
    for name, digest in inventory['files'].items():
        require(sha(safe_file(root, name)) == digest, 'Original source changed: ' + name)
    return inventory


def state_digest(value):
    """Canonical semantic digest: tensor dtype/shape/bytes, mapping key order neutral."""
    h = hashlib.sha256()

    def put(data):
        h.update(struct.pack('>Q', len(data)))
        h.update(data)

    def visit(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous()
            if a.is_floating_point() or a.is_complex():
                require(bool(torch.isfinite(a).all()), 'Nonfinite checkpoint tensor')
            put(b'tensor'); put(str(a.dtype).encode()); put(canonical(list(a.shape)))
            put(a.view(torch.uint8).numpy().tobytes() if a.ndim else
                a.reshape(1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            put(b'dict')
            for k in sorted(x, key=lambda k: (type(k).__name__, repr(k))):
                visit(k); visit(x[k])
        elif isinstance(x, (tuple, list)):
            put(type(x).__name__.encode()); put(str(len(x)).encode())
            for v in x: visit(v)
        elif x is None:
            put(b'none')
        elif isinstance(x, bool):
            put(b'bool'); put(str(x).encode())
        elif isinstance(x, int):
            put(b'int'); put(str(x).encode())
        elif isinstance(x, float):
            require(math.isfinite(x), 'Nonfinite checkpoint scalar')
            put(b'float'); put(struct.pack('>d', x))
        elif isinstance(x, str):
            put(b'str'); put(x.encode())
        else:
            raise ValueError('Unsupported checkpoint value: ' + str(type(x)))
    visit(value)
    return h.hexdigest()


def state_payload_digest(state):
    return state_digest({k: v for k, v in state.items() if k != 'identity'})


def validate_environment(source, current):
    require(set(source) == set(current), 'Runtime environment field inventory changed')
    for key in source:
        if key not in ENV_CHANGE_ALLOWED:
            require(source[key] == current[key], 'Runtime compatibility mismatch: ' + key)


def recovery_model(variant, seed, device='cpu', threads=2):
    adapter.seed_all(seed, threads)
    model = MemoryBoundedMPNN(variant, edge_chunk_size=16384).to(device)
    model.variant = variant
    require(sum(p.numel() for p in model.parameters()) == adapter.PARAMETERS[variant],
            'Recovery parameter count changed')
    return model


def validate_protocol(protocol, expected_current, data, model, seed, device):
    """Require all original scientific settings and all non-GPU runtime fields."""
    require(protocol.get('schema') == 'fims_site_gnn_protocol_v1', 'Wrong source protocol schema')
    expected = copy.deepcopy(expected_current)
    validate_environment(protocol['environment'], expected['environment'])
    expected['environment'] = protocol['environment']
    require(protocol == expected, 'Original scientific protocol/data/source differs')
    return dict(protocol_sha256=hashlib.sha256(canonical(protocol)).hexdigest(),
                model=model, seed=seed, parameters=adapter.PARAMETERS[model],
                train_events=sum(e['split'] == 'train' for e in data.events.values()))


def validate_resume(state, identity, model, initial_hash, batches):
    require(set(state) == STATE_KEYS, 'Resume checkpoint field inventory changed')
    require(state['identity'] == identity, 'Resume checkpoint/source protocol mismatch')
    require(state['initialization_sha256'] == initial_hash, 'Original initialization differs')
    epoch, batch = state['epoch'], state['next_batch']
    require(isinstance(epoch, int) and isinstance(batch, int) and
            0 <= epoch <= train.EPOCHS and 0 <= batch <= batches, 'Invalid saved progress')
    require(epoch < train.EPOCHS or batch == 0, 'Completed-epoch checkpoint has pending batches')
    require(len(state['history']) == epoch and len(state['stats']) == batch,
            'Checkpoint history/stats do not match progress')
    require([r['epoch'] for r in state['history']] == list(range(1, epoch + 1)),
            'Checkpoint epoch history is incomplete')
    if epoch:
        best = min(state['history'], key=lambda r: r['validation_class_balanced_bce'])
        require(state['best_epoch'] == best['epoch'] and
                state['best_bce'] == best['validation_class_balanced_bce'],
                'Checkpoint selection history changed')
        require(state['best_model'] is not None, 'Selected checkpoint weights missing')
        model.load_state_dict(state['best_model'], strict=True)
    else:
        require(state['best_epoch'] is None and state['best_bce'] is None and
                state['best_model'] is None, 'Unvalidated epoch has selected weights')
    model.load_state_dict(state['model'], strict=True)
    require(all(torch.isfinite(v).all() for v in state['model'].values()), 'Nonfinite saved weights')
    optimizer = train.make_optimizer(model)
    expected_optimizer = optimizer.state_dict()
    saved_optimizer = state['optimizer']
    require(set(saved_optimizer) == {'state', 'param_groups'} and
            saved_optimizer['param_groups'] == expected_optimizer['param_groups'],
            'Saved Adam hyperparameters or parameter ordering changed')
    params = list(model.parameters())
    require(set(saved_optimizer['state']).issubset(set(range(len(params)))),
            'Saved Adam parameter identity changed')
    for index, values in saved_optimizer['state'].items():
        require(set(values) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Invalid Adam state inventory')
        step = values['step']
        require(torch.is_tensor(step) and step.numel() == 1 and
                bool(torch.isfinite(step).all()) and float(step) >= 0 and
                float(step) == int(float(step)), 'Invalid Adam step count')
        for name in ('exp_avg', 'exp_avg_sq'):
            tensor = values[name]
            require(torch.is_tensor(tensor) and tensor.shape == params[index].shape and
                    tensor.dtype == params[index].dtype and bool(torch.isfinite(tensor).all()),
                    'Invalid Adam moment tensor: ' + name)
        require(bool((values['exp_avg_sq'] >= 0).all()), 'Negative Adam second moment')
    optimizer.load_state_dict(state['optimizer'])
    current_rng = train.capture_rng(next(model.parameters()).device)
    try:
        train.restore_rng(state['rng'], next(model.parameters()).device)
    finally:
        train.restore_rng(current_rng, next(model.parameters()).device)
    require(state['fit_seconds'] >= 0, 'Invalid saved fit duration')
    # Digest also validates finite scalar metadata and supported serialization.
    state_payload_digest(state)


def immutable_copy(source, dest):
    raw = source.read_bytes()
    train.atomic_bytes(dest, raw, immutable=True)
    require(sha(source) == sha(dest), 'Source copy changed during snapshot')


def checked_complete(folder):
    p = safe_file(folder, 'complete.json')
    marker = json.loads(p.read_text())
    require(marker.get('status') == 'COMPLETE', 'Invalid complete marker')
    require(train.verify_complete(folder, marker['identity']), 'Complete verification failed')
    protocol = json.loads(safe_file(folder, 'protocol.json').read_text())
    require(marker['identity']['protocol_sha256'] == sha(folder / 'protocol.json'),
            'Completed protocol identity differs')
    meta = json.loads(safe_file(folder, 'metrics.json').read_text())
    require(meta['identity'] == marker['identity'], 'Completed metrics identity differs')
    return marker, protocol


def snapshot_source(original, out, data, variant, seed, device, expected_protocol, initial_hash):
    """Commit a source snapshot once. Original fit lock prevents active snapshots."""
    lineage = out / 'lineage'
    receipt_path = lineage / 'source_receipt.json'
    if receipt_path.exists():
        receipt = json.loads(safe_file(lineage, 'source_receipt.json').read_text())
        for name, digest in receipt['files'].items():
            require(sha(safe_file(lineage, name)) == digest, 'Saved lineage snapshot changed')
        require(receipt['original_study'] == str(original) and receipt['model'] == variant and
                receipt['seed'] == seed and receipt['original_package_sha256'] == ORIGINAL_MANIFEST_SHA,
                'Saved lineage belongs to another source')
        return receipt
    require(not lineage.exists(), 'Incomplete lineage directory; do not overwrite it')
    source = original / 'results/fits' / variant / f'seed{seed}'
    require(source.is_dir() and not source.is_symlink(), 'Original fit folder is missing/unsafe')
    lock_path = source / 'fit.lock'
    require(lock_path.is_file() and not lock_path.is_symlink(), 'Original fit lock is missing/unsafe')
    # Open existing lock without creating or changing the old directory.
    with lock_path.open('r') as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Original fit is still active; recovery must wait') from exc
        protocol = json.loads(safe_file(source, 'protocol.json').read_text())
        identity = validate_protocol(protocol, expected_protocol, data, variant, seed, device)
        require(sha(source / 'protocol.json') == identity['protocol_sha256'],
                'Source protocol is not the exact canonical protocol')
        complete = (source / 'complete.json').exists()
        names = ['protocol.json']
        if complete:
            marker, _ = checked_complete(source)
            require(marker['identity'] == identity, 'Completed source identity differs')
            names += list(marker['outputs']) + ['complete.json']
            mode = 'completed_source_reuse'
            payload_digest = None
            progress = None
        else:
            checkpoint = safe_file(source, 'resume_checkpoint.pt')
            before = sha(checkpoint)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model = ORIGINAL_CREATE_MODEL(variant, seed, device)
            validate_resume(state, identity, model, initial_hash,
                            data.settings['optimizer_batches_per_epoch'])
            require(sha(checkpoint) == before, 'Source checkpoint changed during locked snapshot')
            names += ['resume_checkpoint.pt']
            mode = 'migrated_optimizer_state'
            payload_digest = state_payload_digest(state)
            progress = dict(epoch=state['epoch'], next_batch=state['next_batch'])
            del model, state
        stage = Path(tempfile.mkdtemp(prefix='.lineage.', dir=out))
        inventory = {}
        for name in sorted(set(names)):
            immutable_copy(safe_file(source, name), stage / name)
            inventory[name] = sha(stage / name)
        receipt = dict(schema='fims_gnn_recovery_source_v1', mode=mode,
                       original_study=str(original), original_fit=str(source),
                       original_package_sha256=ORIGINAL_MANIFEST_SHA,
                       model=variant, seed=seed, original_identity=identity,
                       files=inventory, preserved_state_sha256=payload_digest,
                       progress=progress,
                       note='Original fit lock held through snapshot. Original files not overwritten.')
        train.write_json(stage / 'source_receipt.json', receipt, immutable=True)
        os.rename(stage, lineage)
        return receipt


def migrated_protocol(original_protocol, receipt, receipt_sha, current_protocol):
    protocol = copy.deepcopy(original_protocol)
    protocol['environment'] = current_protocol['environment']
    protocol['runtime_recovery'] = dict(
        schema='fims_gnn_memory_recovery_v1',
        source_receipt_sha256=receipt_sha,
        parent_protocol_sha256=receipt['original_identity']['protocol_sha256'],
        source_checkpoint_sha256=receipt['files']['resume_checkpoint.pt'],
        preserved_state_sha256=receipt['preserved_state_sha256'],
        code={n: sha(ROOT / 'code' / n) for n in ('recovery_train.py', 'memory_model.py')},
        edge_chunk_size=16384,
        operation='Nonreentrant checkpointed message chunks; full destination sum; no graph truncation',
        unchanged='All graph nodes/edges/queries; parameter tensors; Adam state; RNG; progress; history; loss; optimizer batches',
        arithmetic='Mathematically equivalent aggregation; floating-point grouping can change at roundoff',
        runtime='GPU model may differ; compatible library/build/runtime fields required; each session recorded')
    return protocol


def recover_fit(original, recovery, data, variant, seed, device):
    out = recovery / 'results/fits' / variant / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    # This outer lock spans setup AND the original trainer's inner fit lock.
    with (out / 'recovery.lock').open('a') as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This recovery fit is already active') from exc
        reference = ORIGINAL_CREATE_MODEL(variant, seed, device)
        initial_hash = adapter.parameter_hash(reference)
        current_protocol, _ = ORIGINAL_IDENTITY(data, variant, seed, device)
        del reference
        receipt = snapshot_source(original, out, data, variant, seed, device,
                                  current_protocol, initial_hash)
        lineage = out / 'lineage'
        original_protocol = json.loads((lineage / 'protocol.json').read_text())
        original_identity = validate_protocol(original_protocol, current_protocol, data, variant, seed, device)
        require(receipt['original_identity'] == original_identity, 'Saved lineage source identity changed')
        if receipt['mode'] == 'completed_source_reuse':
            # Copy the verified original protocol/predictions/checkpoints unchanged.
            for name in receipt['files']:
                if name != 'complete.json': immutable_copy(lineage / name, out / name)
            immutable_copy(lineage / 'complete.json', out / 'complete.json')
            checked_complete(out)
            train.write_json(out / 'recovery_receipt.json', dict(
                status='REUSED_COMPLETE', source_receipt_sha256=sha(lineage / 'source_receipt.json'),
                no_training=True, original_identity=receipt['original_identity']), immutable=True)
            train.write_json(out / 'status.json', dict(status='COMPLETE', reused_original=True,
                                                     **receipt['original_identity']))
            print(f'{variant} seed={seed}: completed original fit verified and reused; no training', flush=True)
            return
        expected = migrated_protocol(original_protocol, receipt,
                                     sha(lineage / 'source_receipt.json'), current_protocol)
        protocol_path = out / 'protocol.json'
        if protocol_path.exists():
            protocol = json.loads(safe_file(out, 'protocol.json').read_text())
            validate_environment(protocol['environment'], current_protocol['environment'])
            expected['environment'] = protocol['environment']
            require(protocol == expected, 'Recovery protocol or implementation changed')
        else:
            protocol = expected
        identity = dict(receipt['original_identity'],
                        protocol_sha256=hashlib.sha256(canonical(protocol)).hexdigest())
        train.write_json(protocol_path, protocol, immutable=True)
        migration_path = out / 'migration_receipt.json'
        resume = out / 'resume_checkpoint.pt'
        if not migration_path.exists():
            state = torch.load(lineage / 'resume_checkpoint.pt', map_location='cpu', weights_only=True)
            verifier_model = ORIGINAL_CREATE_MODEL(variant, seed, device)
            validate_resume(state, original_identity, verifier_model, initial_hash,
                            data.settings['optimizer_batches_per_epoch'])
            del verifier_model
            require(state_payload_digest(state) == receipt['preserved_state_sha256'],
                    'Preserved source state changed')
            state['identity'] = identity
            require(state_payload_digest(state) == receipt['preserved_state_sha256'],
                    'Migration altered model/optimizer/RNG/progress/history')
            if resume.exists():
                existing = torch.load(resume, map_location='cpu', weights_only=True)
                require(existing['identity'] == identity and
                        state_payload_digest(existing) == receipt['preserved_state_sha256'],
                        'Uncommitted migration checkpoint differs; refusing reset')
            else:
                train.atomic_torch(resume, state)
            train.write_json(migration_path, dict(
                schema='fims_gnn_checkpoint_migration_v1', status='VERIFIED',
                parent_identity=receipt['original_identity'], recovery_identity=identity,
                source_receipt_sha256=sha(lineage / 'source_receipt.json'),
                source_checkpoint_sha256=receipt['files']['resume_checkpoint.pt'],
                migrated_checkpoint_sha256=sha(resume),
                unchanged_payload_sha256=receipt['preserved_state_sha256'],
                changed_fields=['identity'], progress=receipt['progress']), immutable=True)
        migration = json.loads(safe_file(out, 'migration_receipt.json').read_text())
        require(migration['recovery_identity'] == identity and
                migration['source_receipt_sha256'] == sha(lineage / 'source_receipt.json') and
                migration['unchanged_payload_sha256'] == receipt['preserved_state_sha256'],
                'Migration receipt differs')
        require(resume.is_file(), 'Migrated/resumed checkpoint is missing; refusing to restart fit')
        sessions = out / 'runtime_sessions'
        sessions.mkdir(exist_ok=True)
        session = dict(environment=current_protocol['environment'], protocol_sha256=identity['protocol_sha256'],
                       utc=datetime.now(timezone.utc).isoformat(),
                       process_id=os.getpid(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                       slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'))
        fd, filename = tempfile.mkstemp(prefix='session_', suffix='.json', dir=sessions)
        with os.fdopen(fd, 'wb') as f: f.write(canonical(session)); f.flush(); os.fsync(f.fileno())
        # Stable identity permits an explicitly recorded GPU-card change, never a
        # silent library, graph, loss, optimizer or implementation change.
        old_factory, old_identity = adapter.create_model, train.make_identity
        adapter.create_model = recovery_model
        train.make_identity = lambda d, m, s, dev: (protocol, identity)
        try:
            probe = recovery_model(variant, seed, device)
            require(adapter.parameter_hash(probe) == initial_hash, 'Recovery initialization changed')
            del probe
            train.fit(data, recovery / 'results/fits', variant, seed, device)
        finally:
            adapter.create_model, train.make_identity = old_factory, old_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-study', type=Path)
    p.add_argument('--recovery-root', type=Path, default=ROOT)
    p.add_argument('--model', choices=adapter.VARIANTS)
    p.add_argument('--seed', type=int, choices=train.SEEDS)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        import test_recovery_train
        test_recovery_train.run_tests()
        return
    require(a.original_study is not None and a.model is not None and a.seed is not None,
            '--original-study, --model and --seed are required')
    recovery, original = a.recovery_root.resolve(), a.original_study.resolve()
    require(recovery == ROOT and original != recovery, 'Unexpected recovery/original directory')
    validate_package(original)
    validate_package(ROOT / 'vendor/original')
    device = torch.device(a.device)
    require(device.type == 'cuda' and torch.cuda.is_available(),
            'Real recovery requires an allocated CUDA GPU; no CPU fallback')
    def request_stop(signum, frame): train.STOP = True
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1): signal.signal(sig, request_stop)
    data = train.SiteData(original / 'data')
    try: recover_fit(original, recovery, data, a.model, a.seed, device)
    except train.Paused: raise SystemExit(75)


if __name__ == '__main__':
    main()
