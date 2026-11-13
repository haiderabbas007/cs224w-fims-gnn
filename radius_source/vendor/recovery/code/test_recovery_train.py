"""Invented-event tests for explicit state migration, preserving scientific batches."""
from __future__ import annotations
import copy
import fcntl
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import recovery_train as r
from test_site_pipeline_v1 import fixture


def expect_failure(function, description):
    try: function()
    except (ValueError, RuntimeError): return
    raise AssertionError('Did not reject ' + description)


def test_eight_event_updates(data):
    device = torch.device('cpu')
    weights = torch.tensor(3727159 / (2 * np.asarray([3427421, 299738])), dtype=torch.float32)
    for variant in r.adapter.VARIANTS:
        old = r.ORIGINAL_CREATE_MODEL(variant, 20260924, device)
        new = r.recovery_model(variant, 20260924, device)
        assert r.adapter.parameter_hash(old) == r.adapter.parameter_hash(new)
        old_opt = r.train.make_optimizer(old)
        new_opt = r.train.make_optimizer(new)
        arrays = [data.get(i % 3) for i in range(8)]
        for _ in range(2):
            a = r.train.training_batch(old, old_opt, arrays, weights, 4007.697721890993, device)
            b = r.train.training_batch(new, new_opt, arrays, weights, 4007.697721890993, device)
            for name in a:
                assert np.isclose(a[name], b[name], atol=2e-5, rtol=2e-5), (variant, name)
            for p, q in zip(old.parameters(), new.parameters()):
                assert torch.allclose(p, q, atol=3e-6, rtol=3e-5), variant
        # Recovery itself is exactly repeatable when no implementation changes.
        state = dict(model=r.train.clone(new), optimizer=copy.deepcopy(new_opt.state_dict()),
                     rng=r.train.capture_rng(device))
        r.train.training_batch(new, new_opt, arrays, weights, 4007.697721890993, device)
        resumed = r.recovery_model(variant, 20260924, device)
        resumed.load_state_dict(state['model'], strict=True)
        resumed_opt = r.train.make_optimizer(resumed)
        resumed_opt.load_state_dict(state['optimizer'])
        r.train.restore_rng(state['rng'], device)
        r.train.training_batch(resumed, resumed_opt, arrays, weights, 4007.697721890993, device)
        assert all(torch.equal(x, y) for x, y in zip(new.parameters(), resumed.parameters())), variant


def interrupted_original(data, original, variant, seed):
    real_batch = r.train.training_batch
    calls = [0]
    def batch(*args, **kwargs):
        result = real_batch(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 3: r.train.STOP = True
        return result
    r.train.training_batch = batch
    r.train.STOP = False
    try:
        try: r.train.fit(data, original / 'results/fits', variant, seed, torch.device('cpu'))
        except r.train.Paused: pass
        else: raise AssertionError('Expected synthetic interruption')
    finally:
        r.train.training_batch = real_batch
        r.train.STOP = False


def test_migration(root):
    original, recovery = root / 'original', root / 'recovery'
    original.mkdir(); recovery.mkdir()
    data = fixture(original / 'data')
    variant, seed, device = 'full1', 20260924, torch.device('cpu')
    interrupted_original(data, original, variant, seed)
    source = original / 'results/fits' / variant / f'seed{seed}'
    source_hashes = {p.name: r.sha(p) for p in source.iterdir() if p.is_file()}
    source_state = torch.load(source / 'resume_checkpoint.pt', weights_only=True)
    assert source_state['epoch'] == 2 and source_state['next_batch'] == 1
    assert len(source_state['history']) == 2 and source_state['best_model'] is not None
    old_fit = r.train.fit
    seen = []
    def inspect(d, out, m, s, dev):
        f = out / m / f'seed{s}'
        migrated = torch.load(f / 'resume_checkpoint.pt', weights_only=True)
        assert migrated['identity'] != source_state['identity']
        assert r.state_payload_digest(migrated) == r.state_payload_digest(source_state)
        receipt = json.loads((f / 'migration_receipt.json').read_text())
        assert receipt['changed_fields'] == ['identity']
        assert receipt['unchanged_payload_sha256'] == r.state_payload_digest(source_state)
        assert receipt['migrated_checkpoint_sha256'] == r.sha(f / 'resume_checkpoint.pt')
        seen.append(receipt)
    r.train.fit = inspect
    try:
        r.recover_fit(original, recovery, data, variant, seed, device)
        r.recover_fit(original, recovery, data, variant, seed, device)
    finally: r.train.fit = old_fit
    assert len(seen) == 2 and seen[0] == seen[1]
    target = recovery / 'results/fits' / variant / f'seed{seed}'
    # Corruption in a frozen source snapshot is rejected before a new fit.
    receipt_path = target / 'lineage/source_receipt.json'
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw); receipt['files']['resume_checkpoint.pt'] = '0' * 64
    receipt_path.write_text(json.dumps(receipt))
    expect_failure(lambda: r.recover_fit(original, recovery, data, variant, seed, device), 'snapshot corruption')
    receipt_path.write_bytes(raw)
    # Finish a real tiny fit from the migrated optimizer/RNG state.
    r.recover_fit(original, recovery, data, variant, seed, device)
    before = r.sha(target / 'complete.json')
    r.recover_fit(original, recovery, data, variant, seed, device)
    assert r.sha(target / 'complete.json') == before
    assert all(r.sha(source / n) == h for n, h in source_hashes.items()), 'Original files were changed'

    # A separate target can reuse a fully completed original, byte-for-byte.
    r.train.fit(data, original / 'results/fits', variant, seed, device)
    reuse = root / 'reuse'; reuse.mkdir()
    r.recover_fit(original, reuse, data, variant, seed, device)
    reused = reuse / 'results/fits' / variant / f'seed{seed}'
    marker = json.loads((source / 'complete.json').read_text())
    assert r.sha(source / 'complete.json') == r.sha(reused / 'complete.json')
    assert all(r.sha(source / n) == r.sha(reused / n) for n in marker['outputs'])
    r.recover_fit(original, reuse, data, variant, seed, device)

    # A running source cannot be snapshotted.
    blocked = root / 'blocked'; blocked.mkdir()
    with (source / 'fit.lock').open('r') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        expect_failure(lambda: r.recover_fit(original, blocked, data, variant, seed, device), 'active source')

    # Neither unknown state fields nor changed optimizer settings get migrated.
    identity = source_state['identity']
    model = r.ORIGINAL_CREATE_MODEL(variant, seed, device)
    initial = r.adapter.parameter_hash(model)
    def check(s): r.validate_resume(s, identity, model, initial, 1)
    check(source_state)
    for mutation in ('lr', 'moment', 'best', 'identity', 'progress'):
        bad = copy.deepcopy(source_state)
        if mutation == 'lr': bad['optimizer']['param_groups'][0]['lr'] = .5
        elif mutation == 'moment': next(iter(bad['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
        elif mutation == 'best': next(iter(bad['best_model'].values())).fill_(float('inf'))
        elif mutation == 'identity': bad['identity']['seed'] += 1
        elif mutation == 'progress': bad['epoch'] += 1
        expect_failure(lambda: check(bad), mutation)
    r.validate_environment({'python': 'x', 'gpu_name': 'a', 'gpu_capability': [7, 5]},
                           {'python': 'x', 'gpu_name': 'b', 'gpu_capability': [8, 0]})
    expect_failure(lambda: r.validate_environment({'python': 'x'}, {'python': 'y'}), 'library drift')


def run_tests():
    r.validate_package(r.ROOT / 'vendor/original')
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data = fixture(root / 'toy_data')
        test_eight_event_updates(data)
        test_migration(root)
    print('RECOVERY TRAINER SELF-TEST: PASS (all four 8-event fixed-D update parity; exact patched-runtime resume; '
          'model/Adam/RNG/history preservation; committed migration idempotence; original files preserved; '
          'completed-fit reuse; active-source rejection; state/environment tamper rejection)', flush=True)


if __name__ == '__main__': run_tests()
