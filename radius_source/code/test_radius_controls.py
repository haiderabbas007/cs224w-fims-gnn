#!/usr/bin/env python3
"""Synthetic-only regression checks for the radius-matched count/shape controls."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np

import control_radius as controls
from site_data_v1 import atomic_npz, geometry_only, sha


def geometry(x, y, t, side):
    return dict(x_mm=np.asarray(x, np.float64), y_mm=np.asarray(y, np.float64),
                time_bin=np.asarray(t, np.int64), side=np.asarray(side, np.int8))


def reference(g, radius, window):
    """Independent all-pairs scalar membership and covariance-eigenvalue reference."""
    n = len(g['x_mm'])
    out = np.zeros((n, 9), np.float64)
    for q in range(n):
        nearby = []
        for j in range(n):
            dt = int(g['time_bin'][q]) - int(g['time_bin'][j])
            dx = float(g['x_mm'][j] - g['x_mm'][q])
            dy = float(g['y_mm'][j] - g['y_mm'][q])
            distance2 = dx * dx + dy * dy
            if (j != q and g['side'][j] == g['side'][q] and
                    0 <= dt < window and distance2 <= radius * radius):
                nearby.append((j, dx, dy, distance2))
        for k, r in enumerate((radius / 4, radius / 2, radius)):
            ids = [j for j, dx, dy, d2 in nearby if d2 <= r * r]
            out[q, k] = np.log1p(len(ids))
            out[q, k + 3] = np.log1p(len(set(int(g['time_bin'][j]) for j in ids)))
        if nearby:
            dxy = np.array([[dx, dy] for j, dx, dy, d2 in nearby])
            distances = np.sqrt([d2 for j, dx, dy, d2 in nearby])
            out[q, 6:8] = [np.mean(distances) / 2, np.std(distances) / 2]
            if len(nearby) >= 2:
                centered = dxy - dxy.mean(0)
                eig = np.linalg.eigvalsh(centered.T @ centered / len(nearby))
                if eig.sum() > 0:
                    out[q, 8] = (eig[1] - eig[0]) / eig.sum()
    return out


def expect_error(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Expected rejection did not occur')


def test_features():
    # Query0: nested radii and outer boundary; past7 included, past8/future/endcap excluded.
    g = geometry([0, .25, .5, 1, 2, 4, 4.00001, .1, .2, .3],
                 [0] * 10, [10, 10, 9, 3, 10, 10, 10, 11, 2, 10],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    for radius in controls.RADII:
        x8, x1 = controls.extract_features(g, radius)
        np.testing.assert_allclose(x8, reference(g, radius, 8), rtol=0, atol=2e-15)
        np.testing.assert_allclose(x1, reference(g, radius, 1), rtol=0, atol=2e-15)
        # Only signals known at the query's time/endcap can change its row.
        hidden = {k: v.copy() for k, v in g.items()}
        hidden['x_mm'][[7, 8, 9]] = [0.001, 0.002, 0.003]
        np.testing.assert_array_equal(controls.extract_features(hidden, radius)[0][0], x8[0])
        perm = np.array([9, 3, 6, 1, 7, 0, 2, 4, 5, 8])
        shuffled = {k: v[perm] for k, v in g.items()}
        for actual, expected in zip(controls.extract_features(shuffled, radius), (x8[perm], x1[perm])):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
        moved = {k: v.copy() for k, v in g.items()}
        moved['x_mm'] += 16
        moved['y_mm'] -= 8
        for actual, expected in zip(controls.extract_features(moved, radius), (x8, x1)):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)
        assert controls.extract_features(geometry([], [], [], []), radius)[0].shape == (0, 9)
        assert not controls.extract_features(geometry([0], [0], [0], [0]), radius)[0].any()
        expect_error(lambda: controls.extract_features(dict(g, target=np.zeros(10)), radius))
        duplicate = {k: np.concatenate((v, v[:1])) for k, v in g.items()}
        expect_error(lambda: controls.extract_features(duplicate, radius))
        spec = controls.feature_spec(radius)
        assert spec['nested_count_radii_mm'] == [radius / 4, radius / 2, radius]
        assert spec['shape_distance_normalization_mm'] == 2
    # Exact (not merely tolerance) parity with the historical2mm extractor on valid events.
    rng = np.random.default_rng(9131)
    random = geometry(rng.normal(size=90), rng.normal(size=90),
                      rng.integers(0, 12, 90), rng.integers(0, 2, 90))
    for event in (g, random, geometry([], [], [], [])):
        for actual, expected in zip(controls.extract_features(event, 2),
                                    controls.original.extract_features(event)):
            np.testing.assert_array_equal(actual, expected)
    expect_error(lambda: controls.extract_features(g, 3))
    expect_error(lambda: controls.extract_features(g, True))


def test_scaler_and_optimizer():
    # Validation is never passed to fit_scaler; ignored training outlier is masked.
    train = np.array([[0., 1.], [2., 1.], [1e12, 1e12]])
    targets = np.array([0, 1, -1], np.int8)
    mean, scale = controls.original.fitted_scaler(train, targets)
    np.testing.assert_array_equal(mean, [1., 1.])
    np.testing.assert_array_equal(scale, [1., 1.])
    validation = np.array([[1e30, 2e30]])
    assert ((validation - mean) / scale)[0, 0] > 1e20
    theta = np.array([.2, -.4, .3])
    loss, grad = controls.original.bce_gradient(theta, train, targets)
    masked_loss, masked_grad = controls.original.bce_gradient(theta, train[:2], targets[:2])
    assert loss == masked_loss
    np.testing.assert_array_equal(grad, masked_grad)
    a, status = controls.original.fit_logistic(train, targets)
    b, _ = controls.original.fit_logistic(train, targets)
    assert status['converged'] and np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)


def synthetic_data(base):
    events, values, files = {}, {}, {}
    for index, split in enumerate(('train', 'train', 'validation', 'validation')):
        if index == 3:
            g = geometry([], [], [], [])
            y = np.empty(0, np.int8)
        else:
            g = geometry([0, .25, .75, 1.5, 3, .125], [0, 0, .125, 0, 0, 0],
                         [7, 7, 6, 4, 7, 8], [0] * 6)
            if index == 1:
                g['y_mm'] += .25
            y = np.array([0, 1, 1, 0, -1, 0], np.int8)
        a = dict(g, target=y, primary_electron_weight=np.where(y == 0, 2, 0).astype(np.int64),
                 microcurler_electron_weight=np.where(y == 1, 1, 0).astype(np.int64))
        p = base / f'input{index}.npz'
        atomic_npz(p, a)
        files[p.name], values[index] = sha(p), a
        events[index] = dict(path=p.name, split=split, index=index, raw_id=index,
                             run_uid='invented-test-only', shard='synthetic')
    return SimpleNamespace(events=events, manifest_sha='a' * 64,
        manifest=dict(portable_manifest_sha256='b' * 64, files=files),
        settings=dict(supervised_class_counts=[6, 4]), validation=[2, 3],
        get=lambda i: values[i], verify_unchanged=lambda: None)


def test_fit_cache_resume():
    before = controls.original.extract_features
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        data = synthetic_data(base)
        identities = []
        for radius in controls.RADII:
            controls.run_controls(data, base, radius)
            out = base / 'results/controls' / controls.radius_key(radius)
            with controls.configured(radius) as engine:
                cache = engine.FeatureCache(data, base / 'results/control_feature_cache' / controls.radius_key(radius))
                np.testing.assert_array_equal(cache.event(0)['x8'],
                    controls.extract_features(geometry_only(data.get(0)), radius)[0])
                for variant in controls.VARIANTS:
                    protocol = engine.make_protocol(data, variant)
                    identity = engine.make_identity(protocol, variant)
                    fit = out / variant / f'seed{controls.SEED}'
                    assert engine.verified_complete(fit, identity)
                    assert protocol['radius_mm'] == radius
                    assert protocol['site_manifest_sha256'] == data.manifest_sha
                    assert identity['train_events'] == 2
                    assert identity['parameters'] == (7 if variant == 'counts8' else 10)
                    identities.append(identity['protocol_sha256'])
                    with np.load(fit / 'selected_predictions.npz', allow_pickle=False) as predictions:
                        assert predictions['event_index'].tolist() == [2, 3]
                        assert predictions['event_ptr'].tolist() == [0, 6, 6]
                        np.testing.assert_array_equal(predictions['target'], data.get(2)['target'])
                old_hash = sha(out / 'shape8' / f'seed{controls.SEED}' / 'selected_predictions.npz')
            controls.run_controls(data, base, radius)
            assert sha(out / 'shape8' / f'seed{controls.SEED}' / 'selected_predictions.npz') == old_hash
        assert len(set(identities)) == 6
        # A radius-specific cache cannot be reused under another scientific identity.
        with controls.configured(1) as engine:
            expect_error(lambda: engine.FeatureCache(data, base / 'results/control_feature_cache/r2'))
            cache = engine.FeatureCache(data, base / 'results/control_feature_cache/r1')
            path = cache.base / 'event00000.npz'
            path.write_bytes(path.read_bytes() + b'tamper')
            expect_error(lambda: cache.event(0))
        with controls.configured(4) as engine:
            fit = base / 'results/controls/r4/shape8' / f'seed{controls.SEED}'
            path = fit / 'selected_predictions.npz'
            path.write_bytes(path.read_bytes() + b'tamper')
            identity = engine.make_identity(engine.make_protocol(data, 'shape8'), 'shape8')
            expect_error(lambda: engine.verified_complete(fit, identity))
    assert controls.original.extract_features is before


def run_tests():
    test_features()
    test_scaler_and_optimizer()
    test_fit_cache_resume()
    print('RADIUS CONTROL SELF-TEST: PASS (exact historical 2mm parity; scalar 1/2/4mm references; '
          'nested radii; fixed 2mm normalization; time/endcap/query exclusion; permutation/translation; '
          'empty events; truth rejection; training-only scaler; deterministic logistic fits; '
          'six completed outputs; idempotent resume; cross-radius/tamper rejection)', flush=True)


if __name__ == '__main__':
    run_tests()
