#!/usr/bin/env python3
"""Radius-matched deterministic count and shape controls on frozen site data.

Each radius gets one counts8 fit and one shape8 fit, not three stochastic
replicates. The original optimizer, loss, training-only scaler and saved-output
engine are imported unchanged. Only the observable neighborhood features and
their protocol identity are versioned here. No confirmation data are read.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'vendor/recovery/vendor/original/code'
sys.path.insert(0, str(ORIGINAL))
import fit_controls_v1 as original
from site_data_v1 import (SiteData, GEOMETRY_KEYS, geometry_only, require, sha,
                          canonical, write_json)
import numpy as np
from scipy.spatial import cKDTree

RADII = (1.0, 2.0, 4.0)
VARIANTS = ('counts8', 'shape8')
SEED = original.SEED
NORMALIZATION_MM = 2.0
FEATURE_NAMES = (
    'log1p_count_within_R_over_4', 'log1p_count_within_R_over_2',
    'log1p_count_within_R', 'log1p_distinct_bins_within_R_over_4',
    'log1p_distinct_bins_within_R_over_2', 'log1p_distinct_bins_within_R',
    'mean_radius_over_2mm', 'population_std_radius_over_2mm',
    'centered_xy_covariance_anisotropy',
)
ORIGINAL_MAKE_PROTOCOL = original.make_protocol


def checked_radius(radius):
    if isinstance(radius, bool) or float(radius) not in RADII:
        raise ValueError('The fixed experiment radii are 1, 2 and 4 mm')
    return float(radius)


def radius_key(radius):
    return 'r' + str(int(checked_radius(radius)))


def feature_spec(radius):
    radius = checked_radius(radius)
    return dict(
        observation_radius_mm=radius,
        nested_count_radii_mm=[radius / 4.0, radius / 2.0, radius],
        nested_count_definition='radii scale with observation R: R/4, R/2, R',
        shape_distance_normalization_mm=NORMALIZATION_MM,
        history_bins=8, cached_auxiliary_current_bin_features=True,
        same_endcap=True, query_excluded=True,
        all_truth_roles_supply_context=True, feature_names=list(FEATURE_NAMES),
        shape_definition='mean/std of neighbor distance to query; centered neighbor xy covariance; (lambda_max-lambda_min)/trace',
        insufficient_support='empty: all shape zero; one neighbor: mean available, std and anisotropy zero; zero covariance trace: anisotropy zero',
        timing='t-7..t inclusive; prediction after current 25 ns bin closes',
        interpretation='same observable spatial/time context as the GNN; no claim of architecture-only comparison',
    )


def extract_features(geometry, radius):
    """Return float64 [N,9] eight-bin/current-bin features, using geometry only.

The current-bin array is retained solely for compatibility with the immutable
cache engine; this study fits the eight-bin controls only. At R=2 mm every
calculation and its reduction order match the original feature extractor.
"""
    radius = checked_radius(radius)
    require(set(geometry) == set(GEOMETRY_KEYS), 'Encoder accepts geometry only')
    x, y, bins, side = [np.asarray(geometry[k]) for k in GEOMETRY_KEYS]
    require(x.ndim == 1, 'Geometry arrays must be one-dimensional')
    n = len(x)
    require(all(a.ndim == 1 and len(a) == n for a in (x, y, bins, side)), 'Geometry alignment')
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite geometry')
    require(bins.dtype.kind in 'iu' and side.dtype.kind in 'iu' and np.isin(side, [0, 1]).all(), 'Invalid bins/endcaps')
    require(not n or (np.min(bins) >= np.iinfo(np.int64).min + 7 and
                     np.max(bins) <= np.iinfo(np.int64).max), 'Unsafe bin range')
    bins = bins.astype(np.int64, copy=False)
    xy = np.column_stack((x, y)).astype(np.float64, copy=False)
    if n:
        addresses = np.rec.fromarrays([side, bins, x, y])
        require(len(np.unique(addresses)) == n, 'Duplicate binary site address/time/endcap')
    out8, out1 = np.zeros((n, 9), np.float64), np.zeros((n, 9), np.float64)
    nested_radii = (radius / 4.0, radius / 2.0, radius)
    for cap in (0, 1):
        ids = np.flatnonzero(side == cap)
        if not len(ids):
            continue
        points = xy[ids]
        tree = cKDTree(points)
        for start in range(0, len(ids), 256):
            neighborhoods = tree.query_ball_point(points[start:start + 256], radius, return_sorted=True)
            for local_q, candidates in enumerate(neighborhoods, start):
                q = int(ids[local_q])
                js = ids[np.asarray(candidates, np.int64)]
                js = js[(js != q) & (bins[js] <= bins[q]) & (bins[js] >= bins[q] - 7)]
                delta = xy[js] - xy[q]
                dist2 = np.einsum('ij,ij->i', delta, delta)
                keep = dist2 <= radius * radius
                js, delta, dist2 = js[keep], delta[keep], dist2[keep]
                for out, window in ((out8, 8), (out1, 1)):
                    use = np.ones(len(js), bool) if window == 8 else bins[js] == bins[q]
                    neighbors, dxy, d2 = js[use], delta[use], dist2[use]
                    row = out[q]
                    for k, r in enumerate(nested_radii):
                        inside = d2 <= r * r
                        row[k] = np.log1p(np.count_nonzero(inside))
                        row[k + 3] = np.log1p(len(np.unique(bins[neighbors[inside]])))
                    if len(neighbors):
                        d = np.sqrt(d2)
                        row[6], row[7] = d.mean() / NORMALIZATION_MM, d.std(ddof=0) / NORMALIZATION_MM
                    if len(neighbors) >= 2:
                        centered = dxy - dxy.mean(axis=0)
                        c = centered.T @ centered / len(neighbors)
                        trace = float(c[0, 0] + c[1, 1])
                        if trace > 0:
                            row[8] = np.clip(np.hypot(c[0, 0] - c[1, 1], 2 * c[0, 1]) / trace, 0.0, 1.0)
    require(np.isfinite(out8).all() and np.isfinite(out1).all(), 'Nonfinite features')
    return out8, out1


def code_hashes():
    return {
        'code/control_radius.py': sha(Path(__file__)),
        'vendor/recovery/vendor/original/code/fit_controls_v1.py': sha(ORIGINAL / 'fit_controls_v1.py'),
        'vendor/recovery/vendor/original/code/site_data_v1.py': sha(ORIGINAL / 'site_data_v1.py'),
    }


@contextmanager
def configured(radius):
    """Scope feature/protocol adapters while retaining the immutable fit engine."""
    radius = checked_radius(radius)
    names = ('extract_features', 'feature_spec', 'code_hashes', 'FEATURE_NAMES',
             'make_protocol', 'make_identity')
    saved = {name: getattr(original, name) for name in names}

    def make_protocol(data, variant):
        require(variant in VARIANTS, 'This study fits eight-bin controls only')
        p = ORIGINAL_MAKE_PROTOCOL(data, variant)
        p.update(
            schema='fims_radius_control_protocol_v1', radius_mm=radius,
            training_events=sum(e['split'] == 'train' for e in data.events.values()),
            validation_events=len(data.validation),
            experiment='joint observation/connection-radius sensitivity; controls have matched observation context',
            detector_changes=False, continuous_hit_branch=False,
            prior_results='fresh matched deterministic fit; old two-mm control is not overwritten',
        )
        return p

    def make_identity(protocol, variant):
        return dict(protocol_sha256=hashlib.sha256(canonical(protocol)).hexdigest(),
                    model=variant, seed=SEED, parameters=7 if variant == 'counts8' else 10,
                    train_events=protocol['training_events'], radius_mm=radius)

    original.extract_features = lambda g: extract_features(g, radius)
    original.feature_spec = lambda: feature_spec(radius)
    original.code_hashes = code_hashes
    original.FEATURE_NAMES = FEATURE_NAMES
    original.make_protocol = make_protocol
    original.make_identity = make_identity
    try:
        yield original
    finally:
        for name, value in saved.items():
            setattr(original, name, value)


def run_controls(data, study_root, radius):
    """Build verified caches and fit both controls; safe to resume this radius."""
    radius = checked_radius(radius)
    study_root = Path(study_root).resolve()
    out = study_root / 'results/controls' / radius_key(radius)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (out / 'controls.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Controls for this radius are already active; no second worker started') from exc
        with configured(radius) as engine:
            if all(engine.verified_complete(out / v / f'seed{SEED}',
                    engine.make_identity(engine.make_protocol(data, v), v)) for v in VARIANTS):
                print(f'R={radius:g} mm: both deterministic controls COMPLETE and verified', flush=True)
                return
            cache_start = time.monotonic()
            cache = engine.FeatureCache(data, study_root / 'results/control_feature_cache' / radius_key(radius))
            cache.prepare()
            cache_seconds = time.monotonic() - cache_start
            for variant in VARIANTS:
                engine.fit_one(data, cache, out, variant)
            data.verify_unchanged()
            # Per-invocation wall time, never represented as aggregate job time.
            write_json(out / 'execution_profile.json', dict(
                radius_mm=radius, invocation_seconds=time.monotonic() - started,
                cache_prepare_or_verify_seconds=cache_seconds,
                fit_count=2, stochastic_replicates=False,
                note='Resume can include verified cached work; per-fit metrics record completed fit time.',
            ))
    print(f'FIMS RADIUS CONTROLS COMPLETE: R={radius:g} mm; counts8 and shape8', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--radius', type=float, choices=RADII)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        from test_radius_controls import run_tests
        run_tests()
        return
    require(a.study_root is not None and a.data is not None and a.radius is not None,
            '--study-root, --data and --radius are required')
    require(a.study_root.resolve() == ROOT,
            '--study-root must identify this verified code package')
    from radius_common import verify_package, verify_data_binding
    verify_package(a.study_root.resolve())
    verify_data_binding(a.study_root, a.data)
    data = SiteData(a.data)
    run_controls(data, a.study_root, a.radius)


if __name__ == '__main__':
    main()
