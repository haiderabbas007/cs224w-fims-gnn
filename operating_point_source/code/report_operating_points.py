#!/usr/bin/env python3
"""Describe fixed saved FIMS predictions; never train, simulate or score models.

The cutoffs are selected and reported on the same development-validation events.
They are not independently calibrated operating points or detector requirements.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/original/code'))
import report_parallel_v1 as old
import metrics_parallel_v1 as metrics

cm = metrics.base
SEEDS = tuple(metrics.SEEDS)
TARGETS = ((99.9, 1000), (99.5, 5000), (99.0, 10000), (95.0, 50000), (90.0, 100000))
MODELS = ('cnn_small', 'cnn_wide', 'full2', 'full1', 'no_xy2', 'current2',
          'counts8', 'shape8', 'counts1', 'shape1')
CONTROLS = frozenset(('counts8', 'shape8', 'counts1', 'shape1'))
LABELS = dict(old.LABELS, no_xy2='GNN without explicit x/y offsets')
EXPECTED = frozenset((model, seed) for model in MODELS
                     for seed in ((SEEDS[0],) if model in CONTROLS else SEEDS))
SUMMARY_KEYS = (
    'microcurler_site_recall', 'macro_microcurler_site_recall',
    'microcurler_charge_recall', 'macro_microcurler_charge_recall',
    'supervised_site_precision', 'other_site_retention', 'balanced_accuracy',
    'primary_charge_retention', 'macro_primary_charge_retention',
    'primary_loss_fraction', 'primary_loss_ppm',
    'fraction_primary_events_losing_above_1_percent',
    'fraction_primary_events_losing_above_5_percent',
    'fraction_primary_events_losing_above_10_percent',
    'maximum_event_primary_loss_fraction',
)


def require(ok, text):
    cm.require(ok, text)


def sha(path):
    return old.sha(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_reference(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/unsafe reviewed metrics reference')
    before = sha(path)
    ref = read_json(path)
    require(ref.get('schema') == 'fims_parallel_development_report_v1'
            and ref.get('status') == 'COMPLETE' and not ref.get('missing_fits'),
            'Reference is not the completed historical parallel report')
    rows = ref.get('results', [])
    keys = [(r['model'], int(r['seed'])) for r in rows]
    require(len(keys) == 22 and len(set(keys)) == 22 and set(keys) == EXPECTED,
            'Reference must identify all 22 reviewed selected fits')
    require(ref.get('data_provenance', {}).get('portable_manifest_sha256') == old.PORTABLE_SHA,
            'Reference uses another dataset')
    for row in rows:
        for k in ('complete_sha256', 'protocol_sha256', 'predictions_sha256'):
            require(isinstance(row.get(k), str) and len(row[k]) == 64,
                    'Reference lacks pinned historical fit hash: ' + k)
    require(sha(path) == before, 'Reference changed during read')
    return {k: r for k, r in zip(keys, rows)}, before


def fit_jobs(capacity, study):
    for model in MODELS:
        deterministic = model in CONTROLS
        for seed in ((SEEDS[0],) if deterministic else SEEDS):
            source_model = model.removeprefix('cnn_') if model.startswith('cnn_') else model
            fit = (capacity / 'results' / source_model / f'seed{seed}'
                   if model.startswith('cnn_') else study / 'results/fits' / model / f'seed{seed}')
            yield model, source_model, seed, fit, deterministic


def inventory_snapshot(fit):
    marker = old.safe_file(fit, 'complete.json')
    before = sha(marker)
    complete = read_json(marker)
    require(complete.get('status') == 'COMPLETE' and isinstance(complete.get('outputs'), dict),
            'Saved fit is not complete: ' + str(fit))
    inventory = {'complete.json': before}
    for name, digest in complete['outputs'].items():
        require(Path(name).name == name, 'Unexpected nested fit output')
        require(sha(old.safe_file(fit, name)) == digest, 'Fit output changed: ' + str(fit / name))
        inventory[name] = digest
    require(sha(marker) == before, 'Completion marker changed during verification')
    return inventory


def verify_reviewed_record(record, reviewed, inventory):
    for key in ('complete_sha256', 'protocol_sha256', 'predictions_sha256', 'epoch', 'parameters'):
        require(record.get(key) == reviewed.get(key), 'Historical selected fit changed: ' + key)
    require(inventory['complete.json'] == reviewed['complete_sha256'], 'Historical completion hash changed')
    require(inventory['selected_predictions.npz'] == reviewed['predictions_sha256'],
            'Historical predictions changed')
    require(inventory['protocol.json'] == reviewed['protocol_sha256'], 'Historical protocol changed')


def analyse_points(arrays, expected_events=2500, targets=TARGETS):
    """Exact tied-logit integer-budget decisions, with independent denominators."""
    population = cm.validate_arrays(arrays, expected_events)
    weights = metrics.validate_charge(arrays)
    population['microcurler_electrons'] = int(weights.sum())
    prefix = cm.rank_prefixes(arrays['logits'], arrays['target'], arrays['primary_electron_weight'])
    points, event_rows = [], []
    for retention, budget in targets:
        require(math.isclose(retention, 100.0 - budget / 10000.0, abs_tol=1e-12),
                'Retention percentage and integer budget disagree')
        cutoff = cm.cutoff_at_budget(prefix, int(budget))
        aggregate, events = cm.metrics_at_point(arrays, cutoff)
        flags = cm.decision_flags(arrays['logits'], cutoff)
        total_mc = int(weights.sum())
        caught_mc = int(weights[flags].sum())
        aggregate.update(microcurler_electrons=total_mc,
                         flagged_microcurler_electrons=caught_mc,
                         microcurler_charge_recall=cm.ratio(caught_mc, total_mc),
                         flagged_microcurler_electrons_on_ignored_sites=int(
                             weights[flags & (arrays['target'] == -1)].sum()))
        for pos, event in enumerate(events):
            lo, hi = map(int, arrays['event_ptr'][pos:pos + 2])
            total = int(weights[lo:hi].sum())
            caught = int(weights[lo:hi][flags[lo:hi]].sum())
            event.update(microcurler_electrons=total, flagged_microcurler_electrons=caught,
                         microcurler_charge_recall=cm.ratio(caught, total))
        loss = [e['primary_loss_fraction'] for e in events if e['primary_electrons']]
        aggregate['primary_event_loss_quantiles'] = {
            str(q): float(np.quantile(loss, q, method='linear')) if loss else None
            for q in (.5, .9, .95, .99, 1.0)
        }
        recalls = [e['microcurler_charge_recall'] for e in events if e['microcurler_electrons']]
        aggregate['macro_microcurler_charge_recall'] = float(np.mean(recalls)) if recalls else None
        points.append(dict(retention_percent=float(retention), operating_point=cutoff, metrics=aggregate))
        event_rows.extend(dict(retention_percent=float(retention), operating_point=cutoff['kind'],
                               target=cutoff['target'], target_units='ppm', **e) for e in events)
    return population, points, event_rows


def reproduce_historical(points, reviewed):
    """The new analysis must exactly reproduce unchanged historical endpoints."""
    previous = {int(p['operating_point']['target']): p for p in reviewed['points']
                if p['operating_point']['kind'] == 'primary_loss_budget'}
    for p in points:
        budget = int(p['operating_point']['target'])
        if budget not in previous:
            continue
        oldp = previous[budget]
        for key in ('threshold_kind', 'threshold_logit', 'lost_primary_electrons',
                    'flagged_microcurler_sites', 'flagged_sites'):
            require(p['operating_point'][key] == oldp['operating_point'][key],
                    'Historical threshold/prefix no longer reproduces: ' + key)
        for key in ('microcurler_site_recall', 'macro_microcurler_site_recall',
                    'microcurler_charge_recall', 'primary_charge_retention',
                    'lost_primary_electrons_on_ignored_sites'):
            a, b = p['metrics'][key], oldp['metrics'][key]
            require(a == b or (a is not None and b is not None and
                              math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)),
                    'Historical metric no longer reproduces: ' + key)
        require(p['metrics']['primary_loss_tails'] == oldp['metrics']['primary_loss_tails'],
                'Historical event-loss counts no longer reproduce')


def flatten(record, point):
    p, m = point['operating_point'], point['metrics']
    row = dict(model=record['model'], seed=record['seed'], epoch=record['epoch'],
               parameters=record['parameters'], deterministic=record['deterministic'],
               retention_percent=point['retention_percent'], operating_point=p['kind'],
               target=p['target'], target_units='ppm', threshold_kind=p['threshold_kind'],
               threshold_logit=p['threshold_logit'], status=p['status'])
    for key in SUMMARY_KEYS:
        if not key.startswith('fraction_primary_events_losing_above_'):
            row[key] = m[key]
    for pct in (1, 5, 10):
        tail = m['primary_loss_tails'][f'above_{pct}_percent']
        row[f'fraction_primary_events_losing_above_{pct}_percent'] = tail['fraction']
        row[f'primary_events_losing_above_{pct}_percent'] = tail['events']
    for key in ('events', 'primary_bearing_events', 'microcurler_bearing_events',
                'microcurler_sites', 'flagged_microcurler_sites', 'primary_electrons',
                'lost_primary_electrons', 'lost_primary_electrons_on_ignored_sites',
                'microcurler_electrons', 'flagged_microcurler_electrons'):
        row[key] = m[key]
    for q, value in m['primary_event_loss_quantiles'].items():
        row['event_primary_loss_quantile_' + q] = value
    return row


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['model'], row['target']), []).append(row)
    result = []
    for model in MODELS:
        for retention, target in TARGETS:
            values = groups.get((model, target), [])
            expected = {SEEDS[0]} if model in CONTROLS else set(SEEDS)
            require(len(values) == len(expected) and {v['seed'] for v in values} == expected,
                    'Cannot summarize missing/duplicate fits: ' + model)
            st = {}
            for key in SUMMARY_KEYS:
                a = [v[key] for v in values if v[key] is not None]
                st[key] = dict(mean=float(np.mean(a)) if a else None,
                               minimum=min(a) if a else None, maximum=max(a) if a else None,
                               defined_runs=len(a))
            result.append(dict(model=model, retention_percent=retention, target=target,
                               target_units='ppm', operating_point='primary_loss_budget',
                               seeds=sorted(expected), deterministic=model in CONTROLS,
                               coverage='COMPLETE', statistics=st))
    return result


def display_curve(arrays, points, max_samples=2500):
    """Sample only drawing coordinates. Every reported metric uses exact ties."""
    curve = metrics.full_curve(arrays)
    n = len(curve['threshold_logit'])
    ids = set(np.linspace(0, max(n - 1, 0), min(n, max_samples), dtype=np.int64).tolist())
    near = np.flatnonzero(curve['primary_charge_retention'] >= .90)
    if len(near):
        ids.update(near[np.unique(np.linspace(0, len(near) - 1, min(len(near), max_samples),
                                             dtype=np.int64))].tolist())
    for point in points:
        ids.add(int(point['operating_point']['selected_groups']))
    ids = np.asarray(sorted(ids), dtype=np.int64)
    result = {key: value[ids] if np.asarray(value).ndim == 1 and len(value) == n else value
              for key, value in curve.items()}
    result.update(display_only=np.asarray(True), original_complete_tie_points=np.asarray(n),
                  selected_original_indices=ids)
    return result


def summary_text(summary, records):
    pct = lambda v: 'undefined' if v is None else f'{100 * v:.3f}%'
    lines = ['FIMS OPERATING-POINT REVIEW V1: COMPLETE', '',
             '22 existing selected fits; 2,500 fixed development-validation events.',
             'No training, model inference or new simulation. No confirmation data read.',
             'Cutoffs selected and reported on these same development events.', '',
             'Retention target | Model | Microcurler-site recall | Equal-event recall | Actual primary-charge retention | Events losing >5% primary charge | Supervised precision']
    for s in summary:
        label = LABELS.get(s['model'], s['model'])
        label += ' [one deterministic fit]' if s['deterministic'] else ' [three-seed mean]'
        st = s['statistics']
        lines.append(f'{s["retention_percent"]:g}% | {label} | ' + ' | '.join(pct(st[k]['mean']) for k in (
            'microcurler_site_recall', 'macro_microcurler_site_recall', 'primary_charge_retention',
            'fraction_primary_events_losing_above_5_percent', 'supervised_site_precision')))
    lines += ['', 'Interpretation and limitations:',
              '- 90%, 95% and 99% are exploratory comparison targets, not demonstrated detector safety limits.',
              '- 99.9% and 99.5% retain historical references; all five points use each fit\'s unchanged selected checkpoint.',
              '- Primary-charge retention counts every accepted attributed primary electron, including ignored/mixed sites.',
              '- Supervised microcurler-site recall and precision omit ignored labels; microcurler-charge recall includes all attributed accepted microcurler electrons.',
              '- Equal-event recall averages only events with supervised microcurler sites. Events without microcurlers are not assigned artificial recall values.',
              '- Event-loss fractions use only primary-bearing events; empty and zero-primary events remain in the data inventory.',
              '- Whole equal-logit groups stay together. Integer loss budgets are floored. Actual retention can exceed the nominal target.',
              '- Histograms display classifier scores, not calibrated probabilities. Cutoffs are shown separately for every model/seed.',
              '- Seed ranges describe these training repetitions on shared events, not independent-event confidence intervals.',
              '- Graphs, feature controls and CNN patches differ; these are implemented pipeline comparisons.',
              '- Original GNN runtime recovery used chunked messages and activation recomputation; numerical roundoff may differ from the original CUDA path. Existing saved predictions are reused exactly.',
              '- All results describe the historical detector/readout configuration. Newly discussed transport values have NOT been applied to this dataset.',
              '- Charge preservation and the >5% event-loss diagnostic do not establish tracking or particle-identification performance.',
              '- Curves saved under curves/ are sampled for display only. Tables, cutoffs, histogram counts and event rows are computed from the complete saved predictions.',
              '- No single winning threshold is adopted here. Freeze a chosen procedure before a genuinely new calibration/evaluation sample.', '']
    return '\n'.join(lines)


@contextmanager
def report_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'report.lock'
    require(not path.is_symlink(), 'Unsafe report lock')
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('A report writer is already active')
        yield


def checked_output(capacity, study, out):
    require(not out.is_symlink(), 'Report output cannot be a symlink')
    out = out.resolve()
    for source in (capacity, study):
        require(out != source and not out.is_relative_to(source) and not source.is_relative_to(out),
                'Report destination must be separate from every source study')
    return out


def build(capacity_root, study_root, output_dir, reference_metrics=None):
    capacity, study = Path(capacity_root).resolve(), Path(study_root).resolve()
    out = checked_output(capacity, study, Path(output_dir))
    reference = Path(reference_metrics) if reference_metrics else ROOT / 'reference/previous_metrics.json'
    reviewed, reference_sha = load_reference(reference)
    package_sha = sha(old.safe_file(ROOT, 'PACKAGE_MANIFEST.json'))
    with report_lock(out):
        # A previous successful report is reusable only after ALL source and output hashes reverify.
        native, native_provenance = old.load_native(capacity)
        jobs = list(fit_jobs(capacity, study))
        snapshots = {}
        for model, _, seed, fit, _ in jobs:
            inventory = inventory_snapshot(fit)
            require(inventory['complete.json'] == reviewed[(model, seed)]['complete_sha256'],
                    'Historical fit marker differs: ' + model)
            snapshots[(model, seed)] = inventory
        print('ALL 22 REVIEWED FIT INVENTORIES / NATIVE VALIDATION ATTRIBUTION: PASS', flush=True)
        # Build atomically in a new staging directory. Never publish a COMPLETE marker for partial output.
        stage = Path(tempfile.mkdtemp(prefix='.report_build_', dir=out))
        (stage / 'scores').mkdir()
        (stage / 'curves').mkdir()
        records, point_rows, all_events, curves = [], [], [], []
        from matplotlib.backends.backend_pdf import PdfPages
        from report_plots import render_score_page, render_summary
        with PdfPages(stage / 'fims_classifier_score_distributions_v1.pdf') as pdf:
            for ordinal, (model, source_model, seed, fit, deterministic) in enumerate(jobs, 1):
                arrays, record = old.load_fit(fit, source_model, seed, native, deterministic=deterministic)
                verify_reviewed_record(record, reviewed[(model, seed)], snapshots[(model, seed)])
                record['model'] = model
                record['source_fit'] = str(fit)
                population, points, events = analyse_points(arrays)
                reproduce_historical(points, reviewed[(model, seed)])
                record.update(population=population, points=points,
                              class_balanced_bce=cm.class_balanced_bce(arrays['logits'], arrays['target']))
                records.append(record)
                point_rows.extend(flatten(record, p) for p in points)
                all_events.extend(dict(model=model, seed=seed, **e) for e in events)
                hist = render_score_page(pdf, arrays, record, points)
                old.write_npz(stage / 'scores' / f'{model}_seed{seed}_histograms.npz', hist)
                curve_path = stage / 'curves' / f'{model}_seed{seed}_display_curve.npz'
                old.write_npz(curve_path, display_curve(arrays, points))
                curves.append(dict(model=model, seed=seed, path=curve_path))
                print(f'OPERATING-POINT REVIEW: {model} seed={seed} ({ordinal}/22); historical points reproduced', flush=True)
        summary = summarize(point_rows)
        render_summary(stage, summary, curves, all_events)
        old.write_csv(stage / 'fims_operating_points_v1.csv', point_rows)
        old.write_csv(stage / 'fims_operating_point_event_metrics_v1.csv', all_events)
        old.atomic_bytes(stage / 'fims_operating_point_summary_v1.txt', summary_text(summary, records).encode())
        report = dict(schema='fims_operating_point_review_v1', status='COMPLETE',
                      completed_fits=22, expected_fits=22, validation_events=2500,
                      results=records, seed_summaries=summary,
                      primary_retention_targets_percent=[t for t, _ in TARGETS],
                      descriptive_development_only=True, new_detector_settings_applied=False,
                      new_training=False, new_simulation=False, model_inference=False,
                      package_sha256=package_sha,
                      reference_metrics_sha256=reference_sha, data_provenance=native_provenance)
        old.write_json(stage / 'fims_operating_point_metrics_v1.json', report)
        print('Rechecking source inventories and native data after reporting...', flush=True)
        # Full content verification, rather than relying only on mtimes.
        native_again, provenance_again = old.load_native(capacity)
        require(provenance_again == native_provenance, 'Native provenance changed during report')
        for key in native:
            require(np.array_equal(native[key], native_again[key]), 'Native arrays changed during report: ' + key)
        for model, _, seed, fit, _ in jobs:
            require(inventory_snapshot(fit) == snapshots[(model, seed)],
                    'Saved fit changed during report: ' + model)
        require(sha(reference) == reference_sha, 'Reviewed reference changed during report')
        require(sha(ROOT / 'PACKAGE_MANIFEST.json') == package_sha, 'Report package changed during report')
        inventory = {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        provenance = dict(schema='fims_operating_point_provenance_v1',
                          package_sha256=package_sha,
                          reference_metrics_sha256=reference_sha,
                          sources=[dict(model=model, seed=seed, path=str(fit), inventory=snapshots[(model, seed)])
                                   for model, _, seed, fit, _ in jobs],
                          data_provenance=native_provenance,
                          verification='Full input content hashes and native attribution verified before and after reporting')
        old.write_json(stage / 'fims_operating_point_provenance_v1.json', provenance)
        inventory['fims_operating_point_provenance_v1.json'] = sha(stage / 'fims_operating_point_provenance_v1.json')
        marker = dict(schema='fims_operating_point_complete_v1', status='COMPLETE', completed_fits=22,
                      package_sha256=package_sha,
                      reference_metrics_sha256=reference_sha, outputs=inventory)
        old.write_json(stage / 'fims_operating_point_complete_v1.json', marker)
        archive = stage / 'fims_operating_point_review_v1.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in [*inventory, 'fims_operating_point_complete_v1.json']:
                z.write(stage / name, name)
        with zipfile.ZipFile(archive) as z:
            require(z.testzip() is None, 'Output ZIP failed CRC verification')
        # Publish regular outputs first and the completeness certificate last.
        for name in [*inventory, archive.name]:
            dst = out / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            require(not dst.is_symlink(), 'Unsafe report destination')
            os.replace(stage / name, dst)
        os.replace(stage / 'fims_operating_point_complete_v1.json', out / 'fims_operating_point_complete_v1.json')
        for directory in sorted(stage.rglob('*'), reverse=True):
            if directory.is_dir():
                directory.rmdir()
        stage.rmdir()
        print('FIMS OPERATING-POINT REVIEW V1: COMPLETE', flush=True)
        print('Review ZIP: ' + str(out / archive.name), flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity-root', type=Path, required=True)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-metrics', type=Path)
    args = parser.parse_args()
    build(args.capacity_root, args.study_root, args.output_dir, args.reference_metrics)


if __name__ == '__main__':
    main()
