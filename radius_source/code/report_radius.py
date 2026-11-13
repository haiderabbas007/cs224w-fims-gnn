#!/usr/bin/env python3
"""Read-only report for the matched FIMS site-GNN radius experiment.

Every threshold is a descriptive development-validation threshold. Selected
checkpoints form the primary comparison; final checkpoints are separate
diagnostics. This program performs no inference, training, or simulation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor/recovery/vendor/original/code'
sys.path.insert(0, str(VENDOR))

import numpy as np
import metrics_parallel_v1 as metric
import report_parallel_v1 as old_report
from site_data_v1 import SiteData

base = metric.base
require = base.require
RADII = (1, 2, 4)
SEEDS = (20260924, 20260925, 20260926)
CONTROLS = ('counts8', 'shape8')
BUDGETS_PPM = (500, 1000, 5000, 10000, 50000, 100000)
MAIN_BUDGETS = (1000, 5000, 10000)
SUMMARY_KEYS = (
    'microcurler_site_recall', 'macro_microcurler_site_recall',
    'microcurler_charge_recall', 'macro_microcurler_charge_recall',
    'supervised_site_precision', 'primary_charge_retention',
    'primary_loss_fraction', 'primary_loss_ppm',
    'fraction_primary_events_losing_above_5_percent',
    'maximum_event_primary_loss_fraction',
)
LABELS = {'full1': 'GNN, one layer', 'counts8': 'Counts', 'shape8': 'Counts + shape'}
PREFIX = 'fims_radius_'

sha = old_report.sha
safe_file = old_report.safe_file
atomic_bytes = old_report.atomic_bytes
write_json = old_report.write_json
write_npz = old_report.write_npz


def write_csv(path, rows):
    """Always write a file, including a readable empty-file header."""
    stream = io.StringIO(newline='')
    if rows:
        names = list(rows[0])
        require(all(set(r) == set(names) for r in rows), 'CSV row field mismatch')
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    else:
        stream.write('status\nNO_COMPLETED_RESULTS\n')
    atomic_bytes(path, stream.getvalue().encode())


def load_native(data):
    """Use verified geometry sidecars; labels/charge are reporting-only inputs."""
    target, primary, micro, lengths = [], [], [], []
    events = [data.events[i] for i in data.validation]
    for number, index in enumerate(data.validation, 1):
        event = data.get(index)
        target.append(event['target'])
        primary.append(event['primary_electron_weight'])
        micro.append(event['microcurler_electron_weight'])
        lengths.append(len(event['target']))
        if number % 500 == 0:
            print(f'Radius report: native validation attribution {number}/{len(events)}', flush=True)
    concatenate = lambda parts, dtype: np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(0, dtype)
    result = dict(
        target=concatenate(target, np.int8),
        primary_electron_weight=concatenate(primary, np.int64),
        microcurler_electron_weight=concatenate(micro, np.int64),
        event_ptr=np.r_[np.int64(0), np.cumsum(lengths, dtype=np.int64)],
        event_index=np.asarray(data.validation, np.int64),
        event_id_raw=np.asarray([e['raw_id'] for e in events], np.int64),
        event_run_uid=np.asarray([e['run_uid'] for e in events]),
        event_shard=np.asarray([e['shard'] for e in events]),
    )
    return result


def verify_fit(folder, *, radius, model, seed, site_manifest_sha256):
    """Verify the entire committed output inventory before reading predictions."""
    folder = Path(folder)
    marker = safe_file(folder, 'complete.json')
    complete = json.loads(marker.read_text())
    require(complete.get('status') == 'COMPLETE', 'Fit is not complete: ' + str(folder))
    inventory = complete.get('outputs')
    require(isinstance(inventory, dict) and inventory, 'Missing fit output inventory')
    needed = {'protocol.json', 'metrics.json', 'selected_predictions.npz'}
    deterministic = model in CONTROLS
    if deterministic:
        needed.add('model_parameters.json')
    else:
        needed |= {'history.json', 'selected_checkpoint.pt', 'final_checkpoint.pt', 'final_predictions.npz'}
    require(needed.issubset(inventory), 'Incomplete committed fit inventory')
    for name, digest in inventory.items():
        require(Path(name).name == name, 'Fit output must be a flat relative filename')
        require(sha(safe_file(folder, name)) == digest, 'Completed fit hash mismatch: ' + str(folder/name))
    protocol = json.loads((folder/'protocol.json').read_text())
    meta = json.loads((folder/'metrics.json').read_text())
    identity = complete.get('identity')
    require(isinstance(identity, dict) and identity == meta.get('identity'), 'Fit metadata identity mismatch')
    require(identity.get('model') == model and identity.get('seed') == seed, 'Wrong model or seed')
    require(identity.get('protocol_sha256') == sha(folder/'protocol.json'), 'Protocol hash mismatch')
    require(float(protocol.get('radius_mm', -1)) == float(radius), 'Wrong radius protocol')
    require(protocol.get('site_manifest_sha256') == site_manifest_sha256, 'Wrong site data manifest')
    require(protocol.get('portable_manifest_sha256') == SiteDataPortableSHA,
            'Wrong original development data lineage')
    require(identity.get('parameters') == (12929 if model == 'full1' else 7 if model == 'counts8' else 10),
            'Unexpected model parameter count')
    if deterministic:
        require(meta.get('selected_epoch') is None, 'Deterministic control claims epoch selection')
        history = None
    else:
        history = json.loads((folder/'history.json').read_text())
        require(isinstance(history, list) and len(history) == 10, 'Incomplete training history')
        require([r['epoch'] for r in history] == list(range(1, 11)), 'Epoch sequence changed')
        require(all(math.isfinite(r['validation_class_balanced_bce']) for r in history),
                'Nonfinite validation history')
        chosen = min(history, key=lambda r: r['validation_class_balanced_bce'])
        require(meta.get('selected_epoch') == chosen['epoch'], 'Selection is not earliest minimum validation BCE')
        require(math.isclose(meta['selected_bce'], chosen['validation_class_balanced_bce'],
                             rel_tol=2e-6, abs_tol=2e-7), 'Selected loss metadata mismatch')
    profile = None
    profile_path = folder/'execution_profile.json'
    if profile_path.exists():
        # Runtime evidence is committed separately after the scientific marker.
        receipt = json.loads(safe_file(folder, 'execution_profile_complete.json').read_text())
        require(receipt.get('execution_profile_sha256') == sha(profile_path), 'Execution profile hash mismatch')
        profile = json.loads(safe_file(folder, 'execution_profile.json').read_text())
        require(receipt.get('sessions') == len(profile.get('sessions', [])), 'Execution profile session count mismatch')
    return dict(folder=folder, complete=complete, protocol=protocol, meta=meta,
                history=history, deterministic=deterministic, profile=profile,
                complete_sha256=sha(marker), protocol_sha256=identity['protocol_sha256'])


# The SiteData implementation performs the stronger source/split/file checks.
from site_data_v1 import PORTABLE_SHA as SiteDataPortableSHA


def load_predictions(verified, native, checkpoint, *, expected_events):
    require(checkpoint in ('selected', 'final'), 'Unknown checkpoint role')
    require(not (verified['deterministic'] and checkpoint == 'final'), 'Control has no final-epoch endpoint')
    folder = verified['folder']
    name = checkpoint + '_predictions.npz'
    require(name in verified['complete']['outputs'], 'Prediction file is not committed')
    with np.load(safe_file(folder, name), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    base.validate_arrays(arrays, expected_events)
    require(str(arrays['protocol_sha256'].item()) == verified['protocol_sha256'],
            'Prediction protocol mismatch')
    for key in base.IDENTITY_KEYS:
        require(np.array_equal(arrays[key], native[key]), 'Prediction/native alignment mismatch: ' + key)
    if 'microcurler_electron_weight' in arrays:
        require(np.array_equal(arrays['microcurler_electron_weight'], native['microcurler_electron_weight']),
                'Prediction/native microcurler charge mismatch')
    arrays['microcurler_electron_weight'] = native['microcurler_electron_weight']
    observed = base.class_balanced_bce(arrays['logits'], arrays['target'])
    expected = (verified['meta']['selected_bce'] if checkpoint == 'selected'
                else verified['history'][-1]['validation_class_balanced_bce'])
    require(observed is not None and math.isclose(observed, expected, rel_tol=2e-6, abs_tol=2e-7),
            checkpoint + ' predictions do not reproduce validation BCE')
    return arrays


def verify_scientific_protocol(verified, radius, model):
    """Check the pinned experiment settings as well as self-consistent hashes."""
    p = verified['protocol']
    require(verified['complete']['identity'].get('train_events') == 7500,
            'Training population differs from the fixed experiment')
    require(p.get('history_bins') == 8, 'Temporal context changed')
    if model == 'full1':
        require(p.get('schema') == 'fims_radius_gnn_protocol_v1', 'Wrong GNN protocol family')
        require(p.get('observation_radius_mm') == radius and p.get('connection_radius_mm') == radius,
                'Observation and connection radii must match')
        require(p.get('spatial_normalization_mm') == 2 and p.get('temporal_normalization_bins') == 7,
                'Edge coordinate normalization changed')
        require(p.get('epochs') == 10 and p.get('layers') == 1 and p.get('hidden_channels') == 64,
                'Architecture or training duration changed')
        require(p.get('experiment_plan_sha256') == sha(ROOT/'EXPERIMENT_PLAN.json'), 'Fixed experiment plan changed')
        require(set(p.get('code', {})) == {'train_site_gnn_v1.py', 'graph_models_v1.py', 'gnn_adapter_v1.py',
                                         'site_data_v1.py', 'vendor/capacity_model_v2.py'}, 'GNN source inventory changed')
        require(set(p.get('experiment_sources', {})) == {'code/radius_graph.py', 'code/train_radius.py',
                                                       'vendor/recovery/code/memory_model.py'}, 'Radius source inventory changed')
        sources = [(VENDOR, p.get('code', {})), (ROOT, p.get('experiment_sources', {}))]
    else:
        require(p.get('schema') == 'fims_radius_control_protocol_v1' and p.get('model') == model,
                'Wrong control protocol family')
        f = p.get('features', {})
        require(f.get('observation_radius_mm') == radius and f.get('nested_count_radii_mm') == [radius/4, radius/2, radius],
                'Control observation context changed')
        require(f.get('shape_distance_normalization_mm') == 2 and f.get('history_bins') == 8,
                'Control shape normalization or temporal context changed')
        require(f.get('same_endcap') is True and f.get('query_excluded') is True,
                'Control endcap/query rule changed')
        require(len(p.get('columns', [])) == (6 if model == 'counts8' else 9), 'Control feature count changed')
        require(set(p.get('code', {})) == {'code/control_radius.py',
                    'vendor/recovery/vendor/original/code/fit_controls_v1.py',
                    'vendor/recovery/vendor/original/code/site_data_v1.py'}, 'Control source inventory changed')
        sources = [(ROOT, p.get('code', {}))]
    for folder, inventory in sources:
        require(isinstance(inventory, dict) and inventory, 'Missing fit source provenance')
        for name, digest in inventory.items():
            require(sha(safe_file(folder, name)) == digest, 'Fit source differs from reviewed package: '+name)


def analyse(arrays, expected_events):
    """Extend the frozen whole-tie metric implementation to six fixed budgets."""
    population = base.validate_arrays(arrays, expected_events)
    micro = metric.validate_charge(arrays)
    population['microcurler_electrons'] = int(micro.sum())
    prefixes = base.rank_prefixes(arrays['logits'], arrays['target'], arrays['primary_electron_weight'])
    points, event_rows = [], []
    for budget in BUDGETS_PPM:
        cutoff = base.cutoff_at_budget(prefixes, budget)
        measured, events = base.metrics_at_point(arrays, cutoff)
        flags = base.decision_flags(arrays['logits'], cutoff)
        total_mc, caught_mc = int(micro.sum()), int(micro[flags].sum())
        measured.update(microcurler_electrons=total_mc, flagged_microcurler_electrons=caught_mc,
                        microcurler_charge_recall=base.ratio(caught_mc, total_mc),
                        flagged_microcurler_electrons_on_ignored_sites=int(micro[flags & (arrays['target'] == -1)].sum()))
        for pos, event in enumerate(events):
            lo, hi = map(int, arrays['event_ptr'][pos:pos+2])
            total = int(micro[lo:hi].sum())
            caught = int(micro[lo:hi][flags[lo:hi]].sum())
            event.update(microcurler_electrons=total, flagged_microcurler_electrons=caught,
                         microcurler_charge_recall=base.ratio(caught, total))
        losses = [e['primary_loss_fraction'] for e in events if e['primary_electrons']]
        measured['primary_event_loss_quantiles'] = {
            str(q): float(np.quantile(losses, q, method='linear')) if losses else None
            for q in (.5, .9, .95, .99, 1.)
        }
        recalls = [e['microcurler_charge_recall'] for e in events if e['microcurler_electrons']]
        measured['macro_microcurler_charge_recall'] = float(np.mean(recalls)) if recalls else None
        points.append(dict(operating_point=cutoff, metrics=measured))
        event_rows.extend(dict(retention_target=1.-budget/1e6, primary_loss_budget_ppm=budget, **e) for e in events)
    return dict(population=population,
                class_balanced_bce=base.class_balanced_bce(arrays['logits'], arrays['target']),
                points=points), event_rows


def flatten(record, point):
    row = old_report.flatten(record, point)
    return dict(radius_mm=record['radius_mm'], checkpoint=record['checkpoint'],
                retention_target=1.-point['operating_point']['target']/1e6, **row)


def summaries(rows):
    groups = {}
    for row in rows:
        key = (row['checkpoint'], row['radius_mm'], row['model'], row['target'])
        groups.setdefault(key, []).append(row)
    out = []
    for (checkpoint, radius, model, target), values in sorted(groups.items()):
        deterministic = model in CONTROLS
        required = {SEEDS[0]} if deterministic else set(SEEDS)
        seen = [row['seed'] for row in values]
        require(len(set(seen)) == len(seen), 'Repeated seed within radius/model/checkpoint group')
        require(set(seen).issubset(required), 'Unexpected seed in report group')
        complete = set(seen) == required
        row = dict(radius_mm=radius, model=model, checkpoint=checkpoint,
                   primary_loss_budget_ppm=target, retention_target=1.-target/1e6,
                   seeds=sorted(seen), required_seeds=sorted(required),
                   deterministic=deterministic, coverage='COMPLETE' if complete else 'PARTIAL', statistics={})
        if complete:
            for key in SUMMARY_KEYS:
                numbers = [r[key] for r in values if r[key] is not None]
                row['statistics'][key] = dict(
                    mean=float(np.mean(numbers)) if numbers else None,
                    minimum=min(numbers) if numbers else None,
                    maximum=max(numbers) if numbers else None,
                    defined_runs=len(numbers))
        out.append(row)
    return out


def paired_radius_deltas(rows):
    """Descriptive matched-seed deltas; not independent-data significance tests."""
    groups = {(r['checkpoint'], r['model'], r['radius_mm'], r['seed'], r['target']): r for r in rows}
    result = []
    for key, row in sorted(groups.items()):
        checkpoint, model, radius, seed, budget = key
        if radius == 2:
            continue
        reference = groups.get((checkpoint, model, 2, seed, budget))
        if reference is None:
            continue
        result.append(dict(checkpoint=checkpoint, model=model, radius_mm=radius,
                           reference_radius_mm=2, seed=seed, primary_loss_budget_ppm=budget,
                           **{key+'_difference': None if row[key] is None or reference[key] is None
                              else row[key]-reference[key] for key in SUMMARY_KEYS}))
    return result


def compact_curve(arrays):
    """Keep every score tie; avoid storing ratios already determined by counts."""
    curve = metric.full_curve(arrays)
    names = ('threshold_logit', 'threshold_flag_none', 'flagged_sites',
             'flagged_microcurler_sites', 'lost_primary_electrons',
             'flagged_microcurler_electrons', 'total_microcurler_sites',
             'total_microcurler_electrons', 'total_primary_electrons')
    return {key: curve[key] for key in names}


def collect_runtime(verified, radius, model, seed):
    meta = verified['meta']
    row = dict(radius_mm=radius, model=model, seed=seed, deterministic=model in CONTROLS,
               fit_seconds=meta.get('fit_seconds'), selected_epoch=meta.get('selected_epoch'),
               parameters=meta.get('parameters', verified['complete']['identity'].get('parameters')),
               validation_seconds_sum=None, profile_sha256=None, execution_profile=None,
               graph_queries=None, mean_executed_edges_per_query=None,
               graph_preparation_seconds=None)
    if verified['history']:
        values = [e.get('validation_seconds') for e in verified['history']]
        if all(v is not None and math.isfinite(v) and v >= 0 for v in values):
            row['validation_seconds_sum'] = float(sum(values))
    if verified['profile'] is not None:
        row['execution_profile'] = verified['profile']
        row['profile_sha256'] = sha(verified['folder']/'execution_profile.json')
        counts = [s.get('counters', {}) for s in verified['profile'].get('sessions', [])]
        if counts and all(all(k in c for k in ('graph_queries', 'total_executed_edges', 'builder_seconds', 'graph_seconds')) for c in counts):
            queries = sum(c['graph_queries'] for c in counts)
            edges = sum(c['total_executed_edges'] for c in counts)
            row.update(graph_queries=queries,
                       mean_executed_edges_per_query=edges/queries if queries else None,
                       graph_preparation_seconds=sum(c['builder_seconds']+c['graph_seconds'] for c in counts))
    return row


def make_plots(out, summary, curves, runtime, status):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'full1': '#276FBF', 'counts8': '#A86C2C', 'shape8': '#737373'}
    by_point = {(r['model'], r['radius_mm'], r['primary_loss_budget_ppm']): r
                for r in summary if r['checkpoint'] == 'selected' and r['coverage'] == 'COMPLETE'}
    keys = ('microcurler_site_recall', 'macro_microcurler_site_recall',
            'fraction_primary_events_losing_above_5_percent')
    titles = ('Pooled microcurler-site recall', 'Equal-event microcurler-site recall',
              'Primary-bearing events losing >5% charge')
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    for col, budget in enumerate(MAIN_BUDGETS):
        for row, (key, title) in enumerate(zip(keys, titles)):
            ax = axes[row, col]
            for model in ('full1', *CONTROLS):
                available = [(r, by_point[(model, r, budget)]) for r in RADII if (model, r, budget) in by_point]
                if not available:
                    continue
                x = np.asarray([p[0] for p in available])
                stats = [p[1]['statistics'][key] for p in available]
                mean = np.asarray([s['mean'] if s['mean'] is not None else np.nan for s in stats])*100
                low = np.asarray([s['minimum'] if s['minimum'] is not None else np.nan for s in stats])*100
                high = np.asarray([s['maximum'] if s['maximum'] is not None else np.nan for s in stats])*100
                ax.plot(x, mean, 'o-', label=LABELS[model], color=colors[model])
                if model == 'full1':
                    ax.fill_between(x, low, high, alpha=.14, color=colors[model])
            ax.set_xticks(RADII)
            ax.set_xlabel('Neighborhood radius (mm)')
            ax.set_ylabel('Percent')
            ax.grid(alpha=.2)
            if row < 2:
                ax.set_ylim(0, 100)
            else:
                ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(f'{100-budget/10000:g}% primary-retention target')
            if col == 0:
                ax.text(-.22, .5, title, transform=ax.transAxes, rotation=90,
                        va='center', ha='center', fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, fontsize=8, loc='lower right')
    fig.suptitle(f'FIMS radius sensitivity - selected checkpoints - {status}', fontsize=16)
    fig.supxlabel('Development validation; cutoffs chosen on these same events. GNN: three-seed mean/range; controls: one deterministic fit.', fontsize=9)
    p1 = out/'fims_radius_comparison_v1.png'
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    rcolors = {1: '#14968B', 2: '#276FBF', 4: '#D05A36'}
    seen = set()
    for record, path in curves:
        with np.load(path, allow_pickle=False) as c:
            primary, positive = int(c['total_primary_electrons']), int(c['total_microcurler_sites'])
            cost = c['lost_primary_electrons']/primary*100 if primary else np.full(len(c['flagged_sites']), np.nan)
            recall = c['flagged_microcurler_sites']/positive*100 if positive else np.full(len(c['flagged_sites']), np.nan)
        model, radius = record['model'], record['radius_mm']
        for ax, limit in zip(axes, (2, 10)):
            ids = np.flatnonzero(np.isfinite(cost) & np.isfinite(recall) & (cost <= limit))
            if len(ids) and ids[-1]+1 < len(cost):
                # Include the next complete tie so the step extends to the axis boundary.
                ids = np.r_[ids, ids[-1]+1]
            if len(ids) > 3000:
                ids = ids[np.unique(np.linspace(0, len(ids)-1, 3000).astype(int))]
            tag = (model, radius, limit)
            label = f'{LABELS[model]}, {radius} mm' if tag not in seen else None
            ax.step(cost[ids], recall[ids], where='post', color=rcolors[radius],
                    linestyle='-' if model == 'full1' else '--' if model == 'shape8' else ':',
                    alpha=.65, linewidth=1.1, label=label)
            seen.add(tag)
    for ax, limit in zip(axes, (2, 10)):
        ax.set(xlim=(0, limit), ylim=(0, 100), xlabel='Pooled primary charge lost (%)',
               ylabel='Microcurler-site recall (%)')
        ax.grid(alpha=.2)
    axes[0].set_title('Strict-retention region')
    axes[1].set_title('Broader operating region')
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, fontsize=7.5, ncol=2)
    fig.suptitle(f'FIMS radius sensitivity - all completed selected seeds - {status}')
    fig.supxlabel('Exact complete-score-tie curves are saved in NPZ files; plotted points are sampled visually. No independent test confirmation.', fontsize=9)
    p2 = out/'fims_radius_retention_curves_v1.png'
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axes[0]
    for model in ('full1', *CONTROLS):
        items = [r for r in runtime if r['model'] == model and r['fit_seconds'] is not None]
        if not items:
            continue
        ax.scatter([r['radius_mm'] for r in items], [r['fit_seconds']/3600 for r in items],
                   color=colors[model], label=LABELS[model], alpha=.7)
    ax.set_xticks(RADII)
    ax.set(xlabel='Neighborhood radius (mm)', ylabel='Recorded fit time (hours)',
           title='Recorded software cost; all completed runs shown')
    ax.grid(alpha=.2)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax = axes[1]
    items = [r for r in runtime if r['model'] == 'full1' and r['mean_executed_edges_per_query'] is not None]
    if items:
        ax.scatter([r['radius_mm'] for r in items], [r['mean_executed_edges_per_query'] for r in items],
                   color=colors['full1'], alpha=.7)
    else:
        ax.text(.5, .5, 'No committed graph profiles available', transform=ax.transAxes, ha='center')
    ax.set_xticks(RADII)
    ax.set(xlabel='Neighborhood radius (mm)', ylabel='Mean executed incoming edges per query',
           title='Recorded one-layer graph workload')
    ax.grid(alpha=.2)
    fig.supxlabel('Fit time is the trainer\'s recorded invocation time; resume sessions may differ. Profiles count repeated query visits. Queue delay is excluded.', fontsize=9)
    p3 = out/'fims_radius_runtime_v1.png'
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    return [p1, p2, p3]


def summary_text(status, records, missing, summary):
    lines = ['FIMS RADIUS SENSITIVITY V1: '+status, '',
             'Fixed 7,500 training / 2,500 development-validation events; no new detector simulation.',
             'Nine fresh one-layer GNN fits: 1/2/4 mm x three seeds. Six deterministic count/shape fits.',
             'Both observation and connection radii vary together; this does not isolate connectivity from field of view.',
             'Spatial offsets retain the fixed 2 mm normalization; eight 25 ns bins and closed-current-bin access are unchanged.',
             'The GNN uses mathematically equivalent query-only incoming-edge execution for its ONE layer.',
             'Controls use nested count radii R/4, R/2, R and shape moments over R; shape distances are divided by 2 mm.',
             '', 'MAIN COMPARISON: selected checkpoints, all three GNN seeds required per group.',
             'Retention target | radius | model | site recall | equal-event site recall | actual pooled retention | events losing >5% charge']
    pct = lambda x: 'undefined' if x is None else f'{100*x:.4f}%'
    for item in sorted(summary, key=lambda s: (s['primary_loss_budget_ppm'], s['radius_mm'], s['model'])):
        if item['checkpoint'] != 'selected':
            continue
        label = LABELS[item['model']]
        if item['coverage'] != 'COMPLETE':
            lines.append(f'{100*item["retention_target"]:g}% | {item["radius_mm"]} mm | {label} | PARTIAL: seeds {item["seeds"]}; no pooled seed mean')
            continue
        stats = item['statistics']
        values = [stats[k]['mean'] for k in ('microcurler_site_recall', 'macro_microcurler_site_recall',
                                            'primary_charge_retention', 'fraction_primary_events_losing_above_5_percent')]
        lines.append(f'{100*item["retention_target"]:g}% | {item["radius_mm"]} mm | {label} | '+' | '.join(map(pct, values)))
    lines += ['', f'Completed report endpoints (selected and final diagnostic): {len(records)}',
              f'Missing planned fits: {len(missing)}',
              'Final-epoch predictions are reported separately in JSON/CSV; they are not substituted for selected checkpoints.',
              '99.95/99.9/99.5/99.0% are retained for continuity; 95/90% are broader descriptive comparisons.',
              'Thresholds are fitted and reported on the same validation events. These are development results, not independent confirmation.',
              'Equal-event recall averages only events containing supervised microcurler sites; primary-loss tails use only primary-bearing events.',
              'All primary charge counts, including charge at ignored sites; empty and microcurler-free events remain in the sample.',
              'The >5% event-loss boundary is a diagnostic, not an approved detector requirement.',
              'A seed range is not a confidence interval for new data. Site/charge metrics do not establish track-reconstruction efficiency.',
              'Detector upgrades, noise/background overlays and continuous-hit training are not part of this experiment.']
    if missing:
        lines += ['', 'Missing fits:']+[json.dumps(m, sort_keys=True) for m in missing]
    return '\n'.join(lines)+'\n'


def build_report(study, data, *, allow_partial=False, synthetic=False):
    study = Path(study).resolve()
    if not synthetic:
        require(len(data.validation) == 2500, 'Expected exactly 2500 validation events')
    require(not (study/'results/report').resolve().is_relative_to(data.base),
            'Report cannot be inside the frozen dataset')
    out = study/'results/report'
    out.mkdir(parents=True, exist_ok=True)
    (out/'curves').mkdir(exist_ok=True)
    native = load_native(data)
    jobs = [(radius, 'full1', seed, study/'results/fits'/f'r{radius}'/'full1'/f'seed{seed}')
            for radius in RADII for seed in SEEDS]
    jobs += [(radius, model, SEEDS[0], study/'results/controls'/f'r{radius}'/model/f'seed{SEEDS[0]}')
             for radius in RADII for model in CONTROLS]
    records, rows, events, missing, curves, runtime, lineage = [], [], [], [], [], [], []
    for radius, model, seed, folder in jobs:
        if not (folder/'complete.json').exists():
            missing.append(dict(radius_mm=radius, model=model, seed=seed))
            continue
        verified = verify_fit(folder, radius=radius, model=model, seed=seed,
                              site_manifest_sha256=data.manifest_sha)
        if not synthetic:
            verify_scientific_protocol(verified, radius, model)
        lineage.append(dict(radius_mm=radius, model=model, seed=seed,
                            fit_relative_path=str(folder.relative_to(study)),
                            complete_sha256=verified['complete_sha256'],
                            protocol_sha256=verified['protocol_sha256'],
                            protocol=verified['protocol'], output_hashes=verified['complete']['outputs']))
        runtime.append(collect_runtime(verified, radius, model, seed))
        for checkpoint in (('selected',) if verified['deterministic'] else ('selected', 'final')):
            arrays = load_predictions(verified, native, checkpoint, expected_events=len(data.validation))
            record = dict(radius_mm=radius, model=model, seed=seed, checkpoint=checkpoint,
                          epoch=verified['meta']['selected_epoch'] if checkpoint == 'selected' else 10,
                          parameters=verified['complete']['identity']['parameters'],
                          deterministic=verified['deterministic'],
                          protocol_sha256=verified['protocol_sha256'],
                          predictions_sha256=verified['complete']['outputs'][checkpoint+'_predictions.npz'],
                          complete_sha256=verified['complete_sha256'])
            analysis, event_rows = analyse(arrays, len(data.validation))
            record.update(analysis)
            records.append(record)
            rows.extend(flatten(record, point) for point in analysis['points'])
            events.extend(dict(radius_mm=radius, model=model, seed=seed, checkpoint=checkpoint, **e) for e in event_rows)
            if checkpoint == 'selected':
                path = out/'curves'/f'r{radius}_{model}_seed{seed}_selected_full_curve.npz'
                write_npz(path, compact_curve(arrays))
                curves.append((record, path))
            print(f'RADIUS REPORT: {radius} mm {model} seed={seed} {checkpoint}: verified', flush=True)
    status = 'PARTIAL' if missing else 'COMPLETE'
    # Never publish a misleading complete artifact after a failed final report.
    if missing and not allow_partial:
        raise ValueError('Planned fits are incomplete; use --allow-partial for an explicitly partial report: '+json.dumps(missing))
    data.verify_unchanged()
    summary = summaries(rows)
    paired = paired_radius_deltas(rows)
    outputs = []
    payload = dict(schema='fims_radius_sensitivity_report_v1', status=status,
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   site_manifest_sha256=data.manifest_sha,
                   retention_targets=[1.-b/1e6 for b in BUDGETS_PPM],
                   expected_fits=15, completed_fits=len(runtime), missing_fits=missing,
                   checkpoint_roles={'selected': 'primary comparison; earliest minimum ordinary validation BCE',
                                     'final': 'diagnostic only; epoch 10; no alternate target-specific selection'},
                   scope='development validation only; no independent calibration or test data',
                   results=records, seed_summaries=summary, paired_radius_vs_2mm=paired)
    files = {
        'fims_radius_metrics_v1.json': payload,
        'fims_radius_runtime_v1.json': dict(records=runtime, interpretation='software runtime; excludes queue wait; not hardware latency'),
        'fims_radius_provenance_v1.json': dict(
            schema='fims_radius_report_provenance_v1', site_manifest_sha256=data.manifest_sha,
            portable_manifest_sha256=data.manifest.get('portable_manifest_sha256'),
            preserved_manifest_sha256=data.manifest.get('preserved_manifest_sha256'),
            experiment_plan=json.loads((ROOT/'EXPERIMENT_PLAN.json').read_text()),
            source_receipt=(json.loads((study/'results/source_receipt.json').read_text())
                            if (study/'results/source_receipt.json').is_file() else None),
            validation_event_indices=list(map(int, data.validation)), fits=lineage,
            report_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (
                Path(__file__), VENDOR/'metrics_parallel_v1.py', VENDOR/'vendor_metrics/capacity_metrics_v2.py',
                VENDOR/'site_data_v1.py', VENDOR/'report_parallel_v1.py')}),
    }
    for name, value in files.items():
        write_json(out/name, value)
        outputs.append(out/name)
    for name, value in (
        ('fims_radius_operating_points_v1.csv', rows),
        ('fims_radius_event_metrics_v1.csv', events),
        ('fims_radius_paired_differences_v1.csv', paired),
    ):
        write_csv(out/name, value)
        outputs.append(out/name)
    text = out/'fims_radius_summary_v1.txt'
    atomic_bytes(text, summary_text(status, records, missing, summary).encode())
    outputs.append(text)
    outputs += make_plots(out, summary, curves, runtime, status)
    outputs += [path for record, path in curves]
    inventory = {str(path.relative_to(out)): sha(path) for path in outputs}
    marker_name = 'fims_radius_complete_v1.json'
    marker = dict(schema='fims_radius_report_complete_v1', status=status,
                  site_manifest_sha256=data.manifest_sha, expected_fits=15,
                  completed_fits=len(runtime), missing_fits=missing, outputs=inventory)
    marker_bytes = (json.dumps(marker, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    target = out/'fims_radius_sensitivity_review_v1.zip'
    require(not target.is_symlink(), 'Unsafe review archive destination')
    fd, tmp = tempfile.mkstemp(prefix='.radius_review_', suffix='.zip', dir=out)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in inventory:
                z.write(out/name, name)
            z.writestr(marker_name, marker_bytes)
        with zipfile.ZipFile(tmp) as z:
            require(z.testzip() is None, 'Review archive CRC failure')
            for name, digest in inventory.items():
                require(hashlib.sha256(z.read(name)).hexdigest() == digest, 'Review archive hash failure: '+name)
        os.replace(tmp, target)
        # Commit last: an earlier report's ZIP cannot satisfy a new completion marker.
        atomic_bytes(out/marker_name, marker_bytes)
    finally:
        Path(tmp).unlink(missing_ok=True)
    print('FIMS RADIUS SENSITIVITY V1: '+status, flush=True)
    print('Review ZIP: '+str(target), flush=True)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        import test_radius_report
        test_radius_report.run_tests()
        return
    if args.study_root is None or args.data is None:
        p.error('--study-root and --data are required')
    study = args.study_root.resolve()
    require(study == ROOT, 'Study root does not match the executed package')
    from radius_common import verify_package, verify_data_binding
    verify_package(study)
    verify_data_binding(study, args.data)
    data = SiteData(args.data)
    out = study/'results/report'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build_report(study, data, allow_partial=args.allow_partial)


if __name__ == '__main__':
    main()
