"""Plots for the read-only operating-point review of saved FIMS scores.

Histogram bins are for display only. Decisions and metrics always use the
original logits and whole equal-logit groups in the calling report.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LABELS = {
    'cnn_small': 'CNN small', 'cnn_wide': 'CNN wide',
    'full2': 'GNN 2-layer', 'full1': 'GNN 1-layer',
    'no_xy2': 'GNN without explicit x/y offsets',
    'current2': 'GNN current bin', 'counts8': 'Counts / 8 bins',
    'shape8': 'Counts + shape / 8 bins',
    'counts1': 'Counts / current bin',
    'shape1': 'Counts + shape / current bin',
}
MODEL_ORDER = tuple(LABELS)
MODEL_COLORS = {m: plt.get_cmap('tab10')(i) for i, m in enumerate(MODEL_ORDER)}
CUT_COLORS = {99.: '#0072B2', 95.: '#CC79A7', 90.: '#D55E00'}
TARGETS = (99.9, 99.5, 99., 95., 90.)


def sigmoid_for_display(logits):
    """Stable monotone transform, used only for an explicitly uncalibrated view."""
    z = np.asarray(logits, dtype=np.float64)
    p = np.empty(z.shape, dtype=np.float64)
    positive = z >= 0
    p[positive] = 1. / (1. + np.exp(-z[positive]))
    ex = np.exp(z[~positive])
    p[~positive] = ex / (1. + ex)
    return p


def histogram_data(arrays, bins=100):
    """Exact counts/charge per visual bin, including ignored-site primary charge."""
    z = np.asarray(arrays['logits'], dtype=np.float64)
    y = np.asarray(arrays['target'])
    w = np.asarray(arrays['primary_electron_weight'])
    if z.ndim != 1 or z.shape != y.shape or z.shape != w.shape:
        raise ValueError('Histogram arrays are not aligned')
    if not np.isfinite(z).all() or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError('Invalid histogram scores/labels')
    if w.dtype.kind not in 'iu' or np.any(w < 0):
        raise ValueError('Invalid histogram charge')
    if len(z):
        lo, hi = float(z.min()), float(z.max())
        if lo == hi:
            lo, hi = lo - 1., hi + 1.
        padding = max((hi-lo)*.015, 1e-5)
        logit_edges = np.linspace(lo-padding, hi+padding, bins+1)
    else:
        logit_edges = np.linspace(-1., 1., bins+1)
    score_edges = np.linspace(0., 1., bins+1)
    score = sigmoid_for_display(z)
    result = {'logit_edges': logit_edges, 'score_edges': score_edges}
    for space, values, edges in [('logit', z, logit_edges), ('score', score, score_edges)]:
        for name, mask in [('microcurler_sites', y == 1), ('other_sites', y == 0)]:
            result[f'{space}_{name}'] = np.histogram(values[mask], bins=edges)[0].astype(np.int64)
        result[f'{space}_primary_electrons'] = np.histogram(values, bins=edges, weights=w)[0]
        ignored = y == -1
        result[f'{space}_ignored_primary_electrons'] = np.histogram(values[ignored], bins=edges, weights=w[ignored])[0]
    return result


def _stairs_mass(ax, counts, edges, *, label, color, denominator=None, log=False, linestyle='-'):
    total = int(np.sum(counts)) if denominator is None else int(denominator)
    if total <= 0:
        ax.plot([], [], color=color, linestyle=linestyle, label=label+' (none)')
        return
    mass = np.asarray(counts, dtype=np.float64) / total * 100.
    if log:
        mass = np.where(mass > 0, mass, np.nan)
    ax.stairs(mass, edges, label=label, color=color, linewidth=1.5, linestyle=linestyle)


def _draw_cutoffs(ax, points, *, score=False):
    for p in points:
        retention = float(p['retention_percent'])
        if retention not in CUT_COLORS:
            continue
        cut = p['operating_point']
        if cut['threshold_kind'] != 'logit':
            continue
        value = float(cut['threshold_logit'])
        if score:
            value = float(sigmoid_for_display(np.asarray(value)))
        ax.axvline(value, color=CUT_COLORS[retention], linewidth=1.25, linestyle='--', alpha=.9)


def render_score_page(pdf, arrays, record, points):
    """Append one fit's score distributions and exact operating-point table.

Returns a dict of NumPy histogram arrays, suitable for np.savez_compressed.
No distributions or cutoffs are pooled across seeds.
"""
    hist = histogram_data(arrays)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.7))
    fig.subplots_adjust(left=.075, right=.98, top=.86, bottom=.19, hspace=.42, wspace=.28)
    ax_logit, ax_score, ax_primary, ax_table = axes.flat
    for ax, space, log in [(ax_logit, 'logit', True), (ax_score, 'score', False)]:
        edges = hist[space+'_edges']
        _stairs_mass(ax, hist[space+'_microcurler_sites'], edges,
                     label='Microcurler sites', color='#0072B2', log=log)
        _stairs_mass(ax, hist[space+'_other_sites'], edges,
                     label='Other supervised sites', color='#E69F00', log=log)
        _draw_cutoffs(ax, points, score=(space == 'score'))
        if log and (np.any(hist[space+'_microcurler_sites']) or np.any(hist[space+'_other_sites'])):
            ax.set_yscale('log')
        ax.grid(alpha=.2)
        ax.set_ylabel('Percent of each class per display bin')
        ax.set_xlabel('Saved classifier logit' if space == 'logit' else 'sigmoid(logit): classifier score')
        ax.legend(fontsize=8, loc='best')
    ax_logit.set_title('Class separation: logit view; log count scale', fontsize=11)
    ax_score.set_title('Score view: not a calibrated probability', fontsize=11)
    ax_score.set_xlim(0., 1.)
    primary = hist['logit_primary_electrons']
    ignored = hist['logit_ignored_primary_electrons']
    total_primary = int(np.sum(primary))
    _stairs_mass(ax_primary, primary, hist['logit_edges'], label='All primary charge',
                 color='#009E73', log=True, denominator=total_primary)
    _stairs_mass(ax_primary, ignored, hist['logit_edges'], label='Contribution on ignored sites',
                 color='#CC79A7', log=True, denominator=total_primary, linestyle=':')
    _draw_cutoffs(ax_primary, points)
    if total_primary:
        ax_primary.set_yscale('log')
    ax_primary.set_title('Useful charge at risk, including ignored sites', fontsize=11)
    ax_primary.set_xlabel('Saved classifier logit')
    ax_primary.set_ylabel('Percent of all primary charge per display bin')
    ax_primary.grid(alpha=.2)
    ax_primary.legend(fontsize=8, loc='best')
    ax_table.axis('off')
    cells = []
    pct = lambda value: 'undefined' if value is None else f'{100*value:.2f}%'
    for point in sorted(points, key=lambda p: -float(p['retention_percent'])):
        m, cut = point['metrics'], point['operating_point']
        cells.append([
            f'{point["retention_percent"]:g}%',
            'undefined' if m['primary_charge_retention'] is None else f'{100*m["primary_charge_retention"]:.4f}%',
            pct(m['microcurler_site_recall']),
            pct(m['primary_loss_tails']['above_5_percent']['fraction']),
            'none' if cut['threshold_kind'] == 'flag_none' else f'{cut["threshold_logit"]:.4g}',
        ])
    table = ax_table.table(cellText=cells,
        colLabels=['Retention\ntarget', 'Actual\nretention', 'Site\nrecall', 'Primary events\nlosing >5%', 'Logit\ncutoff'],
        cellLoc='center', loc='center', colWidths=[.18, .21, .18, .22, .21])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1., 2.1)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#DDDDDD')
        if row == 0:
            cell.set_facecolor('#EAF0F6')
            cell.set_text_props(weight='bold')
    ax_table.set_title('Metrics from exact cuts; equal-score groups kept intact', fontsize=10.5)
    model = LABELS.get(record['model'], record['model'])
    selection = 'deterministic control' if record.get('deterministic') else f'selected epoch {record.get("epoch")}'
    fig.suptitle(f'{model} | seed {record["seed"]} | {selection}', fontsize=15, y=.965)
    fig.text(.5, .925, 'Saved development-validation scores only; no new training, inference or simulation.',
             ha='center', fontsize=10)
    handles = [Line2D([0], [0], color=CUT_COLORS[t], linestyle='--', label=f'{t:g}% primary-retention target')
               for t in (99., 95., 90.)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .075), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .042, 'Flag scores at or above the cutoff. Cutoffs and metrics use the same 2,500 validation events: development only.',
             ha='center', fontsize=8.5)
    fig.text(.5, .021, 'Histogram bins are visual only; zeros are omitted on log axes. Ignored sites are excluded from class histograms, not from charge loss.',
             ha='center', fontsize=8.5)
    pdf.savefig(fig)
    plt.close(fig)
    return hist


def _metric_stat(summary, key, field):
    value = summary['statistics'][key][field]
    return np.nan if value is None else 100.*float(value)


def _curve(record):
    if 'curve' in record:
        return record['curve']
    with np.load(record['path'], allow_pickle=False) as z:
        return {k: z[k] for k in ('primary_charge_retention', 'microcurler_site_recall')}


def event_loss_survival(rows):
    """Exact fraction of primary-bearing events with loss strictly above x."""
    values = np.asarray([float(r['primary_loss_fraction']) for r in rows
                         if int(r['primary_electrons']) > 0], dtype=np.float64)
    if not len(values):
        return np.asarray([0., 1.]), np.asarray([np.nan, np.nan])
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError('Invalid event loss fraction')
    values.sort()
    x = np.unique(np.r_[0., values, 1.])
    survivors = len(values)-np.searchsorted(values, x, side='right')
    return x, survivors.astype(np.float64)/len(values)


def render_summary(out, seed_summaries, curve_records, event_rows):
    """Write overview PNGs. Neural bands are full seed ranges, not CIs.

seed_summaries uses report_parallel_v1.summaries schema; curve_records accepts
either {'model','seed','curve': array_dict} or {'model','seed','path': npz_path}.
event_rows contains model, seed, target (ppm) and per-event native metrics.
"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    produced = []
    models = [m for m in MODEL_ORDER if any(r['model'] == m for r in curve_records)]
    summary_map = {(r['model'], int(r['target'])): r for r in seed_summaries
                   if r['operating_point'] == 'primary_loss_budget' and r['coverage'] == 'COMPLETE'}
    budgets = (1000, 5000, 10000, 50000, 100000)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=.07, right=.98, top=.89, bottom=.22, hspace=.38, wspace=.22)
    keys = ('microcurler_site_recall', 'macro_microcurler_site_recall',
            'primary_charge_retention', 'fraction_primary_events_losing_above_5_percent')
    titles = ('Microcurler-site recall: pooled across sites', 'Microcurler-site recall: equal weight per positive event',
              'Actual pooled primary-charge retention', 'Primary-bearing events losing >5% of their charge')
    for model in models:
        if not all((model, b) in summary_map for b in budgets):
            continue
        summaries = [summary_map[model, b] for b in budgets]
        for ax, key in zip(axes.flat, keys):
            mean = [_metric_stat(s, key, 'mean') for s in summaries]
            low = [_metric_stat(s, key, 'minimum') for s in summaries]
            high = [_metric_stat(s, key, 'maximum') for s in summaries]
            ax.plot(range(len(budgets)), mean, 'o-', color=MODEL_COLORS[model], markersize=4,
                    linewidth=1.5, label=LABELS[model])
            ax.fill_between(range(len(budgets)), low, high, color=MODEL_COLORS[model], alpha=.1)
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(budgets)), [f'{x:g}%' for x in TARGETS])
        ax.set_xlabel('Development primary-retention target')
        ax.set_ylabel('Percent')
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .055), ncol=3, fontsize=9, frameon=False)
    fig.suptitle('FIMS operating-point review | selected fits | development validation', fontsize=15)
    fig.text(.5, .025, 'Neural models: three-seed means and full ranges. Controls: one deterministic fit. No detector acceptance limit is established.',
             ha='center', fontsize=9)
    dest = out/'fims_operating_point_comparison_v1.png'
    fig.savefig(dest, dpi=170)
    plt.close(fig)
    produced.append(dest)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.subplots_adjust(left=.07, right=.98, top=.85, bottom=.29, wspace=.23)
    seen = set()
    for record in curve_records:
        c = _curve(record)
        cost = 100.*(1.-np.asarray(c['primary_charge_retention']))
        recall = 100.*np.asarray(c['microcurler_site_recall'])
        model = record['model']
        for ax, limit in zip(axes, (1., 10.)):
            ids = np.flatnonzero(np.isfinite(cost)&np.isfinite(recall)&(cost <= limit))
            if len(ids) > 2500:
                ids = ids[np.unique(np.linspace(0, len(ids)-1, 2500).astype(np.int64))]
            label = LABELS.get(model, model) if (model, limit) not in seen else None
            ax.plot(cost[ids], recall[ids], color=MODEL_COLORS[model], alpha=.7, linewidth=1.1, label=label)
            seen.add((model, limit))
    for ax, limit, title in zip(axes, (1., 10.), ('Historical strict region', 'Broader region requested at the meeting')):
        ax.set_xlim(0., limit)
        ax.set_ylim(0., 100.)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Pooled primary charge lost (%)')
        ax.set_ylabel('Microcurler-site recall (%)')
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle('Every completed seed shown | curves from saved logits', fontsize=15)
    fig.text(.5, .01, 'Curve sampling is visual only. Exact whole-score-tie cutoffs are used for all tabulated metrics.', ha='center', fontsize=8.5)
    dest = out/'fims_operating_point_recall_curves_v1.png'
    fig.savefig(dest, dpi=170)
    plt.close(fig)
    produced.append(dest)

    grouped = {}
    for row in event_rows:
        if row.get('operating_point', 'primary_loss_budget') != 'primary_loss_budget':
            continue
        ppm = row.get('target')
        if ppm is None:
            ppm = round((100.-float(row['retention_percent']))*10000)
        key = (row['model'], int(row['seed']), int(ppm))
        grouped.setdefault(key, []).append(row)
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 5.7))
    fig.subplots_adjust(left=.055, right=.98, top=.83, bottom=.31, wspace=.24)
    seen = set()
    for (model, seed, ppm), rows in grouped.items():
        if ppm not in (10000, 50000, 100000):
            continue
        ax = axes[(10000, 50000, 100000).index(ppm)]
        x, survival = event_loss_survival(rows)
        label = LABELS.get(model, model) if (model, ppm) not in seen else None
        visible = np.where(survival > 0., survival*100., np.nan)
        ax.step(x*100., visible, where='post', color=MODEL_COLORS[model], alpha=.7, linewidth=1.1, label=label)
        seen.add((model, ppm))
    for ax, target in zip(axes, (99, 95, 90)):
        ax.set_xlim(0., 100.)
        ax.set_yscale('log')
        ax.set_ylim(.02, 100.)
        ax.set_title(f'{target}% pooled retention target', fontsize=11)
        ax.axvline(5., color='#222222', linestyle='--', linewidth=1.)
        ax.grid(alpha=.2)
        ax.set_xlabel('Primary charge lost within an event (%)')
        ax.set_ylabel('Primary-bearing events losing more (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .055), ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle('How charge loss is distributed across events | logarithmic frequency', fontsize=15)
    fig.text(.5, .015, 'Every seed shown separately; zero tails omitted. Dashed line: 5% diagnostic boundary, not an established acceptable loss.',
             ha='center', fontsize=9)
    dest = out/'fims_operating_point_event_loss_v1.png'
    fig.savefig(dest, dpi=170)
    plt.close(fig)
    produced.append(dest)
    return produced
