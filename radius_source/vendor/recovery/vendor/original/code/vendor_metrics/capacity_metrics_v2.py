#!/usr/bin/env python3
"""Development-only, whole-score-tie operating points for FIMS CNN capacity.

Recall counts supervised microcurler binary sites (target == 1). Primary charge
counts every primary electron, including contributions at ignored sites. These
quantities have different denominators. Thresholds below are selected and
described on the same development-validation events, not independent calibration.
"""
from __future__ import annotations
import math
import numpy as np

BUDGETS_PPM = (500, 1000, 5000, 10000)
RECALL_TARGETS_PERCENT = (80, 90)
IDENTITY_KEYS = ('target', 'primary_electron_weight', 'event_ptr', 'event_index',
                 'event_id_raw', 'event_run_uid', 'event_shard')


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def validate_arrays(arrays, expected_events=None):
    require(set(IDENTITY_KEYS + ('logits', 'protocol_sha256')).issubset(arrays), 'Missing prediction fields')
    y = np.asarray(arrays['target']); z = np.asarray(arrays['logits'])
    w = np.asarray(arrays['primary_electron_weight']); ptr = np.asarray(arrays['event_ptr'])
    require(y.ndim == 1 and y.dtype.kind in 'iu' and np.isin(y, [-1, 0, 1]).all(), 'Invalid target')
    require(z.shape == y.shape and z.dtype.kind == 'f' and np.isfinite(z).all(), 'Invalid logits')
    require(w.shape == y.shape and w.dtype.kind in 'iu' and np.all(w >= 0), 'Invalid primary weights')
    require(sum(map(int, w)) <= np.iinfo(np.int64).max, 'Primary weights overflow int64')
    require(ptr.ndim == 1 and len(ptr) >= 1 and ptr.dtype.kind in 'iu' and ptr[0] == 0
            and ptr[-1] == len(y) and np.all(ptr[1:] >= ptr[:-1]), 'Invalid event pointers')
    ne = len(ptr) - 1
    if expected_events is not None:
        require(ne == expected_events, 'Wrong validation event count')
    for key in ('event_index', 'event_id_raw'):
        v = np.asarray(arrays[key])
        require(v.shape == (ne,) and v.dtype.kind in 'iu' and np.all(v >= 0), 'Invalid ' + key)
    for key in ('event_run_uid', 'event_shard'):
        v = np.asarray(arrays[key])
        require(v.shape == (ne,) and v.dtype.kind in 'US', 'Invalid ' + key)
    require(len(set(map(int, arrays['event_index']))) == ne, 'Duplicate dataset event indices')
    keys = [(str(arrays['event_shard'][i]), str(arrays['event_run_uid'][i]), int(arrays['event_id_raw'][i])) for i in range(ne)]
    require(len(set(keys)) == ne, 'Duplicate physical event identities')
    p = np.asarray(arrays['protocol_sha256'])
    require(p.ndim == 0 and p.dtype.kind in 'US' and len(str(p.item())) == 64, 'Invalid protocol hash')
    return dict(events=ne, sites=len(y), microcurler_sites=int(np.count_nonzero(y == 1)),
                other_sites=int(np.count_nonzero(y == 0)), ignored_sites=int(np.count_nonzero(y == -1)),
                primary_electrons=sum(map(int, w)), empty_events=int(np.count_nonzero(np.diff(ptr) == 0)))


def class_balanced_bce(logits, target):
    z = np.asarray(logits, np.float64); y = np.asarray(target)
    require(z.shape == y.shape and z.ndim == 1 and np.isfinite(z).all()
            and np.isin(y, [-1, 0, 1]).all(), 'Invalid BCE arrays')
    terms = []
    for cls in (0, 1):
        a = z[y == cls]
        if not len(a):
            return None
        signed = a if cls == 0 else -a
        terms.append(float(np.mean(np.maximum(signed, 0) + np.log1p(np.exp(-np.abs(signed))))))
    return float(sum(terms) / 2)


def rank_prefixes(logits, target, primary):
    z, y, w = map(np.asarray, (logits, target, primary))
    require(z.ndim == y.ndim == w.ndim == 1 and z.shape == y.shape == w.shape, 'Rank alignment error')
    require(z.dtype.kind == 'f' and np.isfinite(z).all(), 'Nonfinite/nonfloating scores')
    require(y.dtype.kind in 'iu' and np.isin(y, [-1, 0, 1]).all(), 'Invalid ranked target')
    require(w.dtype.kind in 'iu' and np.all(w >= 0), 'Invalid ranked primary counts')
    total = sum(map(int, w)); require(total <= np.iinfo(np.int64).max, 'Primary total overflow')
    order = np.argsort(-z, kind='stable')
    if len(z):
        ranked = z[order]
        ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]) + 1, len(z)]
        cost = np.cumsum(w[order], dtype=np.int64)[ends - 1]
        caught = np.cumsum(y[order] == 1, dtype=np.int64)[ends - 1]
        cut = ranked[ends - 1]
    else:
        ends = cost = caught = np.array([], np.int64); cut = np.array([], z.dtype)
    return dict(group_ends=ends, cost=cost, caught=caught, cut=cut,
                total_primary=total, total_microcurler=int(np.count_nonzero(y == 1)))


def _point(prefix, index):
    return dict(threshold_kind='logit' if index >= 0 else 'flag_none',
                threshold_logit=float(prefix['cut'][index]) if index >= 0 else None,
                selected_groups=index + 1,
                flagged_sites=int(prefix['group_ends'][index]) if index >= 0 else 0,
                flagged_microcurler_sites=int(prefix['caught'][index]) if index >= 0 else 0,
                lost_primary_electrons=int(prefix['cost'][index]) if index >= 0 else 0,
                decision_rule='flag logits >= cutoff; never split equal-logit groups',
                purpose='Descriptive development-validation operating point; not an adopted deployment cutoff')


def cutoff_at_budget(prefix, allowed_loss_ppm):
    require(isinstance(allowed_loss_ppm, (int, np.integer)) and 0 <= allowed_loss_ppm <= 1000000, 'Invalid ppm budget')
    cap = prefix['total_primary'] * int(allowed_loss_ppm) // 1000000
    index = int(np.searchsorted(prefix['cost'], cap, side='right')) - 1
    point = _point(prefix, index)
    point.update(kind='primary_loss_budget', target=int(allowed_loss_ppm), target_units='ppm',
                 allowed_primary_electron_loss=cap,
                 status='defined' if prefix['total_primary'] else 'no_primary_electrons')
    return point


def cutoff_at_recall(prefix, recall_percent):
    """Least permissive complete prefix achieving ceil(Npositive * percent/100)."""
    require(isinstance(recall_percent, (int, np.integer)) and 0 <= recall_percent <= 100, 'Invalid recall target')
    count = prefix['total_microcurler']
    needed = (count * int(recall_percent) + 99) // 100
    index = int(np.searchsorted(prefix['caught'], needed, side='left')) if needed else -1
    point = _point(prefix, index)
    point.update(kind='microcurler_recall_target', target=int(recall_percent), target_units='percent',
                 required_microcurler_sites=needed,
                 status='defined' if count else 'no_microcurler_sites')
    return point


def decision_flags(logits, point):
    if point['threshold_kind'] == 'flag_none':
        require(point['threshold_logit'] is None, 'Invalid flag-none cutoff')
        return np.zeros(len(logits), bool)
    require(point['threshold_kind'] == 'logit' and np.isfinite(point['threshold_logit']), 'Invalid cutoff')
    return np.asarray(logits) >= point['threshold_logit']


def row_metrics(flags, y, primary):
    npos = int(np.count_nonzero(y == 1)); nother = int(np.count_nonzero(y == 0))
    tp = int(np.count_nonzero(flags & (y == 1))); fp = int(np.count_nonzero(flags & (y == 0)))
    charge = sum(map(int, primary)); lost = sum(map(int, primary[flags]))
    recall = ratio(tp, npos); retention = ratio(nother - fp, nother)
    return dict(sites=len(y), microcurler_sites=npos, other_sites=nother,
                ignored_sites=int(np.count_nonzero(y == -1)), flagged_sites=int(flags.sum()),
                flagged_microcurler_sites=tp, flagged_other_sites=fp,
                flagged_ignored_sites=int(np.count_nonzero(flags & (y == -1))),
                microcurler_site_recall=recall, supervised_site_precision=ratio(tp, tp + fp),
                other_site_retention=retention,
                balanced_accuracy=(recall + retention) / 2 if recall is not None and retention is not None else None,
                primary_electrons=charge, lost_primary_electrons=lost,
                lost_primary_electrons_on_ignored_sites=sum(map(int, primary[flags & (y == -1)])),
                primary_loss_fraction=ratio(lost, charge), primary_loss_ppm=ratio(lost * 1000000, charge),
                primary_charge_retention=ratio(charge - lost, charge))


def metrics_at_point(arrays, point):
    flags = decision_flags(arrays['logits'], point)
    y, w, ptr = [arrays[k] for k in ('target', 'primary_electron_weight', 'event_ptr')]
    aggregate = row_metrics(flags, y, w); events = []
    for i in range(len(ptr) - 1):
        lo, hi = map(int, ptr[i:i+2])
        row = row_metrics(flags[lo:hi], y[lo:hi], w[lo:hi])
        row.update(event_position=i, event_index=int(arrays['event_index'][i]),
                   event_id_raw=int(arrays['event_id_raw'][i]), event_run_uid=str(arrays['event_run_uid'][i]),
                   event_shard=str(arrays['event_shard'][i]))
        events.append(row)
    positive = [e for e in events if e['microcurler_sites']]
    primary = [e for e in events if e['primary_electrons']]
    tails = {}
    for pct, divisor in ((1, 100), (5, 20), (10, 10)):
        n = sum(e['lost_primary_electrons'] * divisor > e['primary_electrons'] for e in primary)
        tails[f'above_{pct}_percent'] = dict(events=n, primary_bearing_events=len(primary), fraction=ratio(n, len(primary)), comparison='>')
    aggregate.update(events=len(events), microcurler_bearing_events=len(positive), primary_bearing_events=len(primary),
                     macro_microcurler_site_recall=float(np.mean([e['microcurler_site_recall'] for e in positive])) if positive else None,
                     microcurler_events_with_any_true_flag=sum(e['flagged_microcurler_sites'] > 0 for e in positive),
                     fraction_microcurler_events_with_any_true_flag=ratio(sum(e['flagged_microcurler_sites'] > 0 for e in positive), len(positive)),
                     macro_primary_charge_retention=float(np.mean([e['primary_charge_retention'] for e in primary])) if primary else None,
                     maximum_event_primary_loss_fraction=max((e['primary_loss_fraction'] for e in primary), default=None),
                     primary_loss_tails=tails)
    if 'selected_groups' in point:
        for key in ('flagged_sites', 'flagged_microcurler_sites', 'lost_primary_electrons'):
            require(aggregate[key] == point[key], 'Whole-tie prefix and actual flags disagree: ' + key)
    if point.get('kind') == 'primary_loss_budget':
        require(aggregate['lost_primary_electrons'] <= point['allowed_primary_electron_loss'], 'Integer charge budget exceeded')
    if point.get('kind') == 'microcurler_recall_target':
        require(aggregate['flagged_microcurler_sites'] >= point['required_microcurler_sites'], 'Recall target not reached')
    return aggregate, events


def analyse(arrays, expected_events=2500):
    population = validate_arrays(arrays, expected_events)
    prefix = rank_prefixes(arrays['logits'], arrays['target'], arrays['primary_electron_weight'])
    points = [cutoff_at_budget(prefix, p) for p in BUDGETS_PPM]
    points += [cutoff_at_recall(prefix, p) for p in RECALL_TARGETS_PERCENT]
    points += [dict(kind='fixed_logit', target=0, target_units='logit', threshold_kind='logit', threshold_logit=0.0,
                    status='defined', purpose='Fixed 0.5 probability threshold; not retention matched')]
    results = []; rows = []
    for point in points:
        metrics, event_rows = metrics_at_point(arrays, point)
        results.append(dict(operating_point=point, metrics=metrics))
        rows.extend([dict(operating_point=point['kind'], target=point['target'], target_units=point['target_units'], **r) for r in event_rows])
    return dict(population=population, class_balanced_bce=class_balanced_bce(arrays['logits'], arrays['target']), points=results), rows
