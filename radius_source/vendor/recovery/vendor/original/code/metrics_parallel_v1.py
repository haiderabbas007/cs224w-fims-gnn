#!/usr/bin/env python3
"""Development metrics with exact tie/budget semantics and charge attribution.

The original capacity metrics are vendored unchanged. Additional charge recall
uses every accepted microcurler electron, including those at ignored/mixed
sites. It is deliberately distinct from supervised-site recall.
"""
from __future__ import annotations
import numpy as np
from vendor_metrics import capacity_metrics_v2 as base

SEEDS = (20260924, 20260925, 20260926)
BUDGETS_PPM = (500, 1000, 5000, 10000)
RECALL_TARGETS_PERCENT = (80, 90)
require = base.require


def validate_charge(arrays):
    w = np.asarray(arrays['microcurler_electron_weight'])
    require(w.shape == arrays['target'].shape and w.dtype.kind in 'iu' and
            np.all(w >= 0), 'Invalid microcurler electron attribution')
    require(sum(map(int, w)) <= np.iinfo(np.int64).max, 'Microcurler charge overflow')
    return w


def full_curve(arrays):
    """Every complete equal-score group, plus flag-none; no interpolation."""
    z, y, w = [np.asarray(arrays[k]) for k in
               ('logits', 'target', 'primary_electron_weight')]
    mw = validate_charge(arrays)
    p = base.rank_prefixes(z, y, w)
    order = np.argsort(-z, kind='stable')
    ends = p['group_ends']
    mcharge = np.cumsum(mw[order], dtype=np.int64)[ends - 1] if len(ends) else np.empty(0, np.int64)
    npositive = p['total_microcurler']; nprimary = p['total_primary']; nmcharge = int(mw.sum())
    # Index zero encodes flag-none with NaN cutoff only in numeric NPZ, not JSON.
    return dict(threshold_logit=np.r_[np.nan, p['cut']],
                threshold_flag_none=np.r_[True, np.zeros(len(ends), bool)],
                flagged_sites=np.r_[np.int64(0), ends],
                flagged_microcurler_sites=np.r_[np.int64(0), p['caught']],
                lost_primary_electrons=np.r_[np.int64(0), p['cost']],
                flagged_microcurler_electrons=np.r_[np.int64(0), mcharge],
                microcurler_site_recall=np.r_[0., p['caught'] / npositive] if npositive else np.full(len(ends)+1, np.nan),
                microcurler_charge_recall=np.r_[0., mcharge / nmcharge] if nmcharge else np.full(len(ends)+1, np.nan),
                primary_charge_retention=np.r_[1., 1.-p['cost'] / nprimary] if nprimary else np.full(len(ends)+1, np.nan),
                total_microcurler_sites=np.asarray(npositive, np.int64),
                total_microcurler_electrons=np.asarray(nmcharge, np.int64),
                total_primary_electrons=np.asarray(nprimary, np.int64))


def analyse(arrays, expected_events=2500):
    population = base.validate_arrays(arrays, expected_events)
    mweights = validate_charge(arrays)
    population['microcurler_electrons'] = int(mweights.sum())
    prefix = base.rank_prefixes(arrays['logits'], arrays['target'], arrays['primary_electron_weight'])
    cutoffs = [base.cutoff_at_budget(prefix, p) for p in BUDGETS_PPM]
    cutoffs += [base.cutoff_at_recall(prefix, p) for p in RECALL_TARGETS_PERCENT]
    points, rows = [], []
    for cutoff in cutoffs:
        metrics, event_rows = base.metrics_at_point(arrays, cutoff)
        flags = base.decision_flags(arrays['logits'], cutoff)
        total_mc = int(mweights.sum()); caught_mc = int(mweights[flags].sum())
        metrics.update(microcurler_electrons=total_mc,
                       flagged_microcurler_electrons=caught_mc,
                       microcurler_charge_recall=base.ratio(caught_mc, total_mc),
                       flagged_microcurler_electrons_on_ignored_sites=int(mweights[flags & (arrays['target'] == -1)].sum()))
        for i, e in enumerate(event_rows):
            lo, hi = map(int, arrays['event_ptr'][i:i+2])
            total = int(mweights[lo:hi].sum()); caught = int(mweights[lo:hi][flags[lo:hi]].sum())
            e.update(microcurler_electrons=total, flagged_microcurler_electrons=caught,
                     microcurler_charge_recall=base.ratio(caught, total))
        primary_losses = [e['primary_loss_fraction'] for e in event_rows if e['primary_electrons']]
        metrics['primary_event_loss_quantiles'] = {
            str(q): float(np.quantile(primary_losses, q, method='linear')) if primary_losses else None
            for q in (.5, .9, .95, .99, 1.)}
        mc_events = [e['microcurler_charge_recall'] for e in event_rows if e['microcurler_electrons']]
        metrics['macro_microcurler_charge_recall'] = float(np.mean(mc_events)) if mc_events else None
        points.append(dict(operating_point=cutoff, metrics=metrics))
        rows.extend(dict(operating_point=cutoff['kind'], target=cutoff['target'],
                         target_units=cutoff['target_units'], **e) for e in event_rows)
    return dict(population=population,
                class_balanced_bce=base.class_balanced_bce(arrays['logits'], arrays['target']),
                points=points), rows


def self_test():
    # A tied group mixes supervised positive and ignored-primary charge. It
    # cannot be split to fit a one-electron budget.
    a = dict(logits=np.array([2., 2., 1., -1.], np.float32),
             target=np.array([1, -1, 1, 0], np.int8),
             primary_electron_weight=np.array([0, 2, 0, 998], np.int64),
             microcurler_electron_weight=np.array([3, 4, 1, 0], np.int64),
             event_ptr=np.array([0, 2, 4, 4], np.int64),
             event_index=np.array([0, 1, 2], np.int64), event_id_raw=np.array([0, 1, 2], np.int64),
             event_run_uid=np.array(['run']*3), event_shard=np.array(['block00']*3),
             protocol_sha256=np.asarray('0'*64))
    ans, rows = analyse(a, 3)
    p = next(p for p in ans['points'] if p['operating_point']['kind'] == 'primary_loss_budget' and p['operating_point']['target'] == 1000)
    require(p['operating_point']['threshold_kind'] == 'flag_none', 'Tie was split')
    p = next(p for p in ans['points'] if p['operating_point']['kind'] == 'primary_loss_budget' and p['operating_point']['target'] == 5000)
    require(p['metrics']['lost_primary_electrons_on_ignored_sites'] == 2, 'Ignored primary charge omitted')
    require(p['metrics']['microcurler_charge_recall'] == 1., 'Common microcurler charge denominator wrong')
    require(p['metrics']['primary_loss_tails']['above_5_percent']['fraction'] == .5, 'Event tail denominator wrong')
    require(ans['population']['empty_events'] == 1, 'Empty event lost')
    c = full_curve(a)
    require(np.array_equal(c['flagged_sites'], [0,2,3,4]), 'Curve split score tie')
    require(np.array_equal(c['flagged_microcurler_electrons'], [0,7,8,8]), 'Curve charge aggregation wrong')
    # Fractional loss allowance must floor: 999*1000/1e6 = 0.
    prefix = base.rank_prefixes(np.array([2.,1.]), np.array([1,0]), np.array([1,998]))
    require(base.cutoff_at_budget(prefix,1000)['threshold_kind'] == 'flag_none', 'Integer budget floor violated')
    b = {k:v.copy() for k,v in a.items()}; b['primary_electron_weight'][:] = 0; b['microcurler_electron_weight'][:] = 0
    r,_=analyse(b,3)
    require(r['points'][0]['metrics']['primary_charge_retention'] is None and
            r['points'][0]['metrics']['microcurler_charge_recall'] is None, 'Undefined denominator not preserved')
    print('PARALLEL METRICS SELF-TEST: PASS (whole ties, integer budgets, ignored charge, event tails, empty events, native versus charge recall)')


if __name__ == '__main__':
    self_test()
