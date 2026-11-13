"""Semantic tests for saved-prediction reporting; no real events or training."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import report_operating_points as report


def toy():
    # First event loses all of its two primary electrons at permissive cuts;
    # pooled retention is nevertheless excellent. The fourth event is empty.
    return dict(logits=np.asarray([3., 3., 2., 1., 0., -1., -2.], np.float32),
                target=np.asarray([1, -1, 1, 0, 0, 1, 0], np.int8),
                primary_electron_weight=np.asarray([0, 2, 0, 18, 180, 0, 800], np.int64),
                microcurler_electron_weight=np.asarray([3, 4, 1, 0, 0, 6, 0], np.int64),
                event_ptr=np.asarray([0, 2, 5, 7, 7], np.int64),
                event_index=np.arange(4, dtype=np.int64), event_id_raw=np.arange(4, dtype=np.int64),
                event_run_uid=np.asarray(['run'] * 4), event_shard=np.asarray(['block00'] * 4),
                protocol_sha256=np.asarray('0' * 64))


def point_for(points, target):
    return next(p for p in points if p['operating_point']['target'] == target)


class MetricsTests(unittest.TestCase):
    def test_ties_cannot_be_split(self):
        _, points, _ = report.analyse_points(toy(), 4)
        strict = point_for(points, 1000)
        self.assertEqual(strict['operating_point']['threshold_kind'], 'flag_none')
        self.assertEqual(strict['metrics']['flagged_sites'], 0)
        self.assertGreater(point_for(points, 5000)['metrics']['flagged_sites'], 0)

    def test_exact_budget_exhaustive(self):
        a = toy()
        _, points, _ = report.analyse_points(a, 4)
        cutoffs = sorted(set(a['logits'].tolist()), reverse=True)
        for point in points:
            cap = int(a['primary_electron_weight'].sum()) * point['operating_point']['target'] // 1000000
            allowed = [c for c in cutoffs if int(a['primary_electron_weight'][a['logits'] >= c].sum()) <= cap]
            expected = min(allowed) if allowed else None
            self.assertEqual(point['operating_point']['threshold_logit'], expected)

    def test_ignored_charge_still_protected(self):
        population, points, _ = report.analyse_points(toy(), 4)
        p = point_for(points, 5000)['metrics']
        self.assertEqual(population['primary_electrons'], 1000)
        self.assertEqual(p['lost_primary_electrons_on_ignored_sites'], 2)
        self.assertEqual(p['flagged_microcurler_electrons_on_ignored_sites'], 4)
        self.assertEqual(p['primary_loss_tails']['above_5_percent']['fraction'], 1 / 3)

    def test_integer_budget_floors(self):
        a = toy()
        a['primary_electron_weight'][-1] = 799
        _, points, _ = report.analyse_points(a, 4)
        self.assertEqual(point_for(points, 1000)['operating_point']['allowed_primary_electron_loss'], 0)

    def test_equal_event_recall_not_pooled(self):
        a = toy()
        a['event_ptr'] = np.asarray([0, 5, 7, 7], np.int64)
        for k in ('event_index', 'event_id_raw', 'event_run_uid', 'event_shard'):
            a[k] = a[k][:3]
        _, points, _ = report.analyse_points(a, 3)
        p = point_for(points, 5000)['metrics']
        self.assertAlmostEqual(p['microcurler_site_recall'], 2 / 3)
        self.assertAlmostEqual(p['macro_microcurler_site_recall'], .5)
        self.assertEqual(p['microcurler_bearing_events'], 2)

    def test_empty_and_undefined_denominators(self):
        a = toy()
        a['primary_electron_weight'][:] = 0
        a['microcurler_electron_weight'][:] = 0
        a['target'][a['target'] == 1] = 0
        population, points, rows = report.analyse_points(a, 4)
        self.assertEqual(population['empty_events'], 1)
        for p in points:
            self.assertIsNone(p['metrics']['microcurler_site_recall'])
            self.assertIsNone(p['metrics']['macro_microcurler_site_recall'])
            self.assertIsNone(p['metrics']['primary_charge_retention'])
            self.assertIsNone(p['metrics']['microcurler_charge_recall'])
        self.assertEqual(len(rows), 20)

    def test_damage_boundary_is_strict(self):
        a = toy()
        a['primary_electron_weight'] = np.asarray([0, 2, 0, 38, 0, 0, 960], np.int64)
        a['event_ptr'] = np.asarray([0, 5, 7, 7], np.int64)
        for k in ('event_index', 'event_id_raw', 'event_run_uid', 'event_shard'):
            a[k] = a[k][:3]
        _, points, _ = report.analyse_points(a, 3)
        p = point_for(points, 5000)['metrics']
        # 2/40 = exactly5%, which must not count as >5%.
        self.assertEqual(p['primary_loss_tails']['above_5_percent']['events'], 0)

    def test_historical_endpoint_reproduction(self):
        a = toy()
        _, points, _ = report.analyse_points(a, 4)
        original, _ = report.metrics.analyse(a, 4)
        report.reproduce_historical(points, original)
        broken = copy.deepcopy(original)
        broken['points'][1]['metrics']['primary_charge_retention'] -= .001
        with self.assertRaisesRegex(ValueError, 'no longer reproduces'):
            report.reproduce_historical(points, broken)

    def test_visual_sampling_preserves_all_metric_points(self):
        a = toy()
        _, points, _ = report.analyse_points(a, 4)
        curve = report.display_curve(a, points, max_samples=2)
        selected = set(curve['selected_original_indices'].tolist())
        self.assertTrue(curve['display_only'])
        self.assertTrue(all(p['operating_point']['selected_groups'] in selected for p in points))

    def test_model_seed_coverage_rejects_incomplete_means(self):
        _, points, _ = report.analyse_points(toy(), 4)
        rows = []
        for model, seed in report.EXPECTED:
            record = dict(model=model, seed=seed, epoch=None if model in report.CONTROLS else 3,
                          parameters=10, deterministic=model in report.CONTROLS)
            rows.extend(report.flatten(record, p) for p in points)
        summaries = report.summarize(rows)
        self.assertEqual(len(summaries), 50)
        with self.assertRaisesRegex(ValueError, 'missing/duplicate'):
            report.summarize(rows[:-1])

    def test_wrong_retention_unit_rejected(self):
        with self.assertRaisesRegex(ValueError, 'budget disagree'):
            report.analyse_points(toy(), 4, targets=((95.0, 5000),))


class ProvenanceTests(unittest.TestCase):
    def test_input_inventory_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'predictions').write_bytes(b'original')
            marker = {'status': 'COMPLETE', 'outputs': {'predictions': report.sha(root / 'predictions')}}
            (root / 'complete.json').write_text(json.dumps(marker))
            inventory = report.inventory_snapshot(root)
            self.assertIn('complete.json', inventory)
            (root / 'predictions').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Fit output changed'):
                report.inventory_snapshot(root)

    def test_output_cannot_overlap_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            capacity, study = root / 'capacity', root / 'recovery'
            capacity.mkdir(); study.mkdir()
            for output in (capacity, capacity / 'report', study / 'report', root):
                with self.assertRaisesRegex(ValueError, 'separate'):
                    report.checked_output(capacity, study, output)
            self.assertEqual(report.checked_output(capacity, study, root / 'new_report'), root / 'new_report')

    def test_reviewed_prediction_hash_is_required(self):
        record = dict(complete_sha256='a' * 64, protocol_sha256='b' * 64,
                      predictions_sha256='c' * 64, epoch=3, parameters=10)
        inventory = {'complete.json': 'a' * 64, 'protocol.json': 'b' * 64,
                     'selected_predictions.npz': 'c' * 64}
        report.verify_reviewed_record(record, record, inventory)
        changed = dict(record, predictions_sha256='d' * 64)
        with self.assertRaisesRegex(ValueError, 'selected fit changed'):
            report.verify_reviewed_record(changed, record, inventory)


if __name__ == '__main__':
    unittest.main()
