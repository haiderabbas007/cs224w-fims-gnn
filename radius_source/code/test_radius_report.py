#!/usr/bin/env python3
"""Synthetic tests only: exact ties, event denominators, provenance and reports."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import report_radius as report


def arrays(target=(1, -1, 1, 0), primary=(0, 2, 0, 998),
           logits=(2., 2., 1., -1.), ptr=(0, 2, 4, 4), micro=(3, 4, 1, 0)):
    ne = len(ptr)-1
    return dict(target=np.asarray(target, np.int8),
                primary_electron_weight=np.asarray(primary, np.int64),
                microcurler_electron_weight=np.asarray(micro, np.int64),
                logits=np.asarray(logits, np.float32), event_ptr=np.asarray(ptr, np.int64),
                event_index=np.arange(ne, dtype=np.int64),
                event_id_raw=np.arange(ne, dtype=np.int64),
                event_run_uid=np.asarray(['synthetic']*ne),
                event_shard=np.asarray(['block00']*ne), protocol_sha256=np.asarray('0'*64))


def native(a):
    return {k: v.copy() for k, v in a.items() if k not in ('logits', 'protocol_sha256')}


class FakeData:
    def __init__(self, folder, a):
        self.base = Path(folder).resolve()
        self.base.mkdir(parents=True, exist_ok=True)
        self.manifest_sha = 'd'*64
        self.manifest = dict(portable_manifest_sha256=report.SiteDataPortableSHA,
                             preserved_manifest_sha256='e'*64)
        self.validation = list(map(int, a['event_index']))
        self.events = [dict(raw_id=int(a['event_id_raw'][i]), run_uid=str(a['event_run_uid'][i]),
                            shard=str(a['event_shard'][i])) for i in self.validation]
        self.a = a
        self.checked = False

    def get(self, index):
        lo, hi = self.a['event_ptr'][index:index+2]
        return {k: self.a[k][lo:hi].copy() for k in
                ('target', 'primary_electron_weight', 'microcurler_electron_weight')}

    def verify_unchanged(self):
        self.checked = True


def fixture(folder, a, *, radius=2, model='full1', seed=20260924, manifest='d'*64):
    """Write tiny hash-committed outputs in the actual trainer's report schema."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    deterministic = model in report.CONTROLS
    protocol = dict(radius_mm=radius, site_manifest_sha256=manifest,
                    portable_manifest_sha256=report.SiteDataPortableSHA, model=model)
    report.write_json(folder/'protocol.json', protocol)
    psha = report.sha(folder/'protocol.json')
    identity = dict(model=model, seed=seed, parameters=12929 if model == 'full1' else 7 if model == 'counts8' else 10,
                    radius_mm=radius, train_events=7500, protocol_sha256=psha)
    selected = {k: v.copy() for k, v in a.items() if k != 'microcurler_electron_weight'}
    selected['protocol_sha256'] = np.asarray(psha)
    report.write_npz(folder/'selected_predictions.npz', selected)
    loss = report.base.class_balanced_bce(selected['logits'], selected['target'])
    meta = dict(identity=identity, selected_bce=loss, selected_epoch=None if deterministic else 2,
                fit_seconds=float(radius*10+seed % 10), parameters=identity['parameters'])
    files = ['protocol.json', 'metrics.json', 'selected_predictions.npz']
    if deterministic:
        report.write_json(folder/'model_parameters.json', dict(coef=[0.]*identity['parameters']))
        files += ['model_parameters.json']
    else:
        final = {k: v.copy() for k, v in selected.items()}
        final['logits'] = -final['logits']
        final_loss = report.base.class_balanced_bce(final['logits'], final['target'])
        # Keep selected/final deliberately different while the selected epoch is
        # the earliest minimum in a complete ten-epoch history.
        if final_loss < loss:
            final['logits'] = np.zeros_like(final['logits'])+5
            final_loss = report.base.class_balanced_bce(final['logits'], final['target'])
        history = [dict(epoch=i, validation_class_balanced_bce=loss+1,
                        validation_seconds=.25) for i in range(1, 11)]
        history[1]['validation_class_balanced_bce'] = loss
        history[2]['validation_class_balanced_bce'] = loss  # earliest tie selects epoch2
        history[-1]['validation_class_balanced_bce'] = final_loss
        report.write_json(folder/'history.json', history)
        report.write_npz(folder/'final_predictions.npz', final)
        (folder/'selected_checkpoint.pt').write_bytes(b'synthetic selected tensor fixture')
        (folder/'final_checkpoint.pt').write_bytes(b'synthetic final tensor fixture')
        files += ['history.json', 'selected_checkpoint.pt', 'final_checkpoint.pt', 'final_predictions.npz']
    report.write_json(folder/'metrics.json', meta)
    marker = dict(status='COMPLETE', identity=identity,
                  outputs={name: report.sha(folder/name) for name in files})
    report.write_json(folder/'complete.json', marker)
    return folder


def recommit(folder, name):
    marker = json.loads((folder/'complete.json').read_text())
    marker['outputs'][name] = report.sha(folder/name)
    report.write_json(folder/'complete.json', marker)


class MetricTests(unittest.TestCase):
    def test_tied_ignored_charge_cannot_be_split(self):
        a = arrays()
        analysis, event_rows = report.analyse(a, 3)
        points = {p['operating_point']['target']: p for p in analysis['points']}
        self.assertEqual(points[1000]['operating_point']['threshold_kind'], 'flag_none')
        self.assertEqual(points[1000]['metrics']['microcurler_site_recall'], 0.)
        m = points[5000]['metrics']
        self.assertEqual(m['microcurler_site_recall'], 1.)
        self.assertEqual(m['lost_primary_electrons_on_ignored_sites'], 2)
        self.assertEqual(m['microcurler_charge_recall'], 1.)
        self.assertEqual(m['primary_bearing_events'], 2)
        self.assertEqual(m['primary_loss_tails']['above_5_percent']['fraction'], .5)
        self.assertEqual(len(event_rows), 3*len(report.BUDGETS_PPM))
        curve = report.compact_curve(a)
        np.testing.assert_array_equal(curve['flagged_sites'], [0, 2, 3, 4])
        np.testing.assert_array_equal(curve['lost_primary_electrons'], [0, 2, 2, 1000])
        self.assertTrue(np.isnan(curve['threshold_logit'][0]))

    def test_strict_damage_boundary_and_zero_primary(self):
        a = arrays(target=(0, 0, 0, 0, 1), primary=(5, 95, 6, 94, 0),
                   logits=(1, -1, 1, -1, 1), ptr=(0, 2, 4, 5, 5), micro=(0, 0, 0, 0, 1))
        point = dict(threshold_kind='logit', threshold_logit=0.)
        m, rows = report.base.metrics_at_point(a, point)
        self.assertEqual(m['events'], 4)
        self.assertEqual(m['primary_bearing_events'], 2)
        self.assertEqual(m['primary_loss_tails']['above_5_percent']['events'], 1)
        self.assertEqual(m['primary_loss_tails']['above_5_percent']['fraction'], .5)
        self.assertIsNone(rows[2]['primary_loss_fraction'])
        self.assertIsNone(rows[3]['primary_loss_fraction'])

    def test_equal_event_recall_is_not_pooled(self):
        a = arrays(target=[1]*10+[0], primary=[0]*10+[1000],
                   logits=[1]*9+[-1, -2], ptr=(0, 9, 10, 11), micro=[1]*10+[0])
        point = dict(threshold_kind='logit', threshold_logit=0.)
        m, _ = report.base.metrics_at_point(a, point)
        self.assertEqual(m['microcurler_site_recall'], .9)
        self.assertEqual(m['macro_microcurler_site_recall'], .5)
        self.assertEqual(m['microcurler_bearing_events'], 2)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = fixture(Path(self.temp.name)/'fit', arrays())

    def tearDown(self):
        self.temp.cleanup()

    def verified(self):
        return report.verify_fit(self.folder, radius=2, model='full1', seed=20260924,
                                 site_manifest_sha256='d'*64)

    def test_selected_and_final_remain_separate(self):
        v = self.verified()
        s = report.load_predictions(v, native(arrays()), 'selected', expected_events=3)
        f = report.load_predictions(v, native(arrays()), 'final', expected_events=3)
        self.assertFalse(np.array_equal(s['logits'], f['logits']))
        self.assertEqual(v['meta']['selected_epoch'], 2)

    def test_completed_file_tamper_rejected(self):
        with (self.folder/'selected_predictions.npz').open('ab') as stream:
            stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.verified()

    def test_recommitted_wrong_event_alignment_rejected(self):
        path = self.folder/'selected_predictions.npz'
        with np.load(path, allow_pickle=False) as z:
            a = dict(z)
        a['event_index'] = a['event_index'][::-1].copy()
        report.write_npz(path, a)
        recommit(self.folder, path.name)
        v = self.verified()
        with self.assertRaisesRegex(ValueError, 'alignment mismatch: event_index'):
            report.load_predictions(v, native(arrays()), 'selected', expected_events=3)

    def test_radius_and_native_manifest_identity_required(self):
        with self.assertRaisesRegex(ValueError, 'Wrong radius'):
            report.verify_fit(self.folder, radius=4, model='full1', seed=20260924,
                              site_manifest_sha256='d'*64)
        with self.assertRaisesRegex(ValueError, 'Wrong site data'):
            report.verify_fit(self.folder, radius=2, model='full1', seed=20260924,
                              site_manifest_sha256='f'*64)

    def test_selection_metadata_not_arbitrary(self):
        path = self.folder/'metrics.json'
        meta = json.loads(path.read_text())
        meta['selected_epoch'] = 3
        report.write_json(path, meta)
        recommit(self.folder, path.name)
        with self.assertRaisesRegex(ValueError, 'earliest minimum'):
            self.verified()

    def test_execution_profile_has_own_receipt(self):
        path = self.folder/'execution_profile.json'
        report.write_json(path, dict(sessions=[dict(status='COMPLETE')]))
        report.write_json(self.folder/'execution_profile_complete.json',
                          dict(execution_profile_sha256=report.sha(path), sessions=1))
        self.assertIsNotNone(self.verified()['profile'])
        path.write_text('{"sessions": []}')
        with self.assertRaisesRegex(ValueError, 'profile hash'):
            self.verified()

    def test_scientific_settings_and_source_hashes_are_pinned(self):
        v = self.verified()
        p = v['protocol']
        p.update(schema='fims_radius_gnn_protocol_v1', history_bins=8, epochs=10, layers=1,
                 hidden_channels=64, observation_radius_mm=2, connection_radius_mm=2,
                 spatial_normalization_mm=2, temporal_normalization_bins=7,
                 experiment_plan_sha256=report.sha(report.ROOT/'EXPERIMENT_PLAN.json'),
                 code={n: report.sha(report.VENDOR/n) for n in ('train_site_gnn_v1.py', 'graph_models_v1.py',
                       'gnn_adapter_v1.py', 'site_data_v1.py', 'vendor/capacity_model_v2.py')},
                 experiment_sources={n: report.sha(report.ROOT/n) for n in ('code/radius_graph.py', 'code/train_radius.py',
                       'vendor/recovery/code/memory_model.py')})
        report.verify_scientific_protocol(v, 2, 'full1')
        p['spatial_normalization_mm'] = 4
        with self.assertRaisesRegex(ValueError, 'normalization changed'):
            report.verify_scientific_protocol(v, 2, 'full1')
        p['spatial_normalization_mm'] = 2
        p['experiment_sources']['code/radius_graph.py'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Fit source differs'):
            report.verify_scientific_protocol(v, 2, 'full1')


class ReportTests(unittest.TestCase):
    def test_partial_groups_do_not_claim_three_seed_means(self):
        row = dict(checkpoint='selected', radius_mm=1, model='full1', target=1000,
                   seed=20260924, **{k: .5 for k in report.SUMMARY_KEYS})
        group = report.summaries([row])[0]
        self.assertEqual(group['coverage'], 'PARTIAL')
        self.assertEqual(group['statistics'], {})
        rows = [dict(row, seed=s) for s in report.SEEDS]
        self.assertEqual(report.summaries(rows)[0]['statistics']['microcurler_site_recall']['mean'], .5)
        with self.assertRaisesRegex(ValueError, 'Repeated seed'):
            report.summaries(rows+[rows[0]])
        final = [dict(r, checkpoint='final', microcurler_site_recall=.9) for r in rows]
        groups = report.summaries(rows+final)
        self.assertEqual(len(groups), 2)
        self.assertNotEqual(groups[0]['statistics']['microcurler_site_recall'],
                            groups[1]['statistics']['microcurler_site_recall'])

    def test_paired_differences_match_seed(self):
        rows = [dict(checkpoint='selected', radius_mm=radius, model='full1', target=1000,
                     seed=seed, **{k: radius*.1+(seed-20260924)*.01 for k in report.SUMMARY_KEYS})
                for radius in report.RADII for seed in report.SEEDS]
        delta = report.paired_radius_deltas(rows)
        self.assertEqual(len(delta), 6)
        for row in delta:
            self.assertAlmostEqual(row['microcurler_site_recall_difference'],
                                   -.1 if row['radius_mm'] == 1 else .2)

    def test_full_report_and_explicit_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = FakeData(root/'native', arrays())
            study = root/'study'
            for radius in report.RADII:
                for seed in report.SEEDS:
                    fixture(study/'results/fits'/f'r{radius}'/'full1'/f'seed{seed}', arrays(),
                            radius=radius, seed=seed)
                for model in report.CONTROLS:
                    fixture(study/'results/controls'/f'r{radius}'/model/f'seed20260924', arrays(),
                            radius=radius, model=model)
            with contextlib.redirect_stdout(io.StringIO()):
                result = report.build_report(study, data, synthetic=True)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(result['completed_fits'], 15)
            self.assertEqual(len(result['results']), 24)
            self.assertTrue(data.checked)
            out = study/'results/report'
            marker = json.loads((out/'fims_radius_complete_v1.json').read_text())
            with zipfile.ZipFile(out/'fims_radius_sensitivity_review_v1.zip') as z:
                self.assertFalse(any(name.endswith('.pt') for name in z.namelist()))
                self.assertEqual(json.loads(z.read('fims_radius_complete_v1.json')), marker)
                for name, digest in marker['outputs'].items():
                    self.assertEqual(report.hashlib.sha256(z.read(name)).hexdigest(), digest)
                self.assertEqual(len([n for n in z.namelist() if n.endswith('_full_curve.npz')]), 15)
            # An absent fit cannot silently turn the surviving two seeds into a mean.
            missing = study/'results/fits/r4/full1/seed20260926/complete.json'
            missing.unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'Planned fits are incomplete'):
                    report.build_report(study, data, synthetic=True)
                partial = report.build_report(study, data, synthetic=True, allow_partial=True)
            self.assertEqual(partial['status'], 'PARTIAL')
            groups = [s for s in partial['seed_summaries'] if s['radius_mm'] == 4 and s['model'] == 'full1']
            self.assertTrue(all(s['coverage'] == 'PARTIAL' and not s['statistics'] for s in groups))
            with zipfile.ZipFile(out/'fims_radius_sensitivity_review_v1.zip') as z:
                self.assertEqual(json.loads(z.read('fims_radius_complete_v1.json'))['status'], 'PARTIAL')
                self.assertFalse(any('r4_full1_seed20260926' in n for n in z.namelist()))


def run_tests():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print('RADIUS REPORT SELF-TEST: PASS (synthetic data only)')


if __name__ == '__main__':
    run_tests()
