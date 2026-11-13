"""Standard-library checks of provenance, source binding and completed outputs.

Every input is an invented temporary fixture. Real package/data pins are never
rewritten: unittest.mock patches constants only for the fixture lifetime.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import radius_common as common


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(common.canonical(value))


def inventory(root, schema):
    root = Path(root)
    files = {p.relative_to(root).as_posix(): common.sha(p)
             for p in sorted(root.rglob('*'))
             if p.is_file() and p != root / 'PACKAGE_MANIFEST.json'}
    write_json(root / 'PACKAGE_MANIFEST.json', {'schema': schema, 'files': files})
    return common.sha(root / 'PACKAGE_MANIFEST.json')


class SourceFixture:
    def __init__(self, base):
        self.base = Path(base)
        self.study = self.base / 'new_study'
        self.original = self.base / 'old_study'
        self.recovery = self.base / 'old_recovery'
        self.original.mkdir()
        (self.original / 'source.txt').write_text('invented original code\n')
        original_sha = inventory(self.original, 'fims_parallel_package_v1')
        (self.recovery / 'vendor').mkdir(parents=True)
        shutil.copytree(self.original, self.recovery / 'vendor/original')
        (self.recovery / 'source.txt').write_text('invented recovery code\n')
        recovery_sha = inventory(self.recovery, 'fims_gnn_recovery_package_v1')
        (self.study / 'vendor').mkdir(parents=True)
        shutil.copytree(self.recovery, self.study / 'vendor/recovery')
        (self.study / 'source.txt').write_text('invented radius code\n')
        write_json(self.study / 'EXPERIMENT_PLAN.json', {
            'schema': 'fims_radius_sensitivity_plan_v1',
            'radii_mm': [1, 2, 4], 'seeds': list(common.SEEDS),
            'model': 'full1', 'history_bins': 8,
            'coordinate_normalization_mm': 2, 'epochs': 10,
        })
        inventory(self.study, common.SCHEMA)
        self.data = self.original / 'data'
        portable, preserved = '1' * 64, '2' * 64
        write_json(self.data / 'site_manifest_v1.json', {
            'schema': 'fims_site_geometry_v1', 'synthetic': False,
            'portable_manifest_sha256': portable,
            'preserved_manifest_sha256': preserved,
            'events': [{'index': i, 'split': 'train' if i < 7500 else 'validation'}
                       for i in range(10000)],
        })
        site_sha = common.sha(self.data / 'site_manifest_v1.json')
        historical = {}
        for seed in common.SEEDS:
            folder = self.recovery / 'results/fits/full1' / ('seed' + str(seed))
            write_json(folder / 'protocol.json', {'site_manifest_sha256': site_sha})
            protocol_sha = common.sha(folder / 'protocol.json')
            write_json(folder / 'complete.json', {
                'status': 'COMPLETE',
                'identity': {'protocol_sha256': protocol_sha, 'seed': seed},
            })
            historical[seed] = (protocol_sha, common.sha(folder / 'complete.json'))
        self.pins = dict(ORIGINAL_SHA=original_sha, RECOVERY_SHA=recovery_sha,
                         PORTABLE_SHA=portable, PRESERVED_SHA=preserved,
                         HISTORICAL_FULL1=historical)

    def bind(self):
        receipt = common.verify_sources(self.study, self.original, self.recovery)
        write_json(self.study / 'results/source_receipt.json', receipt)
        return receipt


def completed_fixture(base, *, radius=2, model='full1', seed=20260924):
    folder = Path(base) / ('r' + str(radius)) / model / ('seed' + str(seed))
    folder.mkdir(parents=True)
    write_json(folder / 'protocol.json', {'radius_mm': radius})
    identity = {'model': model, 'seed': seed,
                'protocol_sha256': common.sha(folder / 'protocol.json')}
    write_json(folder / 'metrics.json', {'identity': identity})
    outputs = ['protocol.json', 'metrics.json', 'selected_predictions.npz']
    if model == 'full1':
        outputs += ['selected_checkpoint.pt', 'final_checkpoint.pt',
                    'final_predictions.npz', 'history.json']
    else:
        outputs += ['model_parameters.json']
    for name in outputs:
        if not (folder / name).exists():
            (folder / name).write_bytes(('invented output: ' + name).encode())
    write_json(folder / 'complete.json', {
        'status': 'COMPLETE', 'identity': identity,
        'outputs': {name: common.sha(folder / name) for name in outputs},
    })
    return folder


class RadiusCommonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fims_radius_common_test_')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def source_fixture(self):
        fixture = SourceFixture(self.base)
        patcher = patch.multiple(common, **fixture.pins)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fixture

    def test_package_and_binding_roundtrip_are_read_only(self):
        f = self.source_fixture()
        before = {p: common.sha(p) for root in (f.original, f.recovery)
                  for p in root.rglob('*') if p.is_file()}
        receipt = f.bind()
        self.assertEqual(common.verify_data_binding(f.study, f.data), receipt)
        self.assertEqual(before, {p: common.sha(p) for p in before})

    def test_package_file_tamper_rejected(self):
        f = self.source_fixture()
        common.verify_package(f.study)
        (f.study / 'source.txt').write_text('changed code\n')
        with self.assertRaisesRegex(ValueError, 'Package file changed'):
            common.verify_package(f.study)

    def test_pinned_package_manifest_tamper_rejected(self):
        f = self.source_fixture()
        marker = f.study / 'vendor/recovery/PACKAGE_MANIFEST.json'
        marker.write_bytes(marker.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            common.verify_package(f.study)

    def test_inventory_symlink_rejected_even_with_matching_bytes(self):
        root = self.base / 'package'
        root.mkdir()
        source = root / 'source.txt'
        source.write_text('same bytes\n')
        digest = inventory(root, 'invented')
        outside = self.base / 'outside.txt'
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Missing/unsafe'):
            common.verify_inventory(root, expected=digest, schema='invented')

    def test_unsafe_relative_and_directory_escape_rejected(self):
        root = self.base / 'package'
        root.mkdir()
        outside = self.base / 'outside'
        outside.mkdir()
        (outside / 'x').write_text('fixture')
        (root / 'alias').symlink_to(outside, target_is_directory=True)
        for name in ('../outside/x', '/etc/passwd', 'alias/x', 'a\\b', './x'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                common.safe_file(root, name)

    def test_historical_pin_mismatch_rejected(self):
        f = self.source_fixture()
        altered = copy.deepcopy(f.pins['HISTORICAL_FULL1'])
        seed = common.SEEDS[0]
        altered[seed] = (altered[seed][0], 'f' * 64)
        with patch.object(common, 'HISTORICAL_FULL1', altered):
            with self.assertRaisesRegex(ValueError, 'Historical full1 reference differs'):
                common.verify_sources(f.study, f.original, f.recovery)

    def test_data_manifest_change_breaks_historical_binding(self):
        f = self.source_fixture()
        f.bind()
        marker = f.data / 'site_manifest_v1.json'
        data = json.loads(marker.read_text())
        data['extra_metadata'] = 'changed after historical fit'
        write_json(marker, data)
        with self.assertRaisesRegex(ValueError, 'Site dataset is not the one'):
            common.verify_data_binding(f.study, f.data)

    def test_receipt_mismatch_rejected(self):
        f = self.source_fixture()
        receipt = f.bind()
        receipt['package_sha256'] = '0' * 64
        write_json(f.study / 'results/source_receipt.json', receipt)
        with self.assertRaisesRegex(ValueError, 'Task dataset differs'):
            common.verify_data_binding(f.study, f.data)

    def test_identical_manifest_at_other_data_path_rejected(self):
        f = self.source_fixture()
        f.bind()
        copied = self.base / 'other_data'
        shutil.copytree(f.data, copied)
        with self.assertRaisesRegex(ValueError, 'Task dataset differs'):
            common.verify_data_binding(f.study, copied)

    def test_completed_gnn_and_controls_verify_but_unfinished_returns_false(self):
        self.assertFalse(common.checked_fit(self.base / 'not_started'))
        for model in ('full1', 'counts8', 'shape8'):
            with self.subTest(model=model):
                path = completed_fixture(self.base, model=model)
                self.assertTrue(common.checked_fit(path))

    def test_completed_output_tamper_rejected(self):
        path = completed_fixture(self.base)
        (path / 'selected_predictions.npz').write_bytes(b'changed predictions')
        with self.assertRaisesRegex(ValueError, 'Completed output changed'):
            common.checked_fit(path)

    def test_completed_output_symlink_rejected(self):
        path = completed_fixture(self.base)
        leaf = path / 'selected_predictions.npz'
        outside = self.base / 'same_predictions'
        outside.write_bytes(leaf.read_bytes())
        leaf.unlink()
        leaf.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Missing/unsafe'):
            common.checked_fit(path)

    def test_completed_radius_model_and_seed_path_mismatch_rejected(self):
        path = completed_fixture(self.base / 'reference')
        alternatives = [self.base / 'wrong_radius/r4/full1/seed20260924',
                        self.base / 'wrong_model/r2/counts8/seed20260924',
                        self.base / 'wrong_seed/r2/full1/seed20260925']
        for destination in alternatives:
            with self.subTest(destination=destination):
                shutil.copytree(path, destination)
                with self.assertRaisesRegex(ValueError, 'different radius/model/seed path'):
                    common.checked_fit(destination)

    def test_completed_metrics_identity_mismatch_rejected_after_rehash(self):
        path = completed_fixture(self.base)
        metrics = json.loads((path / 'metrics.json').read_text())
        metrics['identity']['seed'] += 1
        write_json(path / 'metrics.json', metrics)
        marker = json.loads((path / 'complete.json').read_text())
        marker['outputs']['metrics.json'] = common.sha(path / 'metrics.json')
        write_json(path / 'complete.json', marker)
        with self.assertRaisesRegex(ValueError, 'Completed metrics identity differs'):
            common.checked_fit(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
