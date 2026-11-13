"""Synthetic 22-fit end-to-end report, using the real saved-fit verifier."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import report_operating_points as report
from test_report import toy


def prepare_synthetic(folder):
    root = Path(folder)
    capacity = root / 'capacity'; capacity.mkdir()
    study = root / 'recovery'; study.mkdir()
    package = root / 'package'; package.mkdir()
    report.old.write_json(package / 'PACKAGE_MANIFEST.json', {'synthetic_test_only': True})
    a = toy()
    a['event_ptr'] = np.r_[a['event_ptr'], np.full(2500 - 4, 7, np.int64)]
    a['event_index'] = np.arange(2500, dtype=np.int64)
    a['event_id_raw'] = np.arange(2500, dtype=np.int64)
    a['event_run_uid'] = np.asarray(['synthetic'] * 2500)
    a['event_shard'] = np.asarray(['synthetic'] * 2500)
    native = {k: v for k, v in a.items() if k not in ('logits', 'protocol_sha256')}
    records = []
    for model, source_model, seed, fit, deterministic in report.fit_jobs(capacity, study):
        fit.mkdir(parents=True)
        protocol = {'portable_manifest_sha256': report.old.PORTABLE_SHA,
                    'synthetic_test_only': True, 'model': source_model, 'seed': seed}
        report.old.write_json(fit / 'protocol.json', protocol)
        phash = report.sha(fit / 'protocol.json')
        identity = {'model': source_model, 'seed': seed, 'protocol_sha256': phash, 'parameters': 10}
        predictions = dict(a, protocol_sha256=np.asarray(phash))
        report.old.write_npz(fit / 'selected_predictions.npz', predictions)
        bce = report.cm.class_balanced_bce(a['logits'], a['target'])
        meta = {'identity': identity, 'selected_epoch': None if deterministic else 2,
                'selected_bce': bce, 'parameters': 10}
        report.old.write_json(fit / 'metrics.json', meta)
        names = ['protocol.json', 'selected_predictions.npz', 'metrics.json']
        if not deterministic:
            report.old.write_json(fit / 'history.json', [
                {'epoch': 1, 'validation_class_balanced_bce': bce + .1},
                {'epoch': 2, 'validation_class_balanced_bce': bce},
                {'epoch': 3, 'validation_class_balanced_bce': bce},
            ])
            names.append('history.json')
        report.old.write_json(fit / 'complete.json', {'status': 'COMPLETE', 'identity': identity,
                              'outputs': {name: report.sha(fit / name) for name in names}})
        arrays, record = report.old.load_fit(fit, source_model, seed, native, deterministic=deterministic)
        analysis, _ = report.metrics.analyse(arrays, 2500)
        record.update(analysis, model=model)
        records.append(record)
    provenance = {'portable_manifest_sha256': report.old.PORTABLE_SHA,
                  'synthetic_test_only': True, 'native_files': {}}
    reference = package / 'reference' / 'previous_metrics.json'
    report.old.write_json(reference, {'schema': 'fims_parallel_development_report_v1',
                           'status': 'COMPLETE', 'missing_fits': [], 'results': records,
                           'data_provenance': provenance})
    return capacity, study, package, native, provenance, reference


class SyntheticIntegration(unittest.TestCase):
    def test_all22_saved_fits_through_report_and_zip(self):
        with tempfile.TemporaryDirectory(prefix='fims_report_synthetic_') as folder:
            capacity, study, package, native, provenance, reference = prepare_synthetic(folder)
            output = Path(folder) / 'output'
            before = {str(p): report.sha(p) for source in (capacity, study) for p in source.rglob('*') if p.is_file()}
            with patch.object(report, 'ROOT', package), patch.object(report.old, 'load_native', return_value=(native, provenance)) as loader:
                result = report.build(capacity, study, output, reference)
                self.assertEqual(loader.call_count, 2)
            self.assertEqual(result['completed_fits'], 22)
            self.assertEqual(len(result['seed_summaries']), 50)
            self.assertEqual(len(result['results']), 22)
            marker = json.loads((output / 'fims_operating_point_complete_v1.json').read_text())
            self.assertEqual(marker['status'], 'COMPLETE')
            self.assertEqual(marker['package_sha256'], report.sha(package / 'PACKAGE_MANIFEST.json'))
            for name, digest in marker['outputs'].items():
                self.assertEqual(report.sha(output / name), digest)
            with zipfile.ZipFile(output / 'fims_operating_point_review_v1.zip') as z:
                self.assertIsNone(z.testzip())
                self.assertEqual(set(z.namelist()), set(marker['outputs']) | {'fims_operating_point_complete_v1.json'})
            after = {str(p): report.sha(p) for source in (capacity, study) for p in source.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertFalse(any(p.name.startswith('.report_build_') for p in output.iterdir()))


if __name__ == '__main__':
    unittest.main()
