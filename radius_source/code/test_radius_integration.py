#!/usr/bin/env python3
"""Train all 15 tiny synthetic fits, then consume their real report artifacts.

This test never opens development or confirmation events. The only population
override in scientific protocol checks is the invented three-event train count.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import train_radius as training
import control_radius as controls
import report_radius as report
from site_data_v1 import SiteData, write_json
from test_site_pipeline_v1 import fixture
import torch


def run_tests():
    with tempfile.TemporaryDirectory(prefix='fims_radius_integration_') as tmp:
        temp = Path(tmp)
        study, source = temp/'study', temp/'invented_data'
        study.mkdir()
        shutil.copyfile(report.ROOT/'EXPERIMENT_PLAN.json', study/'EXPERIMENT_PLAN.json')
        fixture(source)
        # Preserve the synthetic flag and all invented arrays. The report
        # requires a stable dataset lineage string even in its synthetic mode.
        mp = source/'site_manifest_v1.json'
        manifest = json.loads(mp.read_text())
        manifest['portable_manifest_sha256'] = report.SiteDataPortableSHA
        write_json(mp, manifest)
        data = SiteData(source, synthetic=True)
        assert data.manifest['synthetic'] is True
        assert len(data.events) == 5 and data.validation == [3, 4]
        with contextlib.redirect_stdout(io.StringIO()):
            for radius in report.RADII:
                for seed in report.SEEDS:
                    training.fit_radius(study, data, radius, seed, torch.device('cpu'))
                controls.run_controls(data, study, radius)
        jobs = [(r, 'full1', s, study/'results/fits'/f'r{r}'/'full1'/f'seed{s}')
                for r in report.RADII for s in report.SEEDS]
        jobs += [(r, m, report.SEEDS[0], study/'results/controls'/f'r{r}'/m/f'seed{report.SEEDS[0]}')
                 for r in report.RADII for m in report.CONTROLS]
        native = report.load_native(data)
        scientific_checks = 0
        for radius, model, seed, path in jobs:
            verified = report.verify_fit(path, radius=radius, model=model, seed=seed,
                                         site_manifest_sha256=data.manifest_sha)
            for checkpoint in ('selected',) if model in report.CONTROLS else ('selected', 'final'):
                predictions = report.load_predictions(verified, native, checkpoint, expected_events=2)
                assert predictions['event_ptr'].tolist() == [0, 5, 5]
                assert predictions['target'][2] == -1
                assert predictions['primary_electron_weight'][2] == 2
            probe = copy.deepcopy(verified)
            assert probe['complete']['identity']['train_events'] == 3
            probe['complete']['identity']['train_events'] = 7500
            report.verify_scientific_protocol(probe, radius, model)
            scientific_checks += 1
        with contextlib.redirect_stdout(io.StringIO()):
            payload = report.build_report(study, data, synthetic=True)
        assert payload['status'] == 'COMPLETE'
        assert payload['completed_fits'] == 15 and len(payload['results']) == 24
        assert all(group['coverage'] == 'COMPLETE' for group in payload['seed_summaries'])
        out = study/'results/report'
        marker = json.loads((out/'fims_radius_complete_v1.json').read_text())
        with zipfile.ZipFile(out/'fims_radius_sensitivity_review_v1.zip') as z:
            assert z.testzip() is None
            assert json.loads(z.read('fims_radius_complete_v1.json')) == marker
            for name, digest in marker['outputs'].items():
                assert report.hashlib.sha256(z.read(name)).hexdigest() == digest, name
        runtime = json.loads((out/'fims_radius_runtime_v1.json').read_text())['records']
        gnn_profiles = [r for r in runtime if r['model'] == 'full1']
        assert len(gnn_profiles) == 9
        assert all(r['graph_queries'] > 0 and r['mean_executed_edges_per_query'] is not None for r in gnn_profiles)
        # Completed fits must be reused, and their saved predictions must remain
        # reportable without retraining or a changed protocol/profile identity.
        before = {str(path): report.sha(path/'complete.json') for *_, path in jobs}
        with contextlib.redirect_stdout(io.StringIO()):
            for radius in report.RADII:
                for seed in report.SEEDS:
                    training.fit_radius(study, data, radius, seed, torch.device('cpu'))
                controls.run_controls(data, study, radius)
        assert all(report.sha(Path(path)/'complete.json') == digest for path, digest in before.items())
        data.verify_unchanged()
        print(f'RADIUS INTEGRATION SELF-TEST: PASS ({scientific_checks} actual tiny fits; '
              'nine GNN + six controls; 24 saved prediction endpoints; protocol/source pins; '
              'native alignment and ignored charge; profiles; complete report/archive; resume reuse; synthetic data only)')


if __name__ == '__main__':
    run_tests()
