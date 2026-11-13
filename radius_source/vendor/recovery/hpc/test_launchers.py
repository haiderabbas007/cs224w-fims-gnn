#!/usr/bin/env python3
"""Synthetic scheduler tests; never calls Slurm or reads detector arrays."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import common
import submit

class RecoverySchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/'recovery'
        (self.root/'results').mkdir(parents=True)
        self.original = Path(self.temp.name)/'old'
        self.capacity = Path(self.temp.name)/'capacity'
        self.digest = '1'*64
        self.journal = dict(package_sha256=self.digest, submissions=[])

    def tearDown(self): self.temp.cleanup()

    def scheduler(self, rows=(), journal=None, **kw):
        return submit.Submitter(self.root, self.original, self.capacity, '/known/python',
                                self.digest, list(rows), journal or self.journal, **kw)

    def test_original_array_afterany_and_full_gpu(self):
        rows = [dict(id='15133168_11',name='fims-6b4fc507-fit',state='RUNNING',reason='gpu-node')]
        calls = []
        def run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=str(6000+len(calls))+'\n')
        with patch.dict(os.environ, {}, clear=True), patch.object(submit.subprocess, 'run', side_effect=run):
            submit.plan(self.root,self.original,self.capacity,'/known/python',self.digest,rows,self.journal)
        self.assertEqual(len(calls),3)
        gpu = calls[1]
        self.assertIn('--dependency=afterany:15133168',gpu)
        self.assertIn('--gres=gpu:NV-A30:1',gpu)
        self.assertIn('--time=3-00:00:00',gpu)
        self.assertIn('--signal=B:USR1@300',gpu)
        self.assertIn('--array=0,1,2,3,4,5,6,7,8,9,10,11%12',gpu)
        self.assertIn('--dependency=afterok:6002',calls[2])
        self.assertTrue(all(c[0]=='sbatch' for c in calls))

    def test_live_recovery_array_reused(self):
        rows = [dict(id='777_[0-11%12]',name='fims-gr-11111111-fit',state='PENDING',reason='Priority')]
        with patch.object(submit.subprocess,'run') as run:
            self.assertEqual(self.scheduler(rows).submit('fit',indices=list(range(12))),'777')
            run.assert_not_called()

    def test_squeue_failure_is_not_empty_queue(self):
        with patch.object(submit.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['squeue'])):
            with self.assertRaises(subprocess.CalledProcessError):submit.queue_rows()

    def test_parse_failure_aborts(self):
        with patch.object(submit.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout='malformed\n')):
            with self.assertRaises(ValueError):submit.queue_rows()

    def test_duplicate_active_arrays_aborts(self):
        rows = [dict(id=i,name='exact',state='RUNNING',reason='node') for i in ('200_0','201_1')]
        with self.assertRaises(ValueError):submit.one_active(rows,'exact')

    def test_cancel_only_recorded_blocked_pending_report(self):
        rows=[dict(id='700',name='fims-gr-11111111-final_report',state='PENDING',reason='DependencyNeverSatisfied')]
        j=copy.deepcopy(self.journal);j['submissions']=[dict(stage='final_report',id='700')]
        calls=[]
        def run(cmd,**kwargs):
            calls.append(cmd);return subprocess.CompletedProcess(cmd,0,stdout='701\n')
        with patch.object(submit.subprocess,'run',side_effect=run):
            self.scheduler(rows,j).submit('final_report')
        self.assertEqual(calls[0],['scancel','700'])
        self.assertEqual(calls[1][0],'sbatch')
        with patch.object(submit.subprocess,'run') as run:
            with self.assertRaises(ValueError):self.scheduler(rows).submit('final_report')
            run.assert_not_called()

    def test_blocked_fit_never_cancelled(self):
        rows=[dict(id='700',name='fims-gr-11111111-fit',state='PENDING',reason='DependencyNeverSatisfied')]
        with patch.object(submit.subprocess,'run') as run:
            with self.assertRaises(ValueError):self.scheduler(rows).submit('fit',indices=[0])
            run.assert_not_called()

    def test_dry_run_makes_no_scheduler_mutations(self):
        with patch.object(submit.subprocess,'run') as run:
            self.scheduler(dry_run=True).submit('fit',indices=[0],dependencies=[('afterany','123')])
            run.assert_not_called()
        self.assertFalse((self.root/'results/jobs.json').exists())

    def test_single_gpu_override_validation(self):
        with patch.dict(os.environ,{'FIMS_GPU_GRES':'gpu:NV-A100:2'}):
            with self.assertRaises(ValueError):self.scheduler()

    def test_inventory_tamper_and_completed_fit(self):
        r=self.root
        (r/'code.py').write_text('original\n')
        common.atomic_json(r/'PACKAGE_MANIFEST.json',dict(schema=common.SCHEMA,files={'code.py':common.sha(r/'code.py')}))
        common.verify_inventory(r,schema=common.SCHEMA)
        (r/'code.py').write_text('changed\n')
        with self.assertRaises(ValueError):common.verify_inventory(r,schema=common.SCHEMA)
        fit=r/'fit';fit.mkdir()
        required=['metrics.json','protocol.json','selected_predictions.npz','history.json','selected_checkpoint.pt','final_checkpoint.pt','final_predictions.npz']
        for name in required:(fit/name).write_bytes(name.encode())
        common.atomic_json(fit/'complete.json',dict(status='COMPLETE',outputs={n:common.sha(fit/n) for n in required}))
        self.assertTrue(common.checked_fit(fit))
        (fit/'history.json').write_bytes(b'tampered')
        with self.assertRaises(ValueError):common.checked_fit(fit)

    def test_stage_uses_absolute_arguments_not_spool_path(self):
        stage=(Path(__file__).parent/'stage.sh').read_text()
        self.assertNotIn('BASH_SOURCE',stage)
        self.assertIn('"$fims_root/code/recovery_train.py"',stage)
        self.assertIn('--cuda-stress',stage)
        self.assertIn('trap fims_forward_stop USR1 TERM INT',stage)
        self.assertIn('kill -USR1 "$fims_child"',stage)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RecoverySchedulerTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    print('RECOVERY SCHEDULER SELF-TEST: PASS (dependencies, active guard, narrow cancellation, dry-run, hashes, signal forwarding)')
