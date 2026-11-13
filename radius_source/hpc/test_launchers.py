#!/usr/bin/env python3
"""No-cluster scheduler tests, including signal delivery to a synthetic worker."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import submit


class RadiusSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/'radius'
        (self.root/'results').mkdir(parents=True)
        self.old = Path(self.tmp.name)/'old'
        self.recovery = Path(self.tmp.name)/'recovery'
        self.digest = '1'*64
        self.journal = dict(package_sha256=self.digest, submissions=[])

    def tearDown(self):
        self.tmp.cleanup()

    def scheduler(self, rows=(), journal=None, **kwargs):
        return submit.Submitter(self.root, self.old, self.recovery, '/known/python',
                                self.digest, list(rows), journal or self.journal, **kwargs)

    def fake_submissions(self):
        calls = []
        def run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=str(6000+len(calls))+'\n')
        return calls, run

    def test_initial_plan_full_gpu_and_afterany_report(self):
        calls, run = self.fake_submissions()
        with patch.dict(os.environ, {}, clear=True), patch.object(submit, 'checked_fit', return_value=False), \
                patch.object(submit.subprocess, 'run', side_effect=run):
            submit.plan(self.root,self.old,self.recovery,'/known/python',self.digest,[],self.journal)
        self.assertEqual(len(calls), 3)
        self.assertIn('--gres=gpu:NV-A30:1',calls[0])
        self.assertIn('--time=3-00:00:00',calls[0])
        self.assertIn('--signal=B:USR1@300',calls[0])
        self.assertIn('--array=0,1,2,3,4,5,6,7,8%9',calls[0])
        self.assertIn('--array=0,1,2%3',calls[1])
        self.assertIn('--dependency=afterany:6001,afterany:6002',calls[2])
        self.assertTrue(all(c[0]=='sbatch' for c in calls))
        self.assertTrue(all('--chdir='+str(self.root) in c for c in calls))
        self.assertTrue(all(str(self.root/'hpc/stage.sh') in c for c in calls))

    def test_only_missing_tasks_submitted(self):
        def complete(path):
            path=str(path)
            return not ('/r4/' in path and ('/full1/' in path or '/shape8/' in path))
        calls, run = self.fake_submissions()
        with patch.object(submit, 'checked_fit', side_effect=complete), \
                patch.object(submit.subprocess, 'run', side_effect=run):
            submit.plan(self.root,self.old,self.recovery,'/known/python',self.digest,[],self.journal)
        self.assertIn('--array=6,7,8%9',calls[0])
        self.assertIn('--array=2%3',calls[1])

    def test_active_array_is_reused(self):
        rows=[dict(id='777_[0-8%9]',name='fims-rad-11111111-fit',state='PENDING',reason='Priority')]
        with patch.object(submit.subprocess,'run') as run:
            self.assertEqual(self.scheduler(rows).submit('fit',indices=list(range(9))),'777')
            run.assert_not_called()

    def test_report_name_changes_with_dependencies(self):
        calls, run = self.fake_submissions()
        with patch.object(submit.subprocess,'run',side_effect=run):
            s=self.scheduler()
            s.submit('report',dependencies=[('afterany','400')])
            s.submit('report',dependencies=[('afterany','401')])
        names=[next(x for x in c if x.startswith('--job-name=')) for c in calls]
        self.assertNotEqual(names[0],names[1])

    def test_same_pending_report_reused(self):
        deps=[('afterany','400')]
        token=hashlib.sha256(json.dumps(deps).encode()).hexdigest()[:8]
        rows=[dict(id='777',name='fims-rad-11111111-report-'+token,state='PENDING',reason='Dependency')]
        with patch.object(submit.subprocess,'run') as run:
            self.assertEqual(self.scheduler(rows).submit('report',dependencies=deps),'777')
            run.assert_not_called()

    def test_complete_study_submits_nothing(self):
        with patch.object(submit,'checked_fit',return_value=True), \
                patch.object(submit,'checked_report',return_value=True), \
                patch.object(submit.subprocess,'run') as run:
            submit.plan(self.root,self.old,self.recovery,'/known/python',self.digest,[],self.journal)
            run.assert_not_called()

    def test_live_worker_with_complete_marker_still_dependency(self):
        rows=[dict(id='700_8',name='fims-rad-11111111-fit',state='RUNNING',reason='gpu')]
        calls, run=self.fake_submissions()
        with patch.object(submit,'checked_fit',return_value=True), \
                patch.object(submit.subprocess,'run',side_effect=run):
            submit.plan(self.root,self.old,self.recovery,'/known/python',self.digest,rows,self.journal)
        self.assertEqual(len(calls),1)
        self.assertIn('--dependency=afterany:700',calls[0])

    def test_queue_failure_is_not_empty_queue(self):
        with patch.object(submit.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['squeue'])):
            with self.assertRaises(subprocess.CalledProcessError):submit.queue_rows()

    def test_bad_scheduler_response_is_rejected(self):
        with patch.object(submit.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout='malformed\n')):
            with self.assertRaises(ValueError):submit.queue_rows()

    def test_duplicate_arrays_are_rejected(self):
        rows=[dict(id=jid,name='exact',state='RUNNING',reason='node') for jid in ('10_1','11_2')]
        with self.assertRaises(ValueError):submit.one_active(rows,'exact')

    def test_cancel_only_recorded_pending_blocked_report(self):
        token=hashlib.sha256(json.dumps([]).encode()).hexdigest()[:8]
        rows=[dict(id='700',name='fims-rad-11111111-report-'+token,state='PENDING',reason='DependencyNeverSatisfied')]
        j=copy.deepcopy(self.journal)
        j['submissions']=[dict(stage='report',id='700')]
        calls, run=self.fake_submissions()
        with patch.object(submit.subprocess,'run',side_effect=run):
            self.scheduler(rows,j).submit('report')
        self.assertEqual(calls[0],['scancel','700'])
        self.assertEqual(calls[1][0],'sbatch')
        with patch.object(submit.subprocess,'run') as run:
            with self.assertRaises(ValueError):self.scheduler(rows).submit('report')
            run.assert_not_called()

    def test_blocked_fit_is_never_cancelled(self):
        rows=[dict(id='700',name='fims-rad-11111111-fit',state='PENDING',reason='DependencyNeverSatisfied')]
        with patch.object(submit.subprocess,'run') as run:
            with self.assertRaises(ValueError):self.scheduler(rows).submit('fit',indices=[0])
            run.assert_not_called()

    def test_dry_run_has_no_scheduler_mutations(self):
        with patch.object(submit.subprocess,'run') as run:
            self.scheduler(dry_run=True).submit('fit',indices=[0])
            run.assert_not_called()
        self.assertFalse((self.root/'results/jobs.json').exists())

    def test_invalid_resource_overrides_rejected(self):
        for env in ({'FIMS_GPU_GRES':'gpu:NV-A100:2'},{'FIMS_RADIUS_CONCURRENCY':'10'}):
            with patch.dict(os.environ,env):
                with self.assertRaises(ValueError):self.scheduler()

    def test_report_hash_tamper_rejected(self):
        out=self.root/'results/report';out.mkdir()
        summary=out/'summary.txt';summary.write_text('verified\n')
        marker={'status':'COMPLETE','outputs':{'summary.txt':hashlib.sha256(summary.read_bytes()).hexdigest()}}
        (out/'fims_radius_complete_v1.json').write_text(json.dumps(marker))
        (out/'fims_radius_sensitivity_review_v1.zip').write_bytes(b'archive')
        self.assertTrue(submit.checked_report(self.root))
        summary.write_text('tampered\n')
        with self.assertRaises(ValueError):submit.checked_report(self.root)

    def test_absolute_spool_paths_and_no_environment_install(self):
        stage=(Path(__file__).parent/'stage.sh').read_text()
        bootstrap=(Path(__file__).parent/'bootstrap.sh').read_text()
        self.assertNotIn('BASH_SOURCE',stage)
        self.assertIn('"$fims_root/code/train_radius.py"',stage)
        self.assertIn('"$fims_root/code/control_radius.py"',stage)
        self.assertIn('"$fims_root/code/report_radius.py"',stage)
        self.assertIn('--allow-partial',stage)
        self.assertNotIn('pip install',bootstrap)
        self.assertIn('prior = json.loads',bootstrap)
        self.assertIn('immutable=True',bootstrap)

    def test_shell_signal_reaches_child_and_preserves_pause_exit(self):
        stage=Path(__file__).parent/'stage.sh'
        fake=Path(self.tmp.name)/'fake_python'
        ready=Path(self.tmp.name)/'ready.txt'
        stopped=Path(self.tmp.name)/'stopped.txt'
        fake.write_text('#!'+sys.executable+'\n'+
            'import os,signal,sys,time\n'+
            'from pathlib import Path\n'+
            'if sys.argv[1] == "-":\n sys.stdin.read();sys.exit(0)\n'+
            'def pause(sig,frame):\n Path(os.environ["FIMS_STOPPED"]).write_text("saved");sys.exit(75)\n'+
            'signal.signal(signal.SIGUSR1,pause)\n'+
            'Path(os.environ["FIMS_READY"]).write_text(str(os.getpid()))\n'+
            'while True:time.sleep(.02)\n')
        fake.chmod(0o700)
        env=dict(os.environ,SLURM_ARRAY_TASK_ID='0',FIMS_READY=str(ready),FIMS_STOPPED=str(stopped))
        proc=subprocess.Popen(['bash',str(stage),'fit',str(self.root),str(self.old),str(self.recovery),str(fake)],
                              env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        try:
            deadline=time.monotonic()+5
            while not ready.exists() and proc.poll() is None and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertTrue(ready.exists(),'Synthetic worker never started')
            proc.send_signal(signal.SIGUSR1)
            output,_=proc.communicate(timeout=5)
            self.assertEqual(proc.returncode,75,output)
            self.assertEqual(stopped.read_text(),'saved')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                proc.wait(timeout=5)

    def test_login_status_uses_legacy_stdlib_and_checks_inventory(self):
        import ast
        status=Path(__file__).parent/'status.sh'
        script=status.read_text()
        python=script.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
        # Parse using Python 3.6 grammar even though the scientific runtime is newer.
        ast.parse(python,feature_version=(3,6))
        self.assertNotIn('radius_common',python)
        self.assertNotIn('is_relative_to',python)
        self.assertNotIn('text=True',python)
        source=self.root/'tiny_source.py'
        source.write_text('verified')
        marker={'schema':'fims_radius_sensitivity_package_v1',
                'files':{'tiny_source.py':hashlib.sha256(source.read_bytes()).hexdigest()}}
        (self.root/'PACKAGE_MANIFEST.json').write_text(json.dumps(marker))
        result=subprocess.run(['bash',str(status),str(self.root)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('No radius job submission journal yet',result.stdout)
        source.write_text('corrupted')
        result=subprocess.run(['bash',str(status),str(self.root)],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Package file changed',result.stderr)


if __name__ == '__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RadiusSchedulerTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    print('RADIUS SCHEDULER SELF-TEST: PASS (missing-only arrays, dependencies, duplicate guard, hashes, signal forwarding)')
