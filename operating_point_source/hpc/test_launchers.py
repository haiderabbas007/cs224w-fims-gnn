"""Regression checks for report-only submission, identity and downloaded bytes."""
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'hpc'))
import common
import submit


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


receive = module('op_receive', ROOT/'local/receive_report.py')
unpack = module('op_unpack', ROOT/'local/unpack_remote.py')
DIGEST = 'e'*64


def report_fixture(root, package=DIGEST):
    out = root/'results/report'; out.mkdir(parents=True)
    text = b'SYNTHETIC TEST ONLY\n'
    name = 'fims_operating_point_summary_v1.txt'
    (out/name).write_bytes(text)
    marker = {'status':'COMPLETE', 'package_sha256':package,
              'outputs':{name:hashlib.sha256(text).hexdigest()}}
    (out/common.MARKER).write_text(json.dumps(marker))
    with zipfile.ZipFile(out/common.ARCHIVE, 'w') as z:
        z.write(out/name, name); z.write(out/common.MARKER, common.MARKER)
    return out


class LauncherTests(unittest.TestCase):
    def test_request_cpu_only_absolute_stage(self):
        cmd = submit.command(Path('/new'), Path('/cap'), Path('/rec'), '/old/.venv/bin/python', DIGEST)
        self.assertIn('--partition=shared',cmd)
        self.assertIn('--cpus-per-task=2',cmd)
        self.assertIn('--mem=16G',cmd)
        self.assertIn('--time=04:00:00',cmd)
        self.assertIn('/new/hpc/stage.sh',cmd)
        self.assertFalse(any('--array=' in v or '--gres=' in v or '--dependency=' in v for v in cmd))

    def test_active_report_prevents_resubmit(self):
        rows=[{'name':'fims-op-'+DIGEST[:8]+'-report','id':'123','state':'RUNNING'}]
        with patch.object(submit.subprocess,'run',side_effect=AssertionError('must not submit')):
            self.assertEqual(submit.submit(Path('/new'),Path('/cap'),Path('/rec'),'/py',DIGEST,rows,{}),'123')

    def test_completed_report_prevents_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); report_fixture(root)
            with patch.object(submit.subprocess,'run',side_effect=AssertionError('must not submit')):
                self.assertIsNone(submit.submit(root,Path('/cap'),Path('/rec'),'/py',DIGEST,[],{}))

    def test_failed_queue_not_empty_queue(self):
        with patch.object(submit.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'squeue')):
            with self.assertRaises(subprocess.CalledProcessError): submit.queue_rows()

    def test_submit_persists_job_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'results').mkdir()
            journal={'package_sha256':DIGEST,'submissions':[]}
            result=subprocess.CompletedProcess([],0,stdout='456;koa\n')
            with patch.object(submit.subprocess,'run',return_value=result) as run:
                self.assertEqual(submit.submit(root,Path('/cap'),Path('/rec'),'/py',DIGEST,[],journal),'456')
                self.assertEqual(run.call_args[0][0][0],'sbatch')
            self.assertEqual(json.loads((root/'results/jobs.json').read_text())['submissions'][0]['id'],'456')

    def test_output_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=report_fixture(root)
            self.assertTrue(common.checked_report(root,DIGEST))
            (out/'fims_operating_point_summary_v1.txt').write_text('changed')
            with self.assertRaises(ValueError): common.checked_report(root,DIGEST)

    def test_receive_wrong_package_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=report_fixture(root)
            with self.assertRaises(ValueError): receive.receive(out/common.ARCHIVE,root/'download','a'*64)

    def test_receive_verified_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=report_fixture(root)
            receive.receive(out/common.ARCHIVE,root/'download',DIGEST)
            self.assertEqual((root/'download/fims_operating_point_summary_v1.txt').read_text(),'SYNTHETIC TEST ONLY\n')

    def test_archive_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=report_fixture(root)
            with zipfile.ZipFile(out/common.ARCHIVE,'a') as z:z.writestr('extra.txt','unexpected')
            with self.assertRaises(ValueError): receive.receive(out/common.ARCHIVE,root/'download',DIGEST)

    def test_code_package_modified_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'code.py').write_bytes(b'pass\n')
            manifest={'schema':common.SCHEMA,'files':{'code.py':common.sha(root/'code.py')}}
            (root/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest))
            digest=common.verify_package(root)
            (root/'code.py').write_bytes(b'raise SystemExit\n')
            with self.assertRaises(ValueError): common.verify_package(root,digest)

    def test_unsafe_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); archive=root/'unsafe.tar.gz'
            with tarfile.open(archive,'w:gz') as z:
                obj=tarfile.TarInfo('../outside');obj.size=1;z.addfile(obj,io.BytesIO(b'x'))
            with self.assertRaises(ValueError): unpack.unpack(archive,root/'base',unpack.sha(archive),DIGEST)

    def test_login_sources_python36_syntax(self):
        for path in [ROOT/'hpc/common.py',ROOT/'hpc/status.py',ROOT/'hpc/submit.py',ROOT/'local/unpack_remote.py']:
            ast.parse(path.read_text(),feature_version=(3,6))

    def test_shell_syntax(self):
        for path in [ROOT/'hpc/stage.sh',ROOT/'local/koa.sh']:
            subprocess.run(['bash','-n',str(path)],check=True)

    def test_no_scientific_python_on_login(self):
        local=(ROOT/'local/koa.sh').read_text()
        self.assertNotIn('srun ',local)
        self.assertIn("/usr/bin/python3 '$fims_root/hpc/submit.py'",local)
        stage=(ROOT/'hpc/stage.sh').read_text()
        self.assertIn('"$fims_root/code/report_operating_points.py"',stage)
        self.assertNotIn('BASH_SOURCE',stage)


if __name__=='__main__':unittest.main()
