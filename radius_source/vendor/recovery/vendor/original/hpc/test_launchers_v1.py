#!/usr/bin/env python3
"""Synthetic scheduler/transfer tests; never contacts Slurm or SSH."""
import hashlib,importlib.util,io,json,os,sys,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import submit
from verify_package import sha,verify
spec=importlib.util.spec_from_file_location('unpack_remote',HERE.parent/'local/unpack_remote.py')
unpack=importlib.util.module_from_spec(spec);spec.loader.exec_module(unpack)

class LaunchTests(unittest.TestCase):
    def test_package_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d);src=b/'src';src.mkdir();(src/'code').mkdir();(src/'code/a.py').write_text('print(1)\n')
            manifest={'schema':'synthetic','files':{'code/a.py':sha(src/'code/a.py')}}
            (src/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest));digest=sha(src/'PACKAGE_MANIFEST.json')
            archive=b/'source.tar.gz'
            with tarfile.open(archive,'w:gz') as t:
                for p in [src/'PACKAGE_MANIFEST.json',src/'code/a.py']:t.add(p,arcname=str(p.relative_to(src)))
            root=unpack.unpack(archive,b/'dest',sha(archive),digest)
            self.assertEqual(root.name,'study_'+digest[:12]);self.assertEqual(verify(root),digest)
            self.assertEqual(unpack.unpack(archive,b/'dest',sha(archive),digest),root)
            (root/'code/a.py').write_text('changed')
            with self.assertRaises(ValueError):unpack.unpack(archive,b/'dest',sha(archive),digest)
            with self.assertRaises(ValueError):verify(root)

    def test_unsafe_archive_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d);archive=b/'bad.tar.gz'
            with tarfile.open(archive,'w:gz') as t:
                m=tarfile.TarInfo('../outside');m.size=1;t.addfile(m,io.BytesIO(b'x'))
            with self.assertRaises(ValueError):unpack.unpack(archive,b/'dest',sha(archive),'0'*64)

    def test_dependency_matrix_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();capacity=Path(d)/'capacity';capacity.mkdir()
            called=[];queue=[]
            def fake_run(command,**kwargs):
                self.assertEqual(command[0],'sbatch')
                called.append(command);jid=str(100+len(called))
                name=next(x.split('=',1)[1] for x in command if x.startswith('--job-name='))
                queue.append(dict(id=jid,name=name,state='PENDING',reason='Priority'))
                class R:stdout=jid+'\n'
                return R()
            args=['submit.py','--study',str(root),'--capacity-root',str(capacity),'--python','/safe/bin/python']
            with patch.object(submit,'verify',return_value='a'*64),patch.object(submit,'query_queue',side_effect=lambda:list(queue)),patch.object(submit.subprocess,'run',side_effect=fake_run),patch.object(sys,'argv',args):
                submit.main();self.assertEqual(len(called),5)
                stages=[c[-4] for c in called];self.assertEqual(stages,['initial_report','prepare','controls','fit','final_report'])
                fit=called[3];self.assertIn('--dependency=afterok:102',fit)
                self.assertIn('--array=0,1,2,3,4,5,6,7,8,9,10,11%3',fit)
                self.assertIn('--dependency=afterok:104:103',called[4])
                self.assertTrue(str(root/'hpc/stage.sh') in fit)
                submit.main();self.assertEqual(len(called),5)

    def test_queue_failure_prevents_submission(self):
        with tempfile.TemporaryDirectory() as d:
            args=['submit.py','--study',d,'--capacity-root',d,'--python','/safe/python']
            with patch.object(submit,'verify',return_value='b'*64),patch.object(submit,'query_queue',side_effect=RuntimeError('scheduler unavailable')),patch.object(submit.subprocess,'run') as run,patch.object(sys,'argv',args):
                with self.assertRaises(RuntimeError):submit.main()
                run.assert_not_called()

    def test_completion_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'metrics.json').write_text('{}')
            (p/'complete.json').write_text(json.dumps({'status':'COMPLETE','outputs':{'metrics.json':sha(p/'metrics.json')}}))
            self.assertTrue(submit.checked_fit(p));(p/'metrics.json').write_text('changed')
            with self.assertRaises(ValueError):submit.checked_fit(p)

if __name__=='__main__':unittest.main()
