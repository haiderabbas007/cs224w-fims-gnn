#!/usr/bin/env python3
"""Offline transfer/install/report tests. No SSH or cluster access is performed."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


installer = module('unpack_remote')
receiver = module('receive_report')
digest = lambda value: hashlib.sha256(value).hexdigest()


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='fims_radius_local_test_')
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_tar(self, change=None, extra=None):
        content = b'print("synthetic only")\n'
        manifest = json.dumps({'schema': 'synthetic', 'files': {'code/toy.py': digest(content)}}, sort_keys=True).encode()
        entries = [('PACKAGE_MANIFEST.json', manifest), ('code/toy.py', change if change is not None else content)]
        if extra:
            entries.append(extra)
        path = self.root / 'package.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            for name, raw in entries:
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        return path, digest(path.read_bytes()), digest(manifest)

    def install(self, package):
        path, archive_sha, manifest_sha = package
        return installer.unpack(path, self.root / 'installed', archive_sha, manifest_sha)

    def test_install_and_resume_preserve_results(self):
        package = self.make_tar()
        first = self.install(package)
        self.assertEqual(first.name, 'study_' + package[2][:12])
        (first / 'results').mkdir()
        (first / 'results' / 'saved').write_text('training checkpoint')
        self.assertEqual(self.install(package), first)
        self.assertEqual((first / 'results' / 'saved').read_text(), 'training checkpoint')

    def test_existing_source_corruption_rejected(self):
        package = self.make_tar()
        root = self.install(package)
        (root / 'code/toy.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Existing package changed'):
            self.install(package)

    def test_archive_digest_corruption_rejected(self):
        path, archive_sha, manifest_sha = self.make_tar()
        with path.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'archive checksum'):
            self.install((path, archive_sha, manifest_sha))

    def test_manifest_digest_corruption_rejected(self):
        path, archive_sha, _ = self.make_tar()
        with self.assertRaisesRegex(ValueError, 'Manifest checksum'):
            self.install((path, archive_sha, '0' * 64))

    def test_member_hash_corruption_rejected_and_stage_removed(self):
        with self.assertRaisesRegex(ValueError, 'source checksum'):
            self.install(self.make_tar(change=b'changed'))
        self.assertEqual(list((self.root / 'installed').iterdir()), [])

    def test_unsafe_duplicate_and_unlisted_members_rejected(self):
        for extra, message in [(('../escape', b'x'), 'unsafe'),
                               (('code/toy.py', b'x'), 'Duplicate'),
                               (('extra.py', b'x'), 'inventory')]:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                self.install(self.make_tar(extra=extra))

    def test_archive_symlink_rejected(self):
        path, _, manifest_sha = self.make_tar()
        with tarfile.open(path, 'r:gz') as archive:
            raw = [(m, archive.extractfile(m).read()) for m in archive.getmembers()]
        with tarfile.open(path, 'w:gz') as archive:
            for member, content in raw:
                archive.addfile(member, io.BytesIO(content))
            link = tarfile.TarInfo('link')
            link.type, link.linkname = tarfile.SYMTYPE, '/tmp/target'
            archive.addfile(link)
        with self.assertRaisesRegex(ValueError, 'links'):
            self.install((path, digest(path.read_bytes()), manifest_sha))

    def make_zip(self, status='COMPLETE', corrupt=False, extra=None, omit=False):
        path = self.root / 'download.zip'
        name, raw = 'fims_radius_summary_v1.txt', b'Synthetic report\n'
        marker = {'schema': 'fims_radius_report_complete_v1', 'status': status,
                  'outputs': {name: digest(raw)}}
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('fims_radius_complete_v1.json', json.dumps(marker))
            if not omit:
                archive.writestr(name, b'changed' if corrupt else raw)
            if extra:
                archive.writestr(*extra)
        return path

    def receive(self, path):
        with contextlib.redirect_stdout(io.StringIO()):
            return receiver.receive(path, self.root / 'downloads')

    def test_complete_and_partial_reports_publish_verified_outputs(self):
        for status in ('COMPLETE', 'PARTIAL'):
            with self.subTest(status=status):
                result = self.receive(self.make_zip(status=status))
                self.assertEqual(result['status'], status)
                destination = self.root / 'downloads'
                self.assertTrue((destination / 'fims_radius_sensitivity_review_v1.zip').is_file())
                self.assertEqual((destination / 'fims_radius_summary_v1.txt').read_text(), 'Synthetic report\n')

    def test_corrupt_or_missing_report_output_rejected_before_publish(self):
        for args in ({'corrupt': True}, {'omit': True}):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, 'hash mismatch'):
                self.receive(self.make_zip(**args))
            self.assertEqual(list((self.root / 'downloads').iterdir()), [])

    def test_unknown_report_status_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid radius report marker'):
            self.receive(self.make_zip(status='RUNNING'))

    def test_unsafe_report_path_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe ZIP path'):
            self.receive(self.make_zip(extra=('../escape', 'x')))

    def test_undeclared_report_file_is_not_published_beside_zip(self):
        self.receive(self.make_zip(extra=('fims_radius_unverified.txt', 'unverified')))
        self.assertFalse((self.root / 'downloads/fims_radius_unverified.txt').exists())

    def test_report_output_symlink_rejected(self):
        destination = self.root / 'downloads'
        destination.mkdir()
        protected = self.root / 'protected.txt'
        protected.write_text('keep')
        (destination / 'fims_radius_summary_v1.txt').symlink_to(protected)
        with self.assertRaisesRegex(ValueError, 'Unsafe output destination'):
            self.receive(self.make_zip())
        self.assertEqual(protected.read_text(), 'keep')

    def stub_transfer(self, action, master_fail=False):
        bindir = self.root / 'bin'
        bindir.mkdir(exist_ok=True)
        calls = self.root / 'calls.jsonl'
        calls.unlink(missing_ok=True)
        stub = '''#!/usr/bin/env python3
import json, os, pathlib, sys
with open(os.environ['FIMS_STUB_CALLS'], 'a') as stream:
    stream.write(json.dumps({'tool': pathlib.Path(sys.argv[0]).name, 'args': sys.argv[1:]})+'\\n')
if '-MNf' in sys.argv and os.environ.get('FIMS_STUB_MASTER_FAIL') == '1':
    sys.exit(255)
'''
        for tool in ('ssh', 'scp'):
            target = bindir / tool
            target.write_text(stub)
            target.chmod(0o755)
        package = self.root / 'package with spaces'
        package.mkdir(exist_ok=True)
        (package / 'PACKAGE_MANIFEST.json').write_text('{}')
        archive = self.root / 'portable.tar.gz'
        archive.write_bytes(b'synthetic archive; transport stub only')
        env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
                   FIMS_STUB_CALLS=str(calls), FIMS_STUB_MASTER_FAIL=str(int(master_fail)))
        proc = subprocess.run(['bash', str(HERE / 'koa.sh'), action, str(package), str(archive)],
                              env=env, capture_output=True, text=True)
        return proc, [json.loads(line) for line in calls.read_text().splitlines()]

    def test_push_resume_and_status_share_one_authentication(self):
        for action in ('push', 'resume', 'remote-status'):
            with self.subTest(action=action):
                proc, calls = self.stub_transfer(action)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(sum('-MNf' in c['args'] for c in calls), 1)
                paths = [[v for v in c['args'] if v.startswith('ControlPath=')] for c in calls]
                self.assertTrue(all(len(p) == 1 for p in paths))
                self.assertEqual(len({p[0] for p in paths}), 1)
                self.assertEqual(calls[-1]['args'][-3:-1], ['-O', 'exit'])
                commands = '\n'.join(c['args'][-1] for c in calls if c['tool'] == 'ssh')
                if action in ('push', 'resume'):
                    self.assertIn('srun --partition=sandbox', commands)
                    self.assertIn('/hpc/bootstrap.sh', commands)
                    self.assertIn('--expected-sha', commands)
                    self.assertIn('--manifest-sha', commands)
                    self.assertEqual(sum(c['tool'] == 'scp' for c in calls), 1)
                else:
                    self.assertIn('/hpc/status.sh', commands)
                    self.assertFalse(any(c['tool'] == 'scp' for c in calls))

    def test_failed_authentication_does_not_transfer_or_submit(self):
        proc, calls = self.stub_transfer('push', master_fail=True)
        self.assertEqual(proc.returncode, 255)
        self.assertEqual(len(calls), 2)  # Failed master plus cleanup only.
        self.assertFalse(any(c['tool'] == 'scp' for c in calls))


if __name__ == '__main__':
    unittest.main(verbosity=2)
