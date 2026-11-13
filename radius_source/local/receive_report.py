"""Verify an already downloaded report ZIP and publish its top-level outputs."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile

def receive(path, folder):
    path, folder = Path(path), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder/'fims_radius_sensitivity_review_v1.zip'
    if final.is_symlink(): raise ValueError('Unsafe ZIP destination')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or archive.testzip() is not None:
            raise ValueError('Review ZIP integrity failure')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or name.endswith('/'):
                raise ValueError('Unsafe ZIP path')
        marker_name = 'fims_radius_complete_v1.json'
        status = json.loads(archive.read(marker_name))
        if (status.get('status') not in ('COMPLETE', 'PARTIAL') or
            not isinstance(status.get('outputs'), dict) or not status['outputs']):
            raise ValueError('Invalid radius report marker')
        for name, digest in status['outputs'].items():
            if name not in names or hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('Report hash mismatch: '+name)
        # Only declared and verified report files are made visible beside the ZIP.
        for name in [*status['outputs'], marker_name]:
            if '/' in name or not name.startswith('fims_radius_') or not name.endswith(('.txt','.json','.csv','.png','.pdf','.html')):
                continue
            target = folder/name
            if target.is_symlink(): raise ValueError('Unsafe output destination: '+str(target))
            fd, temporary = tempfile.mkstemp(prefix='.'+name+'.',dir=folder)
            try:
                with archive.open(name) as src, os.fdopen(fd,'wb') as dst:
                    shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
                os.replace(temporary,target)
            finally:
                Path(temporary).unlink(missing_ok=True)
    if final.is_symlink(): raise ValueError('Unsafe ZIP destination')
    os.replace(path,final)
    print('Review status: '+status['status'])
    print('Saved: '+str(final))
    if status['status'] != 'COMPLETE':
        print('Interim results only: unfinished models or seeds are explicitly listed in the report.')
    return status

if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('Usage: receive_report.py TEMP_ZIP DESTINATION')
    receive(sys.argv[1],sys.argv[2])
