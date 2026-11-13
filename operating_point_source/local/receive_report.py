#!/usr/bin/env python3
"""Verify a downloaded report and atomically publish its top-level outputs."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile

MARKER = 'fims_operating_point_complete_v1.json'
ARCHIVE = 'fims_operating_point_review_v1.zip'


def receive(path, folder, expected):
    path, folder = Path(path), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder/ARCHIVE
    if final.is_symlink():
        raise ValueError('Unsafe ZIP destination')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(set(names)) != len(names) or z.testzip() is not None:
            raise ValueError('Review ZIP integrity failure')
        for name in names:
            p = PurePosixPath(name)
            if (p.is_absolute() or '..' in p.parts or '\\' in name or name.endswith('/') or
                    str(p) != name or name in ('', '.')):
                raise ValueError('Unsafe ZIP path')
        marker = json.loads(z.read(MARKER))
        outputs = marker.get('outputs')
        if (marker.get('status') != 'COMPLETE' or marker.get('package_sha256') != expected or
                not isinstance(outputs, dict) or not outputs or
                set(names) != set(outputs)|{MARKER} or MARKER in outputs or ARCHIVE in outputs):
            raise ValueError('Report identity or inventory mismatch')
        for name, digest in outputs.items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError('Report hash mismatch: '+name)
        # Only verified top-level report outputs are published beside the ZIP.
        for name in list(outputs)+[MARKER]:
            expected_name = (name.startswith('fims_operating_point_') or
                             name in ('fims_operating_points_v1.csv',
                                      'fims_classifier_score_distributions_v1.pdf'))
            if ('/' in name or not expected_name or
                    not name.endswith(('.txt', '.json', '.csv', '.png', '.pdf', '.html'))):
                continue
            target = folder/name
            if target.is_symlink():
                raise ValueError('Unsafe output destination: '+str(target))
            fd, temporary = tempfile.mkstemp(prefix='.'+name+'.', dir=str(folder))
            try:
                with z.open(name) as src, os.fdopen(fd, 'wb') as dst:
                    shutil.copyfileobj(src, dst); dst.flush(); os.fsync(dst.fileno())
                os.replace(temporary, str(target))
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    if final.is_symlink():
        raise ValueError('Unsafe ZIP destination')
    os.replace(str(path), str(final))
    print('Report COMPLETE; downloaded output hashes verified.')
    print('Saved: '+str(final))
    print('Top-level plots and tables are beside the ZIP; per-fit score histograms are inside it.')
    return marker


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('Usage: receive_report.py TEMP_ZIP DESTINATION EXPECTED_PACKAGE_SHA')
    receive(sys.argv[1], sys.argv[2], sys.argv[3])
