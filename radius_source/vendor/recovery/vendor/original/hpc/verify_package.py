#!/usr/bin/env python3
"""Read-only package/data identity checks. No model or data transformation."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

CAPACITY_MANIFEST_SHA = '4cbf351c982f4fd683b8e3cc87a1fc40bd15ac45d7f9345fc6f26b7e45636f06'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def safe_file(root, name):
    r = PurePosixPath(name)
    if r.is_absolute() or '..' in r.parts or '\\' in name or str(r) != name:
        raise ValueError('Unsafe package path: '+name)
    p = root/name
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
        raise ValueError('Missing or unsafe package file: '+name)
    return p

def verify(root, capacity=None):
    root = Path(root).resolve()
    mf = safe_file(root, 'PACKAGE_MANIFEST.json')
    manifest = json.loads(mf.read_text())
    if not isinstance(manifest.get('files'), dict) or not manifest['files']:
        raise ValueError('Package inventory missing')
    for name, digest in manifest['files'].items():
        if sha(safe_file(root, name)) != digest:
            raise ValueError('Package hash mismatch: '+name)
    digest = sha(mf)
    if root.name.startswith('study_') and root.name != 'study_'+digest[:12]:
        raise ValueError('Versioned study directory disagrees with its package')
    if capacity is not None:
        cm = Path(capacity).resolve()/'data/portable_manifest_v2.json'
        if not cm.is_file() or cm.is_symlink() or sha(cm) != CAPACITY_MANIFEST_SHA:
            raise ValueError('Wrong/missing frozen capacity dataset')
    return digest

def main():
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('--capacity-root');a=p.parse_args()
    print('PACKAGE / FROZEN DEVELOPMENT IDENTITY: PASS '+verify(a.study,a.capacity_root))

if __name__=='__main__':main()
