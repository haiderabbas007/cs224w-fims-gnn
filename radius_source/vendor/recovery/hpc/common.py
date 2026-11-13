#!/usr/bin/env python3
"""Read-only identity checks shared by the versioned recovery launchers."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

ORIGINAL_SHA = '6b4fc507d9df8340fc7c728f1576f6565073f4670dfe6e65f1c763261734045a'
CAPACITY_SHA = '4cbf351c982f4fd683b8e3cc87a1fc40bd15ac45d7f9345fc6f26b7e45636f06'
SCHEMA = 'fims_gnn_recovery_package_v1'
MODELS = ('full2', 'full1', 'no_xy2', 'current2')
SEEDS = (20260924, 20260925, 20260926)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()

def safe_file(root, name):
    root = Path(root).resolve()
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or str(rel) != name:
        raise ValueError('Unsafe relative path: '+name)
    target = root/name
    if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root):
        raise ValueError('Missing/unsafe file: '+str(target))
    return target

def verify_inventory(root, *, expected=None, schema=None):
    root = Path(root).resolve()
    marker = safe_file(root, 'PACKAGE_MANIFEST.json')
    digest = sha(marker)
    if expected is not None and digest != expected:
        raise ValueError('Unexpected package manifest: '+str(root))
    manifest = json.loads(marker.read_text())
    if schema is not None and manifest.get('schema') != schema:
        raise ValueError('Unexpected recovery package schema')
    if not isinstance(manifest.get('files'), dict) or not manifest['files']:
        raise ValueError('Package inventory missing')
    for name, expected_hash in manifest['files'].items():
        if sha(safe_file(root, name)) != expected_hash:
            raise ValueError('Package changed: '+str(root/name))
    return digest

def verify(root, original=None, capacity=None):
    root = Path(root).resolve()
    digest = verify_inventory(root, schema=SCHEMA)
    verify_inventory(root/'vendor/original', expected=ORIGINAL_SHA)
    if original is not None:
        original = Path(original).resolve()
        if original == root or root.is_relative_to(original) or original.is_relative_to(root):
            raise ValueError('Recovery must be separate from the original study')
        verify_inventory(original, expected=ORIGINAL_SHA)
        marker = safe_file(original/'data', 'site_manifest_v1.json')
        m = json.loads(marker.read_text())
        if m.get('portable_manifest_sha256') != CAPACITY_SHA:
            raise ValueError('Original site data refer to another development sample')
    if capacity is not None:
        if sha(safe_file(Path(capacity)/'data', 'portable_manifest_v2.json')) != CAPACITY_SHA:
            raise ValueError('Capacity reference manifest changed')
    return digest

# Public wrapper name used by recovery report/launcher code.
verify_packages = verify

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink(): raise ValueError('Unsafe output: '+str(path))
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def checked_fit(path):
    path = Path(path)
    marker = path/'complete.json'
    if not marker.exists(): return False
    m = json.loads(safe_file(path, 'complete.json').read_text())
    required = {'metrics.json', 'protocol.json', 'selected_predictions.npz', 'history.json',
                'selected_checkpoint.pt', 'final_checkpoint.pt', 'final_predictions.npz'}
    if m.get('status') != 'COMPLETE' or not required.issubset(m.get('outputs', {})):
        raise ValueError('Invalid fit completion inventory: '+str(path))
    for name, digest in m['outputs'].items():
        if Path(name).name != name or sha(safe_file(path, name)) != digest:
            raise ValueError('Completed fit output changed: '+str(path/name))
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument('root'); p.add_argument('--original-study'); p.add_argument('--capacity-root')
    a = p.parse_args()
    print('RECOVERY / ORIGINAL CODE / FROZEN DEVELOPMENT: PASS '+verify(a.root, a.original_study, a.capacity_root))

if __name__ == '__main__': main()
