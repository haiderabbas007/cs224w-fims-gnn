"""Standard-library verification and scheduling helpers (login Python >=3.6)."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

SCHEMA = 'fims_operating_point_review_package_v1'
MARKER = 'fims_operating_point_complete_v1.json'
ARCHIVE = 'fims_operating_point_review_v1.zip'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def safe_name(name):
    if not isinstance(name, str):
        return False
    p = PurePosixPath(name)
    return (name not in ('', '.', '..') and not p.is_absolute() and
            '..' not in p.parts and '\\' not in name and str(p) == name)


def safe_file(root, name):
    root = Path(root).resolve()
    if not safe_name(name):
        raise ValueError('Unsafe relative file name: '+str(name))
    p = root/name
    if (p.is_symlink() or not p.is_file() or
            os.path.commonpath([str(root), str(p.resolve())]) != str(root)):
        raise ValueError('Missing or unsafe file: '+str(p))
    return p


def verify_package(root, expected=None):
    root = Path(root).resolve()
    p = safe_file(root, 'PACKAGE_MANIFEST.json')
    digest = sha(p)
    if expected is not None and digest != expected:
        raise ValueError('Installed package identity differs')
    m = json.loads(p.read_text())
    if m.get('schema') != SCHEMA or not isinstance(m.get('files'), dict) or not m['files']:
        raise ValueError('Invalid operating-point package manifest')
    for name, value in m['files'].items():
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('Invalid file digest')
        if sha(safe_file(root, name)) != value:
            raise ValueError('Package file changed: '+name)
    return digest


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Unsafe output path')
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checked_report(root, expected=None, check_zip=True):
    out = Path(root)/'results/report'
    marker = out/MARKER
    if not marker.exists():
        return False
    marker = safe_file(out, MARKER)
    obj = json.loads(marker.read_text())
    if obj.get('status') != 'COMPLETE':
        return False
    if expected is not None and obj.get('package_sha256') != expected:
        raise ValueError('Completed report package identity differs')
    outputs = obj.get('outputs')
    if not isinstance(outputs, dict) or not outputs or MARKER in outputs or ARCHIVE in outputs:
        raise ValueError('Invalid completed report inventory')
    for name, digest in outputs.items():
        if sha(safe_file(out, name)) != digest:
            raise ValueError('Completed report changed: '+name)
    if not (out/ARCHIVE).exists():
        return False
    archive = safe_file(out, ARCHIVE)
    if check_zip:
        with zipfile.ZipFile(str(archive)) as z:
            names = z.namelist()
            if (len(set(names)) != len(names) or
                    set(names) != set(outputs)|{MARKER} or z.testzip() is not None):
                raise ValueError('Review ZIP inventory/integrity mismatch')
            if z.read(MARKER) != marker.read_bytes():
                raise ValueError('Review ZIP marker differs')
            for name, digest in outputs.items():
                if hashlib.sha256(z.read(name)).hexdigest() != digest:
                    raise ValueError('Review ZIP output differs: '+name)
    return True
