"""Standard-library-only identity and filesystem checks for the radius study."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA = '6b4fc507d9df8340fc7c728f1576f6565073f4670dfe6e65f1c763261734045a'
RECOVERY_SHA = '3e6c80de0602e77c9dc4b8baa5e25663b047a4ba66eb276975d46f96aa349adf'
PORTABLE_SHA = '4cbf351c982f4fd683b8e3cc87a1fc40bd15ac45d7f9345fc6f26b7e45636f06'
PRESERVED_SHA = '68f9bf045e3db4bb3713296041053be26d9ea07502f9bd5af7afb15414766163'
SCHEMA = 'fims_radius_sensitivity_package_v1'
RADII = (1, 2, 4)
SEEDS = (20260924, 20260925, 20260926)
HISTORICAL_FULL1 = {
    20260924: ('a93313338179b5bfb1bf3a89f4c17e4ea3f35517b5b32eb13301b08a2537e69a', '3b7601ec9d76b7bff593c711178f33d6fd963c64668bfb01f714095f47748f60'),
    20260925: ('ea9f0f1730703175b8c4e49a3efb1c4b86ac2a16cfd04780264801307143c0cb', 'd5baee98e0c40adfa85fd63d9edafa956c984759e076bf88fe3bc502371813f3'),
    20260926: ('1bb10f7133ade9c9d429688fe8b93834de70078d6f76250625f46a887cc8a82f', '347ccdabe4dd471a239687536309c56235a725ecb3116fc6babdedd83a180322'),
}

def setup_paths():
    for path in (ROOT/'vendor/recovery/vendor/original/code', ROOT/'vendor/recovery/code', ROOT/'code'):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return ROOT

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n'

def safe_file(root, name):
    root = Path(root).resolve()
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or str(rel) != name or name in ('', '.'):
        raise ValueError('Unsafe relative path: '+name)
    target = root/name
    if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root):
        raise ValueError('Missing/unsafe file: '+str(target))
    return target

def verify_inventory(root, expected=None, schema=None):
    marker = safe_file(root, 'PACKAGE_MANIFEST.json')
    digest = sha(marker)
    if expected is not None and digest != expected:
        raise ValueError('Source package identity changed: '+str(root))
    inventory = json.loads(marker.read_text())
    if schema is not None and inventory.get('schema') != schema:
        raise ValueError('Wrong package schema: '+str(root))
    if not isinstance(inventory.get('files'), dict) or not inventory['files']:
        raise ValueError('Empty package inventory')
    for name, value in inventory['files'].items():
        if sha(safe_file(root, name)) != value:
            raise ValueError('Package file changed: '+str(Path(root)/name))
    return digest

def verify_package(root=ROOT):
    root = Path(root).resolve()
    digest = verify_inventory(root, schema=SCHEMA)
    verify_inventory(root/'vendor/recovery', RECOVERY_SHA, 'fims_gnn_recovery_package_v1')
    verify_inventory(root/'vendor/recovery/vendor/original', ORIGINAL_SHA, 'fims_parallel_package_v1')
    plan = json.loads(safe_file(root, 'EXPERIMENT_PLAN.json').read_text())
    if (plan.get('schema') != 'fims_radius_sensitivity_plan_v1' or
        plan.get('radii_mm') != list(RADII) or plan.get('seeds') != list(SEEDS) or
        plan.get('model') != 'full1' or plan.get('history_bins') != 8 or
        plan.get('coordinate_normalization_mm') != 2 or plan.get('epochs') != 10):
        raise ValueError('Unexpected radius experiment plan')
    return digest

def verify_sources(root, original, recovery):
    root, original, recovery = (Path(p).resolve() for p in (root, original, recovery))
    package_sha = verify_package(root)
    for prior in (original, recovery):
        if root == prior or root.is_relative_to(prior) or prior.is_relative_to(root):
            raise ValueError('New study must be separate from the existing studies')
    verify_inventory(original, ORIGINAL_SHA, 'fims_parallel_package_v1')
    verify_inventory(recovery, RECOVERY_SHA, 'fims_gnn_recovery_package_v1')
    marker = safe_file(original/'data', 'site_manifest_v1.json')
    data = json.loads(marker.read_text())
    if (data.get('schema') != 'fims_site_geometry_v1' or data.get('synthetic', False) or
        data.get('portable_manifest_sha256') != PORTABLE_SHA or
        data.get('preserved_manifest_sha256') != PRESERVED_SHA):
        raise ValueError('Expected the original frozen development site dataset')
    events = data.get('events', [])
    if len(events) != 10000 or sum(e['split']=='train' for e in events) != 7500 or sum(e['split']=='validation' for e in events) != 2500:
        raise ValueError('Expected original 7,500/2,500 event split')
    site_sha = sha(marker)
    reference = {}
    for seed, (protocol_sha, complete_sha) in HISTORICAL_FULL1.items():
        folder = recovery/'results/fits/full1'/('seed'+str(seed))
        pp, cp = safe_file(folder, 'protocol.json'), safe_file(folder, 'complete.json')
        if sha(pp) != protocol_sha or sha(cp) != complete_sha:
            raise ValueError('Historical full1 reference differs from reviewed results: '+str(seed))
        protocol = json.loads(pp.read_text())
        complete = json.loads(cp.read_text())
        if (protocol.get('site_manifest_sha256') != site_sha or
            complete.get('status') != 'COMPLETE' or
            complete.get('identity', {}).get('protocol_sha256') != protocol_sha):
            raise ValueError('Site dataset is not the one used by the reviewed GNN fits')
        reference[str(seed)] = dict(protocol_sha256=protocol_sha, complete_sha256=complete_sha)
    return dict(schema='fims_radius_source_receipt_v1', package_sha256=package_sha,
                original_package_sha256=ORIGINAL_SHA, recovery_package_sha256=RECOVERY_SHA,
                original_study=str(original), recovery_study=str(recovery),
                site_manifest_sha256=site_sha, data=str(original/'data'), historical_full1_references=reference,
                note='Read-only source verification. Each task separately hashes every event through SiteData.')

def verify_data_binding(root, data):
    root, data = Path(root).resolve(), Path(data).resolve()
    receipt = json.loads(safe_file(root, 'results/source_receipt.json').read_text())
    current = verify_sources(root, receipt['original_study'], receipt['recovery_study'])
    if (receipt.get('package_sha256') != verify_package(root) or
        receipt != current or
        Path(receipt.get('data', '')).resolve() != data or
        sha(safe_file(data, 'site_manifest_v1.json')) != receipt.get('site_manifest_sha256')):
        raise ValueError('Task dataset differs from the reviewed source receipt')
    return receipt

def atomic_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink(): raise ValueError('Unsafe output: '+str(path))
    raw = canonical(value)
    if immutable and path.exists():
        if path.read_bytes() != raw: raise ValueError('Immutable output differs: '+str(path))
        return
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)

def checked_fit(path):
    """False only for unfinished work; corrupt completed work is an error."""
    path = Path(path)
    if not (path/'complete.json').exists(): return False
    marker = json.loads(safe_file(path, 'complete.json').read_text())
    protocol = json.loads(safe_file(path, 'protocol.json').read_text())
    identity = marker.get('identity', {})
    if (path.name != 'seed'+str(identity.get('seed')) or
        path.parent.name != identity.get('model') or
        path.parent.parent.name != 'r'+str(int(protocol.get('radius_mm', -1)))):
        raise ValueError('Completed fit is in a different radius/model/seed path: '+str(path))
    required = {'metrics.json', 'protocol.json', 'selected_predictions.npz'}
    if identity.get('model') == 'full1':
        required |= {'selected_checkpoint.pt', 'final_checkpoint.pt', 'final_predictions.npz', 'history.json'}
    elif identity.get('model') in ('counts8', 'shape8'):
        required |= {'model_parameters.json'}
    else:
        raise ValueError('Unexpected radius model in completion: '+str(path))
    if (marker.get('status') != 'COMPLETE' or not required.issubset(marker.get('outputs', {})) or
        identity.get('protocol_sha256') != sha(path/'protocol.json') or
        protocol.get('radius_mm') not in RADII):
        raise ValueError('Invalid completed fit identity: '+str(path))
    for name, digest in marker['outputs'].items():
        if Path(name).name != name or sha(safe_file(path, name)) != digest:
            raise ValueError('Completed output changed: '+str(path/name))
    metrics = json.loads(safe_file(path, 'metrics.json').read_text())
    if metrics.get('identity') != identity:
        raise ValueError('Completed metrics identity differs: '+str(path))
    return True

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--study-root', type=Path, default=ROOT)
    p.add_argument('--original-study', type=Path)
    p.add_argument('--recovery-study', type=Path)
    args = p.parse_args()
    if args.original_study is not None and args.recovery_study is not None:
        print(json.dumps(verify_sources(args.study_root,args.original_study,args.recovery_study),indent=2))
    else:
        print('RADIUS PACKAGE: VERIFIED '+verify_package(args.study_root))
