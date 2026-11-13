#!/usr/bin/env python3
"""Replay a pinned ideal readout and retain its pre-binning arrival times.

This is an additive export of the original 10,000 development events. It does
not run Geant4, train a model, or read calibration/confirmation data. Every
native electron field and reconstructed binary-site field must match exactly.
"""
from __future__ import annotations
import argparse
import ast
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_MANIFEST = 'expanded_transport_data_manifest_v2.json'
DATA_SHA = 'fd62d8c55c4a2455d95e4ef4152ad29405c15afacd99cceb6386091dec78424d'
SPLIT_SHA = 'f1333977acf03370f8bdb3c850abf75b6ea6a344e2b25b3734d12bcbbb61a9e6'
READOUT_SHA = 'c730d745ebe8870661693d0c74e99dc1ac8c7860b02517f9033d1ab8d621bce3'
PREFIX = 'fims_common_source_'
SCHEMA = 'fims_continuous_arrival_export_v1'
FROZEN = ('ideal_electron_inputs_v1.npz', 'ideal_electron_targets_v1.npz',
          'binary_spike_inputs_v1.npz', 'binary_spike_targets_v1.npz',
          'hit_truth_labels_v1.npz', 'readout_audit_v1.json')
FUNCTIONS = {'require','integer_array','validate_root_arrays','event_rng',
             'sample_event','make_targets_and_collision_mask','concatenate',
             'export_electrons','validate_electrons','label_tracks','encode'}
ASSIGNMENTS = {'PREFIX','PARTITION_SPECS','TRUTH_RULE','IDENTITIES','BRANCHES',
               'GAIN','HALF_Z_MM','RADIUS_MM','DRIFT_MM_NS','SIGMA_T_MM_SQRT_CM',
               'SIGMA_L_MM_SQRT_CM','PIXEL_MM','GLOBAL_PIXELS','CLOCK_NS',
               'ELECTRON_ASSUMPTIONS','BINARY_ASSUMPTIONS'}
LIMITATIONS = [
    'Arrival timestamps are replayed samples from the pinned ideal readout, not measured hardware hits.',
    'All electrons from a source step use its post-step Gtime; individual creation times are not measured.',
    'No electronics noise, thresholds, efficiency, pulse merging, or deadtime model has been added.',
    'Stored electron order remains native bin/address order solely for exact alignment. It is NOT arrival order.',
    'Hit inference must use arrival_time_ns and strictly earlier arrivals; exact-time ties use a common pre-update state.',
    'Known event boundaries remain supplied. Continuous time within events is not yet an overlaid continuous detector stream.',
    'The original electron supervision mask is retained, including exclusions for mixed 25 ns sites. No truth rule was relaxed.',
    'Native site recall and electron recall have different denominators. Compare charge loss through the explicit electron-to-site mapping.',
]


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), 'Refusing symlink output: '+str(path))
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, obj):
    atomic_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())


def save_npz(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
        with np.load(name, allow_pickle=False) as z:
            require(set(z.files)==set(arrays), 'Serialized NPZ keys differ')
            for k in arrays:
                equal_array(z[k], arrays[k], 'serialized/'+k)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def safe_file(base, relative, digest=None):
    base = Path(base).resolve(); p = base/relative
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts
            and not p.is_symlink() and p.resolve().is_relative_to(base) and p.is_file(),
            'Missing/unsafe frozen input: '+str(p))
    if digest is not None:
        require(sha(p)==digest, 'Frozen input hash differs: '+str(p))
    return p


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def equal_array(a, b, label):
    require(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),
            'Exact native replay mismatch: '+label)


def compare_fields(replayed, saved, label, expected_saved_extras=()):
    require(set(saved)==set(replayed)|set(expected_saved_extras), label+' field set changed')
    for key in replayed:
        equal_array(replayed[key],saved[key],label+'/'+key)


def frozen_namespace(continuous=False):
    """Only compile the allowlisted pure definitions from the pinned source.

    The two additions below merely retain a local value already computed by
    the sampler. No draw, sort, target rule, or arithmetic is replaced.
    """
    source = safe_file(HERE, 'readout_v2_reference.py', READOUT_SHA)
    tree = ast.parse(source.read_text())
    nodes = []
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in FUNCTIONS:
            nodes.append(copy.deepcopy(n))
        elif isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ASSIGNMENTS:
            nodes.append(copy.deepcopy(n))
    require({n.name for n in nodes if isinstance(n,ast.FunctionDef)}==FUNCTIONS,
            'Pinned pure-function inventory changed')
    if continuous:
        additions = 0
        for fn in nodes:
            if not isinstance(fn,ast.FunctionDef):
                continue
            for n in ast.walk(fn):
                if not isinstance(n,ast.Assign) or len(n.targets)!=1 or not isinstance(n.targets[0],ast.Name):
                    continue
                if fn.name=='sample_event' and n.targets[0].id=='token':
                    require(isinstance(n.value,ast.Dict), 'Sampler token structure changed')
                    n.value.keys.append(ast.Constant('arrival_time_ns'))
                    n.value.values.append(ast.parse('arrival[accepted][order]', mode='eval').body)
                    additions += 1
                if fn.name=='export_electrons' and n.targets[0].id=='ikeys':
                    require(isinstance(n.value,ast.Dict), 'Exporter field structure changed')
                    n.value.keys.append(ast.Constant('arrival_time_ns'))
                    n.value.values.append(ast.parse('np.float64', mode='eval').body)
                    additions += 1
        require(additions==2, 'Expected exactly two value-retention additions')
    module = ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    ns = {'np':np, 'json':json, 'hashlib':hashlib, 'Path':Path, '__name__':'fims_pinned_sampler'}
    exec(compile(module,str(source),'exec'),ns)
    return ns


def arrival_order(inputs):
    """Observation-only order; tied times still require joint/pre-update handling."""
    ptr=inputs['event_ptr']; order=[]; ties=0
    for a,b in zip(ptr[:-1],ptr[1:]):
        a,b=int(a),int(b)
        ix=np.lexsort((inputs['global_pixel_y'][a:b],inputs['global_pixel_x'][a:b],
                       inputs['readout_side'][a:b],inputs['arrival_time_ns'][a:b]))
        times=inputs['arrival_time_ns'][a:b][ix]
        ties += int(np.count_nonzero(np.diff(times)==0))
        order.append(ix.astype(np.int64)+a)
    return np.concatenate(order) if order else np.empty(0,np.int64), ties


def load_root(production, manifest, ns):
    import uproot
    root_path=safe_file(production,manifest['root_name'],manifest['root_sha256'])
    require(root_path.stat().st_size==manifest['root_bytes'], 'ROOT size changed')
    names=set(ns['BRANCHES'])|{'MomIn','HitPathLength_fromTrack','ParticleName','CreatorProcess'}
    root={}
    with uproot.open(root_path) as f:
        require('HitNtuple' in f and 'EventNtuple' in f,'Required ROOT trees absent')
        h,e=f['HitNtuple'],f['EventNtuple']
        require(e.num_entries==500 and names.issubset(h.keys()),'ROOT event count/schema changed')
        for name in sorted(names):
            values=h[name].array(library='np')
            require(values.ndim==1 and len(values)==h.num_entries,'Non-scalar ROOT branch '+name)
            if name in {'ParticleName','CreatorProcess'}:
                require(all(isinstance(v,str) for v in values),'Invalid string branch '+name)
                root[name]=np.asarray(values,dtype=str)
            elif name in ns['IDENTITIES']:
                root[name]=ns['integer_array'](values,name)
            else:
                require(np.isfinite(values).all(),'Nonfinite ROOT branch '+name)
                root[name]=values.astype(np.float64)
        require(np.all((root['EventID']>=0)&(root['EventID']<500)),'Invalid ROOT event ID')
        require(np.array_equal(e['NumberOfHits'].array(library='np'),np.bincount(root['EventID'],minlength=500)),
                'ROOT event/hit counts disagree')
    checked,residuals=ns['validate_root_arrays'](root,manifest['run_uid']);root.update(checked)
    require(len(root['EventID'])==manifest['results']['hit_rows'],'Manifest ROOT row count changed')
    return root,residuals


def load_plan(source):
    p=safe_file(source,DATA_MANIFEST,DATA_SHA); manifest=json.loads(p.read_text())
    require(manifest['schema']=='fims_expanded_transport_data_v2' and manifest['status']=='COMPLETE',
            'Wrong development production identity')
    require(manifest['original_split_assignments_preserved'] and not manifest['independent_test_claim'],
            'Wrong development/confirmation scope')
    outputs=manifest['outputs']
    require(all(name in outputs for name in ('frozen_protocol.json','split_lock.json')),
            'Frozen protocol or split lock absent from pinned inventory')
    require(outputs['frozen_protocol.json']==manifest['freeze_sha256'],
            'Frozen protocol digest disagrees with production freeze')
    protocol=json.loads(safe_file(source,'frozen_protocol.json',outputs['frozen_protocol.json']).read_text())
    split_lock=json.loads(safe_file(source,'split_lock.json',outputs['split_lock.json']).read_text())
    require(protocol['schema']=='fims_expanded_transport_protocol_v2', 'Wrong frozen protocol schema')
    require(split_lock['schema']=='fims_expanded_transport_split_lock_v2' and
            split_lock['freeze_sha256']==manifest['freeze_sha256'] and
            split_lock['event_split_sha256']==SPLIT_SHA, 'Split lock lineage disagreement')
    require(split_lock['assignment_precedes_production'] and
            split_lock['assignment_preserved_from_lineage'] and
            not split_lock['assignment_uses_truth'] and not split_lock['reranked_after_transport_version'],
            'Development split assignment policy changed')
    split=safe_file(source,'event_split.csv',SPLIT_SHA)
    require(manifest['outputs']['event_split.csv']==SPLIT_SHA,'Split hash disagreement')
    with split.open(newline='') as f: rows=list(csv.DictReader(f))
    require(len(rows)==10000 and len({(r['shard'],int(r['event_id_raw'])) for r in rows})==10000,
            'Split event identity count changed')
    require(sum(r['split']=='train' for r in rows)==7500 and
            sum(r['split']=='development_validation' for r in rows)==2500,'Wrong split counts')
    for block in range(20):
        selected=[r for r in rows if r['shard']==f'block{block:02d}']
        require(sorted(int(r['event_id_raw']) for r in selected)==list(range(500)), 'Shard split incomplete')
    return manifest,rows


def native_shard(source, all_manifest, block):
    needed={}
    for suffix in FROZEN:
        rel=f'{block}/readout/{PREFIX}{suffix}'
        require(rel in all_manifest['outputs'],'Native file absent from frozen inventory: '+rel)
        needed[rel]=all_manifest['outputs'][rel]
    rel=f'{block}/production/production_manifest.json'
    needed[rel]=all_manifest['outputs'][rel]
    paths={n:safe_file(source,n,h) for n,h in needed.items()}
    pm=json.loads(paths[rel].read_text())
    root_rel=f'{block}/production/{pm["root_name"]}'
    require(all_manifest['outputs'].get(root_rel)==pm['root_sha256'],'ROOT manifest disagreement')
    needed[root_rel]=pm['root_sha256']
    paths[root_rel]=safe_file(source,root_rel,pm['root_sha256'])
    audit=json.loads(paths[f'{block}/readout/{PREFIX}readout_audit_v1.json'].read_text())
    pv=audit['provenance']
    require(audit['status']=='PASS' and audit['partition_status']==block and pm['status']=='PASS'
            and pm['partition']==block and pm['events']==500,'Native stage not complete')
    require(pv['script_sha256']==READOUT_SHA and pv['readout_adapter_sha256']==READOUT_SHA,
            'Unrecognized native readout producer')
    require(pv['root_sha256']==pm['root_sha256'] and pv['production_run_uid']==pm['run_uid']
            and pv['manifest_sha256']==needed[rel] and pv['freeze_sha256']==all_manifest['freeze_sha256'],
            'Native source lineage disagreement')
    require(pv['production_seeds']==[2026100601+2*int(block[-2:]),2026100602+2*int(block[-2:])]
            and pv['sampler_seed']==2026100701+int(block[-2:]),'Frozen RNG schedule changed')
    return needed, paths, pm, audit


def export_block(source,out,block,all_manifest,split_rows,ns,identity):
    print(f'{block}: verifying frozen ROOT, readout and truth hashes...',flush=True)
    digests,paths,pm,audit=native_shard(source,all_manifest,block)
    require(np.__version__==audit['provenance']['numpy_version'],
            f'{block}: replay needs original NumPy {audit["provenance"]["numpy_version"]}; '
            f'current {np.__version__}. Use the original g4tpc interpreter; do not alter the source environment.')
    dest=out/block;dest.mkdir(exist_ok=True)
    marker=dest/'complete.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        require(old['identity']==identity and old['source_hashes']==digests,'Completed shard provenance changed')
        for n,h in old['output_sha256'].items():safe_file(dest,n,h)
        print(f'{block}: verified completed timestamp export; reusing',flush=True)
        return old
    root,residuals=load_root(source/block/'production',pm,ns)
    uid=pm['run_uid'];role,tracks=ns['label_tracks'](root,500,uid)
    native={suffix:load_npz(paths[f'{block}/readout/{PREFIX}{suffix}']) for suffix in FROZEN if suffix.endswith('.npz')}
    hit=native['hit_truth_labels_v1.npz']
    equal_array(role,hit['truth_class'],block+'/ROOT hit truth roles')
    for key in ns['IDENTITIES']:equal_array(root[key],hit[key],block+'/ROOT identities/'+key)
    require(str(hit['root_sha256'])==pm['root_sha256'] and str(hit['truth_rule'])==ns['TRUTH_RULE'],
            'Saved hit-truth rule/root hash changed')
    print(f'{block}: replaying original sampler; saving pre-binning times...',flush=True)
    x,y,events=ns['export_electrons'](root,role,500,uid,audit['provenance']['sampler_seed'],
                                     int(audit['totals']['generated_electrons']),block)
    times=x.pop('arrival_time_ns')
    compare_fields(x,native['ideal_electron_inputs_v1.npz'],block+'/ideal inputs')
    compare_fields(y,native['ideal_electron_targets_v1.npz'],block+'/ideal truth',
                   ('inputs_npz_sha256','hit_truth_labels_sha256','root_sha256'))
    bx,by=ns['encode'](x,y)
    compare_fields(bx,native['binary_spike_inputs_v1.npz'],block+'/binary inputs')
    compare_fields(by,native['binary_spike_targets_v1.npz'],block+'/binary truth',
                   ('inputs_npz_sha256','source_electron_inputs_sha256','source_electron_targets_sha256','root_sha256'))
    require(times.dtype==np.float64 and np.isfinite(times).all() and np.all(times>=0),'Invalid exact arrivals')
    equal_array(np.floor(times/25.0).astype(np.int64),x['time_bin_abs'],block+'/arrival rebinned')
    require(sum(e['generated_electrons'] for e in events)==audit['totals']['generated_electrons'],
            'Generated electron count changed')
    require(len(times)==audit['totals']['accepted_electrons'],'Accepted electron count changed')
    # Truth never enters the observable-only inputs archive. Native bin fields
    # remain a mapping check in the target sidecar, not a new continuous feature.
    inputs={k:x[k].copy() for k in ('event_ptr','global_pixel_x','global_pixel_y','readout_side','pixel_pitch_mm')}
    inputs.update(schema_version=np.asarray('fims_continuous_ideal_electron_inputs_v1'),
                  arrival_time_ns=times, time_unit=np.asarray('ns'),
                  storage_order=np.asarray('native bin/address order; sort by arrival_time_ns before inference'),
                  tie_policy=np.asarray('strictly earlier arrivals; equal timestamps see common pre-update context'))
    order,ties=arrival_order(inputs)
    targets={k:y[k].copy() for k in ('event_ptr','run_uid','event_run_uid','event_id_raw','truth_class',
                                     'supervision_mask','ambiguous_site_mask','source_hit_row','track_id','parent_id','pdg')}
    targets.update(schema_version=np.asarray('fims_continuous_ideal_electron_targets_v1'),
                   target_microcurler=y['target_core'].copy(),
                   electron_to_site=by['electron_to_site'].copy(),
                   native_time_bin_abs=x['time_bin_abs'].copy(),
                   arrival_order_native_index=order,
                   primary_electron_weight=((y['truth_class']==0)&(y['parent_id']==0)).astype(np.int64),
                   partition_status=np.asarray(block),root_sha256=np.asarray(pm['root_sha256']))
    save_npz(dest/'continuous_inputs_v1.npz',inputs)
    targets['inputs_npz_sha256']=np.asarray(sha(dest/'continuous_inputs_v1.npz'))
    save_npz(dest/'continuous_targets_v1.npz',targets)
    # Ship byte-identical sites beside the new times for direct paired analyses.
    for suffix,newname in [('binary_spike_inputs_v1.npz','site_inputs_v1.npz'),
                           ('binary_spike_targets_v1.npz','site_targets_v1.npz')]:
        atomic_bytes(dest/newname,paths[f'{block}/readout/{PREFIX}{suffix}'].read_bytes())
    for n,h in digests.items():safe_file(source,n,h)
    rows=[r for r in split_rows if r['shard']==block]
    require(all(r['run_uid']==uid for r in rows),'ML split run UID changed')
    meta=dict(schema=SCHEMA,status='COMPLETE',identity=identity,source_hashes=digests,
              run_uid=uid,shard=block,events=500,accepted_electrons=len(times),sites=int(bx['event_ptr'][-1]),
              empty_events=int(np.count_nonzero(np.diff(inputs['event_ptr'])==0)),
              equal_timestamp_adjacent_pairs=ties,
              nonzero_subbin_fraction=float(np.mean(times!=25.0*np.floor(times/25.0))) if len(times) else None,
              exact_native_electron_inputs=True,exact_native_electron_truth=True,
              exact_binary_site_roundtrip=True,all_original_masks_preserved=True,
              native_environment={k:audit['provenance'][k] for k in ('python_version','numpy_version','uproot_version')},
              source_drift_contract=residuals,
              split_counts={s:sum(r['split']==s for r in rows) for s in ('train','development_validation')},
              output_sha256={n:sha(dest/n) for n in ('continuous_inputs_v1.npz','continuous_targets_v1.npz','site_inputs_v1.npz','site_targets_v1.npz')})
    write_json(marker,meta)
    print(f'{block}: EXACT ELECTRON/TRUTH/SITE ROUNDTRIP PASS; {len(times):,} arrivals',flush=True)
    return meta


def archive_outputs(out,markers,source,copy_dir):
    # Recheck the pinned metadata after all shard work, before any publication.
    manifest,_=load_plan(source)
    metadata_hashes={DATA_MANIFEST:DATA_SHA, **{
        name:manifest['outputs'][name]
        for name in ('event_split.csv','frozen_protocol.json','split_lock.json')}}
    summary='FIMS CONTINUOUS IDEAL-ARRIVAL EXPORT V1: COMPLETE\n\n'
    summary+=f'20/20 frozen development shards; {sum(m["events"] for m in markers):,} events; '
    summary+=f'{sum(m["accepted_electrons"] for m in markers):,} accepted ideal electrons.\n'
    summary+='Exact native electron inputs/truth and binary-site reconstruction: PASS for every shard.\n'
    summary+='No new Geant4 simulation, training, model scoring, or cutoff fitting ran.\n'
    summary+='The original 7,500/2,500 event split and original supervision masks are retained.\n\n'
    summary+='\n'.join('- '+line for line in LIMITATIONS)+'\n'
    atomic_bytes(out/'fims_continuous_arrival_summary_v1.txt',summary.encode())
    for name,digest in metadata_hashes.items():
        data=safe_file(source,name,digest).read_bytes()
        require(hashlib.sha256(data).hexdigest()==digest,
                'Frozen metadata changed during export: '+name)
        atomic_bytes(out/name,data)
    files={str(p.relative_to(out)):sha(p) for block in range(20) for p in sorted((out/f'block{block:02d}').iterdir()) if p.is_file()}
    files.update({n:sha(out/n) for n in ('event_split.csv',DATA_MANIFEST,'frozen_protocol.json','split_lock.json')})
    write_json(out/'continuous_dataset_manifest_v1.json',dict(schema=SCHEMA,status='COMPLETE',
               development_only=True,source_data_manifest_sha256=DATA_SHA,event_split_sha256=SPLIT_SHA,
               events=10000,train_events=7500,validation_events=2500,files=files,
               native_model_features=['global_pixel_x','global_pixel_y','readout_side','arrival_time_ns'],
               limitations=LIMITATIONS))
    review=out/'fims_continuous_arrival_review_v1.zip'
    fd,tmp=tempfile.mkstemp(prefix='.review.',dir=out);os.close(fd)
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ('continuous_dataset_manifest_v1.json','fims_continuous_arrival_summary_v1.txt','event_split.csv'):
                z.write(out/name,name)
            for block in range(20):
                n=f'block{block:02d}/complete.json';z.write(out/n,n)
            z.write(__file__,'source/export_continuous_hits_v1.py')
        with zipfile.ZipFile(tmp) as z:require(z.testzip() is None,'Review archive corrupt')
        os.replace(tmp,review)
    finally:Path(tmp).unlink(missing_ok=True)
    portable=out/'FIMS_Continuous_Arrivals_v1.tar.gz'
    fd,tmp=tempfile.mkstemp(prefix='.portable.',dir=out);os.close(fd)
    try:
        with tarfile.open(tmp,'w:gz') as t:
            for name in sorted(files):t.add(out/name,arcname=name,recursive=False)
            t.add(out/'continuous_dataset_manifest_v1.json',arcname='continuous_dataset_manifest_v1.json',recursive=False)
        os.replace(tmp,portable)
    finally:Path(tmp).unlink(missing_ok=True)
    atomic_bytes(portable.with_name(portable.name+'.sha256'),(sha(portable)+'  '+portable.name+'\n').encode())
    if copy_dir is not None:
        copy_dir.mkdir(parents=True,exist_ok=True)
        for p in (review,out/'fims_continuous_arrival_summary_v1.txt',portable,portable.with_name(portable.name+'.sha256')):
            require(p.resolve()!= (copy_dir/p.name).resolve(),'Copy destination equals output file')
            fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=copy_dir);os.close(fd)
            try:
                shutil.copyfile(p,tmp);require(sha(tmp)==sha(p),'Copied archive checksum mismatch')
                os.replace(tmp,copy_dir/p.name)
            finally:Path(tmp).unlink(missing_ok=True)
            print('Copied: '+str(copy_dir/p.name),flush=True)
    print(summary,flush=True)


def self_test():
    a=frozen_namespace(False);b=frozen_namespace(True)
    root={k:np.zeros(7) for k in a['BRANCHES']}
    root.update(EventID=np.array([0,0,0,0,2,2,2],np.int64),TrackID=np.array([1,2,3,4,1,2,3]),
                ParentID=np.array([0,1,2,1,0,1,2]),PDG=np.full(7,11),
                MomIn=np.array([100.,.2,.03,.01,100.,.3,.01]),HitPathLength_fromTrack=np.zeros(7),
                ParticleName=np.full(7,'e-'),CreatorProcess=np.full(7,'eIoni'))
    root['charge']=np.array([1000.,1000.,1000.,0.,1000.,1000.,1000.])
    root['prePos_z']=np.array([-50.,-50.,-50.,0.,10.,-20.,45.]);root['postPos_z']=root['prePos_z'].copy()
    d=50.-np.abs(root['postPos_z']);root['driftTime_notSmeared']=d/.01
    root['sigmaT_mm']=.3*np.sqrt(d/10);root['sigmaL_mm']=.2*np.sqrt(d/10)
    a['validate_root_arrays'](root,'synthetic');role,_=a['label_tracks'](root,3,'test')
    x,y,_=a['export_electrons'](root,role,3,'test',1729,10000,'block00')
    xx,yy,_=b['export_electrons'](root,role,3,'test',1729,10000,'block00')
    times=xx.pop('arrival_time_ns');compare_fields(xx,x,'synthetic original inputs');compare_fields(yy,y,'synthetic original labels')
    bx,by=a['encode'](x,y);bbx,bby=b['encode'](xx,yy)
    compare_fields(bbx,bx,'synthetic binary inputs');compare_fields(bby,by,'synthetic binary labels')
    equal_array(np.floor(times/25.).astype(np.int64),x['time_bin_abs'],'synthetic exact binning')
    require(np.any(times!=25*np.floor(times/25)),'Synthetic drift failed to exercise within-bin values')
    require(x['event_ptr'][1]==x['event_ptr'][2],'Empty event lost')
    require(np.any(yy['ambiguous_site_mask']),'Mixed-site supervision policy not exercised')
    xx['arrival_time_ns']=times;order,ties=arrival_order(xx)
    require(ties>0,'Exact ties not exercised')
    require(np.array_equal(np.sort(order),np.arange(len(times))),'Arrival order is not a permutation')
    for lo,hi in zip(x['event_ptr'][:-1],x['event_ptr'][1:]):
        require(np.all(np.diff(times[order[lo:hi]])>=0),'Arrival order is not monotonic')
    # Labels can change without changing physical observations or ordering.
    xxx,_,_=b['export_electrons'](root,np.zeros_like(role),3,'test',1729,10000,'block00')
    for k in xxx:equal_array(xxx[k],xx[k],'truth-independent inputs/'+k)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'roundtrip.npz';save_npz(p,xx);compare_fields(load_npz(p),xx,'file roundtrip')
    print('CONTINUOUS ARRIVAL SELF-TEST: PASS (unchanged RNG draws, native inputs/labels/site parity, exact binning, true sub-bin values, both endcaps, empty events, mixed labels, ties, truth independence, atomic NPZ roundtrip)',flush=True)


def audit_preserved(portable):
    """Real schema/mapping check available without original ROOT files."""
    b=Path(portable).resolve()
    if (b/'data/hit_records').is_dir():b=b/'data/hit_records'
    elif (b/'hit_records').is_dir():b=b/'hit_records'
    manifest,rows=load_plan(b);ns=frozen_namespace(False)
    total=0
    for i in range(20):
        block=f'block{i:02d}';arrays={}
        for suffix in FROZEN[:4]:
            n=f'{block}/readout/{PREFIX}{suffix}'
            arrays[suffix]=load_npz(safe_file(b,n,manifest['outputs'][n]))
        x,y=arrays[FROZEN[0]],arrays[FROZEN[1]];ns['validate_electrons'](x,y)
        bx,by=ns['encode'](x,y)
        compare_fields(bx,arrays[FROZEN[2]],block+'/preserved site inputs')
        compare_fields(by,arrays[FROZEN[3]],block+'/preserved site truth',
                       ('inputs_npz_sha256','source_electron_inputs_sha256','source_electron_targets_sha256','root_sha256'))
        total+=int(x['event_ptr'][-1])
    print(f'PRESERVED REAL SCHEMA/MAPPING AUDIT: PASS (20 shards, 10,000 events, {total:,} electrons; exact timestamps NOT available in these inputs)',flush=True)


def failure_review(out,source,copy_dir,exc):
    """Small diagnostic only; never place raw detector arrays into the review."""
    record=dict(schema=SCHEMA,status='INTERRUPTED_OR_FAILED',source_directory=str(source),
                exporter_sha256=sha(__file__),error_type=type(exc).__name__,error=str(exc),
                traceback=traceback.format_exc(),
                completed_shards=sorted(p.parent.name for p in out.glob('block*/complete.json')),
                no_training=True,no_new_simulation=True)
    write_json(out/'failure.json',record)
    p=out/'fims_continuous_arrival_failure_v1.zip'
    fd,tmp=tempfile.mkstemp(prefix='.failure.',dir=out);os.close(fd)
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            z.write(out/'failure.json','failure.json')
            z.write(out/'run_status.txt','run_status.txt')
            for m in sorted(out.glob('block*/complete.json')):z.write(m,str(m.relative_to(out)))
        os.replace(tmp,p)
    finally:Path(tmp).unlink(missing_ok=True)
    if copy_dir is not None:
        copy_dir.mkdir(parents=True,exist_ok=True)
        atomic_bytes(copy_dir/p.name,p.read_bytes())
        print('Failure review copied: '+str(copy_dir/p.name),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--portable-data',type=Path,help='Optional read-only schema audit of preserved portable inputs')
    p.add_argument('--copy-dir',type=Path,default=Path('/mnt/c/Users/Haider Abbas/Downloads/FIMS Research'))
    p.add_argument('--no-copy',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test()
    if args.portable_data:audit_preserved(args.portable_data)
    if args.source_dir is None:
        if args.self_test or args.portable_data:return
        p.error('Provide --source-dir and --output-dir, or --self-test')
    if args.output_dir is None:p.error('--output-dir is required for replay')
    source=args.source_dir.resolve();out=args.output_dir.resolve()
    require(source!=out and not out.is_relative_to(source),'Output must be separate from frozen source data')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Timestamp export is already active; no duplicate worker started.')
        atomic_bytes(out/'run_status.txt',b'RUNNING\n')
        try:
            import uproot
            manifest,rows=load_plan(source)
            identity=dict(schema=SCHEMA,exporter_sha256=sha(__file__),readout_sha256=READOUT_SHA,
                          source_data_sha256=DATA_SHA,split_sha256=SPLIT_SHA,
                          environment=dict(python=platform.python_version(),numpy=np.__version__,uproot=uproot.__version__))
            ns=frozen_namespace(True);print('Pinned development source and additive sampler retention: VERIFIED',flush=True)
            print('Replay environment: '+json.dumps(identity['environment']),flush=True)
            markers=[export_block(source,out,f'block{i:02d}',manifest,rows,ns,identity) for i in range(20)]
            archive_outputs(out,markers,source,None if args.no_copy else args.copy_dir)
            atomic_bytes(out/'run_status.txt',b'COMPLETE\n')
        except BaseException as exc:
            atomic_bytes(out/'run_status.txt',('INTERRUPTED_OR_FAILED\n'+str(exc)+'\n').encode())
            try:failure_review(out,source,None if args.no_copy else args.copy_dir,exc)
            except Exception as packaging_error:
                print('Could not package failure review: '+str(packaging_error),file=sys.stderr,flush=True)
            raise


if __name__=='__main__':main()
