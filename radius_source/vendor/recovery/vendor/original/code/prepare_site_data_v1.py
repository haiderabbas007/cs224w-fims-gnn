#!/usr/bin/env python3
"""Export audited binary-site geometry, preserving original event and site order."""
from pathlib import Path
import argparse,fcntl,json,time
import numpy as np
from site_data_v1 import *

def prepare(capacity_root,output_dir):
    base=Path(capacity_root).resolve()/'data';out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'prepare.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    mp=safe_file(base,'portable_manifest_v2.json');require(sha(mp)==PORTABLE_SHA,'Wrong portable data identity')
    m=json.loads(mp.read_text());pres=base/'hit_records';rp=safe_file(pres,'PRESERVED_RECORDS.json')
    require(sha(rp)==PRESERVED_SHA,'Wrong preserved readout identity');records=json.loads(rp.read_text())
    def verified(rel):
        p=safe_file(pres,rel);require(rel in records['files'] and sha(p)==records['files'][rel],'Unverified native input '+rel);return p
    by_shard={}
    for e in m['events']:by_shard.setdefault(e['shard'],[]).append(e)
    files={};events=[];counts=np.zeros(2,np.int64);total=0
    for shard,items in sorted(by_shard.items()):
        prefix=shard+'/readout/fims_common_source_'
        ip=verified(prefix+'binary_spike_inputs_v1.npz');tp=verified(prefix+'binary_spike_targets_v1.npz')
        with np.load(ip,allow_pickle=False) as z:x={k:z[k] for k in z.files}
        with np.load(tp,allow_pickle=False) as z:y={k:z[k] for k in z.files}
        require(str(y['inputs_npz_sha256'])==sha(ip),'Native truth/input link mismatch')
        require(np.array_equal(x['event_ptr'],y['event_ptr']) and float(x['clock_ns'])==25.,'Native timing/pointers differ')
        pitch=float(x['pixel_pitch_mm']);require(pitch>0,'Invalid pitch')
        for e in items:
            i=int(e['index']);raw=int(e['raw_id']);slots=np.flatnonzero(y['event_id_raw']==raw)
            require(len(slots)==1,'Ambiguous raw event identity');slot=int(slots[0])
            require(str(y['event_run_uid'][slot])==e['run_uid'],'Run UID mismatch')
            lo,hi=map(int,x['event_ptr'][slot:slot+2]);s=slice(lo,hi)
            a=dict(x_mm=(x['global_pixel_x'][s].astype(np.float64)+.5)*pitch,
                y_mm=(x['global_pixel_y'][s].astype(np.float64)+.5)*pitch,time_bin=x['time_bin_abs'][s].astype(np.int64),
                side=x['readout_side'][s].astype(np.uint8),target=y['target_core'][s].astype(np.int8),
                primary_electron_weight=y['n_primary_other_electrons'][s].astype(np.int64),
                microcurler_electron_weight=y['n_core_electrons'][s].astype(np.int64))
            validate_event(a)
            pp=safe_file(base,e['path']);require(sha(pp)==m['files'][e['path']],'Changed cached portable event')
            with np.load(pp,allow_pickle=False) as z:
                require(np.array_equal(z['target'],a['target']) and np.array_equal(z['primary_electron_weight'],a['primary_electron_weight']),'Native/portable site order differs')
            rel=f'events/event_{i:05d}.npz';dest=out/rel
            if dest.exists():
                require(not dest.is_symlink(),'Unsafe event destination')
                with np.load(dest,allow_pickle=False) as z:
                    require(set(z.files)==set(a) and all(np.array_equal(z[k],a[k]) for k in a),'Cached geometry differs')
            else:atomic_npz(dest,a)
            files[rel]=sha(dest);events.append(dict(e,path=rel));total+=len(a['target'])
            if e['split']=='train':counts+=np.bincount(a['target'][a['target']>=0],minlength=2)
        print(f'Site geometry: {shard} verified; {len(events)}/10000 events; {total:,} sites',flush=True)
    require(len(events)==10000 and counts.tolist()==m['training_settings']['supervised_class_counts'],'Incomplete geometry/train support')
    result=dict(schema='fims_site_geometry_v1',synthetic=False,portable_manifest_sha256=PORTABLE_SHA,
        preserved_manifest_sha256=PRESERVED_SHA,events=sorted(events,key=lambda e:e['index']),files=files,
        validation_indices=m['validation_indices'],training_settings=m['training_settings'],epoch_orders=m['epoch_orders'],
        context='25 ns binary sites; geometry-only encoder allowlist; truth/electron weights sidecars only',
        timing='Current bin may be used only after it closes. No future bin.',total_sites=total)
    write_json(out/'site_manifest_v1.json',result,immutable=True)
    SiteData(out).verify_unchanged();print('FIMS SITE GEOMETRY V1: COMPLETE',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--capacity-root');p.add_argument('--output-dir');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:
        from test_site_pipeline_v1 import test_geometry_contract
        test_geometry_contract();print('SITE GEOMETRY SELF-TEST: PASS');return
    require(a.capacity_root and a.output_dir,'--capacity-root and --output-dir required');prepare(a.capacity_root,a.output_dir)
if __name__=='__main__':main()
