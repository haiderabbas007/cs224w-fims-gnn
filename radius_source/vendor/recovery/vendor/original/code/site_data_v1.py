"""Immutable, event-separated geometry and evaluation sidecars for FIMS sites."""
from __future__ import annotations
import hashlib, json, os, tempfile
from collections import OrderedDict
from pathlib import Path
import numpy as np

PORTABLE_SHA = '4cbf351c982f4fd683b8e3cc87a1fc40bd15ac45d7f9345fc6f26b7e45636f06'
PRESERVED_SHA = '68f9bf045e3db4bb3713296041053be26d9ea07502f9bd5af7afb15414766163'
METADATA_PINS = {
    'training_settings':'ee103e58bcf368a9269dfbe69d03670513e5ab460b95f084adc36596b5712f5a',
    'epoch_orders':'f0a2c8aabb5197659ccf30e2838c6f72cfe715f69fedab91b7e33ee99022b36b',
    'validation_indices':'d29be28e86ea1a144635af1920f16e00362948b64fc0107cc381f13002170c1d',
}
GEOMETRY_KEYS = ('x_mm','y_mm','time_bin','side')
ARRAY_KEYS = set(GEOMETRY_KEYS) | {'target','primary_electron_weight','microcurler_electron_weight'}

def require(ok, message):
    if not bool(ok): raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def canonical(v):return json.dumps(v,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'

def atomic_bytes(path,raw,immutable=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    require(not p.is_symlink(),'Unsafe output '+str(p))
    if immutable and p.exists():
        require(p.read_bytes()==raw,'Immutable output differs: '+str(p));return
    fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        Path(tmp).unlink(missing_ok=True)

def write_json(path,v,immutable=False):atomic_bytes(path,canonical(v),immutable)

def atomic_npz(path,arrays):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
    try:
        with os.fdopen(fd,'wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        with np.load(tmp,allow_pickle=False) as z:
            require(set(z.files)==set(arrays),'NPZ inventory mismatch')
            for k in arrays:require(np.array_equal(z[k],arrays[k]),'NPZ mismatch '+k)
        os.replace(tmp,p)
    finally:Path(tmp).unlink(missing_ok=True)

def safe_file(base,name):
    base=Path(base).resolve();r=Path(name);p=base/r
    require(not r.is_absolute() and '..' not in r.parts and not p.is_symlink()
            and p.resolve().is_relative_to(base) and p.is_file(),'Unsafe/missing file '+str(p))
    return p

def signature(p):
    s=Path(p).stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns

def validate_event(a):
    require(set(a)==ARRAY_KEYS,'Wrong geometry/sidecar inventory')
    n=len(a['target'])
    require(all(x.ndim==1 and len(x)==n for x in a.values()),'Unaligned site arrays')
    require(a['target'].dtype==np.int8 and np.isin(a['target'],[-1,0,1]).all(),'Invalid labels')
    require(a['time_bin'].dtype==np.int64 and np.isin(a['side'],[0,1]).all(),'Invalid time/endcap')
    require(np.isfinite(a['x_mm']).all() and np.isfinite(a['y_mm']).all(),'Nonfinite geometry')
    for k in ['primary_electron_weight','microcurler_electron_weight']:
        require(a[k].dtype==np.int64 and (a[k]>=0).all(),'Invalid charge sidecar')
    if n:
        addresses=np.rec.fromarrays([a[k] for k in ('side','time_bin','x_mm','y_mm')])
        require(len(np.unique(addresses))==n,'Duplicate binary sites')

class SiteData:
    def __init__(self,folder,*,synthetic=False,cache_events=12):
        self.base=Path(folder).resolve();self.path=safe_file(self.base,'site_manifest_v1.json')
        self.manifest_sha=sha(self.path);self.manifest=json.loads(self.path.read_text());m=self.manifest
        require(m['schema']=='fims_site_geometry_v1','Wrong site geometry schema')
        require(bool(m.get('synthetic',False))==synthetic,'Synthetic/real mode mismatch')
        if not synthetic:
            require(m['portable_manifest_sha256']==PORTABLE_SHA and m['preserved_manifest_sha256']==PRESERVED_SHA,'Wrong native development lineage')
            for key,pin in METADATA_PINS.items():
                require(hashlib.sha256(canonical(m[key])).hexdigest()==pin,'Frozen metadata changed: '+key)
        self.events={int(e['index']):e for e in m['events']}
        require(len(self.events)==len(m['events']),'Duplicate event indices')
        train={i for i,e in self.events.items() if e['split']=='train'}
        val={i for i,e in self.events.items() if e['split']=='validation'}
        require(train.isdisjoint(val) and train|val==set(self.events),'Invalid event split')
        self.validation=list(map(int,m['validation_indices']))
        require(len(self.validation)==len(set(self.validation)) and set(self.validation)==val,'Validation order mismatch')
        self.settings=m['training_settings'];self.orders=m['epoch_orders']
        eligible=self.settings['eligible_event_indices']
        require(set(eligible).issubset(train) and len(set(eligible))==len(eligible),'Invalid train eligibility')
        for seed,orders in self.orders.items():
            require(len(orders)==10 and all(len(o)==len(eligible) and set(o)==set(eligible) for o in orders),'Training orders changed')
        if not synthetic:
            require(len(train)==7500 and len(val)==2500,'Expected frozen 7500/2500 split')
            require(self.settings['supervised_class_counts']==[3427421,299738],'Training label support changed')
        self.signatures={}
        for name,digest in m['files'].items():
            p=safe_file(self.base,name);before=signature(p)
            require(sha(p)==digest and signature(p)==before,'Changed geometry file '+name)
            self.signatures[name]=before
        require(all(e['path'] in self.signatures for e in self.events.values()),'Unpinned geometry path')
        require(len({e['path'] for e in self.events.values()})==len(self.events),'Shared event file')
        self.cache=OrderedDict();self.cache_events=cache_events

    def get(self,index):
        e=self.events[int(index)];p=safe_file(self.base,e['path'])
        require(signature(p)==self.signatures[e['path']],'Geometry changed during use')
        if index in self.cache:self.cache.move_to_end(index);return self.cache[index]
        with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        validate_event(a)
        for v in a.values():v.setflags(write=False)
        self.cache[index]=a
        while len(self.cache)>self.cache_events:self.cache.popitem(last=False)
        return a

    def verify_unchanged(self):
        require(sha(self.path)==self.manifest_sha,'Geometry manifest changed')
        for n,s in self.signatures.items():require(signature(safe_file(self.base,n))==s,'Changed input '+n)

def geometry_only(a):return {k:a[k] for k in GEOMETRY_KEYS}

def prediction_bundle(data,logits,targets,weights,lengths,core_weights=None):
    records=[data.events[i] for i in data.validation]
    out=dict(logits=np.concatenate(logits).astype(np.float32),target=np.concatenate(targets).astype(np.int8),
        primary_electron_weight=np.concatenate(weights).astype(np.int64),event_ptr=np.r_[0,np.cumsum(lengths,dtype=np.int64)],
        event_index=np.asarray(data.validation,np.int64),event_id_raw=np.asarray([e['raw_id'] for e in records],np.int64),
        event_run_uid=np.asarray([e['run_uid'] for e in records]),event_shard=np.asarray([e['shard'] for e in records]))
    if core_weights is not None:out['microcurler_electron_weight']=np.concatenate(core_weights).astype(np.int64)
    return out
