#!/usr/bin/env python3
"""Resumable site GNN study. Checkpoint engine derived from train_capacity_v2.

Uses the original frozen event orders and loss normalization.

Each invocation owns one fit. Run both widths and all three fixed seeds on the
same backend/environment. This program never reads confirmation data, creates
geometry, chooses a retention threshold, or fits a readout transformation.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import tempfile
import time

# Required by deterministic CUDA GEMM; set before CUDA is initialized.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
import gnn_adapter_v1 as models
from site_data_v1 import SiteData

EPOCHS = 10
BATCH_EVENTS = 8
MICROBATCH = 64
SEEDS = (20260924, 20260925, 20260926)
STOP = False
PROVENANCE_PINS = {
    'training_complete_sha256':'ce055f030ddf67d7a11424cad7a57182945d0ea32259e8faf332d5377e45cb8e',
    'learning_protocol_sha256':'8b13eee3068274c08e7619b26ccc659734162e1c0202ceaafa2bf374ce58cce9',
    'training_settings_sha256':'ee103e58bcf368a9269dfbe69d03670513e5ab460b95f084adc36596b5712f5a',
    'epoch_orders_sha256':'46f4b63f4ee297c277d3fd60372fcb6f557ad81fafe8adf471af6b297b421097',
    'cache_manifest_sha256':'a09f1971835590e5f4ad444f234b888b2ac9da1d9053f69718f6fbc916fcf551',
    'data_manifest_sha256':'fd62d8c55c4a2455d95e4ef4152ad29405c15afacd99cceb6386091dec78424d',
    'event_split_sha256':'f1333977acf03370f8bdb3c850abf75b6ea6a344e2b25b3734d12bcbbb61a9e6',
}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()+b'\n'


def atomic_bytes(path, data, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), 'Unsafe output: '+str(path))
    if immutable and path.exists():
        require(path.read_bytes() == data, 'Existing immutable output differs: '+str(path))
        return
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def write_json(path, value, immutable=False):
    atomic_bytes(path, canonical(value), immutable)


def atomic_torch(path, value):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(value, f); f.flush(); os.fsync(f.fileno())
        torch.load(tmp, map_location='cpu', weights_only=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_npz(path, arrays):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
        with np.load(tmp, allow_pickle=False) as z:
            require(set(z.files) == set(arrays), 'Prediction inventory changed')
            for k, a in arrays.items():
                require(np.array_equal(z[k], a), 'Prediction roundtrip failed: '+k)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def signature(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def safe_file(base, name):
    rel = Path(name)
    p = base / rel
    require(not rel.is_absolute() and '..' not in rel.parts and not p.is_symlink()
            and p.resolve().is_relative_to(base) and p.is_file(), 'Unsafe/missing input: '+str(p))
    return p


def validate_event(a):
    require(set(a) == {'patches', 'target', 'primary_electron_weight'}, 'Wrong portable event fields')
    p, y, w = a['patches'], a['target'], a['primary_electron_weight']
    require(y.ndim == 1 and y.dtype == np.int8 and np.isin(y, [-1,0,1]).all(), 'Invalid native labels')
    require(w.shape == y.shape and w.dtype == np.int64 and np.all(w >= 0), 'Invalid primary charge weights')
    require(p.dtype == np.uint8 and p.shape == (len(y),8,11,11), 'Invalid cached site patches')
    require(not p.size or int(p.max()) <= 25, 'Cached patch count exceeds 25')


class PortableData:
    """Hash-checked immutable event files with bounded RAM caching."""
    def __init__(self, folder, *, synthetic=False, budget_bytes=1024**3):
        self.base = Path(folder).resolve()
        self.path = safe_file(self.base, 'portable_manifest_v2.json')
        self.manifest_sha = sha(self.path)
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        require(m.get('schema') == 'fims_cnn_portable_v2', 'Wrong portable dataset schema')
        require(m.get('synthetic', False) == synthetic, 'Synthetic/real dataset mode mismatch')
        self.events = {int(e['index']):e for e in m['events']}
        require(len(self.events) == len(m['events']) and set(self.events) == set(range(len(self.events))),
                'Portable event IDs must be unique contiguous original indices')
        self.validation = list(map(int, m['validation_indices']))
        self.settings, self.orders = m['training_settings'], m['epoch_orders']
        train = {i for i,e in self.events.items() if e['split'] == 'train'}
        val = {i for i,e in self.events.items() if e['split'] == 'validation'}
        require(train.isdisjoint(val) and train|val == set(self.events), 'Unknown or overlapping split')
        require(len(self.validation) == len(set(self.validation)) and set(self.validation) == val,
                'Validation order/support differs')
        if not synthetic:
            require(len(train) == 7500 and len(val) == 2500, 'Only frozen 7500/2500 development split allowed')
            require(self.settings['supervised_class_counts'] == [3427421,299738], 'Training truth support differs')
            require(self.settings['fixed_denominator'] == 4007.697721890993, 'Frozen loss denominator differs')
            provenance = m['original_provenance']
            require(provenance.get('schema') == 'fims_cnn_event_balance_inputs_v1' and provenance.get('status') == 'PASS',
                    'Missing verified native provenance')
            for key,value in PROVENANCE_PINS.items():
                require(provenance.get(key) == value, 'Wrong original development lineage: '+key)
            require(provenance.get('cnn_patches_loaded_from_verified_cache') is True,
                    'Export is not from the frozen site cache')
            expected_weights = np.asarray(3727159/(2*np.asarray([3427421,299738])),dtype=np.float32)
            require(np.array_equal(np.asarray(self.settings['class_weights'],np.float32),expected_weights),
                    'Original class-balanced training weights changed')
        eligible = list(map(int, self.settings['eligible_event_indices']))
        require(len(eligible) == len(set(eligible)) and set(eligible).issubset(train), 'Bad training eligibility')
        if not synthetic: require(len(eligible) == 7435, 'Frozen optimizer event eligibility changed')
        require(self.settings['optimizer_batches_per_epoch'] == math.ceil(len(eligible)/BATCH_EVENTS),
                'Batch denominator/support differs')
        for seed in SEEDS:
            order = self.orders[str(seed)]
            require(len(order) == EPOCHS and all(len(o) == len(eligible) and set(o) == set(eligible) for o in order),
                    'Frozen epoch order changed')
        self.signatures = {}
        start = last = time.monotonic()
        for ordinal,(name,digest) in enumerate(m['files'].items(),1):
            p = safe_file(self.base, name)
            s = signature(p)
            require(sha(p) == digest and signature(p) == s, 'Input checksum changed: '+name)
            self.signatures[name] = s
            if time.monotonic()-last > 30:
                print(f'Portable input hashes: {ordinal}/{len(m["files"])} verified', flush=True)
                last = time.monotonic()
        for e in self.events.values():
            require(e['path'] in m['files'], 'Event missing from immutable inventory')
        require(len({e['path'] for e in self.events.values()}) == len(self.events),
                'Two native events point to the same exported event file')
        if not synthetic:
            for filename,pin,key in [('training_settings_7500.json','training_settings_sha256','settings'),
                                     ('epoch_orders_v1.json','epoch_orders_sha256','orders')]:
                names = [n for n in m['files'] if Path(n).name == filename]
                require(len(names) == 1 and m['files'][names[0]] == PROVENANCE_PINS[pin],
                        'Missing original settings/order bytes: '+filename)
                native = json.loads(safe_file(self.base,names[0]).read_text())
                require(native == self.settings if key == 'settings' else native['7500'] == self.orders,
                        'Embedded settings/orders disagree with frozen original bytes')
        self.values, self.bytes, self.budget = OrderedDict(), 0, int(budget_bytes)
        require(self.budget > 0, 'Cache budget must be positive')
        print(f'PORTABLE INPUT HASH / FROZEN SPLIT / ORDERS: PASS ({time.monotonic()-start:.1f} s)', flush=True)

    def get(self, index):
        index = int(index)
        e = self.events[index]
        p = safe_file(self.base, e['path'])
        require(signature(p) == self.signatures[e['path']], 'Event changed during fit')
        if index in self.values:
            self.values.move_to_end(index)
            return self.values[index][0]
        with np.load(p, allow_pickle=False) as z:
            a = {k:z[k] for k in z.files}
        validate_event(a)
        for v in a.values(): v.setflags(write=False)
        size = sum(v.nbytes for v in a.values())
        while self.values and self.bytes+size > self.budget:
            _,(_,removed) = self.values.popitem(last=False)
            self.bytes -= removed
        if size <= self.budget:
            self.values[index] = a,size
            self.bytes += size
        return a

    def verify_unchanged(self):
        require(sha(self.path) == self.manifest_sha, 'Portable manifest changed during training')
        for name,s in self.signatures.items():
            require(signature(safe_file(self.base,name)) == s, 'Input changed during training: '+name)


def clone(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def capture_rng(device):
    n = np.random.get_state()
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
        numpy=dict(kind=n[0], keys=n[1].tolist(), pos=n[2], has_gauss=n[3], cached_gaussian=n[4]),
        cuda=[s.cpu() for s in torch.cuda.get_rng_state_all()] if device.type == 'cuda' else [])


def restore_rng(state, device):
    torch.set_rng_state(state['torch'].cpu())
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['kind'],np.asarray(n['keys'],np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
    if device.type == 'cuda':
        require(len(state['cuda']) == torch.cuda.device_count(), 'CUDA RNG device count changed')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    else:
        require(not state['cuda'], 'CUDA checkpoint cannot be resumed silently on CPU')


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=.001, betas=(.9,.999), eps=1e-8,
                            weight_decay=0, amsgrad=False, foreach=False, fused=False)


def graph_chunks(a,variant,indices=None,max_queries=64,max_edges=250000):
    from graph_models_v1 import QueryGraphBuilder
    from site_data_v1 import geometry_only
    builder=QueryGraphBuilder(geometry_only(a),variant,max_edges=2000000)
    pending=[];ids=[];edges=0
    if indices is None:indices=range(len(a['target']))
    for q in indices:
        g=builder.build(int(q));n=g.edge_index.shape[1]
        if pending and (len(pending)>=max_queries or edges+n>max_edges):
            yield ids,pending;pending=[];ids=[];edges=0
        pending.append(g);ids.append(int(q));edges+=n
    if pending:yield ids,pending



def training_batch(model,optimizer,arrays,weights,denominator,device,microbatch=64):
    from graph_models_v1 import batch_graphs
    require(denominator>0 and arrays,'Invalid optimizer batch')
    optimizer.zero_grad(set_to_none=True);numerator=actual=0.
    for a in arrays:
        eligible=np.flatnonzero(a['target']>=0)
        for ids,graphs in graph_chunks(a,model.variant,eligible,max_queries=microbatch):
            y=torch.tensor(a['target'][ids],dtype=torch.float32,device=device)
            logits=model(batch_graphs(graphs,device))
            loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights[y.long()]).sum()
            require(bool(torch.isfinite(loss)),'Nonfinite graph loss')
            (loss/denominator).backward();numerator+=float(loss.detach().cpu());actual+=float(weights[y.long()].sum().detach().cpu())
    require(actual>0,'No supervised queries in optimizer batch')
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True).detach().cpu())
    optimizer.step();require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Nonfinite graph update')
    return dict(numerator=numerator,actual_denominator=actual,gradient_norm=norm)



def class_balanced_bce(logits, target):
    x, y = np.asarray(logits,dtype=np.float64), np.asarray(target)
    losses = np.logaddexp(0,x)-x*np.maximum(y,0)
    require(np.isfinite(losses).all(), 'Nonfinite validation loss')
    if not np.any(y == 0) or not np.any(y == 1): return None
    return float(.5*(losses[y == 0].mean()+losses[y == 1].mean()))


class Paused(Exception): pass


def check_stop():
    if STOP: raise Paused('Signal received; resume from saved optimizer state')


def evaluate(model,data,device,label='validation',stop_callback=check_stop):
    from graph_models_v1 import batch_graphs
    from site_data_v1 import prediction_bundle
    model.eval();blocks=[];targets=[];weights=[];lengths=[];start=last=time.monotonic()
    with torch.no_grad():
        for ordinal,index in enumerate(data.validation,1):
            stop_callback();a=data.get(index);scores=np.empty(len(a['target']),np.float32)
            for ids,graphs in graph_chunks(a,model.variant):
                stop_callback();scores[ids]=model(batch_graphs(graphs,device)).detach().cpu().numpy()
            blocks.append(scores);targets.append(a['target']);weights.append(a['primary_electron_weight']);lengths.append(len(scores))
            if time.monotonic()-last>=30:
                print(f'{label}: {ordinal}/{len(data.validation)} events; {ordinal/max(time.monotonic()-start,1e-9):.2f} events/s',flush=True);last=time.monotonic()
    out=prediction_bundle(data,blocks,targets,weights,lengths);require(np.isfinite(out['logits']).all(),'Nonfinite logits')
    return out,time.monotonic()-start



def environment(device):
    result = dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),
        backend=device.type,cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        threads=torch.get_num_threads(),deterministic=True,amp=False,tf32=False,
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    if device.type == 'cuda':
        result.update(gpu_name=torch.cuda.get_device_name(device),
            gpu_capability=list(torch.cuda.get_device_capability(device)),
            visible_device_count=torch.cuda.device_count())
    return result


def make_identity(data,width,seed,device):
    import torch_geometric,scipy
    here=Path(__file__).resolve().parent
    names=['train_site_gnn_v1.py','graph_models_v1.py','gnn_adapter_v1.py','site_data_v1.py','vendor/capacity_model_v2.py']
    protocol=dict(schema='fims_site_gnn_protocol_v1',portable_manifest_sha256=data.manifest['portable_manifest_sha256'],
        site_manifest_sha256=data.manifest_sha,code={n:sha(here/n) for n in names},environment=dict(environment(device),pyg=torch_geometric.__version__,scipy=scipy.__version__),
        epochs=EPOCHS,batch_events=BATCH_EVENTS,query_microbatch_max=64,query_microbatch_edge_budget=250000,
        individual_graph_edge_safety_limit=2000000,on_graph_limit='FAIL entire fit; never truncate or omit a query',
        optimizer=dict(name='Adam',lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=0,gradient_clip_norm=1.),
        loss='original class-balanced BCE; fixed training denominator; ignored query targets masked; all sites supply context',
        checkpoint_selection='earliest strict minimum ordinary validation class-balanced BCE',
        input='query-induced directed same-endcap 2 mm site graph; bins t-7..t (current2 uses t only); relative edge coordinates; occupancy=1',
        variants=dict(full2='2 layers, 64 channels, relative x/y/time edge offsets',full1='1 layer, 64 channels',no_xy2='2 layers; spatial edge attributes zero; geometric edges retained',current2='2 layers; current bin only'),
        comparison='fixed site-data pipeline comparison; existing CNN reference uses raster patches',
        timing='prediction after current 25 ns bin closes; known event boundaries',thresholds='development diagnostic curves at 99.95/99.9/99.5/99.0%; no evaluation tuning or deployment choice')
    h=hashlib.sha256(canonical(protocol)).hexdigest()
    return protocol,dict(protocol_sha256=h,model=width,seed=seed,parameters=models.PARAMETERS[width],train_events=sum(e['split']=='train' for e in data.events.values()))



def verify_complete(out, identity):
    marker = out/'complete.json'
    if not marker.exists(): return False
    m = json.loads(marker.read_text())
    require(m.get('status') == 'COMPLETE' and m.get('identity') == identity,'Completed fit identity differs')
    required = {'metrics.json','history.json','protocol.json','selected_checkpoint.pt','final_checkpoint.pt',
                'selected_predictions.npz','final_predictions.npz'}
    require(required.issubset(m['outputs']), 'Completed fit inventory missing')
    for n,h in m['outputs'].items():
        require(Path(n).name == n and sha(safe_file(out,n)) == h,'Completed fit output changed: '+n)
    return True


def fit(data, output, width, seed, device):
    out = Path(output).resolve()/width/f'seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'fit.lock').open('a')
    try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('This exact fit is already active; no duplicate worker started')
    model = models.create_model(width,seed,device)
    protocol,identity = make_identity(data,width,seed,device)
    if verify_complete(out,identity):
        print(f'{width} seed={seed}: COMPLETE outputs verified; skipped',flush=True)
        return
    write_json(out/'protocol.json',protocol,immutable=True)
    initial_hash = models.parameter_hash(model)
    optimizer = make_optimizer(model)
    weights = torch.tensor(data.settings['class_weights'],dtype=torch.float32,device=device)
    denominator = float(data.settings['fixed_denominator'])
    resume = out/'resume_checkpoint.pt'
    if resume.exists():
        state = torch.load(resume,map_location='cpu',weights_only=True)
        require(state['identity'] == identity and state['initialization_sha256'] == initial_hash,
                'Resume environment/model/data/initialization changed; use the original runtime')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'],device)
        require(0 <= state['epoch'] <= EPOCHS and
                0 <= state['next_batch'] <= data.settings['optimizer_batches_per_epoch'], 'Invalid saved progress')
        print(f'Resuming {width} seed={seed}: epoch index {state["epoch"]}, batch index {state["next_batch"]}',flush=True)
    else:
        state = dict(identity=identity,initialization_sha256=initial_hash,epoch=0,next_batch=0,
                     history=[],stats=[],best_bce=None,best_epoch=None,best_model=None,fit_seconds=0.)
    saved_time = time.monotonic()

    def save(status='RUNNING'):
        nonlocal saved_time
        state['fit_seconds'] += time.monotonic()-saved_time
        state.update(model=clone(model),optimizer=optimizer.state_dict(),rng=capture_rng(device))
        atomic_torch(resume,state)
        write_json(out/'status.json',dict(status=status,**identity,epoch=state['epoch'],next_batch=state['next_batch']))
        saved_time = time.monotonic()

    if not resume.exists(): save()
    try:
        for epoch in range(state['epoch'],EPOCHS):
            model.train()
            order = data.orders[str(seed)][epoch]
            batches = data.settings['optimizer_batches_per_epoch']
            start,count = time.monotonic(),0
            for b in range(state['next_batch'],batches):
                check_stop()
                arrays = [data.get(i) for i in order[b*BATCH_EVENTS:(b+1)*BATCH_EVENTS]]
                stats = training_batch(model,optimizer,arrays,weights,denominator,device)
                state['stats'].append(stats); state['next_batch'] = b+1; count += 1
                if True:  # Graph batches are slower: persist every completed optimizer step.
                    save()
                    print(f'{width} seed={seed} epoch={epoch+1}/{EPOCHS}: batch {b+1}/{batches} saved; '
                          f'{count/max(time.monotonic()-start,1e-9):.2f} batches/s',flush=True)
            prediction,seconds = evaluate(model,data,device,label=f'{width} epoch {epoch+1} validation')
            bce = class_balanced_bce(prediction['logits'],prediction['target'])
            require(bce is not None and math.isfinite(bce),'Undefined validation BCE')
            y = prediction['target']; positive = prediction['logits'] >= 0
            recall = float(positive[y == 1].mean()); retention = float((~positive[y == 0]).mean())
            stats = state['stats']
            row = dict(epoch=epoch+1,validation_class_balanced_bce=bce,
                validation_balanced_accuracy=.5*(recall+retention),validation_microcurler_recall=recall,
                validation_other_retention=retention,validation_seconds=seconds,
                online_weighted_bce=sum(s['numerator'] for s in stats)/sum(s['actual_denominator'] for s in stats),
                clipped_batch_fraction=float(np.mean([s['gradient_norm']>1 for s in stats])))
            state['history'].append(row)
            if state['best_bce'] is None or bce < state['best_bce']:
                state.update(best_bce=bce,best_epoch=epoch+1,best_model=clone(model))
            state.update(epoch=epoch+1,next_batch=0,stats=[])
            save(); write_json(out/'history.json',state['history'])
            print(f'{width} seed={seed} epoch={epoch+1}: BCE={bce:.6f}; '
                  f'microcurler recall={recall:.6f}; other retention={retention:.6f}',flush=True)
        require(state['best_epoch'] == min(state['history'],key=lambda r:r['validation_class_balanced_bce'])['epoch'],
                'Selection is not earliest minimum BCE')
        final_state = clone(model)
        for name,weights_state,epoch in [('final',final_state,EPOCHS),('selected',state['best_model'],state['best_epoch'])]:
            check_stop()
            model.load_state_dict(weights_state,strict=True)
            prediction,_ = evaluate(model,data,device,label=name+' validation')
            prediction['protocol_sha256'] = np.asarray(identity['protocol_sha256'])
            atomic_npz(out/(name+'_predictions.npz'),prediction)
            atomic_torch(out/(name+'_checkpoint.pt'),dict(identity=identity,epoch=epoch,state_dict=weights_state,
                initialization_sha256=initial_hash,model_contract={'representation':'query-induced site graph','variant':width,'hidden_channels':64,'parameters':models.PARAMETERS[width]}))
        data.verify_unchanged()
        write_json(out/'metrics.json',dict(identity=identity,protocol_sha256=identity['protocol_sha256'],model=width,
            seed=seed,parameters=models.PARAMETERS[width],selected_epoch=state['best_epoch'],selected_bce=state['best_bce'],
            initialization_sha256=initial_hash,fit_seconds=state['fit_seconds'],training_support=data.settings,
            validation_events=len(data.validation),interpretation='development validation only; no independent confirmation'))
        outputs = {n:sha(out/n) for n in ['metrics.json','history.json','protocol.json','selected_checkpoint.pt',
                                        'final_checkpoint.pt','selected_predictions.npz','final_predictions.npz']}
        write_json(out/'complete.json',dict(status='COMPLETE',identity=identity,outputs=outputs))
        require(verify_complete(out,identity),'Completion verification failed')
        write_json(out/'status.json',dict(status='COMPLETE',**identity))
        print(f'FIMS SITE GNN FIT COMPLETE: {width} seed={seed}',flush=True)
    except Paused:
        # If interrupted while writing final/selected outputs, restore the final
        # training state before the resume save; otherwise selected weights could
        # accidentally become the optimizer's final state on the next resume.
        if state['epoch'] == EPOCHS:
            model.load_state_dict(state['model'],strict=True)
        save(status='PAUSED')
        print('PAUSED: checkpoint saved. Rerun this exact fit to continue.',flush=True)
        raise
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--model',choices=list(models.VARIANTS))
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--self-test',action='store_true')
    args = p.parse_args()
    if args.self_test:
        import test_site_pipeline_v1
        test_site_pipeline_v1.run_tests()
        return
    require(all(v is not None for v in (args.data,args.output_dir,args.model,args.seed)),
            '--data, --output-dir, --model, --seed required')
    device = torch.device(args.device)
    require(device.type != 'cuda' or torch.cuda.is_available(),
            'CUDA GPU unavailable. Run inside a GPU allocation; CPU fallback is not automatic.')
    def request_stop(signum,frame):
        global STOP
        STOP = True
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]: signal.signal(sig,request_stop)
    data = SiteData(args.data)
    try: fit(data,args.output_dir,args.model,args.seed,device)
    except Paused: raise SystemExit(75)


if __name__ == '__main__': main()
