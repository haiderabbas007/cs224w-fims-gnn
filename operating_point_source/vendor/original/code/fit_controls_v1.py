#!/usr/bin/env python3
"""Deterministic count/shape controls for the fixed FIMS site development set.

No GPU, PyG, simulation, holdout scoring, or threshold fitting. --output-dir is
STUDY/results/fits. Interrupted feature extraction reuses verified event caches;
an unfinished L-BFGS fit restarts deterministically. Completed fits are verified
and reused. The seed in paths is an identity tag, not an optimization replicate.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import time

# A fixed single-threaded numerical reduction avoids BLAS thread-count drift.
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import expit
from site_data_v1 import (SiteData, GEOMETRY_KEYS, geometry_only, prediction_bundle,
    require, sha, canonical, atomic_npz, write_json, safe_file)

SEED = 20260924
VARIANTS = ('counts8', 'shape8', 'counts1', 'shape1')
RADII = (.5, 1., 2.)
L2 = .01
FEATURE_NAMES = ('log1p_count_r0p5', 'log1p_count_r1', 'log1p_count_r2',
                 'log1p_bins_r0p5', 'log1p_bins_r1', 'log1p_bins_r2',
                 'mean_radius_over_2mm', 'population_std_radius_over_2mm',
                 'centered_xy_covariance_anisotropy')


def feature_spec():
    return dict(radius_mm=list(RADII), history_bins=[8, 1], same_endcap=True,
        query_excluded=True, all_truth_roles_supply_context=True,
        feature_names=list(FEATURE_NAMES),
        shape_definition='mean/std of distance to query; centered neighbor xy covariance; (lambda_max-lambda_min)/trace',
        insufficient_support='empty: all shape zero; one neighbor: mean available, std and anisotropy zero; zero covariance trace: anisotropy zero',
        timing='t-7..t inclusive, or t only; predictions after current 25 ns bin closes')


def extract_features(geometry):
    """Return (eight-bin, current-bin) float64 [N,9], using geometry only."""
    require(set(geometry) == set(GEOMETRY_KEYS), 'Encoder accepts geometry only')
    x, y, bins, side = [np.asarray(geometry[k]) for k in GEOMETRY_KEYS]
    n = len(x)
    require(all(a.ndim == 1 and len(a) == n for a in (x,y,bins,side)), 'Geometry alignment')
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite geometry')
    require(bins.dtype.kind in 'iu' and np.isin(side,[0,1]).all(), 'Invalid bins/endcaps')
    # Signed arithmetic avoids unsigned underflow in causal windows.
    require(not n or (np.min(bins) >= np.iinfo(np.int64).min+7 and np.max(bins) <= np.iinfo(np.int64).max), 'Unsafe bin range')
    bins = bins.astype(np.int64, copy=False)
    xy = np.column_stack((x,y)).astype(np.float64, copy=False)
    out8, out1 = np.zeros((n,9),np.float64), np.zeros((n,9),np.float64)
    for cap in (0,1):
        ids = np.flatnonzero(side == cap)
        if not len(ids): continue
        points = xy[ids]; tree = cKDTree(points)
        for start in range(0,len(ids),256):
            neighborhoods = tree.query_ball_point(points[start:start+256], 2., return_sorted=True)
            for local_q, candidates in enumerate(neighborhoods,start):
                q = int(ids[local_q]); js = ids[np.asarray(candidates,np.int64)]
                js = js[(js != q) & (bins[js] <= bins[q]) & (bins[js] >= bins[q]-7)]
                delta = xy[js]-xy[q]
                dist2 = np.einsum('ij,ij->i',delta,delta)
                # Explicit inclusive geometric boundary, independent of tree result ordering.
                keep = dist2 <= 4.
                js,delta,dist2 = js[keep],delta[keep],dist2[keep]
                for out,window in ((out8,8),(out1,1)):
                    use = np.ones(len(js),bool) if window == 8 else bins[js] == bins[q]
                    neighbors = js[use]; dxy = delta[use]; d2 = dist2[use]
                    row = out[q]
                    for k,r in enumerate(RADII):
                        inside = d2 <= r*r
                        row[k] = np.log1p(np.count_nonzero(inside))
                        row[k+3] = np.log1p(len(np.unique(bins[neighbors[inside]])))
                    if len(neighbors):
                        d = np.sqrt(d2)
                        row[6],row[7] = d.mean()/2.,d.std(ddof=0)/2.
                    if len(neighbors) >= 2:
                        centered = dxy-dxy.mean(axis=0)
                        c = centered.T@centered/len(neighbors)
                        trace = float(c[0,0]+c[1,1])
                        if trace > 0:
                            row[8] = np.clip(np.hypot(c[0,0]-c[1,1],2*c[0,1])/trace,0.,1.)
    require(np.isfinite(out8).all() and np.isfinite(out1).all(), 'Nonfinite features')
    return out8,out1


def fitted_scaler(features, targets):
    features=np.asarray(features,np.float64);targets=np.asarray(targets)
    require(features.ndim==2 and len(features)==len(targets),'Scaler alignment')
    selected=features[targets>=0]
    require(len(selected)>0 and np.isfinite(selected).all(),'No finite supervised training features')
    mean=selected.mean(axis=0);scale=selected.std(axis=0,ddof=0)
    scale=np.where(scale>0,scale,1.)
    return mean,scale


def bce_gradient(theta, features, targets, l2=L2):
    """0.5*mean(BCE|other)+0.5*mean(BCE|microcurler)+0.5*L2*||w||²."""
    x=np.asarray(features,np.float64);y=np.asarray(targets)
    require(x.ndim==2 and len(x)==len(y) and len(theta)==x.shape[1]+1,'Loss alignment')
    valid=y>=0;x=x[valid];y=y[valid].astype(np.float64)
    require(np.isin(y,[0,1]).all() and np.isfinite(x).all(),'Invalid supervised training arrays')
    counts=np.bincount(y.astype(np.int64),minlength=2)
    require((counts>0).all(),'Both training classes required')
    weights=.5/counts[y.astype(np.int64)]
    z=x@theta[:-1]+theta[-1]
    loss=float(np.sum(weights*(np.logaddexp(0.,z)-y*z))+.5*l2*np.dot(theta[:-1],theta[:-1]))
    error=weights*(expit(z)-y)
    grad=np.r_[x.T@error+l2*theta[:-1],np.sum(error)]
    require(np.isfinite(loss) and np.isfinite(grad).all(),'Nonfinite objective/gradient')
    return loss,grad


def fit_logistic(x,y,l2=L2):
    # Mask once to keep each optimizer iteration compact. Ignored rows affect no scaler or loss.
    mask=y>=0;x=np.asarray(x[mask],np.float64);y=np.asarray(y[mask],np.int8)
    result=minimize(bce_gradient,np.zeros(x.shape[1]+1,np.float64),args=(x,y,l2),
        method='L-BFGS-B',jac=True,options=dict(maxiter=500,ftol=1e-12,gtol=1e-8,maxls=40))
    require(result.success,'L-BFGS did not converge: '+str(result.message))
    return result.x,dict(converged=bool(result.success),message=str(result.message),
        iterations=int(result.nit),function_evaluations=int(result.nfev),objective=float(result.fun),
        gradient_inf_norm=float(np.max(np.abs(result.jac))))


def balanced_bce(logits,targets):
    z=np.asarray(logits,np.float64);y=np.asarray(targets)
    terms=[]
    for label in (0,1):
        values=z[y==label];require(len(values)>0,'Validation class missing')
        terms.append(float(np.mean(np.logaddexp(0.,values)-label*values)))
    return float(np.mean(terms))


def code_hashes():
    here=Path(__file__).resolve().parent
    return {n:sha(here/n) for n in ('fit_controls_v1.py','site_data_v1.py')}


def environment():
    return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        backend='cpu',numerical_threads=1)


def make_protocol(data,variant):
    return dict(schema='fims_matched_control_protocol_v1',model=variant,
        portable_manifest_sha256=data.manifest['portable_manifest_sha256'],
        site_manifest_sha256=data.manifest_sha,code=code_hashes(),environment=environment(),
        features=feature_spec(),columns=list(FEATURE_NAMES[:6] if variant.startswith('counts') else FEATURE_NAMES),
        history_bins=8 if variant.endswith('8') else 1,
        scaler='unweighted population mean/std over supervised training sites only; constant std replaced by 1',
        loss='0.5 mean BCE per supervised class + 0.5 * 0.01 * squared weight norm; unpenalized intercept',
        optimizer=dict(name='scipy L-BFGS-B',maxiter=500,ftol=1e-12,gtol=1e-8,maxls=40),
        checkpoint_selection='not applicable; fixed deterministic logistic fit',
        seed_semantics='20260924 is an output identity tag; one deterministic fit, not three seed replicates',
        training='all fixed training events; ignored targets masked; no validation fitted preprocessing',
        evaluation='all 2500 fixed development validation events including ignored and empty events; no holdout access',
        resume='verified event feature caches and completed fits reused; interrupted unfinished optimizer fit restarts')


def make_identity(protocol,variant):
    return dict(protocol_sha256=hashlib.sha256(canonical(protocol)).hexdigest(),model=variant,
        seed=SEED,parameters=(7 if variant.startswith('counts') else 10),train_events=7500)


def verified_complete(out,identity):
    marker=out/'complete.json'
    if not marker.exists():return False
    complete=json.loads(marker.read_text())
    require(complete.get('status')=='COMPLETE' and complete.get('identity')==identity,'Completed control identity mismatch')
    require({'protocol.json','model_parameters.json','selected_predictions.npz','metrics.json'}.issubset(complete['outputs']), 'Completed control outputs missing')
    for name,digest in complete['outputs'].items():
        require(sha(safe_file(out,name))==digest,'Completed control output changed: '+name)
    return True


class FeatureCache:
    def __init__(self,data,folder):
        self.data=data;self.base=Path(folder);self.base.mkdir(parents=True,exist_ok=True)
        self.protocol=dict(schema='fims_control_feature_cache_v1',site_manifest_sha256=data.manifest_sha,
            code=code_hashes(),feature_spec=feature_spec(),numpy=np.__version__,scipy=scipy.__version__)
        write_json(self.base/'protocol.json',self.protocol,immutable=True)
        self.identity=sha(self.base/'protocol.json');self.records={}

    def event(self,index):
        event=self.data.events[index];stem=f'event{index:05d}'
        path=self.base/(stem+'.npz');meta=self.base/(stem+'.json')
        expected=dict(cache_protocol_sha256=self.identity,index=int(index),
            input_sha256=self.data.manifest['files'][event['path']])
        if meta.exists():
            record=json.loads(meta.read_text())
            require(all(record.get(k)==v for k,v in expected.items()),'Feature event identity differs')
            require(sha(safe_file(self.base,stem+'.npz'))==record['features_sha256'],'Feature cache changed')
        else:
            a=self.data.get(index);x8,x1=extract_features(geometry_only(a))
            atomic_npz(path,dict(x8=x8,x1=x1))
            record=dict(expected,features_sha256=sha(path),sites=len(a['target']))
            write_json(meta,record,immutable=True)
        with np.load(path,allow_pickle=False) as z:
            require(set(z.files)=={'x8','x1'},'Feature inventory changed')
            arrays={k:z[k] for k in z.files}
        n=len(self.data.get(index)['target'])
        require(all(v.dtype==np.float64 and v.shape==(n,9) and np.isfinite(v).all() for v in arrays.values()),'Invalid cached features')
        self.records[stem+'.npz']=record['features_sha256'];self.records[stem+'.json']=sha(meta)
        return arrays

    def prepare(self):
        last=time.monotonic()
        for pos,index in enumerate(sorted(self.data.events),1):
            self.event(index)
            if pos%500==0 or time.monotonic()-last>30:
                print(f'Control feature cache: {pos}/{len(self.data.events)} events verified',flush=True);last=time.monotonic()
        write_json(self.base/'complete.json',dict(status='COMPLETE',protocol_sha256=self.identity,files=self.records),immutable=True)


def fit_one(data,cache,root,variant):
    out=root/variant/f'seed{SEED}';out.mkdir(parents=True,exist_ok=True)
    protocol=make_protocol(data,variant);identity=make_identity(protocol,variant)
    if verified_complete(out,identity):
        print(variant+': completed deterministic fit verified; skipping',flush=True);return
    write_json(out/'protocol.json',protocol,immutable=True)
    write_json(out/'status.json',dict(status='RUNNING',identity=identity))
    start=time.monotonic()
    try:
        key='x8' if variant.endswith('8') else 'x1';dim=6 if variant.startswith('counts') else 9
        train=sorted(i for i,e in data.events.items() if e['split']=='train')
        xs=[];ys=[]
        for index in train:
            a=data.get(index);mask=a['target']>=0
            xs.append(cache.event(index)[key][mask,:dim]);ys.append(a['target'][mask])
        x=np.concatenate(xs);y=np.concatenate(ys);del xs,ys
        counts=np.bincount(y,minlength=2)
        require(counts.tolist()==data.settings['supervised_class_counts'],'Training support differs')
        mean,scale=fitted_scaler(x,y);x-=mean;x/=scale
        print(f'{variant}: fitting deterministic L-BFGS, {len(y):,} supervised training sites',flush=True)
        theta,opt=fit_logistic(x,y);train_bce=balanced_bce(x@theta[:-1]+theta[-1],y);del x,y
        logits=[];targets=[];weights=[];micro=[];lengths=[]
        for index in data.validation:
            a=data.get(index);features=cache.event(index)[key][:,:dim]
            z=(features-mean)/scale@theta[:-1]+theta[-1]
            logits.append(z.astype(np.float32));targets.append(a['target']);weights.append(a['primary_electron_weight'])
            micro.append(a['microcurler_electron_weight']);lengths.append(len(z))
        pred=prediction_bundle(data,logits,targets,weights,lengths,core_weights=micro)
        pred['protocol_sha256']=np.asarray(identity['protocol_sha256'])
        val_bce=balanced_bce(pred['logits'],pred['target'])
        atomic_npz(out/'selected_predictions.npz',pred)
        write_json(out/'model_parameters.json',dict(identity=identity,columns=protocol['columns'],mean=mean.tolist(),
            scale=scale.tolist(),weights=theta[:-1].tolist(),intercept=float(theta[-1]),optimizer=opt))
        write_json(out/'metrics.json',dict(identity=identity,model=variant,seed=SEED,parameters=dim+1,
            selected_epoch=None,selected_bce=val_bce,training_class_balanced_bce=train_bce,optimizer=opt,
            train_events=len(train),supervised_training_class_counts=counts.tolist(),validation_events=len(data.validation),
            fit_seconds=time.monotonic()-start,interpretation='development validation only; one fixed deterministic fit; no threshold selection'))
        data.verify_unchanged()
        outputs={n:sha(out/n) for n in ('protocol.json','model_parameters.json','selected_predictions.npz','metrics.json')}
        write_json(out/'complete.json',dict(status='COMPLETE',identity=identity,outputs=outputs))
        require(verified_complete(out,identity),'Completion verification failed')
        write_json(out/'status.json',dict(status='COMPLETE',identity=identity))
        print(f'CONTROL COMPLETE: {variant}; validation class-balanced BCE={val_bce:.6f}',flush=True)
    except BaseException as exc:
        write_json(out/'status.json',dict(status='INTERRUPTED_OR_FAILED',identity=identity,error=str(exc)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path);parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        from test_controls_v1 import run_tests
        run_tests();return
    require(args.data is not None and args.output_dir is not None,'--data and --output-dir required')
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'deterministic_controls.lock').open('a')
    try:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Matched controls already active; no second worker started')
        data=SiteData(args.data)
        if all(verified_complete(root/v/f'seed{SEED}',make_identity(make_protocol(data,v),v)) for v in VARIANTS):
            print('All four deterministic control fits: COMPLETE and verified',flush=True);return
        cache=FeatureCache(data,root.parent/'control_feature_cache_v1');cache.prepare()
        for variant in VARIANTS:fit_one(data,cache,root,variant)
        data.verify_unchanged()
        print('FIMS MATCHED COUNT/SHAPE CONTROLS V1: COMPLETE',flush=True)
    finally:lock.close()


if __name__=='__main__':main()
