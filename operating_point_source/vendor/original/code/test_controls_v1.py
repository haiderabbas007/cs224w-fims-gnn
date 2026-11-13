#!/usr/bin/env python3
"""Synthetic-only checks for truth-blind FIMS count/shape controls."""
from __future__ import annotations
from types import SimpleNamespace
from pathlib import Path
import tempfile
import numpy as np
from fit_controls_v1 import (extract_features,fitted_scaler,bce_gradient,fit_logistic,
    balanced_bce,FeatureCache,make_protocol,make_identity,fit_one,verified_complete,
    SEED,VARIANTS)
from site_data_v1 import atomic_npz,write_json,sha,geometry_only


def geometry(x,y,t,s):
    return dict(x_mm=np.asarray(x,np.float64),y_mm=np.asarray(y,np.float64),
        time_bin=np.asarray(t,np.int64),side=np.asarray(s,np.int8))


def reference(g,window):
    n=len(g['x_mm']);out=np.zeros((n,9))
    for q in range(n):
        others=[]
        for j in range(n):
            d=np.array([g['x_mm'][j]-g['x_mm'][q],g['y_mm'][j]-g['y_mm'][q]])
            if j!=q and g['side'][j]==g['side'][q] and 0<=int(g['time_bin'][q]-g['time_bin'][j])<window and np.dot(d,d)<=4:
                others.append((j,d))
        for k,r in enumerate((.5,1.,2.)):
            close=[j for j,d in others if np.dot(d,d)<=r*r]
            out[q,k]=np.log1p(len(close));out[q,k+3]=np.log1p(len(set(int(g['time_bin'][j]) for j in close)))
        if others:
            d=np.asarray([d for j,d in others]);r=np.linalg.norm(d,axis=1)
            out[q,6]=np.mean(r)/2.;out[q,7]=np.std(r)/2.
            if len(others)>=2:
                v=d-d.mean(0);eig=np.linalg.eigvalsh(v.T@v/len(v))
                if eig.sum()>0:out[q,8]=(eig[1]-eig[0])/eig.sum()
    return out


def run_tests():
    # Inclusive radius/history boundaries, same-bin context and excluded future/query/endcap.
    g=geometry([0,.5,1,2,2.00001,.1,.2,.3,0],[0]*9,[10,10,9,3,10,11,2,10,10],[0,0,0,0,0,0,0,1,1])
    x8,x1=extract_features(g)
    np.testing.assert_allclose(x8,reference(g,8),rtol=0,atol=2e-15)
    np.testing.assert_allclose(x1,reference(g,1),rtol=0,atol=2e-15)
    np.testing.assert_allclose(np.expm1(x8[0,:6]),[1,2,3,1,2,3],atol=1e-14)
    np.testing.assert_allclose(np.expm1(x1[0,:6]),[1,1,1,1,1,1],atol=1e-14)
    singleton=geometry([0],[0],[5],[0]);assert not extract_features(singleton)[0].any()
    empty=geometry([],[],[],[]);assert extract_features(empty)[0].shape==(0,9)
    for w in extract_features(singleton):assert not w.any()
    altered={k:v.copy() for k,v in g.items()};altered['x_mm'][5]=.01;altered['x_mm'][7]=.01
    np.testing.assert_array_equal(extract_features(altered)[0][0],x8[0])
    order=np.array([8,3,6,1,7,0,2,4,5]);permuted={k:v[order] for k,v in g.items()}
    for actual,expected in zip(extract_features(permuted),(x8[order],x1[order])):
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-15)
    translated={k:v.copy() for k,v in g.items()};translated['x_mm']+=100;translated['y_mm']-=30
    for actual,expected in zip(extract_features(translated),(x8,x1)):
        np.testing.assert_allclose(actual,expected,rtol=0,atol=5e-13)
    try:extract_features(dict(g,target=np.ones(9)))
    except ValueError:pass
    else:raise AssertionError('Truth accepted by encoder')
    # Mean/std use supervised training rows only, with ignored outlier masked.
    train=np.array([[0.,1.],[2.,1.],[1e12,1e12]]);y=np.array([0,1,-1],np.int8)
    mean,scale=fitted_scaler(train,y)
    np.testing.assert_array_equal(mean,[1.,1.]);np.testing.assert_array_equal(scale,[1.,1.])
    validation=np.array([[1e30,2e30]])
    transformed=(validation-mean)/scale
    assert transformed[0,0]>1e20
    np.testing.assert_array_equal(fitted_scaler(train,y)[0],mean)
    theta=np.array([.2,-.4,.3]);f,grad=bce_gradient(theta,train,y)
    f2,g2=bce_gradient(theta,train[:2],y[:2])
    assert f==f2;np.testing.assert_array_equal(grad,g2)
    numerical=[]
    for k in range(len(theta)):
        delta=np.zeros_like(theta);delta[k]=1e-6
        numerical.append((bce_gradient(theta+delta,train,y)[0]-bce_gradient(theta-delta,train,y)[0])/2e-6)
    np.testing.assert_allclose(grad,numerical,rtol=2e-7,atol=2e-10)
    # L2 never penalizes the intercept, and steepest negative-gradient motion lowers objective.
    no_l2=bce_gradient(theta,train,y,0.)
    np.testing.assert_allclose(grad-no_l2[1],[.002,-.004,0.],atol=1e-15)
    assert bce_gradient(theta-1e-3*grad,train,y)[0]<f
    fitted,opt=fit_logistic(train,y)
    assert opt['converged'] and np.isfinite(fitted).all()
    fitted_again,_=fit_logistic(train,y);np.testing.assert_array_equal(fitted,fitted_again)
    assert balanced_bce(train[:2]@fitted[:-1]+fitted[-1],y[:2])<.6932
    # Hash-pinned synthetic cache rejects tampering and completed outputs are not blindly reused.
    with tempfile.TemporaryDirectory() as temp:
        base=Path(temp);events={};values={};files={}
        for index,split in enumerate(('train','train','validation')):
            gg=geometry([0,.2,.8,1.8],[0,0,.2,0],[5,5,5,5],[0]*4)
            a=dict(gg,target=np.array([0,1,1,-1],np.int8),
                primary_electron_weight=np.array([2,0,0,1],np.int64),
                microcurler_electron_weight=np.array([0,1,1,0],np.int64))
            p=base/f'input{index}.npz';atomic_npz(p,a);files[p.name]=sha(p);values[index]=a
            events[index]=dict(path=p.name,split=split,index=index,raw_id=index,run_uid='toy',shard='synthetic')
        data=SimpleNamespace(events=events,manifest_sha='a'*64,
            manifest=dict(portable_manifest_sha256='b'*64,files=files),settings=dict(supervised_class_counts=[2,4]),
            validation=[2],get=lambda i:values[i],verify_unchanged=lambda:None)
        cache=FeatureCache(data,base/'cache');cache.prepare()
        np.testing.assert_array_equal(cache.event(0)['x8'],extract_features(geometry_only(values[0]))[0])
        out=base/'fits';fit_one(data,cache,out,'shape8')
        ident=make_identity(make_protocol(data,'shape8'),'shape8')
        assert verified_complete(out/'shape8'/f'seed{SEED}',ident)
        fit_one(data,cache,out,'shape8')
        path=cache.base/'event00000.npz';path.write_bytes(path.read_bytes()+b'tamper')
        try:cache.event(0)
        except ValueError:pass
        else:raise AssertionError('Modified feature cache accepted')
    print('MATCHED CONTROL SELF-TEST: PASS (scalar features, exact radius/history, future/query/endcap exclusion, empty events, permutation/translation, training-only scaler, masked class-balanced loss, finite gradient, deterministic fit, cache/completion checks)')


if __name__=='__main__':run_tests()
