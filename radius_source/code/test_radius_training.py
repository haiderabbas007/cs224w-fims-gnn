"""Invented-event checks; no development or confirmation data are opened."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile

import train_radius as radius_train
import numpy as np
import torch
import graph_models_v1 as original
from radius_graph import RadiusQueryBuilder, RADII, graph_chunks
from site_data_v1 import geometry_only
from test_site_pipeline_v1 import fixture


def expect_failure(fn):
    try:
        fn()
    except (ValueError, IndexError, RuntimeError):
        return
    raise AssertionError('Expected rejection')


def toy_geometry(radius=2.0):
    # Includes both spatial/time boundaries, future sites, opposite endcap,
    # repeated coordinates at different times, and current-bin neighbors.
    return dict(x_mm=np.asarray([0, radius, radius+.001, .2, .3, .4, .5, -.7, -.3], np.float64),
                y_mm=np.asarray([0, 0, 0, .1, .1, 0, 0, .2, -.1], np.float64),
                time_bin=np.asarray([10, 10, 10, 3, 2, 11, 10, 8, 10], np.int64),
                side=np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0], np.int8))


def reference_graph(geometry, query, radius):
    """Original complete induced graph, with an explicit fixed physical scale."""
    prior = original.RADIUS_MM
    original.RADIUS_MM = float(radius)
    try:
        g = original.QueryGraphBuilder(geometry, 'full1', max_edges=None).build(query)
    finally:
        original.RADIUS_MM = prior
    # Original encoding divides by its radius. Recompute attrs directly from
    # float64 geometry rather than rescale rounded float32 values.
    ids = g.observation_indices.numpy()
    s, t = g.edge_index.numpy()
    xy = np.column_stack((geometry['x_mm'], geometry['y_mm']))
    times = geometry['time_bin']
    attrs = np.column_stack(((xy[ids[s]]-xy[ids[t]])/2.,
                             (times[ids[s]]-times[ids[t]])/7.)).astype(np.float32)
    g.edge_attr = torch.from_numpy(attrs)
    return g


def test_geometry():
    for radius in RADII:
        geometry = toy_geometry(radius)
        builder = RadiusQueryBuilder(geometry, radius)
        for q in range(len(geometry['x_mm'])):
            graph = builder.build(q)
            xy = np.column_stack((geometry['x_mm'], geometry['y_mm']))
            expected = [j for j in range(len(xy))
                        if geometry['side'][j] == geometry['side'][q]
                        and geometry['time_bin'][q]-7 <= geometry['time_bin'][j] <= geometry['time_bin'][q]
                        and np.sum((xy[j]-xy[q])**2) <= radius**2]
            assert set(graph.observation_indices.tolist()) == set(expected)
            sources = graph.observation_indices[graph.edge_index[0]].tolist()
            assert set(sources) == set(expected)-{q}
            assert graph.edge_index.shape[1] == len(expected)-1
            assert torch.all(graph.edge_index[1] == graph.query_index.item())
            full = reference_graph(geometry, q, radius)
            mask = full.edge_index[1] == full.query_index.item()
            assert torch.equal(full.observation_indices, graph.observation_indices)
            assert torch.equal(full.edge_index[:,mask], graph.edge_index)
            assert torch.equal(full.edge_attr[mask], graph.edge_attr)
        qgraph = builder.build(0)
        ids = qgraph.observation_indices.tolist()
        assert 1 in ids and 3 in ids and all(j not in ids for j in [2,4,5,6])
        assert any(abs(float(x)-radius/2) < 1e-6 for x in qgraph.edge_attr[:,0])
        # Coordinate-order construction must not depend on incoming file order.
        perm = np.asarray([8,4,2,6,0,1,7,3,5])
        changed = {k:v[perm] for k,v in geometry.items()}
        other = RadiusQueryBuilder(changed, radius).build(int(np.flatnonzero(perm == 0)[0]))
        assert torch.equal(qgraph.x, other.x)
        assert torch.equal(qgraph.edge_index, other.edge_index)
        assert torch.equal(qgraph.edge_attr, other.edge_attr)
        # Translation invariance with exactly representable offsets.
        shifted = dict(geometry, x_mm=geometry['x_mm']+16, y_mm=geometry['y_mm']-8)
        sg = RadiusQueryBuilder(shifted,radius).build(0)
        assert torch.equal(sg.observation_indices,qgraph.observation_indices)
        assert torch.allclose(sg.edge_attr,qgraph.edge_attr,atol=1e-7,rtol=0)
        expect_failure(lambda:RadiusQueryBuilder(dict(geometry,target=np.ones(9)),radius))
        expect_failure(lambda:RadiusQueryBuilder(geometry,radius,variant='full2'))
        expect_failure(lambda:RadiusQueryBuilder(geometry,radius,max_edges=0).build(0))
        expect_failure(lambda:builder.build(-1))
        duplicate={k:np.r_[v,v[0]] for k,v in geometry.items()}
        expect_failure(lambda:RadiusQueryBuilder(duplicate,radius))
    expect_failure(lambda:RadiusQueryBuilder(toy_geometry(),3))
    empty={k:v[:0] for k,v in toy_geometry().items()}
    assert len(RadiusQueryBuilder(empty,2))==0
    expect_failure(lambda:RadiusQueryBuilder(empty,2).build(0))


def assert_model_equal(a,b,atol=5e-6,rtol=5e-5):
    assert list(a.state_dict()) == list(b.state_dict())
    for (name,p),(name2,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==name2 and torch.allclose(p,q,atol=atol,rtol=rtol),(name,torch.max(abs(p-q)).item())


def runtime_parity_checks(device):
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA GPU unavailable')
    seed=20260924
    for radius in RADII:
        geometry=toy_geometry(radius)
        for dtype in (torch.float32,torch.float64):
            radius_train.adapter.seed_all(seed,2)
            old=original.GeometricMPNN('full1').to(device=device,dtype=dtype)
            new=radius_train.create_model('full1',seed,device).to(dtype=dtype)
            assert_model_equal(old,new,atol=0,rtol=0)
            full=original.batch_graphs([reference_graph(geometry,q,radius) for q in range(9)],device)
            star=original.batch_graphs([RadiusQueryBuilder(geometry,radius).build(q) for q in range(9)],device)
            full.x=full.x.to(dtype);full.edge_attr=full.edge_attr.to(dtype)
            star.x=star.x.to(dtype);star.edge_attr=star.edge_attr.to(dtype)
            y=torch.tensor([0,1,1,0,1,0,0,1,0],device=device,dtype=dtype)
            lo,ln=old(full),new(star)
            atol,rtol=(3e-6,3e-5) if dtype==torch.float32 else (1e-11,1e-10)
            assert torch.allclose(lo,ln,atol=atol,rtol=rtol),(device,radius,dtype,'logits')
            weights=torch.tensor([.75,1.5],device=device,dtype=dtype)[y.long()]
            for m,z in ((old,lo),(new,ln)):
                (torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')*weights).sum().backward()
            for (name,p),(_,q) in zip(old.named_parameters(),new.named_parameters()):
                assert p.grad is not None and q.grad is not None,name
                assert torch.allclose(p.grad,q.grad,atol=atol*5,rtol=rtol*5),(radius,dtype,name)
            # Matched Adam steps also detect missing-vs-zero gradient semantics.
            oo=radius_train.train.make_optimizer(old);no=radius_train.train.make_optimizer(new)
            for _ in range(2):
                for m,o,b in ((old,oo,full),(new,no,star)):
                    o.zero_grad(set_to_none=True)
                    z=m(b)
                    loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')*weights).sum()/4007.697721890993
                    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
                assert_model_equal(old,new,atol=5e-6 if dtype==torch.float32 else 1e-10,
                                   rtol=5e-5 if dtype==torch.float32 else 1e-8)
        # An isolated query must still give phi parameters zero, not None, gradients.
        one={k:v[:1] for k,v in geometry.items()}
        radius_train.adapter.seed_all(seed,2)
        old=original.GeometricMPNN('full1').to(device)
        new=radius_train.create_model('full1',seed,device)
        old(reference_graph(one,0,radius).to(device)).sum().backward()
        new(RadiusQueryBuilder(one,radius).build(0).to(device)).sum().backward()
        for p,q in zip(old.parameters(),new.parameters()):
            assert p.grad is not None and q.grad is not None
            assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-6)
    # Exercise a full-graph chunk boundary across query messages. This checks
    # the previous recovery execution as well as the original dense reference.
    n=160
    dense=dict(x_mm=np.linspace(-.49,.49,n),y_mm=np.zeros(n),
               time_bin=np.full(n,10,np.int64),side=np.zeros(n,np.int8))
    q=103
    radius_train.adapter.seed_all(seed,2)
    dense_model=original.GeometricMPNN('full1').to(device)
    bounded_model=radius_train.create_model('full1',seed,device)
    star_model=radius_train.create_model('full1',seed,device)
    full=reference_graph(dense,q,2).to(device)
    star=RadiusQueryBuilder(dense,2).build(q).to(device)
    assert full.edge_index.shape[1]>16384
    outputs=[dense_model(full),bounded_model(full),star_model(star)]
    for out in outputs[1:]:
        assert torch.allclose(outputs[0],out,atol=2e-5,rtol=5e-5)
    for out in outputs:out.sum().backward()
    for models in (bounded_model,star_model):
        for p,v in zip(dense_model.parameters(),models.parameters()):
            assert p.grad is not None and v.grad is not None
            assert torch.allclose(p.grad,v.grad,atol=2e-4,rtol=2e-4)
    print('RADIUS QUERY EXECUTION PARITY: PASS ('+device.type+'; all radii; full graph vs query edges; '
          'float32/64 logits, gradients, initialization, Adam, zero-edge gradients)',flush=True)


def test_resume_and_masks():
    train=radius_train.train
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);data=fixture(root/'data');device=torch.device('cpu')
        arrays=[data.get(i%3) for i in range(8)]
        weights=torch.tensor(data.settings['class_weights'])
        denominator=data.settings['fixed_denominator']
        with radius_train.radius_runtime(root,4):
            a=radius_train.create_model('full1',20260924,device)
            b=radius_train.create_model('full1',20260924,device)
            oa,ob=train.make_optimizer(a),train.make_optimizer(b)
            train.training_batch(a,oa,arrays,weights,denominator,device,microbatch=1)
            train.training_batch(b,ob,arrays,weights,denominator,device,microbatch=64)
            assert_model_equal(a,b)
            pred,_=train.evaluate(a,data,device,stop_callback=lambda:None)
            assert pred['event_ptr'].tolist()==[0,5,5]
            assert len(pred['logits'])==5 and pred['target'][2]==-1
            assert pred['primary_electron_weight'][2]==2
        straight=root/'straight';resumed=root/'resumed'
        with contextlib.redirect_stdout(io.StringIO()):
            radius_train.fit_radius(straight,data,4,20260924,device)
            with radius_train.radius_runtime(resumed,4):
                baseline_batch=train.training_batch
                calls=[0]
                def interrupt(*args,**kwargs):
                    answer=baseline_batch(*args,**kwargs);calls[0]+=1
                    if calls[0]==3:train.STOP=True
                    return answer
                train.training_batch=interrupt
                try:
                    try:train.fit(data,resumed/'results/fits/r4','full1',20260924,device)
                    except train.Paused:pass
                    else:raise AssertionError('Did not interrupt')
                finally:
                    train.training_batch=baseline_batch;train.STOP=False
            radius_train.fit_radius(resumed,data,4,20260924,device)
            for stage in ('selected','final'):
                base=Path('results/fits/r4/full1/seed20260924')
                x=torch.load(straight/base/(stage+'_checkpoint.pt'),weights_only=True)['state_dict']
                y=torch.load(resumed/base/(stage+'_checkpoint.pt'),weights_only=True)['state_dict']
                assert all(torch.equal(x[k],y[k]) for k in x),stage
                with np.load(straight/base/(stage+'_predictions.npz')) as p,np.load(resumed/base/(stage+'_predictions.npz')) as q:
                    assert all(np.array_equal(p[k],q[k]) for k in p.files)
            folder=resumed/'results/fits/r4/full1/seed20260924'
            hashes={p.name:train.sha(p) for p in folder.iterdir() if p.is_file()}
            radius_train.fit_radius(resumed,data,4,20260924,device)
            assert all(train.sha(folder/n)==digest for n,digest in hashes.items())
            # Resume metadata rejects a changed radius under an existing identity.
            protocol,identity=radius_train.make_identity(data,'full1',20260924,device,
                                                         study_root=resumed,radius=2)
            expect_failure(lambda:train.verify_complete(folder,identity))
        data.verify_unchanged()


def run_tests(device='cpu'):
    test_geometry()
    runtime_parity_checks(device)
    if torch.device(device).type=='cpu':
        test_resume_and_masks()
    print('RADIUS TRAINER SELF-TEST: PASS (radius/time/endcap boundaries, permutation, truth exclusion, '
          'full-graph equivalence; CPU suite also covers fixed-D batches, ignored charge, empty events, '
          'optimizer/RNG resume, completed-run reuse and changed-identity rejection)',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    run_tests(p.parse_args().device)
