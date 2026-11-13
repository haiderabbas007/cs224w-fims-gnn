"""Synthetic checks for data separation, masked loss, accumulation and recovery."""
import copy,json,tempfile
from pathlib import Path
import numpy as np
from site_data_v1 import *

def fixture(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    files={};events=[];counts=np.zeros(2,np.int64)
    for i in range(5):
        target=np.array([0,1,-1,0,1],np.int8) if i!=4 else np.empty(0,np.int8)
        n=len(target)
        a=dict(x_mm=np.array([0.,.2,.8,1.9,2.1])[:n]+i*3,
            y_mm=np.array([0.,.1,.2,0.,.4])[:n],time_bin=np.array([8,9,9,10,10],np.int64)[:n],side=np.zeros(n,np.uint8),target=target,
            primary_electron_weight=np.array([1,0,2,1,0],np.int64)[:n],microcurler_electron_weight=np.array([0,1,0,0,1],np.int64)[:n])
        name=f'events/event_{i:05d}.npz';atomic_npz(folder/name,a);files[name]=sha(folder/name)
        split='train' if i<3 else 'validation';events.append(dict(index=i,raw_id=i,run_uid='synthetic',shard='toy',path=name,split=split))
        if i<3:counts+=np.bincount(target[target>=0],minlength=2)
    weights=(counts.sum()/(2*counts)).astype(np.float32).tolist()
    settings=dict(eligible_event_indices=[0,1,2],supervised_class_counts=counts.tolist(),class_weights=weights,
        fixed_denominator=float(counts.sum()),optimizer_batches_per_epoch=1)
    orders={str(s):[[0,2,1] for _ in range(10)] for s in [20260924,20260925,20260926]}
    write_json(folder/'site_manifest_v1.json',dict(schema='fims_site_geometry_v1',synthetic=True,
        portable_manifest_sha256='synthetic',preserved_manifest_sha256='synthetic',events=events,files=files,validation_indices=[3,4],training_settings=settings,epoch_orders=orders))
    return SiteData(folder,synthetic=True)

def test_geometry_contract():
    with tempfile.TemporaryDirectory() as d:
        data=fixture(d);a=data.get(0);validate_event(a)
        require(set(geometry_only(a))==set(GEOMETRY_KEYS),'Truth leaked into geometry')
        require(len(data.get(4)['target'])==0,'Empty event lost')
        try:SiteData(d)
        except ValueError:pass
        else:raise AssertionError('Synthetic data accepted as real')
        p=Path(d)/data.events[0]['path'];p.write_bytes(p.read_bytes()+b'corruption')
        try:data.get(0)
        except ValueError:pass
        else:raise AssertionError('Mutation not detected')

def run_tests():
    import torch
    import train_site_gnn_v1 as train
    import gnn_adapter_v1 as models
    test_geometry_contract()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);data=fixture(root/'data');device=torch.device('cpu');a=data.get(0)
        w=torch.tensor(data.settings['class_weights']);den=data.settings['fixed_denominator']
        for variant in models.VARIANTS:
            m1=models.create_model(variant,20260924,device);m2=models.create_model(variant,20260924,device)
            o1=train.make_optimizer(m1);o2=train.make_optimizer(m2)
            train.training_batch(m1,o1,[a],w,den,device,microbatch=1)
            train.training_batch(m2,o2,[a],w,den,device,microbatch=64)
            for p,q in zip(m1.parameters(),m2.parameters()):
                require(torch.allclose(p,q,atol=2e-6,rtol=2e-5),'Microbatch gradient/update mismatch '+variant)
            # Evaluation must still score the ignored query carrying primary charge.
            pred,_=train.evaluate(m1,data,device,stop_callback=lambda:None)
            require(len(pred['logits'])==5 and pred['event_ptr'].tolist()==[0,5,5],'Masked/empty evaluation alignment failure')
            require(pred['primary_electron_weight'][2]==2,'Ignored primary charge dropped')
        variant='full1'
        train.fit(data,root/'straight',variant,20260924,device)
        old=train.check_stop;calls=[0]
        def stop():
            calls[0]+=1
            if calls[0]==3:raise train.Paused('Synthetic interruption')
        train.check_stop=stop
        try:
            try:train.fit(data,root/'resumed',variant,20260924,device)
            except train.Paused:pass
        finally:train.check_stop=old
        train.fit(data,root/'resumed',variant,20260924,device)
        for stage in ('selected','final'):
            p=root/'straight'/variant/'seed20260924'/(stage+'_checkpoint.pt')
            q=root/'resumed'/variant/'seed20260924'/(stage+'_checkpoint.pt')
            x=torch.load(p,weights_only=True)['state_dict'];y=torch.load(q,weights_only=True)['state_dict']
            require(all(torch.equal(x[k],y[k]) for k in x),'Optimizer/RNG resume parity failed')
        before=sha(root/'resumed'/variant/'seed20260924'/'complete.json')
        train.fit(data,root/'resumed',variant,20260924,device)
        require(sha(root/'resumed'/variant/'seed20260924'/'complete.json')==before,'Completed fit reran')
    print('SITE PIPELINE SELF-TEST: PASS (truth-free geometry, tamper rejection, all four arms, ignored charge, empty events, microbatch gradients, exact optimizer resume)',flush=True)

if __name__=='__main__':run_tests()
