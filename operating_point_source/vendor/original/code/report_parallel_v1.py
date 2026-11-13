#!/usr/bin/env python3
"""Read-only development report: saved CNN controls and the parallel site GNNs.

No cutoff from this report is an independently calibrated deployment choice.
All equal-score groups are preserved and all primary charge counts toward loss.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import metrics_parallel_v1 as metrics

cm = metrics.base
SEEDS = metrics.SEEDS
VARIANTS = ('full2', 'full1', 'no_xy2', 'current2')
CONTROLS = ('counts8', 'shape8', 'counts1', 'shape1')
PORTABLE_SHA = '4cbf351c982f4fd683b8e3cc87a1fc40bd15ac45d7f9345fc6f26b7e45636f06'
PRESERVED_SHA = '68f9bf045e3db4bb3713296041053be26d9ea07502f9bd5af7afb15414766163'
DATA_MANIFEST_SHA = 'fd62d8c55c4a2455d95e4ef4152ad29405c15afacd99cceb6386091dec78424d'
LABELS = {'cnn_small':'CNN small', 'cnn_wide':'CNN wide',
          'full2':'GNN 2-layer', 'full1':'GNN 1-layer', 'no_xy2':'GNN no spatial attributes',
          'current2':'GNN current bin', 'counts8':'Counts / 8 bins', 'shape8':'Counts + shape / 8 bins',
          'counts1':'Counts / current bin', 'shape1':'Counts + shape / current bin'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def safe_file(root, name):
    root=Path(root).resolve(); relative=Path(name); path=root/relative
    cm.require(not relative.is_absolute() and '..' not in relative.parts and
               not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(root),
               'Unsafe or missing file: '+str(path))
    return path


def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    cm.require(not path.is_symlink(),'Unsafe output path')
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally: Path(tmp).unlink(missing_ok=True)


def write_json(path,obj):
    atomic_bytes(path,json.dumps(obj,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n')


def write_csv(path,rows):
    if not rows: return
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);atomic_bytes(path,stream.getvalue().encode())


def write_npz(path,arrays):
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);atomic_bytes(path,stream.getvalue())


def load_native(capacity_root):
    """Load validation attribution only, verifying portable/native alignment."""
    data=Path(capacity_root).resolve()/'data';p=safe_file(data,'portable_manifest_v2.json')
    cm.require(sha(p)==PORTABLE_SHA,'Wrong portable development manifest')
    manifest=json.loads(p.read_text());cm.require(manifest['schema']=='fims_cnn_portable_v2','Wrong data schema')
    event_map={int(e['index']):e for e in manifest['events']}
    order=list(map(int,manifest['validation_indices']))
    cm.require(len(event_map)==10000 and len(order)==2500 and len(set(order))==2500,
               'Wrong development/validation support')
    cm.require(set(order)=={i for i,e in event_map.items() if e['split']=='validation'},'Validation partition differs')
    preserved=data/'hit_records';rp=safe_file(preserved,'PRESERVED_RECORDS.json')
    cm.require(sha(rp)==PRESERVED_SHA,'Wrong preserved-record inventory')
    records=json.loads(rp.read_text());cm.require(records['source_manifest_sha256']==DATA_MANIFEST_SHA,'Wrong native data lineage')
    # Cache just the 20 small truth/attribution sidecars, never the graph/patch input.
    native_shards={};provenance={'portable_manifest_sha256':PORTABLE_SHA,'preserved_manifest_sha256':PRESERVED_SHA,'native_files':{}}
    y_parts=[];w_parts=[];mw_parts=[];lens=[]
    for ordinal,index in enumerate(order,1):
        e=event_map[index];shard=e['shard'];raw=int(e['raw_id'])
        if shard not in native_shards:
            name=shard+'/readout/fims_common_source_binary_spike_targets_v1.npz'
            path=safe_file(preserved,name)
            cm.require(name in records['files'] and sha(path)==records['files'][name], 'Native truth checksum mismatch: '+name)
            with np.load(path,allow_pickle=False) as z:
                wanted=('event_ptr','event_id_raw','event_run_uid','target_core','n_primary_other_electrons','n_core_electrons')
                native_shards[shard]={k:z[k] for k in wanted}
            provenance['native_files'][name]=records['files'][name]
        a=native_shards[shard]
        ids=np.flatnonzero((a['event_id_raw']==raw)&(a['event_run_uid']==e['run_uid']))
        cm.require(len(ids)==1,'Native event identity missing/duplicated')
        pos=int(ids[0]);lo,hi=map(int,a['event_ptr'][pos:pos+2])
        target=a['target_core'][lo:hi];primary=a['n_primary_other_electrons'][lo:hi];micro=a['n_core_electrons'][lo:hi]
        path=safe_file(data,e['path']);cm.require(e['path'] in manifest['files'] and sha(path)==manifest['files'][e['path']], 'Portable event checksum mismatch')
        with np.load(path,allow_pickle=False) as z:
            cm.require(np.array_equal(z['target'],target) and np.array_equal(z['primary_electron_weight'],primary),
                       'Portable prediction order/attribution differs from native sites')
        y_parts.append(target);w_parts.append(primary);mw_parts.append(micro);lens.append(len(target))
        if ordinal%500==0: print(f'Report native validation attribution: {ordinal}/2500 events',flush=True)
    return dict(target=np.concatenate(y_parts),primary_electron_weight=np.concatenate(w_parts),
                microcurler_electron_weight=np.concatenate(mw_parts),
                event_ptr=np.r_[np.int64(0),np.cumsum(lens,dtype=np.int64)],
                event_index=np.asarray(order,np.int64),event_id_raw=np.asarray([event_map[i]['raw_id'] for i in order],np.int64),
                event_run_uid=np.asarray([event_map[i]['run_uid'] for i in order]),
                event_shard=np.asarray([event_map[i]['shard'] for i in order])),provenance


def load_fit(path, model, seed, native, *, deterministic=False, expected_events=2500):
    """Do not consume partially written predictions or a differently selected fit."""
    path=Path(path); marker=safe_file(path,'complete.json');complete=json.loads(marker.read_text())
    cm.require(complete.get('status')=='COMPLETE','Fit is not complete')
    required={'metrics.json','protocol.json','selected_predictions.npz'}
    if not deterministic: required.add('history.json')
    cm.require(required.issubset(complete['outputs']),'Fit completion inventory missing')
    for name,digest in complete['outputs'].items():
        cm.require(sha(safe_file(path,name))==digest,'Completed fit changed: '+str(path/name))
    meta=json.loads((path/'metrics.json').read_text());protocol=json.loads((path/'protocol.json').read_text())
    identity=complete['identity'];psha=sha(path/'protocol.json')
    cm.require(identity==meta['identity'] and identity['model']==model and int(identity['seed'])==seed,
               'Fit metadata identity mismatch')
    cm.require(identity['protocol_sha256']==psha and protocol['portable_manifest_sha256']==PORTABLE_SHA,
               'Fit protocol or dataset mismatch')
    if not deterministic:
        history=json.loads((path/'history.json').read_text())
        cm.require(history and all(np.isfinite(r['validation_class_balanced_bce']) for r in history),'Invalid training history')
        chosen=min(history,key=lambda r:r['validation_class_balanced_bce'])
        cm.require(meta['selected_epoch']==chosen['epoch'] and math.isclose(meta['selected_bce'],chosen['validation_class_balanced_bce'],rel_tol=2e-6,abs_tol=2e-7),
                   'Checkpoint is not the earliest minimum validation BCE')
    else:
        cm.require(meta.get('selected_epoch') is None,'Deterministic control cannot claim epoch selection')
    with np.load(path/'selected_predictions.npz',allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    cm.validate_arrays(arrays,expected_events)
    cm.require(str(arrays['protocol_sha256'].item())==psha,'Prediction protocol differs')
    for key in cm.IDENTITY_KEYS: cm.require(np.array_equal(arrays[key],native[key]),'Prediction/native identity mismatch: '+key)
    arrays['microcurler_electron_weight']=native['microcurler_electron_weight']
    bce=cm.class_balanced_bce(arrays['logits'],arrays['target'])
    cm.require(math.isclose(bce,meta['selected_bce'],rel_tol=2e-6,abs_tol=2e-7),'Selected BCE does not reproduce')
    return arrays,dict(seed=seed,model=model,epoch=meta['selected_epoch'],parameters=meta.get('parameters',identity.get('parameters')),
                       protocol_sha256=psha,predictions_sha256=complete['outputs']['selected_predictions.npz'],
                       complete_sha256=sha(marker),deterministic=deterministic,environment=protocol.get('environment',{}))


def flatten(record,result):
    p,m=result['operating_point'],result['metrics']
    keys=('microcurler_site_recall','macro_microcurler_site_recall','microcurler_charge_recall',
          'macro_microcurler_charge_recall','supervised_site_precision','primary_charge_retention',
          'primary_loss_fraction','primary_loss_ppm','maximum_event_primary_loss_fraction',
          'microcurler_sites','flagged_microcurler_sites','microcurler_electrons','flagged_microcurler_electrons',
          'primary_electrons','lost_primary_electrons','lost_primary_electrons_on_ignored_sites')
    return dict(model=record['model'],seed=record['seed'],epoch=record['epoch'],parameters=record['parameters'],
                deterministic=record['deterministic'],operating_point=p['kind'],target=p['target'],target_units=p['target_units'],
                threshold_kind=p['threshold_kind'],threshold_logit=p['threshold_logit'],status=p['status'],
                **{k:m[k] for k in keys},
                fraction_primary_events_losing_above_5_percent=m['primary_loss_tails']['above_5_percent']['fraction'],
                primary_events_losing_above_5_percent=m['primary_loss_tails']['above_5_percent']['events'],
                primary_bearing_events=m['primary_bearing_events'])


def summaries(rows):
    keys=('microcurler_site_recall','macro_microcurler_site_recall','microcurler_charge_recall','macro_microcurler_charge_recall',
          'supervised_site_precision','primary_charge_retention','primary_loss_fraction','primary_loss_ppm',
          'fraction_primary_events_losing_above_5_percent','maximum_event_primary_loss_fraction')
    groups={}
    for row in rows: groups.setdefault((row['model'],row['operating_point'],row['target']),[]).append(row)
    out=[]
    for (model,kind,target),values in sorted(groups.items()):
        required={SEEDS[0]} if values[0]['deterministic'] else set(SEEDS)
        seen={r['seed'] for r in values};cm.require(len(seen)==len(values),'Repeated model seed')
        complete=seen==required
        entry=dict(model=model,operating_point=kind,target=target,seeds=sorted(seen),required_seeds=sorted(required),
                   coverage='COMPLETE' if complete else 'PARTIAL',deterministic=values[0]['deterministic'],statistics={})
        # Never publish a convenient incomplete subset as the three-seed mean.
        if complete:
            for key in keys:
                a=[r[key] for r in values if r[key] is not None]
                entry['statistics'][key]=dict(mean=float(np.mean(a)) if a else None,minimum=min(a) if a else None,
                                               maximum=max(a) if a else None,defined_runs=len(a))
        out.append(entry)
    return out


def make_plots(out, summary, curve_paths, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    models=list(dict.fromkeys(r['model'] for r in records));colors={m:plt.cm.tab10(i%10) for i,m in enumerate(models)}
    x=np.asarray([99.9,99.5,99.0])
    for model in models:
        matches={r['target']:r for r in summary if r['model']==model and r['operating_point']=='primary_loss_budget' and r['coverage']=='COMPLETE'}
        if not all(t in matches for t in (1000,5000,10000)): continue
        for ax,key in zip(axes,('microcurler_site_recall','microcurler_charge_recall','fraction_primary_events_losing_above_5_percent')):
            mean=np.asarray([matches[t]['statistics'][key]['mean'] for t in (1000,5000,10000)],float)*100
            low=np.asarray([matches[t]['statistics'][key]['minimum'] for t in (1000,5000,10000)],float)*100
            high=np.asarray([matches[t]['statistics'][key]['maximum'] for t in (1000,5000,10000)],float)*100
            ax.plot(np.arange(3),mean,'o-',color=colors[model],label=LABELS.get(model,model));ax.fill_between(np.arange(3),low,high,color=colors[model],alpha=.12)
    for ax,title in zip(axes,('Supervised microcurler-site recall','All accepted microcurler-charge recall','Primary-bearing events losing >5% charge')):
        ax.set_title(title,fontsize=11);ax.set_xticks(np.arange(3),['99.9%','99.5%','99.0%']);ax.set_xlabel('Development primary-retention target');ax.set_ylabel('Percent');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('FIMS parallel study — selected fits — development validation',fontsize=14)
    fig.supxlabel('Neural: three-seed mean and full range; deterministic controls: one fit. Cutoffs selected on these same validation events.',fontsize=9)
    fig.savefig(out/'fims_parallel_operating_points_v1.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    seen=set()
    for record,path in zip(records,curve_paths):
        model=record['model']
        with np.load(path,allow_pickle=False) as c:
            cost=(1-c['primary_charge_retention'])*100;recall=c['microcurler_site_recall']*100;mc=c['microcurler_charge_recall']*100
        # Visual sampling only. The exact full curve is kept in each NPZ.
        for ax,limit in zip(axes,(2.,100.)):
            ids=np.flatnonzero(np.isfinite(cost)&np.isfinite(recall)&(cost<=limit))
            if len(ids)>2500: ids=ids[np.unique(np.linspace(0,len(ids)-1,2500).astype(int))]
            label=LABELS.get(model,model) if (model,limit) not in seen else None
            ax.plot(cost[ids],recall[ids],color=colors[model],alpha=.65,lw=1,label=label);seen.add((model,limit))
            ax.set_xlim(0,limit);ax.set_ylim(0,100);ax.grid(alpha=.2)
            ax.set_xlabel('Pooled primary charge lost (%)');ax.set_ylabel('Supervised microcurler-site recall (%)')
    axes[0].set_title('High-retention region');axes[1].set_title('Full operating range');axes[1].legend(fontsize=8)
    fig.suptitle('Every completed seed retained; plot sampling is visual only',fontsize=13)
    fig.supxlabel('Exact complete-score-tie curves are included as compressed NPZ files. These are developmental curves, not test confirmation.',fontsize=9)
    fig.savefig(out/'fims_parallel_retention_curves_v1.png',dpi=170);plt.close(fig)


def build_report(capacity_root,study_root,output_dir=None):
    capacity=Path(capacity_root).resolve();study=Path(study_root).resolve();out=Path(output_dir).resolve() if output_dir else study/'results/report'
    cm.require(not out.is_relative_to(capacity),'Report must not modify the frozen capacity study')
    out.mkdir(parents=True,exist_ok=True);(out/'curves').mkdir(exist_ok=True)
    native,provenance=load_native(capacity)
    jobs=[('cnn_'+model,model,seed,capacity/'results'/model/f'seed{seed}',False) for model in ('small','wide') for seed in SEEDS]
    jobs += [(model,model,seed,study/'results/fits'/model/f'seed{seed}',False) for model in VARIANTS for seed in SEEDS]
    jobs += [(model,model,SEEDS[0],study/'results/fits'/model/f'seed{SEEDS[0]}',True) for model in CONTROLS]
    records=[];point_rows=[];event_rows=[];missing=[];curve_paths=[]
    for label,source_model,seed,fit,deterministic in jobs:
        if not (fit/'complete.json').is_file():
            if label.startswith('cnn_'): raise ValueError('Required completed CNN control missing: '+str(fit))
            missing.append(dict(model=label,seed=seed));continue
        arrays,record=load_fit(fit,source_model,seed,native,deterministic=deterministic)
        record['model']=label
        analysis,events=metrics.analyse(arrays,2500);record.update(analysis)
        path=out/'curves'/f'{label}_seed{seed}_full_curve.npz';write_npz(path,metrics.full_curve(arrays));curve_paths.append(path)
        records.append(record);point_rows.extend(flatten(record,p) for p in analysis['points'])
        event_rows.extend(dict(model=label,seed=seed,**row) for row in events)
        print(f'PARALLEL REPORT: {label} seed={seed}; selected checkpoint verified',flush=True)
    status='PARTIAL' if missing else 'COMPLETE';summary=summaries(point_rows)
    limits=[
        'Development validation only. Cutoffs are fitted and reported on the same 2500 events, not independently calibrated.',
        '99.9%, 99.5% and 99.0% are comparison settings; no detector retention requirement is established.',
        'One checkpoint is selected by ordinary validation BCE per neural fit, then used for every retention target. No target-specific checkpoint selection.',
        'Supervised-site recall and accepted-electron charge recall use different denominators. Site flags propagate to every constituent electron, including ignored mixed sites.',
        'All primary-electron attribution counts toward loss, including ignored sites. Charge retention does not measure full-track reconstruction or direction accuracy.',
        'Neural seed ranges measure optimization and possibly hardware variability on shared events; they are not independent-data confidence intervals.',
        'The site GNN uses exact query neighborhoods, whereas CNN references use discretized local patches. This is a pipeline comparison, not architecture alone.',
        'Continuous-hit experiments are a separate branch and are not reported as trained here.',
        'Previously inspected confirmation events remain excluded. New independent calibration/evaluation is required after choosing and freezing a final procedure.'
    ]
    source=Path(__file__).resolve().parent
    source_hashes={name:sha(source/name) for name in ('report_parallel_v1.py','metrics_parallel_v1.py','vendor_metrics/capacity_metrics_v2.py')}
    report=dict(schema='fims_parallel_development_report_v1',status=status,missing_fits=missing,
                report_code_sha256=source_hashes,
                data_provenance=provenance,results=records,seed_summaries=summary,limitations=limits)
    write_json(out/'fims_parallel_metrics_v1.json',report)
    write_csv(out/'fims_parallel_operating_points_v1.csv',point_rows)
    write_csv(out/'fims_parallel_event_metrics_v1.csv',event_rows)
    make_plots(out,summary,curve_paths,records)
    lines=['FIMS PARALLEL DEVELOPMENT REPORT V1: '+status,'','Completed selected fits: '+str(len(records)),
           'Pending or incomplete planned new fits: '+str(len(missing)),
           'Fixed development validation: 2500 events; no confirmation sample read.','',
           'Primary retention target | model | site recall | microcurler-charge recall | events losing >5% primary charge']
    pct=lambda v:'undefined' if v is None else f'{100*v:.3f}%'
    for s in summary:
        if s['operating_point']!='primary_loss_budget' or s['target'] not in (1000,5000,10000) or s['coverage']!='COMPLETE':continue
        st=s['statistics'];label=LABELS.get(s['model'],s['model'])
        label+=' [single deterministic fit]' if s['deterministic'] else ' [three-seed mean]'
        lines.append(f'{100-s["target"]/10000:.2f}% | {label} | '+ ' | '.join(pct(st[k]['mean']) for k in
                     ('microcurler_site_recall','microcurler_charge_recall','fraction_primary_events_losing_above_5_percent')))
    lines+=['','Cost of higher site recall | model | achieved site recall | primary charge retained | events losing >5% primary charge']
    for s in summary:
        if s['operating_point']!='microcurler_recall_target' or s['coverage']!='COMPLETE':continue
        st=s['statistics'];label=LABELS.get(s['model'],s['model'])
        label+=' [single deterministic fit]' if s['deterministic'] else ' [three-seed mean]'
        lines.append(f'{s["target"]}% recall target | {label} | '+ ' | '.join(pct(st[k]['mean']) for k in
                     ('microcurler_site_recall','primary_charge_retention','fraction_primary_events_losing_above_5_percent')))
    lines+=['','Incomplete three-seed groups are not shown as three-seed means.','',*limits]
    atomic_bytes(out/'fims_parallel_summary_v1.txt',('\n'.join(lines)+'\n').encode())
    inventory={str(p.relative_to(out)):sha(p) for p in [out/'fims_parallel_metrics_v1.json',out/'fims_parallel_operating_points_v1.csv',
                out/'fims_parallel_event_metrics_v1.csv',out/'fims_parallel_operating_points_v1.png',out/'fims_parallel_retention_curves_v1.png',
                out/'fims_parallel_summary_v1.txt',*curve_paths]}
    write_json(out/'fims_parallel_report_status_v1.json',dict(status=status,completed_fits=len(records),missing_fits=missing,outputs=inventory))
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*inventory,'fims_parallel_report_status_v1.json']:z.write(out/name,name)
    atomic_bytes(out/'fims_parallel_review_v1.zip',buf.getvalue())
    print('FIMS PARALLEL REPORT V1: '+status,flush=True)
    print('Review ZIP: '+str(out/'fims_parallel_review_v1.zip'),flush=True)
    return report


def self_test():
    metrics.self_test()
    rows=[]
    for seed in SEEDS:
        rows.append(dict(model='full2',seed=seed,operating_point='primary_loss_budget',target=1000,deterministic=False,
                         **{k:.5 for k in ('microcurler_site_recall','macro_microcurler_site_recall','microcurler_charge_recall','macro_microcurler_charge_recall',
                         'supervised_site_precision','primary_charge_retention','primary_loss_fraction','primary_loss_ppm',
                         'fraction_primary_events_losing_above_5_percent','maximum_event_primary_loss_fraction')}))
    cm.require(summaries(rows)[0]['coverage']=='COMPLETE','Complete seed group missing')
    partial=summaries(rows[:2])[0]
    cm.require(partial['coverage']=='PARTIAL' and not partial['statistics'],'Incomplete subset misrepresented as seed mean')
    one=dict(rows[0],model='counts8',deterministic=True)
    cm.require(summaries([one])[0]['coverage']=='COMPLETE','Deterministic control demands fake replicate seeds')
    print('PARALLEL REPORT SELF-TEST: PASS (complete seed coverage, partial study, deterministic controls)')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capacity-root');p.add_argument('--study-root');p.add_argument('--output-dir');p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return
    if not a.capacity_root or not a.study_root:p.error('--capacity-root and --study-root are required')
    out=Path(a.output_dir).resolve() if a.output_dir else Path(a.study_root).resolve()/'results/report'
    cm.require(not out.is_relative_to(Path(a.capacity_root).resolve()),'Report output overlaps frozen capacity study')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'report.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('A report writer is already active; no duplicate report started.')
        build_report(a.capacity_root,a.study_root,a.output_dir)


if __name__=='__main__':main()
