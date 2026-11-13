#!/usr/bin/env python3
"""Submit/resume independent FIMS stages with a fail-closed scheduler journal."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from verify_package import verify, sha, CAPACITY_MANIFEST_SHA

MODELS=('full2','full1','no_xy2','current2')
SEEDS=(20260924,20260925,20260926)
CONTROLS=('counts8','shape8','counts1','shape1')

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def checked_fit(path):
    p=path/'complete.json'
    if not p.exists():return False
    if p.is_symlink():raise ValueError('Unsafe fit completion marker')
    m=json.loads(p.read_text())
    if m.get('status') != 'COMPLETE' or not m.get('outputs'):
        raise ValueError('Malformed completed fit: '+str(path))
    for name,digest in m['outputs'].items():
        f=path/name
        if Path(name).name != name or f.is_symlink() or not f.is_file() or sha(f)!=digest:
            raise ValueError('Completed fit output differs: '+str(f))
    return True

def stage_done(root, stage, package_sha):
    p=root/'results/stages'/ (stage+'.json')
    if not p.exists():return False
    m=json.loads(p.read_text())
    if m.get('package_sha256') != package_sha or m.get('capacity_manifest_sha256') != CAPACITY_MANIFEST_SHA:
        raise ValueError('Stage receipt belongs to another study: '+stage)
    if m.get('status') != 'COMPLETE':return False
    # The initial report is deliberately overwritten by later complete reports.
    if stage == 'initial_report':return True
    for name,digest in m.get('outputs',{}).items():
        f=root/name
        if not f.is_file() or f.is_symlink() or sha(f)!=digest:
            raise ValueError('Stage output changed: '+name)
    return True

def mark_stage(root, stage, package_sha):
    outputs={}
    if stage == 'prepare':
        p=root/'data/site_manifest_v1.json'
        if not p.is_file():raise ValueError('Geometry preparation did not create site_manifest_v1.json')
        outputs['data/site_manifest_v1.json']=sha(p)
    elif stage == 'controls':
        for name in CONTROLS:
            path=root/'results/fits'/name/'seed20260924'
            if not checked_fit(path):raise ValueError('Missing completed control: '+name)
            outputs[str((path/'complete.json').relative_to(root))]=sha(path/'complete.json')
    elif stage == 'final_report':
        if any(not checked_fit(root/'results/fits'/m/('seed'+str(s))) for m in MODELS for s in SEEDS):
            raise ValueError('Final report cannot complete before all 12 GNN fits')
        if any(not checked_fit(root/'results/fits'/m/'seed20260924') for m in CONTROLS):
            raise ValueError('Final report cannot complete before all four controls')
        report_status=root/'results/report/fims_parallel_report_status_v1.json'
        if not report_status.is_file():raise ValueError('Final report status is missing')
        report=json.loads(report_status.read_text())
        if report.get('status')!='COMPLETE' or report.get('missing_fits'):
            raise ValueError('Final report is still partial; no complete stage receipt written')
        for name,digest in report.get('outputs',{}).items():
            rp=root/'results/report'/name
            if Path(name).is_absolute() or '..' in Path(name).parts or rp.is_symlink() or not rp.is_file() or sha(rp)!=digest:
                raise ValueError('Final report output hash mismatch: '+name)
        p=root/'results/report/fims_parallel_review_v1.zip'
        if not p.is_file():raise ValueError('Final review ZIP missing')
        outputs[str(p.relative_to(root))]=sha(p)
    elif stage != 'initial_report':raise ValueError('Unknown stage')
    atomic_json(root/'results/stages'/(stage+'.json'),dict(status='COMPLETE',package_sha256=package_sha,
        capacity_manifest_sha256=CAPACITY_MANIFEST_SHA,outputs=outputs,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))

def query_queue():
    q=subprocess.run(['squeue','--noheader','--user',str(os.getuid()),'--format=%i|%j|%T|%R'],
                     check=True,text=True,stdout=subprocess.PIPE)
    rows=[]
    for line in q.stdout.splitlines():
        p=[x.strip() for x in line.strip().split('|',3)]
        if len(p)!=4:raise ValueError('Unexpected squeue output; no duplicate submission attempted')
        rows.append(dict(id=p[0],name=p[1],state=p[2],reason=p[3]))
    return rows

def active_for(rows, name):
    matches=[r for r in rows if r['name']==name]
    if not matches:return None
    ids={re.split(r'[_+]',r['id'])[0] for r in matches}
    if len(ids)!=1 or not next(iter(ids)).isdigit():
        raise ValueError('Multiple/unrecognized live jobs for '+name+'; inspect queue before resubmitting')
    if any('DependencyNeverSatisfied' in r['reason'] for r in matches):
        # Only our exactly named versioned stage may be cancelled on recovery.
        jid=next(iter(ids));subprocess.run(['scancel',jid],check=True)
        print('Cancelled failed-dependency stage '+jid,flush=True)
        return None
    return next(iter(ids))

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--capacity-root',required=True)
    p.add_argument('--python',required=True);p.add_argument('--mark-stage',choices=['prepare','controls','initial_report','final_report'])
    a=p.parse_args();root=Path(a.study).resolve();capacity=Path(a.capacity_root).resolve()
    digest=verify(root,capacity)
    if a.mark_stage:
        mark_stage(root,a.mark_stage,digest);print('STAGE COMPLETE:',a.mark_stage);return
    root.joinpath('results').mkdir(exist_ok=True);root.joinpath('logs').mkdir(exist_ok=True)
    with (root/'results/submission.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        rows=query_queue()  # Scheduler errors abort instead of risking duplicate jobs.
        jp=root/'results/jobs.json';journal=json.loads(jp.read_text()) if jp.exists() else {'package_sha256':digest,'submissions':[]}
        if journal['package_sha256'] != digest:raise ValueError('Submission journal package changed')
        concurrency=int(os.environ.get('FIMS_GNN_CONCURRENCY','3'))
        if not 1<=concurrency<=12:raise ValueError('FIMS_GNN_CONCURRENCY must be 1..12')
        gpu_partition=os.environ.get('FIMS_GPU_PARTITION','gpu');cpu_partition=os.environ.get('FIMS_CPU_PARTITION','shared')
        time_limit=os.environ.get('FIMS_GNN_TIME_LIMIT','08:00:00')
        account=os.environ.get('FIMS_ACCOUNT','')

        def submit(stage,deps=(),indices=None):
            name='fims-'+digest[:8]+'-'+stage
            active=active_for(rows,name)
            if active:
                print('Queued/active, reused:',stage,active,flush=True);return active
            opts=['sbatch','--parsable','--nodes=1','--ntasks=1','--job-name='+name,'--chdir='+str(root)]
            if indices is not None:
                opts+=['--partition='+gpu_partition,'--gres='+os.environ.get('FIMS_GPU_GRES','gpu:1'),
                       '--cpus-per-task=4','--mem=32G','--time='+time_limit,
                       '--array='+','.join(map(str,indices))+'%'+str(concurrency),
                       '--output='+str(root/'logs/fit-%A_%a.out'),
                       '--error='+str(root/'logs/fit-%A_%a.err')]
            else:
                opts+=['--partition='+cpu_partition,'--cpus-per-task=2','--mem=24G','--time=04:00:00',
                       '--output='+str(root/('logs/'+stage+'-%j.out')),'--error='+str(root/('logs/'+stage+'-%j.err'))]
            if account:opts+=['--account='+account]
            deps=[str(x) for x in deps if x]
            if deps:opts+=['--dependency=afterok:'+':'.join(dict.fromkeys(deps))]
            command=opts+[str(root/'hpc/stage.sh'),stage,str(root),str(capacity),a.python]
            q=subprocess.run(command,check=True,text=True,stdout=subprocess.PIPE)
            jid=q.stdout.strip().split(';')[0]
            if not jid.isdigit():raise ValueError('Unrecognized sbatch reply; inspect queue before retrying: '+q.stdout)
            journal['submissions'].append(dict(stage=stage,id=jid,indices=indices,command=command,
                submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            journal[stage]=jid;atomic_json(jp,journal)
            rows.append(dict(id=jid,name=name,state='SUBMITTED',reason=''))
            print('Submitted',stage,jid,flush=True);return jid

        # This report uses existing predictions only, independently of new fits.
        if not stage_done(root,'initial_report',digest):submit('initial_report')
        prep=None if stage_done(root,'prepare',digest) else submit('prepare')
        controls=None if all(checked_fit(root/'results/fits'/m/'seed20260924') for m in CONTROLS) else submit('controls',[prep])
        missing=[i for i,(m,s) in enumerate((m,s) for m in MODELS for s in SEEDS)
                 if not checked_fit(root/'results/fits'/m/('seed'+str(s)))]
        fit=submit('fit',[prep],missing) if missing else None
        if not stage_done(root,'final_report',digest):submit('final_report',[fit,controls])
        print('Study:',root)
        print('12 GNN fits; four deterministic context controls; saved-score CNN tradeoff report.')
        print('Each GPU fit checkpoints progress; rerun the same launcher to resume interrupted fits.')
        print('Accepted Slurm jobs continue after the laptop or SSH terminal closes.')

if __name__=='__main__':main()
