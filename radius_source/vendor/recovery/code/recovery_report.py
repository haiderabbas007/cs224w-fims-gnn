#!/usr/bin/env python3
"""Reuse verified controls and report completed fits with explicit runtime lineage."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'hpc'))
sys.path.insert(0, str(ROOT/'vendor/original/code'))
import report_parallel_v1 as report

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):h.update(b)
    return h.hexdigest()

def verified_inventory(path):
    marker=path/'complete.json'
    if marker.is_symlink() or not marker.is_file():raise ValueError('Missing completed control: '+str(path))
    m=json.loads(marker.read_text())
    if m.get('status')!='COMPLETE' or not m.get('outputs'):raise ValueError('Invalid completed control')
    inventory=dict(m['outputs']);inventory['complete.json']=sha(marker)
    for name,digest in inventory.items():
        p=path/name
        if Path(name).name!=name or p.is_symlink() or not p.is_file() or sha(p)!=digest:
            raise ValueError('Completed control hash mismatch: '+str(p))
    return inventory

def copy_control(source,dest):
    inventory=verified_inventory(source)
    if dest.exists():
        if verified_inventory(dest)!=inventory:raise ValueError('Reused control differs: '+str(dest))
        return inventory
    dest.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.control_',dir=dest.parent))
    try:
        for name,digest in inventory.items():
            shutil.copyfile(source/name,stage/name)
            if sha(stage/name)!=digest:raise ValueError('Control changed during copy')
        if verified_inventory(source)!=inventory:raise ValueError('Source control changed during snapshot')
        os.rename(stage,dest)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return inventory

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-study',type=Path,required=True)
    p.add_argument('--recovery-root',type=Path,required=True)
    p.add_argument('--capacity-root',type=Path,required=True)
    a=p.parse_args();root=a.recovery_root.resolve();old=a.original_study.resolve()
    if root!=ROOT:raise ValueError('Recovery package/root mismatch')
    # Same verification used by the scheduler, and explicit immutable source pin.
    from common import verify_packages
    verify_packages(root,old,a.capacity_root.resolve())
    lockdir=root/'results/locks';lockdir.mkdir(parents=True,exist_ok=True)
    with (lockdir/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        reused={}
        for model in ('counts8','shape8','counts1','shape1'):
            rel=Path(model)/'seed20260924'
            reused[str(rel)]=copy_control(old/'results/fits'/rel,root/'results/fits'/rel)
        out=root/'results/report'
        result=report.build_report(a.capacity_root,root,out)
        note=('Runtime recovery: original data, graph neighborhoods, parameters, optimizer batches, '
              'loss, seeds and checkpoint selection are preserved. Unfinished fits resume from verified '
              'original checkpoints with chunked message evaluation and activation recomputation. '
              'Floating-point summation can differ at roundoff; bitwise identity to the old CUDA path '
              'is not claimed. Completed controls/results are reused unchanged. This is development '
              'validation, not independent confirmation. See runtime lineage for each fit.')
        result['limitations'].append(note)
        lineage={
            'schema':'fims_gnn_recovery_report_lineage_v1',
            'recovery_package_sha256':sha(root/'PACKAGE_MANIFEST.json'),
            'original_package_sha256':sha(old/'PACKAGE_MANIFEST.json'),
            'original_study':str(old),'reused_controls':reused,
            'runtime_note':note,'fit_receipts':{},
        }
        for path in sorted((root/'results/fits').rglob('*.json')):
            if not ('receipt' in path.name or path.parent.name=='runtime_sessions'):continue
            if path.is_symlink():raise ValueError('Unsafe lineage path')
            lineage['fit_receipts'][str(path.relative_to(root))]={'sha256':sha(path),'record':json.loads(path.read_text())}
        report.write_json(out/'fims_parallel_metrics_v1.json',result)
        summary=out/'fims_parallel_summary_v1.txt'
        report.atomic_bytes(summary,summary.read_bytes()+('\n'+note+'\n').encode())
        report.write_json(out/'fims_parallel_recovery_lineage_v1.json',lineage)
        status_path=out/'fims_parallel_report_status_v1.json'
        status=json.loads(status_path.read_text())
        for name in list(status['outputs']):status['outputs'][name]=sha(out/name)
        status['outputs']['fims_parallel_recovery_lineage_v1.json']=sha(out/'fims_parallel_recovery_lineage_v1.json')
        report.write_json(status_path,status)
        target=out/'fims_parallel_recovery_review_v1.zip'
        fd,tmp=tempfile.mkstemp(prefix='.recovery_zip_',dir=out);os.close(fd)
        try:
            with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
                for name in [*status['outputs'],'fims_parallel_report_status_v1.json']:z.write(out/name,name)
            with zipfile.ZipFile(tmp) as z:
                if z.testzip() is not None:raise ValueError('Recovery review ZIP integrity failure')
            os.replace(tmp,target)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        print('RECOVERY REPORT: '+status['status'],flush=True)
        print('Review ZIP: '+str(target),flush=True)

if __name__=='__main__':main()
