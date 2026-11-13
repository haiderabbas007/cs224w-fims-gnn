#!/usr/bin/env bash
set -euo pipefail
fims_root="${1:?Study root required}"
echo "Radius study: $fims_root"
if [ ! -d "$fims_root" ]; then echo 'This radius package has not been transferred yet.'; exit 0; fi
/usr/bin/python3 - "$fims_root" <<'PY'
import hashlib, json, os, re, subprocess, sys
from collections import deque
from pathlib import Path, PurePosixPath
r=Path(sys.argv[1]).resolve()
RADII=(1,2,4)
SEEDS=(20260924,20260925,20260926)
# The login-node system Python can be older than the scientific environment.
# This status-only verifier uses stdlib APIs available in Python 3.6; all
# scientific source, environment and output checks still run on compute nodes.
def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):value.update(block)
    return value.hexdigest()
def safe_file(name):
    p=PurePosixPath(name)
    if (p.is_absolute() or '..' in p.parts or '\\' in name or
        str(p)!=name or name in ('','.')):
        raise SystemExit('Unsafe package inventory path')
    target=r/name
    if (target.is_symlink() or not target.is_file() or
        os.path.commonpath([str(r),str(target.resolve())])!=str(r)):
        raise SystemExit('Missing/unsafe package file: '+name)
    return target
manifest=safe_file('PACKAGE_MANIFEST.json')
digest=sha(manifest)
inventory=json.loads(manifest.read_text())
if inventory.get('schema')!='fims_radius_sensitivity_package_v1' or not isinstance(inventory.get('files'),dict) or not inventory['files']:
    raise SystemExit('Invalid radius package inventory')
for name,value in inventory['files'].items():
    if sha(safe_file(name))!=value:raise SystemExit('Package file changed: '+name)
p=r/'results/jobs.json'
if not p.is_file():
    print('No radius job submission journal yet.');raise SystemExit(0)
if p.is_symlink():raise SystemExit('Unsafe scheduler journal')
j=json.loads(p.read_text())
if j.get('package_sha256')!=digest:raise SystemExit('Submission journal identity differs')
ids=list(dict.fromkeys(str(x['id']) for x in j['submissions']))
if any(not x.isdigit() for x in ids):raise SystemExit('Invalid saved job IDs')
print('\nLIVE SCHEDULER (authoritative):',flush=True)
q=subprocess.run(['squeue','--noheader','--user',str(os.getuid()),
                  '--format=%i|%P|%j|%T|%M|%l|%R'],check=True,universal_newlines=True,stdout=subprocess.PIPE)
print('JOBID | PARTITION | NAME | STATE | ELAPSED | LIMIT | REASON')
visible=0
for line in q.stdout.splitlines():
    fields=[x.strip() for x in line.split('|')]
    if len(fields)!=7:raise SystemExit('Unexpected squeue status output')
    if fields[2].startswith('fims-rad-'+digest[:8]+'-'):
        print(' | '.join(fields));visible+=1
if not visible:print('No active radius jobs in the queue.')
if ids:
    print('\nACCOUNTING (authoritative):',flush=True)
    subprocess.run(['sacct','-j',','.join(ids),'--format=JobID,State,Elapsed,ExitCode,MaxRSS'],check=True)
print('\nSAVED PROGRESS (a RUNNING file is not proof of a live job):',flush=True)
fits=controls=0
for radius in RADII:
    for seed in SEEDS:
        folder=r/'results/fits'/('r'+str(radius))/'full1'/('seed'+str(seed))
        marker=folder/'complete.json'
        if marker.is_file():
            fits+=1
            print('GNN radius=%s mm seed=%s: completion marker present' % (radius,seed))
        elif (folder/'status.json').is_file():
            try:
                v=json.loads((folder/'status.json').read_text())
                print('GNN radius=%s mm seed=%s: %s; epoch index=%s; next batch=%s' %
                      (radius,seed,v.get('status'),v.get('epoch'),v.get('next_batch')))
            except (OSError, ValueError) as exc:print('Cannot read saved progress:',folder,str(exc))
        else:print('GNN radius=%s mm seed=%s: not started' % (radius,seed))
    for model in ('counts8','shape8'):
        folder=r/'results/controls'/('r'+str(radius))/model/'seed20260924'
        if (folder/'complete.json').is_file():
            controls+=1;state='completion marker present'
        elif (folder/'status.json').is_file():
            try:state=json.loads((folder/'status.json').read_text()).get('status','unknown')
            except (OSError,ValueError):state='unreadable status'
        else:state='not started'
        print('Control radius=%s mm %s: %s' % (radius,model,state))
print('Completion markers: %s/9 GNN fits; %s/6 controls. Report/submit verify full hashes.'%(fits,controls))
logs=sorted((r/'logs').glob('*'),key=lambda p:p.stat().st_mtime if p.is_file() else 0)
for p in logs[-18:]:
    if p.is_file() and p.suffix in ('.out','.err') and p.stat().st_size:
        print('\n'+str(p))
        with p.open(errors='replace') as stream:print(''.join(deque(stream,maxlen=4)),end='')
marker=r/'results/report/fims_radius_complete_v1.json'
if marker.is_file():
    m=json.loads(marker.read_text());print('\nReport status:',m.get('status'))
archive=r/'results/report/fims_radius_sensitivity_review_v1.zip'
if archive.is_file():
    print('Review ZIP:',archive,'bytes=',archive.stat().st_size)
    print('A review may be PARTIAL; inspect its summary before interpreting comparisons.')
else:print('\nReview ZIP not available yet.')
print('Queue delay has no reliable ETA. The 72-hour fit allowance is not an estimated duration.')
PY
