#!/usr/bin/env bash
set -euo pipefail
fims_root="${1:?Recovery study root required}"
echo "Recovery study: $fims_root"
if [ ! -d "$fims_root" ];then echo 'Recovery package has not been transferred yet.';exit 0;fi
/usr/bin/python3 - "$fims_root" <<'PY'
import json,os,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'hpc'))
from common import verify,sha
digest=verify(r)
p=r/'results/jobs.json'
if not p.is_file():print('No recovery job submission journal yet.');raise SystemExit(0)
j=json.loads(p.read_text())
if j.get('package_sha256')!=digest:raise SystemExit('Submission journal identity differs')
ids=list(dict.fromkeys(str(x['id']) for x in j['submissions']))
if any(not x.isdigit() for x in ids):raise SystemExit('Invalid saved job IDs')
print('\nLIVE SCHEDULER (authoritative):',flush=True)
q=subprocess.run(['squeue','--user',str(os.getuid()),'--format=%i|%P|%j|%T|%M|%l|%R'],check=True,text=True,stdout=subprocess.PIPE)
print('JOBID | PARTITION | NAME | STATE | ELAPSED | LIMIT | REASON')
for line in q.stdout.splitlines()[1:]:
 fields=line.split('|')
 if len(fields)!=7:raise SystemExit('Unexpected squeue status output')
 if fields[2].startswith('fims-gr-'+digest[:8]+'-') or fields[2]=='fims-6b4fc507-fit':print(' | '.join(fields))
print('\nACCOUNTING (authoritative):',flush=True)
subprocess.run(['sacct','-j',','.join(ids),'--format=JobID,State,Elapsed,ExitCode,MaxRSS'],check=True)
print('\nSAVED CHECKPOINTS (cached progress; RUNNING here is not proof of a live process):',flush=True)
for p in sorted((r/'results/fits').glob('*/seed*/status.json')):
 try:
  v=json.loads(p.read_text())
  print(p.parent.relative_to(r),v.get('status'),'epoch index',v.get('epoch'),'next batch',v.get('next_batch'))
 except (OSError,ValueError) as exc:print('Unreadable saved progress:',p,str(exc))
for p in sorted((r/'results/fits').glob('*/seed*/complete.json')):
 print(p.parent.relative_to(r),': completion marker present; report validates full hashes')
for p in sorted((r/'logs').glob('*')):
 if p.is_file() and p.suffix in ('.out','.err') and p.stat().st_size:
  print('\n'+str(p))
  with p.open(errors='replace') as f:
   from collections import deque
   print(''.join(deque(f,maxlen=5)),end='')
p=r/'results/report/fims_parallel_recovery_review_v1.zip'
if p.is_file():
 print('\nReview ZIP:',p,'bytes=',p.stat().st_size)
 print('A review can be PARTIAL: inspect its summary for completed-fit coverage.')
else:print('\nReview ZIP is not available yet.')
PY
