#!/usr/bin/env bash
set -euo pipefail
fims_root="${1:?Study root required}"
echo "Study: $fims_root"
if [ ! -d "$fims_root" ]; then echo 'Study has not been transferred yet.';exit 0;fi
fims_ids="$(/usr/bin/python3 - "$fims_root/results/jobs.json" <<'PY'
import json,os,sys
p=sys.argv[1]
if os.path.isfile(p):print(','.join(dict.fromkeys(str(r['id']) for r in json.load(open(p))['submissions'])))
PY
)"
squeue --user "$(id -un)" --format='%.22i %.12P %.35j %.12T %.12M %.12l %R'
if [ -n "$fims_ids" ]; then sacct -j "$fims_ids" --format=JobID,State,Elapsed,ExitCode,MaxRSS;fi
/usr/bin/python3 - "$fims_root" <<'PY'
import glob,json,os,sys
r=sys.argv[1]
for p in sorted(glob.glob(r+'/results/fits/*/seed*/status.json')):
 try:
  m=json.load(open(p));print(os.path.relpath(os.path.dirname(p),r),m.get('status'),
   'epoch index',m.get('epoch'),'next batch',m.get('next_batch'))
 except (OSError,ValueError) as e:print('Unreadable status:',p,str(e))
for p in sorted(glob.glob(r+'/results/fits/*/seed*/complete.json')):
 print(os.path.relpath(os.path.dirname(p),r)+': completion marker present (full hashes checked by report)')
PY
for fims_log in "$fims_root"/logs/*.out "$fims_root"/logs/*.err; do
    [ -s "$fims_log" ] || continue
    echo "$fims_log"
    tail -n 4 "$fims_log"
done
fims_review="$fims_root/results/report/fims_parallel_review_v1.zip"
if [ -s "$fims_review" ];then ls -lh "$fims_review";else echo 'Review ZIP is not available yet.';fi
