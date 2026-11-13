#!/usr/bin/env python3
"""Submit only versioned recovery work; preserve original jobs and saved fits."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from common import MODELS, SEEDS, ORIGINAL_SHA, atomic_json, checked_fit, verify

def queue_rows():
    reply = subprocess.run(['squeue', '--noheader', '--user', str(os.getuid()),
                            '--format=%i|%j|%T|%R'], check=True, text=True, stdout=subprocess.PIPE)
    rows = []
    for line in reply.stdout.splitlines():
        if not line.strip(): continue
        parts = [v.strip() for v in line.split('|', 3)]
        if len(parts) != 4: raise ValueError('Unexpected scheduler output; no submissions made')
        rows.append(dict(id=parts[0], name=parts[1], state=parts[2], reason=parts[3]))
    return rows

def parent_id(value):
    parent = re.split(r'[_+.]', value)[0]
    if not parent.isdigit(): raise ValueError('Unrecognized Slurm ID: '+value)
    return parent

def one_active(rows, name):
    matches = [r for r in rows if r['name'] == name]
    if not matches: return None
    ids = {parent_id(r['id']) for r in matches}
    if len(ids) != 1: raise ValueError('Multiple active arrays/jobs with exact name '+name)
    return dict(id=next(iter(ids)), rows=matches)

def resource_options(stage, root, indices, *, gpu_gres, concurrency):
    if stage == 'fit':
        if not indices: raise ValueError('Empty training array')
        return ['--partition=gpu', '--gres='+gpu_gres, '--cpus-per-task=4', '--mem=32G',
                '--time=3-00:00:00', '--signal=B:USR1@300',
                '--array='+','.join(map(str, indices))+'%'+str(concurrency),
                '--output='+str(root/'logs/fit-%A_%a.out'), '--error='+str(root/'logs/fit-%A_%a.err')]
    return ['--partition=shared', '--cpus-per-task=2', '--mem=24G', '--time=04:00:00',
            '--output='+str(root/('logs/'+stage+'-%j.out')), '--error='+str(root/('logs/'+stage+'-%j.err'))]

class Submitter:
    def __init__(self, root, original, capacity, python, digest, rows, journal, *, dry_run=False):
        self.root, self.original, self.capacity = root, original, capacity
        self.python, self.digest, self.rows, self.journal = python, digest, list(rows), journal
        self.dry_run = dry_run
        self.prefix = 'fims-gr-'+digest[:8]+'-'
        self.gres = os.environ.get('FIMS_GPU_GRES', 'gpu:NV-A30:1')
        if not re.fullmatch(r'gpu(?::[A-Za-z0-9_.-]+)?:1', self.gres):
            raise ValueError('FIMS_GPU_GRES must request exactly one GPU')
        self.concurrency = int(os.environ.get('FIMS_GNN_CONCURRENCY', '12'))
        if not 1 <= self.concurrency <= 12: raise ValueError('Concurrency must be 1..12')

    def submit(self, stage, *, dependencies=(), indices=None):
        name = self.prefix+stage
        active = one_active(self.rows, name)
        if active:
            blocked = any('DependencyNeverSatisfied' in r['reason'] for r in active['rows'])
            if not blocked:
                print('Existing recovery job reused:', stage, active['id'], flush=True)
                return active['id']
            # Only our exact versioned report job, already recorded here, may be cancelled.
            recorded = any(r['stage'] == stage and r['id'] == active['id']
                           for r in self.journal.get('submissions', []))
            if stage != 'final_report' or not recorded or any(r['state'] != 'PENDING' for r in active['rows']):
                raise ValueError('Blocked job requires inspection; no broad cancellation: '+active['id'])
            cmd = ['scancel', active['id']]
            if self.dry_run: print('DRY RUN:', shlex.join(cmd))
            else: subprocess.run(cmd, check=True)
            self.rows = [r for r in self.rows if r['name'] != name]
        command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--job-name='+name,
                   '--chdir='+str(self.root)]
        command += resource_options(stage, self.root, indices, gpu_gres=self.gres, concurrency=self.concurrency)
        account = os.environ.get('FIMS_ACCOUNT', '')
        if account:
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', account): raise ValueError('Unsafe account identifier')
            command += ['--account='+account]
        deps = []
        for mode, jid in dependencies:
            if jid is None: continue
            if mode not in ('afterok', 'afterany') or not str(jid).isdigit():
                raise ValueError('Unsafe dependency')
            item = mode+':'+str(jid)
            if item not in deps: deps.append(item)
        if deps: command += ['--dependency='+','.join(deps)]
        command += [str(self.root/'hpc/stage.sh'), stage, str(self.root), str(self.original),
                    str(self.capacity), self.python]
        if self.dry_run:
            print('DRY RUN:', shlex.join(command), flush=True)
            jid = str(99000000+len(self.journal.get('submissions', [])))
        else:
            reply = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
            jid = reply.stdout.strip().split(';')[0]
            if not jid.isdigit(): raise ValueError('Unrecognized sbatch reply; inspect queue before retrying: '+reply.stdout)
        record = dict(stage=stage, id=jid, indices=indices, command=command,
                      submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        self.journal.setdefault('submissions', []).append(record)
        self.journal[stage] = jid
        if not self.dry_run: atomic_json(self.root/'results/jobs.json', self.journal)
        self.rows.append(dict(id=jid, name=name, state='SUBMITTED', reason=''))
        print('Submitted recovery' if not self.dry_run else 'Planned recovery', stage, jid, flush=True)
        return jid

def plan(root, original, capacity, python, digest, rows, journal, *, dry_run=False):
    scheduler = Submitter(root, original, capacity, python, digest, rows, journal, dry_run=dry_run)
    old = one_active(rows, 'fims-'+ORIGINAL_SHA[:8]+'-fit')
    if old:
        # The old array contains completed/failed tasks; afterok would deadlock the repair.
        print('Waiting for existing original GPU work without cancelling it:', old['id'], flush=True)
    missing = [i for i, (model, seed) in enumerate((m, s) for m in MODELS for s in SEEDS)
               if not checked_fit(root/'results/fits'/model/('seed'+str(seed)))]
    # The partial report copies already completed controls and reads completed outputs only.
    # It does not require any GPU job to succeed.
    scheduler.submit('partial_report')
    fit = scheduler.submit('fit', indices=missing,
                           dependencies=[('afterany', old['id'])] if old else []) if missing else None
    scheduler.submit('final_report', dependencies=[('afterok', fit)])
    print('Recovery root:', root)
    print('GPU request:', scheduler.gres, '; concurrency:', scheduler.concurrency, '; limit: 72 hours per allocation')
    print('The original study, completed controls and saved optimizer steps are preserved.')
    print('Closing SSH/laptop does not stop accepted Slurm jobs. Rerun recovery after an interruption to resume.')
    print('Queue delay and remaining fit time are not yet known; CUDA checks gate the memory repair before training.')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery-root', type=Path, required=True)
    p.add_argument('--original-study', type=Path, required=True)
    p.add_argument('--capacity-root', type=Path, required=True)
    p.add_argument('--python', required=True); p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    root, original, capacity = (x.resolve() for x in (a.recovery_root, a.original_study, a.capacity_root))
    digest = verify(root, original, capacity)
    (root/'results').mkdir(exist_ok=True); (root/'logs').mkdir(exist_ok=True)
    with (root/'results/submission.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = queue_rows()  # Fail closed before sbatch or scancel.
        jp = root/'results/jobs.json'
        journal = json.loads(jp.read_text()) if jp.exists() else dict(package_sha256=digest, submissions=[])
        if journal.get('package_sha256') != digest: raise ValueError('Recovery scheduler journal identity changed')
        plan(root, original, capacity, a.python, digest, rows, journal, dry_run=a.dry_run)

if __name__ == '__main__': main()
