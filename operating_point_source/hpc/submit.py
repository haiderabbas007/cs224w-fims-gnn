#!/usr/bin/env python3
"""Submit only the read-only CPU report; never touch training jobs."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from common import atomic_json, checked_report, verify_package


def queue_rows():
    reply = subprocess.run(['squeue', '--noheader', '--user', str(os.getuid()),
                            '--format=%i|%j|%T|%R'], check=True,
                           universal_newlines=True, stdout=subprocess.PIPE)
    rows = []
    for line in reply.stdout.splitlines():
        if not line.strip():
            continue
        parts = [v.strip() for v in line.split('|', 3)]
        if len(parts) != 4:
            raise ValueError('Unexpected queue response; nothing submitted')
        rows.append(dict(zip(('id', 'name', 'state', 'reason'), parts)))
    return rows


def command(root, capacity, recovery, python, digest):
    return ['sbatch', '--parsable', '--nodes=1', '--ntasks=1',
            '--job-name=fims-op-'+digest[:8]+'-report', '--chdir='+str(root),
            '--partition=shared', '--cpus-per-task=2', '--mem=16G',
            '--time=04:00:00', '--output='+str(root/'logs/report-%j.out'),
            '--error='+str(root/'logs/report-%j.err'),
            str(root/'hpc/stage.sh'), str(root), str(capacity), str(recovery), str(python), digest]


def submit(root, capacity, recovery, python, digest, rows, journal, dry_run=False):
    name = 'fims-op-'+digest[:8]+'-report'
    active = [r for r in rows if r['name'] == name]
    if len(active) > 1:
        raise ValueError('Multiple active jobs with this exact report identity; inspect before submitting')
    if active:
        print('Existing report job retained: '+active[0]['id']+' ('+active[0]['state']+')')
        return active[0]['id']
    if checked_report(root, digest):
        print('Report COMPLETE; output hashes and ZIP verified. Use pull to download it.')
        return None
    cmd = command(root, capacity, recovery, python, digest)
    if dry_run:
        print('DRY RUN: '+' '.join(shlex.quote(x) for x in cmd))
        return None
    reply = subprocess.run(cmd, check=True, universal_newlines=True, stdout=subprocess.PIPE)
    jid = reply.stdout.strip().split(';')[0]
    if not re.fullmatch('[0-9]+', jid):
        raise ValueError('Unexpected sbatch response. Inspect status before retrying: '+reply.stdout)
    journal.setdefault('submissions', []).append({
        'id': jid, 'name': name, 'command': cmd,
        'submitted_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    atomic_json(root/'results/jobs.json', journal)
    print('Submitted read-only CPU report job: '+jid)
    print('22 completed fits; saved predictions only. No training, inference or simulation.')
    print('The 4-hour allowance is a limit, not an ETA. Queue delay is unknown.')
    print('The job continues after SSH and the laptop close.')
    return jid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root', required=True)
    p.add_argument('--capacity-root', required=True)
    p.add_argument('--recovery-root', required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--manifest-sha', required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    root, capacity, recovery = map(lambda x: Path(x).resolve(),
                                  (a.study_root, a.capacity_root, a.recovery_root))
    digest = verify_package(root, a.manifest_sha)
    if root.name != 'study_'+digest[:12]:
        raise ValueError('Report root does not match package identity')
    for source in (capacity, recovery):
        if not source.is_dir() or os.path.commonpath([str(root), str(source)]) in (str(root), str(source)):
            raise ValueError('Report and input paths must be separate existing studies')
    # Only the path is checked on the login node. The scientific interpreter is
    # intentionally first executed on a compute node.
    if not Path(a.python).is_absolute() or not Path(a.python).is_file():
        raise ValueError('Scientific Python path is absent')
    for name in ('results', 'logs'):
        dest = root/name
        if dest.is_symlink():
            raise ValueError('Unsafe output directory')
        dest.mkdir(exist_ok=True)
    lockpath = root/'results/submission.lock'
    if lockpath.is_symlink():
        raise ValueError('Unsafe submission lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = queue_rows()
        path = root/'results/jobs.json'
        if path.is_symlink():
            raise ValueError('Unsafe scheduler journal')
        journal = json.loads(path.read_text()) if path.exists() else {'package_sha256': digest, 'submissions': []}
        if journal.get('package_sha256') != digest:
            raise ValueError('Submission journal package identity differs')
        submit(root, capacity, recovery, a.python, digest, rows, journal, a.dry_run)
    print('Report study: '+str(root))


if __name__ == '__main__':
    main()
