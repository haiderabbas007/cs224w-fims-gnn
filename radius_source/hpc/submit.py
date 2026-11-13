#!/usr/bin/env python3
"""Submit missing radius-study work; never mutate the historical studies."""
from __future__ import annotations
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from radius_common import RADII, SEEDS, atomic_json, checked_fit, verify_package, verify_sources


def queue_rows():
    reply = subprocess.run(['squeue', '--noheader', '--user', str(os.getuid()),
                            '--format=%i|%j|%T|%R'], check=True, text=True,
                           stdout=subprocess.PIPE)
    rows = []
    for line in reply.stdout.splitlines():
        if not line.strip():
            continue
        fields = [v.strip() for v in line.split('|', 3)]
        if len(fields) != 4:
            raise ValueError('Unexpected scheduler output; no jobs submitted')
        rows.append(dict(zip(('id', 'name', 'state', 'reason'), fields)))
    return rows


def parent_id(value):
    value = re.split(r'[_+.]', str(value))[0]
    if not value.isdigit():
        raise ValueError('Unrecognized Slurm job ID')
    return value


def one_active(rows, name):
    matches = [r for r in rows if r['name'] == name]
    if not matches:
        return None
    parents = {parent_id(r['id']) for r in matches}
    if len(parents) != 1:
        raise ValueError('Multiple active jobs have exact study name '+name)
    return dict(id=next(iter(parents)), rows=matches)


def resource_options(stage, root, indices=None, *, gres='gpu:NV-A30:1', concurrency=9):
    root = Path(root)
    if stage == 'fit':
        if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < 9 for i in indices):
            raise ValueError('Invalid fit array indices')
        return ['--partition=gpu', '--gres='+gres, '--cpus-per-task=4', '--mem=32G',
                '--time=3-00:00:00', '--signal=B:USR1@300',
                '--array='+','.join(map(str, indices))+'%'+str(concurrency),
                '--output='+str(root/'logs/fit-%A_%a.out'),
                '--error='+str(root/'logs/fit-%A_%a.err')]
    if stage == 'controls':
        if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < 3 for i in indices):
            raise ValueError('Invalid control array indices')
        return ['--partition=shared', '--cpus-per-task=2', '--mem=24G', '--time=04:00:00',
                '--array='+','.join(map(str, indices))+'%3',
                '--output='+str(root/'logs/controls-%A_%a.out'),
                '--error='+str(root/'logs/controls-%A_%a.err')]
    if stage != 'report':
        raise ValueError('Unknown stage')
    return ['--partition=shared', '--cpus-per-task=2', '--mem=24G', '--time=04:00:00',
            '--output='+str(root/'logs/report-%j.out'), '--error='+str(root/'logs/report-%j.err')]


def normalized_dependencies(dependencies):
    output = []
    for mode, jid in dependencies:
        if jid is None:
            continue
        if mode != 'afterany' or not str(jid).isdigit():
            raise ValueError('Invalid report dependency')
        value = (mode, str(jid))
        if value not in output:
            output.append(value)
    return sorted(output)


class Submitter:
    def __init__(self, root, original, recovery, python, digest, rows, journal, *, dry_run=False):
        self.root, self.original, self.recovery = map(Path, (root, original, recovery))
        self.python, self.digest, self.rows = str(python), digest, list(rows)
        self.journal, self.dry_run = journal, dry_run
        self.prefix = 'fims-rad-'+digest[:8]+'-'
        self.gres = os.environ.get('FIMS_GPU_GRES', 'gpu:NV-A30:1')
        if not re.fullmatch(r'gpu(?::[A-Za-z0-9_.-]+)?:1', self.gres):
            raise ValueError('FIMS_GPU_GRES must request exactly one GPU')
        self.concurrency = int(os.environ.get('FIMS_RADIUS_CONCURRENCY', '9'))
        if not 1 <= self.concurrency <= 9:
            raise ValueError('FIMS_RADIUS_CONCURRENCY must be 1..9')

    def submit(self, stage, *, indices=None, dependencies=()):
        deps = normalized_dependencies(dependencies)
        name = self.prefix+stage
        if stage == 'report':
            # A retry may create a new training array. Never reuse a pending
            # report that depends on a different, older set of arrays.
            token = hashlib.sha256(json.dumps(deps).encode()).hexdigest()[:8]
            name += '-'+token
        active = one_active(self.rows, name)
        if active:
            if any('DependencyNeverSatisfied' in r['reason'] for r in active['rows']):
                recorded = any(x.get('stage') == stage and x.get('id') == active['id']
                               for x in self.journal.get('submissions', []))
                if stage != 'report' or not recorded or any(r['state'] != 'PENDING' for r in active['rows']):
                    raise ValueError('Blocked job requires inspection; no broad cancellation: '+active['id'])
                cmd = ['scancel', active['id']]
                if self.dry_run:
                    print('DRY RUN:', shlex.join(cmd))
                else:
                    subprocess.run(cmd, check=True)
                self.rows = [r for r in self.rows if r['name'] != name]
            else:
                print('Existing study job reused:', stage, active['id'], flush=True)
                return active['id']
        command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--job-name='+name,
                   '--chdir='+str(self.root)]
        command += resource_options(stage, self.root, indices, gres=self.gres,
                                    concurrency=self.concurrency)
        account = os.environ.get('FIMS_ACCOUNT', '')
        if account:
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', account):
                raise ValueError('Unsafe Slurm account identifier')
            command += ['--account='+account]
        if deps:
            command += ['--dependency='+','.join(mode+':'+jid for mode, jid in deps)]
        command += [str(self.root/'hpc/stage.sh'), stage, str(self.root), str(self.original),
                    str(self.recovery), self.python]
        if self.dry_run:
            print('DRY RUN:', shlex.join(command), flush=True)
            jid = str(99000000+len(self.journal.get('submissions', [])))
        else:
            reply = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
            jid = reply.stdout.strip().split(';')[0]
            if not jid.isdigit():
                raise ValueError('Unrecognized sbatch reply. Inspect the queue before retrying: '+reply.stdout)
        record = dict(stage=stage, id=jid, name=name, indices=indices, command=command,
                      dependencies=deps, submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        self.journal.setdefault('submissions', []).append(record)
        self.journal[stage] = jid
        if not self.dry_run:
            atomic_json(self.root/'results/jobs.json', self.journal)
        self.rows.append(dict(id=jid, name=name, state='SUBMITTED', reason=''))
        print('Planned' if self.dry_run else 'Submitted', stage, jid, flush=True)
        return jid


def checked_report(root):
    root = Path(root)/'results/report'
    p = root/'fims_radius_complete_v1.json'
    if not p.is_file():
        return False
    if p.is_symlink():
        raise ValueError('Unsafe report marker')
    marker = json.loads(p.read_text())
    if marker.get('status') != 'COMPLETE':
        return False
    if not isinstance(marker.get('outputs'), dict) or not marker['outputs']:
        raise ValueError('Invalid completed report inventory')
    for name, expected in marker['outputs'].items():
        rel = Path(name)
        q = root/rel
        if rel.is_absolute() or '..' in rel.parts or q.is_symlink() or not q.is_file() or not q.resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe report output')
        h = hashlib.sha256()
        with q.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError('Completed report output changed: '+name)
    archive = root/'fims_radius_sensitivity_review_v1.zip'
    return archive.is_file() and not archive.is_symlink() and archive.stat().st_size > 0


def plan(root, original, recovery, python, digest, rows, journal, *, dry_run=False):
    scheduler = Submitter(root, original, recovery, python, digest, rows, journal,
                          dry_run=dry_run)
    missing_fits = [i for i, (radius, seed) in enumerate((r, s) for r in RADII for s in SEEDS)
                    if not checked_fit(root/'results/fits'/('r'+str(radius))/'full1'/('seed'+str(seed)))]
    missing_controls = [i for i, radius in enumerate(RADII)
                        if not all(checked_fit(root/'results/controls'/('r'+str(radius))/model/'seed20260924')
                                   for model in ('counts8', 'shape8'))]
    print('Verified completed GNN fits:', 9-len(missing_fits), '/ 9', flush=True)
    print('Verified completed control radius groups:', 3-len(missing_controls), '/ 3', flush=True)
    fit_active = one_active(rows, scheduler.prefix+'fit')
    control_active = one_active(rows, scheduler.prefix+'controls')
    # Active jobs can still be saving final profile/status outputs even if their
    # completion markers are already present. Reports wait for those jobs too.
    fit = scheduler.submit('fit', indices=missing_fits) if missing_fits else (fit_active['id'] if fit_active else None)
    controls = scheduler.submit('controls', indices=missing_controls) if missing_controls else (control_active['id'] if control_active else None)
    if not missing_fits and not missing_controls and not fit and not controls and checked_report(root):
        print('All nine fits, six controls and the final report are COMPLETE and verified.', flush=True)
        return
    scheduler.submit('report', dependencies=[('afterany', fit), ('afterany', controls)])
    print('Study directory:', root, flush=True)
    print('Three radii: 1, 2, 4 mm; nine fresh one-layer GNN fits and six deterministic controls.', flush=True)
    print('GPU request:', scheduler.gres, '; maximum concurrent fits:', scheduler.concurrency, flush=True)
    print('72 hours is the allocation limit, not a runtime estimate. Queue delay is unknown.', flush=True)
    print('Accepted Slurm jobs continue when SSH or the laptop closes.', flush=True)
    print('Rerun after interruption to resume saved fits. An afterany report clearly labels incomplete results PARTIAL.', flush=True)
    if fit_active or control_active:
        print('Existing arrays were retained. If tasks from an earlier attempt remain incomplete after these finish, rerun to submit them.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root', type=Path, required=True)
    p.add_argument('--original-study', type=Path, required=True)
    p.add_argument('--recovery-study', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    root, original, recovery = (x.resolve() for x in (a.study_root, a.original_study, a.recovery_study))
    digest = verify_package(root)
    verify_sources(root, original, recovery)
    (root/'results').mkdir(exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    with (root/'results/submission.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = queue_rows()  # A failed scheduler query is never treated as an empty queue.
        jp = root/'results/jobs.json'
        if jp.is_symlink():
            raise ValueError('Unsafe scheduler journal')
        journal = json.loads(jp.read_text()) if jp.exists() else dict(package_sha256=digest, submissions=[])
        if journal.get('package_sha256') != digest:
            raise ValueError('Scheduler journal package identity changed')
        plan(root, original, recovery, a.python, digest, rows, copy.deepcopy(journal), dry_run=a.dry_run)


if __name__ == '__main__':
    main()
