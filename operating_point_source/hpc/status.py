#!/usr/bin/env python3
"""Read-only live status and completed-output verification."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import subprocess

from common import ARCHIVE, MARKER, checked_report, verify_package


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root', required=True)
    p.add_argument('--manifest-sha', required=True)
    a = p.parse_args(); root = Path(a.study_root).resolve()
    digest = verify_package(root, a.manifest_sha)
    print('Operating-point report study: '+str(root), flush=True)
    jobname = 'fims-op-'+digest[:8]+'-report'
    reply = subprocess.run(['squeue', '--noheader', '--user', str(os.getuid()),
                            '--format=%i|%P|%j|%T|%M|%l|%R'], check=True,
                           universal_newlines=True, stdout=subprocess.PIPE)
    print('\nLIVE SCHEDULER:')
    active = []
    for line in reply.stdout.splitlines():
        if not line.strip():
            continue
        row = [x.strip() for x in line.split('|')]
        if len(row) != 7:
            raise ValueError('Unexpected squeue output')
        if row[2] == jobname:
            active.append(row)
    if active:
        print('JOBID | PARTITION | NAME | STATE | ELAPSED | LIMIT | REASON')
        for row in active:
            print(' | '.join(row))
    else:
        print('No active job for this report.')
    path = root/'results/jobs.json'
    if path.is_symlink():
        raise ValueError('Unsafe scheduler journal')
    if path.is_file():
        journal = json.loads(path.read_text())
        if journal.get('package_sha256') != digest:
            raise ValueError('Scheduler journal identity differs')
        ids = list(dict.fromkeys(str(x['id']) for x in journal['submissions']))
        if any(not re.fullmatch('[0-9]+', x) for x in ids):
            raise ValueError('Invalid saved job IDs')
        if ids:
            print('\nACCOUNTING:', flush=True)
            subprocess.run(['sacct', '-j', ','.join(ids),
                            '--format=JobID,State,Elapsed,ExitCode,MaxRSS'], check=True)
    else:
        print('No submission recorded yet.')
    logs = sorted((root/'logs').glob('report-*'), key=lambda p: p.stat().st_mtime)
    for path in logs[-6:]:
        if path.is_file() and not path.is_symlink() and path.stat().st_size:
            print('\n'+str(path))
            with path.open(errors='replace') as stream:
                print(''.join(deque(stream, maxlen=8)), end='')
    print('\nSAVED REPORT:')
    if checked_report(root, digest):
        print('COMPLETE: declared output hashes and ZIP verified.')
        print('Review ZIP: '+str(root/'results/report'/ARCHIVE))
        print('Use pull to download the report to FIMS Research.')
    elif active:
        print('No completed report yet; live scheduler state is shown above.')
    else:
        print('No completed report. If the last job failed, inspect its log before rerunning.')
    print('This report uses saved predictions; it does not wait for or change the radius jobs.')


if __name__ == '__main__':
    main()
