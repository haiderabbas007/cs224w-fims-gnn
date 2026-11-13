#!/usr/bin/env python3
"""Local WSL launch/status; cluster training is independent of readout replay."""
import argparse, fcntl, json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def alive(pidfile, script, output):
    try:
        pid = int(pidfile.read_text().strip())
        cmd = Path('/proc')/str(pid)/'cmdline'
        args = cmd.read_bytes().split(b'\0')
        return str(script).encode() in args and str(output).encode() in args
    except (OSError, ValueError):
        return False

def status():
    out=ROOT/'results/continuous'; log=ROOT/'results/continuous_export.log'
    print('Local continuous-arrival output: '+str(out))
    if (out/'run_status.txt').is_file(): print((out/'run_status.txt').read_text().strip())
    else: print('NOT STARTED')
    print('Worker process: '+('ACTIVE' if alive(out/'worker.pid',ROOT/'hit/export_continuous_hits_v1.py',out) else 'not detected; saved status may be stale after WSL restart'))
    if log.is_file():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-12000)); lines=f.read().decode('utf-8',errors='replace').splitlines()
        print('\n'.join(lines[-20:]))

def pick_python():
    override=os.environ.get('FIMS_HIT_PYTHON')
    homes=[Path.home()/'miniforge3',Path.home()/'miniconda3',Path.home()/'anaconda3']
    candidates=[Path(override)] if override else [p/'envs/g4tpc/bin/python' for p in homes]
    errors=[]
    check='import numpy,uproot; assert numpy.__version__=="2.5.0", "Need original NumPy 2.5.0; found "+numpy.__version__; print("Readout dependencies: PASS; NumPy "+numpy.__version__+"; uproot "+uproot.__version__)'
    for p in candidates:
        if not p.is_file():continue
        try:
            r=subprocess.run([str(p),'-c',check],capture_output=True,text=True,timeout=60)
            if r.returncode==0:
                print(r.stdout.strip());return p
            detail=(r.stderr.strip() or r.stdout.strip() or 'exit code '+str(r.returncode)).splitlines()[-1]
            errors.append(str(p)+': '+detail)
        except (OSError, subprocess.TimeoutExpired) as e:errors.append(str(p)+': '+str(e))
    raise RuntimeError('Original readout interpreter unavailable. Set FIMS_HIT_PYTHON to its absolute path.\n'+'\n'.join(errors))

def start_hit():
    source=Path(os.environ.get('FIMS_HIT_SOURCE',str(Path.home()/'TPC-simulation/build/fims_expanded_transport_v2_bjaqa4o_/results'))).resolve()
    if not source.is_dir():raise RuntimeError('Native ROOT production folder missing: '+str(source))
    result=ROOT/'results';out=result/'continuous';result.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    script=ROOT/'hit/export_continuous_hits_v1.py';pidfile=out/'worker.pid'
    with (result/'continuous_launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if alive(pidfile,script,out):
            print('Continuous-arrival worker already active; no duplicate started.');return
        # The exporter owns worker.lock; only probe it here, and release it before launch.
        with (out/'worker.lock').open('a') as worker:
            try:fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print('Continuous-arrival worker lock is held; no duplicate started.');return
        py=pick_python()
        command=[str(py),'-u',str(script),'--source-dir',str(source),'--output-dir',str(out)]
        if os.environ.get('FIMS_WINDOWS_DIR'):command+=['--copy-dir',os.environ['FIMS_WINDOWS_DIR']]
        log=result/'continuous_export.log'
        with log.open('ab',buffering=0) as f:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        fd,tmp=tempfile.mkstemp(dir=out,prefix='.pid_')
        with os.fdopen(fd,'w') as f:f.write(str(proc.pid)+'\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,pidfile)
        print('Started/resumed continuous-arrival export. PID: '+str(proc.pid))
        print('Log: '+str(log))
        print('Read-only ROOT replay; no new simulation or local training.')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','hit-export','hit-status','push','remote-status','pull','install']);a=p.parse_args()
    subprocess.run([sys.executable,str(ROOT/'hpc/verify_package.py'),str(ROOT)],check=True)
    if a.action=='install':print('Installed package: '+str(ROOT));return
    if a.action=='hit-status':status();return
    if a.action in ('start','hit-export'):
        try:start_hit()
        except Exception as e:
            print('Continuous-arrival export did not start: '+str(e),file=sys.stderr)
            if a.action=='hit-export':raise SystemExit(1)
            print('Proceeding with the independent Koa site study.',flush=True)
        if a.action=='hit-export':return
    remote='push' if a.action=='start' else a.action
    subprocess.run(['bash',str(ROOT/'local/koa.sh'),remote,str(ROOT)],check=True)
if __name__=='__main__':main()
