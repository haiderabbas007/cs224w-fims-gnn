#!/usr/bin/env bash
# One normal password/Duo login, reused for this action.
set -euo pipefail
fims_action="${1:-push}"
fims_package="${2:?Verified local package required}"
fims_archive="${3:?Code archive required}"
fims_remote="mhabbas@koa.its.hawaii.edu"
fims_original="/mnt/lustre/koa/scratch/mhabbas/fims/parallel_v1/study_6b4fc507d9df"
fims_capacity="/mnt/lustre/koa/scratch/mhabbas/fims/cnn_capacity_v2/study_4cbf351c982f4fd6"
fims_base="/mnt/lustre/koa/scratch/mhabbas/fims/parallel_gnn_recovery_v1"
fims_windows="/mnt/c/Users/Haider Abbas/Downloads/FIMS Research"
for fims_cmd in ssh scp python3 sha256sum; do command -v "$fims_cmd" >/dev/null; done
fims_manifest_sha="$(sha256sum "$fims_package/PACKAGE_MANIFEST.json")"
fims_manifest_sha="${fims_manifest_sha%% *}"
fims_root="$fims_base/study_${fims_manifest_sha:0:12}"
fims_incoming="$fims_base/incoming/$fims_manifest_sha"
fims_socket_dir="$(mktemp -d /tmp/fims_recovery_ssh_XXXXXXXX)"
fims_socket="$fims_socket_dir/socket"
fims_opts=(-o "ControlPath=$fims_socket" -o ControlMaster=auto -o ControlPersist=600)
fims_cleanup() {
    ssh -o "ControlPath=$fims_socket" -O exit "$fims_remote" >/dev/null 2>&1 || true
    rm -rf -- "$fims_socket_dir"
}
trap fims_cleanup EXIT
echo "Connecting to $fims_remote. Complete your password and Duo prompts normally."
ssh "${fims_opts[@]}" -MNf "$fims_remote"
case "$fims_action" in
    push|resume)
        fims_archive_sha="$(sha256sum "$fims_archive")"; fims_archive_sha="${fims_archive_sha%% *}"
        ssh "${fims_opts[@]}" "$fims_remote" "mkdir -p '$fims_incoming'"
        scp "${fims_opts[@]}" "$fims_archive" "$fims_package/local/unpack_remote.py" "$fims_remote:$fims_incoming/"
        fims_archive_name="$(basename -- "$fims_archive")"
        [[ "$fims_archive_name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Unsafe archive name';exit 1; }
        ssh "${fims_opts[@]}" -t "$fims_remote" \
            "set -e; /usr/bin/python3 '$fims_incoming/unpack_remote.py' --archive '$fims_incoming/$fims_archive_name' --base '$fims_base' --expected-sha '$fims_archive_sha' --manifest-sha '$fims_manifest_sha'; srun --partition=sandbox --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=6G --time=00:10:00 /usr/bin/env -u SLURM_CPU_BIND -u SLURM_CPU_BIND_LIST -u SLURM_CPU_BIND_TYPE -u SLURM_CPU_BIND_VERBOSE /bin/bash '$fims_root/hpc/bootstrap.sh' '$fims_root' '$fims_original' '$fims_capacity' '$fims_original/.venv/bin/python'"
        echo 'Recovery submitted/verified. Accepted cluster jobs continue after SSH or the laptop closes.'
        ;;
    remote-status|status)
        ssh "${fims_opts[@]}" -t "$fims_remote" "/bin/bash '$fims_root/hpc/status.sh' '$fims_root' '$fims_original'"
        ;;
    pull)
        mkdir -p "$fims_windows"
        fims_tmp="$(mktemp "$fims_windows/.fims_recovery_download_XXXXXXXX.zip")"
        scp "${fims_opts[@]}" "$fims_remote:$fims_root/results/report/fims_parallel_recovery_review_v1.zip" "$fims_tmp"
        python3 - "$fims_tmp" "$fims_windows" <<'PY'
from pathlib import Path, PurePosixPath
import hashlib,json,os,shutil,sys,zipfile
p=Path(sys.argv[1]);folder=Path(sys.argv[2]);final=folder/'fims_parallel_recovery_review_v1.zip'
with zipfile.ZipFile(p) as z:
    names=z.namelist()
    if len(set(names))!=len(names) or z.testzip() is not None:raise SystemExit('Review ZIP integrity failure')
    for n in names:
        r=PurePosixPath(n)
        if r.is_absolute() or '..' in r.parts or '\\' in n:raise SystemExit('Unsafe ZIP path')
    status=json.loads(z.read('fims_parallel_report_status_v1.json'))
    for n,digest in status['outputs'].items():
        if hashlib.sha256(z.read(n)).hexdigest()!=digest:raise SystemExit('Report hash mismatch: '+n)
    for n in names:
        if '/' not in n and n.startswith('fims_parallel_') and n.endswith(('.txt','.json','.csv','.png')):
            target=folder/n
            if target.is_symlink():raise SystemExit('Unsafe output destination')
            temp=target.with_name('.'+target.name+'.partial')
            with z.open(n) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst)
            os.replace(temp,target)
if final.is_symlink():raise SystemExit('Unsafe ZIP destination')
os.replace(p,final)
print('Review status: '+status['status'])
print('Saved: '+str(final))
if status['status']!='COMPLETE':print('This is an interim report; unfinished seed groups are not final results.')
PY
        ;;
    *) echo 'Use push, resume, remote-status, or pull.';exit 1;;
esac
