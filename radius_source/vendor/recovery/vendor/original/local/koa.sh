#!/usr/bin/env bash
# WSL transfer helper. One password/Duo authentication is reused for this action.
set -euo pipefail
fims_action="${1:?Usage: koa.sh push|remote-status|pull PACKAGE_ROOT [ARCHIVE]}"
fims_package="${2:?Local verified package root required}"
fims_archive="${3:-$fims_package/FIMS_Parallel_v1.tar.gz}"
fims_windows="${FIMS_WINDOWS_DIR:-/mnt/c/Users/Haider Abbas/Downloads/FIMS Research}"
fims_user="${FIMS_REMOTE_USER:-mhabbas}"
fims_host="${FIMS_REMOTE_HOST:-koa.its.hawaii.edu}"
fims_base="${FIMS_REMOTE_BASE:-/mnt/lustre/koa/scratch/$fims_user/fims/parallel_v1}"
fims_capacity="${FIMS_CAPACITY_ROOT:-/mnt/lustre/koa/scratch/$fims_user/fims/cnn_capacity_v2/study_4cbf351c982f4fd6}"
fims_base_python="${FIMS_BASE_PYTHON:-/home/$fims_user/.conda/envs/fm4npp/bin/python}"
[[ "$fims_user" =~ ^[A-Za-z0-9_-]+$ && "$fims_host" =~ ^[A-Za-z0-9.-]+$ ]] || { echo 'Invalid remote account';exit 1; }
for fims_value in "$fims_base" "$fims_capacity" "$fims_base_python";do
    [[ "$fims_value" =~ ^/[A-Za-z0-9_./-]+$ && "$fims_value" != *'/../'* ]] || { echo 'Invalid remote path';exit 1; }
done
for fims_cmd in ssh scp python3 sha256sum;do command -v "$fims_cmd" >/dev/null;done
python3 "$fims_package/hpc/verify_package.py" "$fims_package"
fims_manifest_sha="$(sha256sum "$fims_package/PACKAGE_MANIFEST.json")";fims_manifest_sha="${fims_manifest_sha%% *}"
fims_root="$fims_base/study_${fims_manifest_sha:0:12}"
fims_incoming="$fims_base/incoming/$fims_manifest_sha"
fims_remote="$fims_user@$fims_host"
fims_socket_dir="$(mktemp -d /tmp/fims_ssh_XXXXXXXX)"
fims_socket="$fims_socket_dir/socket"
fims_ssh_opts=(-o "ControlPath=$fims_socket" -o ControlMaster=auto -o ControlPersist=600)
fims_cleanup() {
    ssh -o "ControlPath=$fims_socket" -O exit "$fims_remote" >/dev/null 2>&1 || true
    rm -rf -- "$fims_socket_dir"
}
trap fims_cleanup EXIT
echo "Connecting to $fims_remote. Complete password and Duo prompts normally."
ssh "${fims_ssh_opts[@]}" -MNf "$fims_remote"
case "$fims_action" in
    push)
        test -s "$fims_archive" || { echo "Code archive not found: $fims_archive";exit 1; }
        fims_archive_sha="$(sha256sum "$fims_archive")";fims_archive_sha="${fims_archive_sha%% *}"
        ssh "${fims_ssh_opts[@]}" "$fims_remote" "mkdir -p '$fims_incoming'"
        scp "${fims_ssh_opts[@]}" "$fims_archive" "$fims_package/local/unpack_remote.py" "$fims_remote:$fims_incoming/"
        fims_archive_name="$(basename -- "$fims_archive")"
        [[ "$fims_archive_name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Unsafe archive filename';exit 1; }
        ssh "${fims_ssh_opts[@]}" -t "$fims_remote" \
            "set -e; /usr/bin/python3 '$fims_incoming/unpack_remote.py' --archive '$fims_incoming/$fims_archive_name' --base '$fims_base' --expected-sha '$fims_archive_sha' --manifest-sha '$fims_manifest_sha'; srun --partition=sandbox --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=6G --time=00:10:00 /usr/bin/env -u SLURM_CPU_BIND -u SLURM_CPU_BIND_LIST -u SLURM_CPU_BIND_TYPE -u SLURM_CPU_BIND_VERBOSE /bin/bash '$fims_root/hpc/bootstrap.sh' '$fims_root' '$fims_capacity' '$fims_base_python'"
        echo 'Transfer/bootstrap finished. Accepted cluster jobs continue after this terminal closes.'
        ;;
    remote-status)
        ssh "${fims_ssh_opts[@]}" -t "$fims_remote" "/bin/bash '$fims_root/hpc/status.sh' '$fims_root'"
        ;;
    pull)
        mkdir -p "$fims_windows"
        fims_tmp="$(mktemp "$fims_windows/.fims_parallel_review_XXXXXXXX.zip")"
        scp "${fims_ssh_opts[@]}" "$fims_remote:$fims_root/results/report/fims_parallel_review_v1.zip" "$fims_tmp"
        python3 - "$fims_tmp" "$fims_windows" <<'PY'
import os,shutil,sys,zipfile
from pathlib import Path,PurePosixPath
p=Path(sys.argv[1]);dest=Path(sys.argv[2]);final=dest/'fims_parallel_review_v1.zip'
with zipfile.ZipFile(p) as z:
    if z.testzip() is not None:raise SystemExit('Downloaded ZIP integrity failure')
    names=z.namelist()
    if len(set(names))!=len(names):raise SystemExit('Duplicate paths in review ZIP')
    for n in names:
        rel=PurePosixPath(n)
        if rel.is_absolute() or '..' in rel.parts or '\\' in n:raise SystemExit('Unsafe review ZIP path')
    # Extract only the human-readable top-level report artifacts.
    for n in names:
        if '/' not in n and n.startswith('fims_parallel_') and n.endswith(('.txt','.png','.csv','.json')):
            target=dest/n
            if target.is_symlink():raise SystemExit('Unsafe report destination')
            temp=target.with_name('.'+target.name+'.partial')
            with z.open(n) as src,temp.open('wb') as out:shutil.copyfileobj(src,out)
            os.replace(temp,target)
if final.is_symlink():raise SystemExit('Unsafe ZIP destination')
os.replace(p,final)
print('Downloaded review and report artifacts to '+str(dest))
PY
        ;;
    *) echo 'Use push, remote-status or pull.';exit 1;;
esac
