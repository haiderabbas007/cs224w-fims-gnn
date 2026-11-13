#!/usr/bin/env bash
# One password/Duo authentication, reused within each action.
set -euo pipefail
fims_action="${1:-push}"
fims_package="${2:?Verified local package required}"
fims_archive="${3:?Verified code archive required}"
fims_remote="mhabbas@koa.its.hawaii.edu"
fims_original="/mnt/lustre/koa/scratch/mhabbas/fims/parallel_v1/study_6b4fc507d9df"
fims_recovery="/mnt/lustre/koa/scratch/mhabbas/fims/parallel_gnn_recovery_v1/study_3e6c80de0602"
fims_base="/mnt/lustre/koa/scratch/mhabbas/fims/gnn_radius_sensitivity_v1"
fims_windows="/mnt/c/Users/Haider Abbas/Downloads/FIMS Research"
case "$fims_action" in push|resume|remote-status|status|pull) ;; *) echo 'Use push, resume, remote-status, or pull.'; exit 2;; esac
for fims_cmd in ssh scp python3 sha256sum; do command -v "$fims_cmd" >/dev/null; done
fims_manifest_sha="$(sha256sum "$fims_package/PACKAGE_MANIFEST.json")"
fims_manifest_sha="${fims_manifest_sha%% *}"
[[ "$fims_manifest_sha" =~ ^[0-9a-f]{64}$ ]] || exit 2
fims_root="$fims_base/study_${fims_manifest_sha:0:12}"
fims_incoming="$fims_base/incoming/$fims_manifest_sha"
fims_socket_dir="$(mktemp -d /tmp/fims_radius_ssh_XXXXXXXX)"
fims_socket="$fims_socket_dir/socket"
fims_opts=(-o "ControlPath=$fims_socket" -o ControlMaster=auto -o ControlPersist=600)
fims_cleanup() {
    ssh -o "ControlPath=$fims_socket" -O exit "$fims_remote" >/dev/null 2>&1 || true
    rm -rf -- "$fims_socket_dir"
}
trap fims_cleanup EXIT
echo "Connecting to $fims_remote. Complete password and Duo normally."
ssh "${fims_opts[@]}" -MNf "$fims_remote"
case "$fims_action" in
    push|resume)
        fims_archive_sha="$(sha256sum "$fims_archive")"; fims_archive_sha="${fims_archive_sha%% *}"
        ssh "${fims_opts[@]}" "$fims_remote" "mkdir -p '$fims_incoming'"
        scp "${fims_opts[@]}" "$fims_archive" "$fims_package/local/unpack_remote.py" "$fims_remote:$fims_incoming/"
        fims_archive_name="$(basename -- "$fims_archive")"
        [[ "$fims_archive_name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Unsafe archive name'; exit 1; }
        ssh "${fims_opts[@]}" -t "$fims_remote" \
          "set -e; /usr/bin/python3 '$fims_incoming/unpack_remote.py' --archive '$fims_incoming/$fims_archive_name' --base '$fims_base' --expected-sha '$fims_archive_sha' --manifest-sha '$fims_manifest_sha'; srun --partition=sandbox --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=6G --time=00:15:00 /usr/bin/env -u SLURM_CPU_BIND -u SLURM_CPU_BIND_LIST -u SLURM_CPU_BIND_TYPE -u SLURM_CPU_BIND_VERBOSE /bin/bash '$fims_root/hpc/bootstrap.sh' '$fims_root' '$fims_original' '$fims_recovery' '$fims_original/.venv/bin/python'"
        echo 'Koa submission/verification finished. Accepted batch jobs continue after this connection closes.'
        ;;
    remote-status|status)
        ssh "${fims_opts[@]}" -t "$fims_remote" "/bin/bash '$fims_root/hpc/status.sh' '$fims_root'"
        ;;
    pull)
        mkdir -p "$fims_windows"
        fims_tmp="$(mktemp "$fims_windows/.fims_radius_download_XXXXXXXX.zip")"
        scp "${fims_opts[@]}" "$fims_remote:$fims_root/results/report/fims_radius_sensitivity_review_v1.zip" "$fims_tmp"
        python3 "$fims_package/local/receive_report.py" "$fims_tmp" "$fims_windows"
        ;;
esac
