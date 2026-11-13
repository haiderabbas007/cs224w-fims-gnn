#!/usr/bin/env bash
# Reuse one password/Duo authentication within each action.
set -euo pipefail
fims_action="${1:-run}"
fims_package="${2:?Verified package path required}"
fims_archive="${3:?Verified package archive required}"
fims_remote="mhabbas@koa.its.hawaii.edu"
fims_capacity="/mnt/lustre/koa/scratch/mhabbas/fims/cnn_capacity_v2/study_4cbf351c982f4fd6"
fims_recovery="/mnt/lustre/koa/scratch/mhabbas/fims/parallel_gnn_recovery_v1/study_3e6c80de0602"
fims_original="/mnt/lustre/koa/scratch/mhabbas/fims/parallel_v1/study_6b4fc507d9df"
fims_base="/mnt/lustre/koa/scratch/mhabbas/fims/operating_point_review_v1"
fims_windows="/mnt/c/Users/Haider Abbas/Downloads/FIMS Research"
case "$fims_action" in run|push|resume|status|remote-status|pull) ;; *) echo 'Use run, status, or pull.'; exit 2;; esac
for fims_cmd in ssh scp python3 sha256sum; do command -v "$fims_cmd" >/dev/null; done
fims_manifest_sha="$(sha256sum "$fims_package/PACKAGE_MANIFEST.json")"
fims_manifest_sha="${fims_manifest_sha%% *}"
[[ "$fims_manifest_sha" =~ ^[0-9a-f]{64}$ ]] || exit 2
fims_root="$fims_base/study_${fims_manifest_sha:0:12}"
fims_incoming="$fims_base/incoming/$fims_manifest_sha"
fims_socket_dir="$(mktemp -d /tmp/fims_operating_point_ssh_XXXXXXXX)"
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
    run|push|resume)
        fims_archive_sha="$(sha256sum "$fims_archive")"; fims_archive_sha="${fims_archive_sha%% *}"
        fims_archive_name="$(basename -- "$fims_archive")"
        [[ "$fims_archive_name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Unsafe archive name'; exit 1; }
        ssh "${fims_opts[@]}" "$fims_remote" "mkdir -p '$fims_incoming'"
        scp "${fims_opts[@]}" "$fims_archive" "$fims_package/local/unpack_remote.py" "$fims_remote:$fims_incoming/"
        ssh "${fims_opts[@]}" -t "$fims_remote" \
          "set -e; /usr/bin/python3 '$fims_incoming/unpack_remote.py' --archive '$fims_incoming/$fims_archive_name' --base '$fims_base' --expected-sha '$fims_archive_sha' --manifest-sha '$fims_manifest_sha'; /usr/bin/python3 '$fims_root/hpc/submit.py' --study-root '$fims_root' --capacity-root '$fims_capacity' --recovery-root '$fims_recovery' --python '$fims_original/.venv/bin/python' --manifest-sha '$fims_manifest_sha'"
        ;;
    status|remote-status)
        ssh "${fims_opts[@]}" -t "$fims_remote" \
          "if [ -d '$fims_root' ]; then /usr/bin/python3 '$fims_root/hpc/status.py' --study-root '$fims_root' --manifest-sha '$fims_manifest_sha'; else echo 'This report package has not been transferred yet. Run the launcher without status.'; fi"
        ;;
    pull)
        mkdir -p "$fims_windows"
        fims_tmp="$(mktemp "$fims_windows/.fims_operating_point_download_XXXXXXXX.zip")"
        scp "${fims_opts[@]}" "$fims_remote:$fims_root/results/report/fims_operating_point_review_v1.zip" "$fims_tmp"
        python3 "$fims_package/local/receive_report.py" "$fims_tmp" "$fims_windows" "$fims_manifest_sha"
        ;;
esac
