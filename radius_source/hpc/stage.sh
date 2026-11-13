#!/usr/bin/env bash
# Slurm relocates its script: use explicit absolute arguments throughout.
set -euo pipefail
fims_stage="${1:?Stage required}"
fims_root="${2:?Study root required}"
fims_original="${3:?Original study required}"
fims_recovery="${4:?Recovery study required}"
fims_python="${5:?Python required}"
cd "$fims_root"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$fims_python" - "$fims_root" "$fims_original" "$fims_recovery" <<'PY'
import json, sys
from pathlib import Path
import torch, numpy, scipy, matplotlib, torch_geometric
r, old, recovery = map(lambda p: Path(p).resolve(), sys.argv[1:])
sys.path.insert(0, str(r/'code'))
from radius_common import verify_package, verify_sources
verify_package(r)
receipt = verify_sources(r, old, recovery)
if json.loads((r/'results/source_receipt.json').read_text()) != receipt:
    raise SystemExit('Pinned launch sources changed.')
actual = dict(python=sys.version.split()[0], torch=str(torch.__version__),
              cuda_build=torch.version.cuda, torch_geometric=torch_geometric.__version__,
              numpy=numpy.__version__, matplotlib=matplotlib.__version__, scipy=scipy.__version__)
if json.loads((r/'results/submitted_environment.json').read_text()) != actual:
    raise SystemExit('Scientific environment changed since submission.')
print('RADIUS PACKAGE / FROZEN DEVELOPMENT SOURCES: PASS', flush=True)
PY
fims_radii=(1 2 4)
fims_seeds=(20260924 20260925 20260926)
case "$fims_stage" in
    fit)
        fims_index="${SLURM_ARRAY_TASK_ID:?Array task index required}"
        [[ "$fims_index" =~ ^[0-9]+$ ]] && [ "$fims_index" -lt 9 ] || { echo 'Invalid radius fit index'; exit 1; }
        fims_radius="${fims_radii[$((fims_index/3))]}"
        fims_seed="${fims_seeds[$((fims_index%3))]}"
        fims_command=("$fims_python" -u "$fims_root/code/train_radius.py" --study-root "$fims_root"
            --data "$fims_original/data" --radius "$fims_radius" --seed "$fims_seed" --device cuda)
        ;;
    controls)
        fims_index="${SLURM_ARRAY_TASK_ID:?Array task index required}"
        [[ "$fims_index" =~ ^[0-9]+$ ]] && [ "$fims_index" -lt 3 ] || { echo 'Invalid radius control index'; exit 1; }
        fims_radius="${fims_radii[$fims_index]}"
        fims_command=("$fims_python" -u "$fims_root/code/control_radius.py" --study-root "$fims_root"
            --data "$fims_original/data" --radius "$fims_radius")
        ;;
    report)
        # afterany makes a review possible even if one fit fails. The report
        # labels incomplete groups PARTIAL and never invents missing results.
        fims_command=("$fims_python" -u "$fims_root/code/report_radius.py" --study-root "$fims_root"
            --data "$fims_original/data" --allow-partial)
        ;;
    *) echo 'Unknown radius study stage'; exit 1;;
esac
fims_child=''
fims_stop=0
fims_forward_stop() {
    fims_stop=1
    if [ -n "$fims_child" ]; then
        if [ "$fims_stage" = fit ]; then
            kill -USR1 "$fims_child" 2>/dev/null || true
        else
            kill -TERM "$fims_child" 2>/dev/null || true
        fi
    fi
}
trap fims_forward_stop USR1 TERM INT
"${fims_command[@]}" &
fims_child=$!
# A trap interrupts wait; continue waiting so the child can save its state.
set +e
while true; do
    wait "$fims_child"
    fims_rc=$?
    if ! kill -0 "$fims_child" 2>/dev/null; then
        # If USR1 interrupted wait, the child can finish between that return
        # and kill -0. Reap its cached exit status rather than returning 138.
        wait "$fims_child"
        fims_rc=$?
        break
    fi
done
set -e
if [ "$fims_stop" -eq 1 ] && [ "$fims_rc" -eq 0 ]; then fims_rc=75; fi
if [ "$fims_rc" -eq 75 ]; then
    echo 'PAUSED: rerun the radius launcher to resume verified saved work.'
fi
exit "$fims_rc"
