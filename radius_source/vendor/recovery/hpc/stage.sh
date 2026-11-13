#!/usr/bin/env bash
# Slurm spools this file elsewhere: all paths use the explicit study argument.
set -euo pipefail
fims_stage="${1:?Stage required}"
fims_root="${2:?Recovery root required}"
fims_original="${3:?Original study required}"
fims_capacity="${4:?Capacity reference required}"
fims_python="${5:?Python required}"
cd "$fims_root"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$fims_root/vendor/original/code${PYTHONPATH:+:$PYTHONPATH}"
"$fims_python" "$fims_root/hpc/common.py" "$fims_root" --original-study "$fims_original" --capacity-root "$fims_capacity"
mkdir -p results/locks
case "$fims_stage" in
    fit)
        fims_index="${SLURM_ARRAY_TASK_ID:?Array index required}"
        [[ "$fims_index" =~ ^[0-9]+$ ]] && [ "$fims_index" -lt 12 ] || { echo 'Invalid recovery array index';exit 1; }
        fims_models=(full2 full1 no_xy2 current2)
        fims_seeds=(20260924 20260925 20260926)
        fims_model="${fims_models[$((fims_index/3))]}"
        fims_seed="${fims_seeds[$((fims_index%3))]}"
        # Gate the changed implementation on invented graphs on this actual GPU.
        "$fims_python" "$fims_root/code/test_memory_model.py" --cuda-stress
        fims_child=''
        fims_stop=0
        fims_forward_stop() {
            fims_stop=1
            if [ -n "$fims_child" ]; then kill -USR1 "$fims_child" 2>/dev/null || true;fi
        }
        trap fims_forward_stop USR1 TERM INT
        "$fims_python" -u "$fims_root/code/recovery_train.py" --original-study "$fims_original" \
            --recovery-root "$fims_root" --model "$fims_model" --seed "$fims_seed" --device cuda &
        fims_child=$!
        # wait can be interrupted by the signal trap before the child has saved.
        set +e
        while true;do
            wait "$fims_child"
            fims_rc=$?
            if ! kill -0 "$fims_child" 2>/dev/null;then break;fi
        done
        set -e
        if [ "$fims_stop" -eq 1 ] && [ "$fims_rc" -eq 0 ];then fims_rc=75;fi
        if [ "$fims_rc" -eq 75 ];then echo 'CHECKPOINT PAUSED: rerun this recovery launcher to resume saved optimizer steps.';fi
        exit "$fims_rc"
        ;;
    partial_report|final_report)
        # The Python report wrapper owns report.lock. Holding it here would
        # deadlock its independent file descriptor in the child process.
        if [ "$fims_stage" = final_report ];then
            "$fims_python" - "$fims_root" <<'PY'
from pathlib import Path
import sys
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'hpc'))
from common import MODELS,SEEDS,checked_fit
for m in MODELS:
 for s in SEEDS:
  if not checked_fit(r/'results/fits'/m/('seed'+str(s))):raise SystemExit('Final report requires all 12 verified fits')
PY
        fi
        "$fims_python" -u "$fims_root/code/recovery_report.py" --original-study "$fims_original" \
            --recovery-root "$fims_root" --capacity-root "$fims_capacity"
        ;;
    *) echo 'Unknown recovery stage';exit 1;;
esac
