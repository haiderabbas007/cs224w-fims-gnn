#!/usr/bin/env bash
# Slurm copies this file into its spool; ALL paths come from explicit arguments.
set -euo pipefail
fims_stage="${1:?Stage required}"
fims_root="${2:?Study root required}"
fims_capacity="${3:?Capacity root required}"
fims_python="${4:?Python required}"
cd "$fims_root"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$fims_python" "$fims_root/hpc/verify_package.py" "$fims_root" --capacity-root "$fims_capacity"
mkdir -p results/stages results/locks
if [ "$fims_stage" != fit ]; then
    exec 9>"results/locks/$fims_stage.lock"
    flock -x 9
fi
case "$fims_stage" in
    prepare)
        "$fims_python" -u "$fims_root/code/prepare_site_data_v1.py" --capacity-root "$fims_capacity" --output-dir "$fims_root/data"
        ;;
    controls)
        "$fims_python" -u "$fims_root/code/fit_controls_v1.py" --data "$fims_root/data" --output-dir "$fims_root/results/fits"
        ;;
    initial_report|final_report)
        # Both reports share an output directory, so serialize their writes.
        exec 8>results/locks/report-output.lock
        flock -x 8
        if [ "$fims_stage" = initial_report ] && [ -s results/stages/final_report.json ]; then
            echo 'Final report is already saved; initial report will not overwrite it.'
        else
            "$fims_python" -u "$fims_root/code/report_parallel_v1.py" --capacity-root "$fims_capacity" --study-root "$fims_root"
        fi
        ;;
    fit)
        fims_index="${SLURM_ARRAY_TASK_ID:?Slurm array task index required}"
        [[ "$fims_index" =~ ^[0-9]+$ ]] && [ "$fims_index" -lt 12 ] || { echo 'Invalid array index';exit 1; }
        fims_models=(full2 full1 no_xy2 current2)
        fims_seeds=(20260924 20260925 20260926)
        fims_model="${fims_models[$((fims_index / 3))]}"
        fims_seed="${fims_seeds[$((fims_index % 3))]}"
        "$fims_python" - "$fims_root" <<'PY'
import sys
sys.path.insert(0,sys.argv[1]+'/code')
import numpy as np,torch
from graph_models_v1 import QueryGraphBuilder,GeometricMPNN,batch_graphs
if not torch.cuda.is_available():raise SystemExit('CUDA allocation unavailable; no CPU fallback')
print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__,'CUDA:',torch.version.cuda,flush=True)
g=dict(x_mm=np.array([0.,.2,.4]),y_mm=np.zeros(3),time_bin=np.array([0,0,1]),side=np.zeros(3,dtype=np.int64))
m=GeometricMPNN('full2').to('cuda')
b=batch_graphs([QueryGraphBuilder(g,'full2').build(2)],'cuda')
y=m(b);assert y.shape==(1,) and bool(torch.isfinite(y).all())
y.sum().backward();assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
print('CUDA graph forward/backward: PASS',flush=True)
PY
        "$fims_python" "$fims_root/code/test_graph_models_v1.py"
        "$fims_python" -u "$fims_root/code/train_site_gnn_v1.py" --data "$fims_root/data" \
            --output-dir "$fims_root/results/fits" --model "$fims_model" --seed "$fims_seed" --device cuda
        ;;
    *) echo 'Unknown stage';exit 1;;
esac
if [ "$fims_stage" != fit ]; then
    "$fims_python" "$fims_root/hpc/submit.py" --study "$fims_root" --capacity-root "$fims_capacity" --python "$fims_python" --mark-stage "$fims_stage"
fi
