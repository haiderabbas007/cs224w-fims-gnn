#!/usr/bin/env bash
# Run only in a compute allocation; the login-node home filesystem is noexec.
set -euo pipefail
fims_root="${1:?Usage: bootstrap.sh RECOVERY_ROOT ORIGINAL_STUDY CAPACITY_ROOT [PYTHON]}"
fims_original="${2:?Original study required}"
fims_capacity="${3:?Capacity reference required}"
fims_python="${4:-$fims_original/.venv/bin/python}"
cd "$fims_root"
mkdir -p results logs
exec 9>results/bootstrap.lock
flock -x 9
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$fims_python" "$fims_root/hpc/common.py" "$fims_root" --original-study "$fims_original" --capacity-root "$fims_capacity"
"$fims_python" - "$fims_root" "$fims_original" "$fims_capacity" <<'PY'
import json,sys
from pathlib import Path
import torch,numpy,scipy,matplotlib,torch_geometric
r,old,capacity=map(lambda p:Path(p).resolve(),sys.argv[1:])
sys.path.insert(0,str(r/'hpc'))
from common import atomic_json
if torch.version.cuda is None:raise SystemExit('Original environment is not a CUDA build; no CPU fallback')
actual=dict(python=sys.version.split()[0],torch=str(torch.__version__),cuda_build=torch.version.cuda,
            torch_geometric=torch_geometric.__version__,numpy=numpy.__version__,
            matplotlib=matplotlib.__version__,scipy=scipy.__version__)
prior=json.loads((old/'results/submitted_environment.json').read_text())
if actual!=prior:raise SystemExit('Original scientific environment changed; no environment installed or modified')
p=r/'results/submitted_environment.json'
if p.exists() and json.loads(p.read_text())!=actual:raise SystemExit('Recovery environment differs from its earlier submission')
atomic_json(p,actual)
paths=dict(original_study=str(old),capacity_root=str(capacity),python=sys.executable)
p=r/'results/paths.json'
if p.exists() and json.loads(p.read_text())!=paths:raise SystemExit('Recovery paths differ from prior submission')
atomic_json(p,paths)
print('REUSED ORIGINAL SCIENTIFIC ENVIRONMENT:',json.dumps(actual,sort_keys=True),flush=True)
PY
"$fims_python" "$fims_root/hpc/test_launchers.py"
"$fims_python" "$fims_root/hpc/submit.py" --recovery-root "$fims_root" \
    --original-study "$fims_original" --capacity-root "$fims_capacity" --python "$fims_python"
