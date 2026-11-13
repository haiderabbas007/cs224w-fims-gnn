#!/usr/bin/env bash
# Run in a compute allocation; Koa login-node home mounts are noexec.
set -euo pipefail
fims_root="${1:?Study root required}"
fims_original="${2:?Original study required}"
fims_recovery="${3:?Recovery study required}"
fims_python="${4:-$fims_original/.venv/bin/python}"
cd "$fims_root"
mkdir -p results logs
exec 9>results/bootstrap.lock
flock -x 9
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$fims_python" - "$fims_root" "$fims_original" "$fims_recovery" <<'PY'
import json, sys
from pathlib import Path
import torch, numpy, scipy, matplotlib, torch_geometric
r, old, recovery = map(lambda p: Path(p).resolve(), sys.argv[1:])
sys.path.insert(0, str(r/'code'))
from radius_common import verify_package, verify_sources, atomic_json
digest = verify_package(r)
receipt = verify_sources(r, old, recovery)
if torch.version.cuda is None:
    raise SystemExit('The original environment is not CUDA-built; no CPU fallback or environment changes.')
actual = dict(python=sys.version.split()[0], torch=str(torch.__version__),
              cuda_build=torch.version.cuda, torch_geometric=torch_geometric.__version__,
              numpy=numpy.__version__, matplotlib=matplotlib.__version__, scipy=scipy.__version__)
prior = json.loads((old/'results/submitted_environment.json').read_text())
if actual != prior:
    raise SystemExit('Original scientific environment changed; no environment was installed or modified.')
for filename, value in [
    ('source_receipt.json', receipt),
    ('submitted_environment.json', actual),
    ('paths.json', dict(original_study=str(old), recovery_study=str(recovery),
                       data=str(old/'data'), python=sys.executable)),
]:
    p = r/'results'/filename
    if p.exists() and json.loads(p.read_text()) != value:
        raise SystemExit('Saved launch configuration differs: '+filename)
    atomic_json(p, value, immutable=True)
print('SOURCE / ORIGINAL SCIENTIFIC ENVIRONMENT: VERIFIED', json.dumps(actual, sort_keys=True), flush=True)
PY
"$fims_python" "$fims_root/hpc/test_launchers.py"
# Training and radius tests run on each allocated GPU before its real-data fit.
"$fims_python" "$fims_root/hpc/submit.py" --study-root "$fims_root" \
    --original-study "$fims_original" --recovery-study "$fims_recovery" --python "$fims_python"
