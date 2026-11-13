#!/usr/bin/env bash
# Slurm runs a spool copy: every source path comes from an explicit argument.
set -euo pipefail
fims_root="${1:?Study root required}"
fims_capacity="${2:?Capacity study required}"
fims_recovery="${3:?Recovery study required}"
fims_python="${4:?Compute-node Python required}"
fims_digest="${5:?Expected package digest required}"
cd "$fims_root"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
"$fims_python" - "$fims_root" "$fims_digest" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root/'hpc'))
from common import verify_package
verify_package(root, sys.argv[2])
import numpy, matplotlib
print('REPORT PACKAGE / NUMPY / MATPLOTLIB: PASS', flush=True)
print('Python '+sys.version.split()[0]+'; NumPy '+numpy.__version__+'; Matplotlib '+matplotlib.__version__, flush=True)
PY
"$fims_python" -u "$fims_root/code/report_operating_points.py" \
    --capacity-root "$fims_capacity" --study-root "$fims_recovery" \
    --output-dir "$fims_root/results/report"
"$fims_python" - "$fims_root" "$fims_digest" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root/'hpc'))
from common import checked_report
if not checked_report(root, sys.argv[2]):
    raise SystemExit('Report process returned without a verified COMPLETE review ZIP')
print('FIMS OPERATING-POINT REVIEW V1: COMPLETE', flush=True)
print('Use the same downloaded launcher with pull to download the report.', flush=True)
PY
