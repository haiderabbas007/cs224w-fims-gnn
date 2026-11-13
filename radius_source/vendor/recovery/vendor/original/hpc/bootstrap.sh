#!/usr/bin/env bash
# Run on a CPU compute allocation: Koa login-node home is mounted noexec.
set -euo pipefail
fims_root="${1:?Usage: bootstrap.sh STUDY CAPACITY_ROOT BASE_PYTHON}"
fims_capacity="${2:?Existing capacity study required}"
fims_base_python="${3:?Existing CUDA-built Python required}"
cd "$fims_root"
mkdir -p results logs
exec 9>results/bootstrap.lock
flock -x 9
"$fims_base_python" "$fims_root/hpc/verify_package.py" "$fims_root" --capacity-root "$fims_capacity"
"$fims_base_python" - <<'PY'
import torch,numpy,matplotlib,scipy
assert torch.version.cuda is not None, 'Existing environment is not CUDA-built'
print('Existing CUDA-built environment:',torch.__version__,torch.version.cuda)
PY
fims_python="$fims_root/.venv/bin/python"
if [ ! -x "$fims_python" ]; then
    "$fims_base_python" -m venv --system-site-packages "$fims_root/.venv"
fi
"$fims_python" - "$fims_root/.venv" <<'PY'
import sys
from pathlib import Path
if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != Path(sys.argv[1]).resolve():
    raise SystemExit('Refusing installation outside the private study virtual environment')
PY
# Constraints preserve the existing scientific stack. Installation is isolated
# in this study's virtual environment; fm4npp is never modified.
"$fims_base_python" - <<'PY' > results/environment_constraints.txt
from importlib.metadata import version
for name in ('torch','numpy','matplotlib','scipy'):
    print(name+'=='+version(name))
print('torch-geometric==2.7.0')
PY
if ! "$fims_python" -c 'import torch_geometric; assert torch_geometric.__version__ == "2.7.0"' >/dev/null 2>&1; then
    "$fims_python" -m pip --isolated --require-virtualenv install --disable-pip-version-check \
        --constraint results/environment_constraints.txt 'torch-geometric==2.7.0'
fi
"$fims_python" - "$fims_root" "$fims_capacity" <<'PY'
import json,os,sys,tempfile
from importlib.metadata import version
from pathlib import Path
import torch,torch_geometric,numpy,matplotlib,scipy
r=Path(sys.argv[1]); p=r/'results/submitted_environment.json'
for line in (r/'results/environment_constraints.txt').read_text().splitlines():
    name,expected=line.split('==',1)
    if version(name) != expected:
        raise SystemExit('Private environment changed the pinned stack: '+name)
value=dict(python=sys.version.split()[0],torch=str(torch.__version__),cuda_build=torch.version.cuda,
           torch_geometric=torch_geometric.__version__,numpy=numpy.__version__,
           matplotlib=matplotlib.__version__,scipy=scipy.__version__)
if p.exists() and json.loads(p.read_text()) != value:
    raise SystemExit('Saved runtime differs. Restore this study environment; do not mix runtimes on resume.')
if not p.exists():
    fd,name=tempfile.mkstemp(dir=p.parent,prefix='.environment_')
    with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(name,p)
(r/'results/python_path.txt').write_text(sys.executable+'\n')
(r/'results/capacity_path.txt').write_text(str(Path(sys.argv[2]).resolve())+'\n')
print('Isolated study environment:',json.dumps(value,sort_keys=True))
PY
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
"$fims_python" "$fims_root/code/test_graph_models_v1.py"
"$fims_python" "$fims_root/code/test_site_pipeline_v1.py"
"$fims_python" "$fims_root/code/fit_controls_v1.py" --self-test
"$fims_python" "$fims_root/code/report_parallel_v1.py" --self-test
"$fims_python" "$fims_root/hpc/test_launchers_v1.py"
"$fims_python" "$fims_root/hpc/submit.py" --study "$fims_root" --capacity-root "$fims_capacity" --python "$fims_python"
