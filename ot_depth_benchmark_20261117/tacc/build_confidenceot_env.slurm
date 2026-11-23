#!/bin/bash
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
#SBATCH -J cot_env
# Rebuild the ConfidenceOT environment on a compute node (aarch64), under /scratch (user 2026-11-23), as a NEW prefix:
# the old /scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot lost bin/python and conda-meta on 2026-11-21
# and is left untouched. Versions: confidenceot_env_requirements.txt (the old env's own versions). The repository is
# installed editable without dependencies, as before (confidenceot 0.5.0).
#   sbatch -o /scratch/10119/ghzheng/OT_project/ot_depth_benchmark_20261117/logs/cot_env_%j.out \
#          ot_depth_benchmark_20261117/tacc/build_confidenceot_env.slurm
# No set -e / -u / pipefail (TACC rules).

P=${COT_ENV_PREFIX:-/scratch/10119/ghzheng/conda_envs/confidenceot_20261123}
REPO=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
REQ=${COT_ENV_REQ:-$REPO/ot_depth_benchmark_20261117/tacc/confidenceot_env_requirements.txt}
CONDA=${CONDA_EXE:-/work/10119/ghzheng/vista/miniconda3/bin/conda}
echo "build $(date -Iseconds) on $(hostname) $(uname -m) -> $P"
if [ ! -x "$P/bin/python" ]; then "$CONDA" create -y -q -p "$P" python=3.11 pip || exit 1; fi
"$P/bin/python" -m pip install -q -U pip
"$P/bin/python" -m pip install -r "$REQ" > "$P.pip.log" 2>&1 || echo "pip install -r returned an error, see $P.pip.log"
"$P/bin/python" -m pip install -q --no-deps -e "$REPO" || echo "editable install of the repo failed"
"$P/bin/python" - <<'EOF'
import platform, numpy, scipy, pandas, anndata, scanpy, sklearn, torch, ot, h5py, confidenceot
from confidenceot import ConfidenceOT  # noqa: F401
print("env check", platform.machine(), "python", platform.python_version(), "numpy", numpy.__version__, "scipy", scipy.__version__,
      "pandas", pandas.__version__, "anndata", anndata.__version__, "scanpy", scanpy.__version__, "sklearn", sklearn.__version__,
      "torch", torch.__version__, "pot", ot.__version__, "confidenceot", confidenceot.__file__)
EOF
"$P/bin/python" -m pip freeze > "$P.freeze.txt" 2>/dev/null
echo "finished $(date -Iseconds)"
