#!/bin/bash
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -t 03:00:00
#SBATCH -A MCB26031
#SBATCH -J otdb_envs
# Build the external-method environments on a Vista compute node (aarch64), as TACC recommends: module python3 + venv +
# pip on a compute node, never on the login node. Versions are the ones of METHOD_PREPROCESSING.md / PLAN.md.
#   sbatch -o /scratch/10119/ghzheng/OT_project/ot_depth_benchmark_20261117/logs/envs_%j.out \
#          ot_depth_benchmark_20261117/tacc/build_external_envs.slurm
# Writes ENVS_REPORT.txt (versions or import errors per environment) next to the environments.
# No set -e / -u / pipefail (TACC rules).

module load gcc python3/3.11.8 2>/dev/null || module load python3
E=${OTDB_ENVS:-/scratch/10119/ghzheng/OT_project/envs}
TOOLS=${OTDB_TOOLS:-/scratch/10119/ghzheng/OT_project/code}
mkdir -p "$E" "$TOOLS"
REPORT="$E/ENVS_REPORT.txt"
echo "build $(date -Iseconds) on $(hostname) $(uname -m); $(python3 --version)" > "$REPORT"

mk() {
    name=$1; shift
    if [ -x "$E/$name/bin/python" ]; then echo "$name exists, packages updated in place" >> "$REPORT"; else python3 -m venv "$E/$name" || { echo "$name: venv failed" >> "$REPORT"; return 1; }; fi
    "$E/$name/bin/pip" install -q -U pip wheel setuptools
    "$E/$name/bin/pip" install "$@" > "$E/$name.pip.log" 2>&1 || echo "$name: pip install returned an error, see $name.pip.log" >> "$REPORT"
}
check() {
    name=$1; code=$2
    "$E/$name/bin/python" -c "$code" >> "$REPORT" 2>&1 || echo "$name: import check FAILED" >> "$REPORT"
}

mk otdb_moscot "moscot==0.4.3" "jax[cpu]==0.6.2" "ott-jax[neural]==0.5.2" scanpy
check otdb_moscot "import platform, moscot, ott, jax, scanpy; print('otdb_moscot', platform.machine(), 'moscot', moscot.__version__, 'ott', ott.__version__, 'jax', jax.__version__)"

mk otdb_wot "wot==1.0.8.post2" scanpy POT "numpy<2" "pandas<2"
check otdb_wot "import platform, wot, numpy, pandas, scanpy; print('otdb_wot', platform.machine(), 'numpy', numpy.__version__, 'pandas', pandas.__version__)"

mk otdb_cfm torch "torchcfm==1.0.7" torchdyn scanpy POT "numpy<2"
check otdb_cfm "import platform, torch, torchcfm, torchdyn; print('otdb_cfm', platform.machine(), 'torch', torch.__version__)"

mk otdb_otbench "numpy<2" scipy pandas "scanpy==1.10.4" "anndata<0.11" scikit-learn statsmodels "cinemaot==0.0.5" "pot==0.9.5" \
    ml-collections absl-py "torch==2.3.1" tables igraph leidenalg pyyaml tqdm
check otdb_otbench "import platform, torch, cinemaot, ot, scanpy, tables, leidenalg, ml_collections; print('otdb_otbench', platform.machine(), 'torch', torch.__version__, 'pot', ot.__version__)"

# CellOT, the authors' repository at the commit tested locally (522d2b9)
if [ ! -d "$TOOLS/cellot/.git" ]; then git clone -q https://github.com/bunnech/cellot.git "$TOOLS/cellot"; fi
git -C "$TOOLS/cellot" checkout -q 522d2b953da8ad244fcf36f64521487fdf763788 && echo "cellot $(git -C "$TOOLS/cellot" rev-parse --short HEAD)" >> "$REPORT"

echo "finished $(date -Iseconds)" >> "$REPORT"
cat "$REPORT"
