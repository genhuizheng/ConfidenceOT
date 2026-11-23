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
REPORT="$E/ENVS_REPORT${OTDB_ONLY:+_only}.txt"
echo "build $(date -Iseconds) on $(hostname) $(uname -m); $(python3 --version)" > "$REPORT"

only=${OTDB_ONLY:-}   # space-separated environment names to build; empty = all
mk() {
    name=$1; shift
    if [ -n "$only" ] && [[ " $only " != *" $name "* ]]; then return 0; fi
    if [ -x "$E/$name/bin/python" ]; then echo "$name exists, packages updated in place" >> "$REPORT"; else python3 -m venv "$E/$name" || { echo "$name: venv failed" >> "$REPORT"; return 1; }; fi
    "$E/$name/bin/pip" install -q -U pip wheel setuptools
    "$E/$name/bin/pip" install "$@" > "$E/$name.pip.log" 2>&1 || echo "$name: pip install returned an error, see $name.pip.log" >> "$REPORT"
    wrap "$name"
}
# The venv interpreter links libpython from the python3 module, which is on the library path only after `module load`.
# The workers run in the conda environment without that module, so bin/python becomes a wrapper that adds the module's
# lib directory for this interpreter only and execs the venv's own python3 (pyvenv.cfg is found from its path).
# First seen 2026-11-18, pilot 1152855: every row exit 127, libpython3.11.so.1.0 not found.
wrap() {
    name=$1; libdir=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
    rm -f "$E/$name/bin/python"
    printf '#!/bin/bash\nexport LD_LIBRARY_PATH=%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\nexec %s "$@"\n' "$libdir" "$E/$name/bin/python3" > "$E/$name/bin/python"
    chmod +x "$E/$name/bin/python"
}
check() {
    name=$1; code=$2
    if [ -n "$only" ] && [[ " $only " != *" $name "* ]]; then return 0; fi
    "$E/$name/bin/python" -c "$code" >> "$REPORT" 2>&1 || echo "$name: import check FAILED" >> "$REPORT"
}

mk otdb_moscot "moscot==0.4.3" "jax[cpu]==0.6.2" "ott-jax[neural]==0.5.2" scanpy
check otdb_moscot "import platform, moscot, ott, jax, scanpy; print('otdb_moscot', platform.machine(), 'moscot', moscot.__version__, 'ott', ott.__version__, 'jax', jax.__version__)"

mk otdb_wot "wot==1.0.8.post2" scanpy POT "numpy<2" "pandas<2"
check otdb_wot "import platform, wot, numpy, pandas, scanpy; print('otdb_wot', platform.machine(), 'numpy', numpy.__version__, 'pandas', pandas.__version__)"

mk otdb_cfm torch "torchcfm==1.0.7" torchdyn scanpy POT "numpy<2"
check otdb_cfm "import platform, torch, torchcfm, torchdyn; print('otdb_cfm', platform.machine(), 'torch', torch.__version__)"

# pandas / scipy / scikit-learn pinned to the local otbench versions (2026-11-18): pandas 3 strings break anndata 0.10
# h5ad writing (CellOT), pilot 1153414. scipy 1.13.1 (2026-11-23): scanpy 1.10.4 calls csr_array.getnnz, removed in scipy
# 1.14, on its pynndescent path (n_obs >= 4096), which CINEMA-OTW reaches on the 5,000-cell units (pilot 1153579); the
# local test units were smaller and never took that path
mk otdb_otbench "numpy<2" "scipy==1.13.1" "pandas==2.3.3" "scanpy==1.10.4" "anndata<0.11" "scikit-learn==1.7.2" statsmodels "cinemaot==0.0.5" "pot==0.9.5" \
    ml-collections absl-py "torch==2.3.1" tables igraph leidenalg pyyaml tqdm
check otdb_otbench "import platform, torch, cinemaot, ot, scanpy, tables, leidenalg, ml_collections, pandas, scipy, sklearn; print('otdb_otbench', platform.machine(), 'torch', torch.__version__, 'pot', ot.__version__, 'pandas', pandas.__version__, 'scipy', scipy.__version__, 'sklearn', sklearn.__version__, 'csr_array.getnnz', hasattr(__import__('scipy.sparse').sparse.csr_array((1, 1)), 'getnnz'))"

# CellOT, the authors' repository at the commit tested locally (522d2b9)
if [ ! -d "$TOOLS/cellot/.git" ]; then git clone -q https://github.com/bunnech/cellot.git "$TOOLS/cellot"; fi
git -C "$TOOLS/cellot" checkout -q 522d2b953da8ad244fcf36f64521487fdf763788 && echo "cellot $(git -C "$TOOLS/cellot" rev-parse --short HEAD)" >> "$REPORT"

echo "finished $(date -Iseconds)" >> "$REPORT"
cat "$REPORT"
