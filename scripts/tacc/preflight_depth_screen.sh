#!/bin/bash
# Check everything the depth screen needs before a single job is submitted.
#
# Four of the eleven configurations depend on packages that are not ours:
# scanpy for the analytic Pearson residual, and Rscript with Seurat and Matrix
# for SCTransform. Finding out from a failed array task wastes a submission, so
# this runs on the login node in seconds and says exactly which indices are
# safe to submit.
#
# No `set -e`: every check must run even after one fails, because the point is
# the full list.
#
# Usage:
#   bash scripts/tacc/preflight_depth_screen.sh

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_DEPTH_SCREEN_ROOT:-/scratch/10119/ghzheng/OT_project/depth_screen}
export PYTHONPATH="$repo/src:$repo/cancer_metastasis:$repo${PYTHONPATH:+:$PYTHONPATH}"

ours_ok=1
external_scanpy=0
external_r=0

echo "== paths"
for pair in "repo:$repo" "root:$root"; do
  name=${pair%%:*}; value=${pair#*:}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "   FAIL $name is not an absolute path: '$value'"; ours_ok=0
  else
    echo "   ok   $name $value"
  fi
done
[[ -f "$repo/scripts/validate_depth_null_specificity.py" ]] \
  && echo "   ok   runner present" \
  || { echo "   FAIL runner missing; did the pull land?"; ours_ok=0; }

echo "== the new flags are in this checkout"
for flag in --equalise-depth --cost --external-representation; do
  if grep -q -- "\"$flag\"" "$repo/scripts/validate_depth_null_specificity.py"; then
    echo "   ok   $flag"
  else
    echo "   FAIL $flag absent; this checkout predates the screen"; ours_ok=0
  fi
done

echo "== our own dependencies"
python - <<'PY'
import importlib, sys
bad = 0
for name in ("numpy", "pandas", "scipy", "sklearn", "anndata", "confidenceot"):
    try:
        module = importlib.import_module(name)
        print(f"   ok   {name:12s} {getattr(module, '__version__', '?')}")
    except Exception as error:
        print(f"   FAIL {name:12s} {type(error).__name__}: {error}")
        bad = 1
sys.exit(bad)
PY
[[ $? -eq 0 ]] || ours_ok=0

echo "== scanpy, for configurations 7 and 8"
if python - <<'PY'
import sys
try:
    import scanpy.experimental as se
    assert hasattr(se.pp, "normalize_pearson_residuals")
    import scanpy
    print(f"   ok   scanpy {scanpy.__version__} with normalize_pearson_residuals")
except Exception as error:
    print(f"   SKIP scanpy unusable: {type(error).__name__}: {error}")
    sys.exit(1)
PY
then external_scanpy=1; fi

echo "== R with Seurat and Matrix, for configurations 9 and 10"
# CONFIDENCEOT_RSCRIPT lets R live in a container rather than on PATH, which is
# how it is set up here: a rocker r-ver image under /scratch/.../containers.
if [[ -n "${CONFIDENCEOT_RSCRIPT:-}" ]]; then
  r_command=$CONFIDENCEOT_RSCRIPT
  echo "   using CONFIDENCEOT_RSCRIPT: $r_command"
elif command -v Rscript > /dev/null 2>&1; then
  r_command=Rscript
  echo "   using Rscript from PATH"
else
  r_command=""
  echo "   SKIP no R. Either module load one, or set CONFIDENCEOT_RSCRIPT, e.g."
  echo "        export CONFIDENCEOT_RSCRIPT='apptainer exec -B /scratch \\"
  echo "          /scratch/10119/ghzheng/containers/r-ver-4.4.sif Rscript'"
fi
if [[ -n "$r_command" ]]; then
  if $r_command --vanilla -e 'suppressMessages({library(Matrix); library(Seurat)}); cat("   ok   Seurat", as.character(packageVersion("Seurat")), "\n")' 2>/dev/null; then
    external_r=1
  else
    echo "   SKIP R runs but Seurat or Matrix will not load; install into a"
    echo "        library the container can see, then re-run this preflight"
  fi
fi

echo
echo "=================================================="
if [[ $ours_ok -eq 1 ]]; then
  echo "configurations 0-6 (ours): READY"
else
  echo "configurations 0-6 (ours): BLOCKED, fix the FAIL lines above"
fi
[[ $external_scanpy -eq 1 ]] && echo "configurations 7-8 (scanpy):     READY" \
                             || echo "configurations 7-8 (scanpy):     NOT AVAILABLE"
[[ $external_r -eq 1 ]]      && echo "configurations 9-10 (SCTransform): READY" \
                             || echo "configurations 9-10 (SCTransform): NOT AVAILABLE"
echo
if [[ $ours_ok -eq 1 ]]; then
  last=6
  [[ $external_scanpy -eq 1 ]] && last=8
  [[ $external_scanpy -eq 1 && $external_r -eq 1 ]] && last=10
  echo "submit:  sbatch --array=0-$last scripts/tacc/screen_depth_configurations.slurm"
  if [[ $external_scanpy -eq 1 && $external_r -ne 1 ]]; then
    echo "  (9-10 need R; add them later with --array=9-10 once Seurat loads)"
  fi
fi
