#!/bin/bash
# Check everything the depth screen needs before a single job is submitted.
#
# Four of the eleven configurations depend on packages that are not ours:
# scanpy for the analytic Pearson residual, and R with sctransform for
# SCTransform. Finding out from a failed array task wastes a submission, so
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

echo "== R with sctransform, for configurations 9 and 10"
# shellcheck source=/dev/null
source "$repo/scripts/tacc/r_environment.sh"
[[ "${CONFIDENCEOT_R_STATUS:-}" == "ready" ]] && external_r=1

echo
echo "=================================================="
if [[ $ours_ok -eq 1 ]]; then
  echo "configurations 0-6 (ours): READY"
else
  echo "configurations 0-6 (ours): BLOCKED, fix the FAIL lines above"
fi
[[ $external_scanpy -eq 1 ]] && echo "configurations 7-8 (scanpy):     READY" \
                             || echo "configurations 7-8 (scanpy):     NOT AVAILABLE"
[[ $external_r -eq 1 ]]      && echo "configurations 9-10 (sctransform): READY" \
                             || echo "configurations 9-10 (sctransform): NOT AVAILABLE"
echo
if [[ $ours_ok -eq 1 ]]; then
  last=6
  [[ $external_scanpy -eq 1 ]] && last=8
  [[ $external_scanpy -eq 1 && $external_r -eq 1 ]] && last=10
  echo "submit:  sbatch --array=0-$last scripts/tacc/screen_depth_configurations.slurm"
  if [[ $external_scanpy -eq 1 && $external_r -ne 1 ]]; then
    echo "  (9-10 need R with sctransform; add them with --array=9-10 later)"
  fi
fi
