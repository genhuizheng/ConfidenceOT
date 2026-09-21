#!/bin/bash
#SBATCH -J cot_cancer_full
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/full_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/full_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -A MCB26031
# Submit a small worker array. Each worker
# processes indices task_id, task_id + task_count, ... to avoid QOS array limits.
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
eligible=${CONFIDENCEOT_MANIFEST:-$result/manifest/pair_manifest_eligible.csv}
output_root=${CONFIDENCEOT_OUTPUT_ROOT:-$result/full}
analysis_scope=${CONFIDENCEOT_ANALYSIS_SCOPE:-all}
annotation_spec=${CONFIDENCEOT_INCLUDE_ANNOTATIONS:-}
shared_budget=${CONFIDENCEOT_REJECTION_BUDGET:-0.95}
source_budget=${CONFIDENCEOT_SOURCE_REJECTION_BUDGET:-$shared_budget}
target_budget=${CONFIDENCEOT_TARGET_REJECTION_BUDGET:-$shared_budget}
maximum_observed_cells=${CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE:-10000}
save_pairing_edges=${CONFIDENCEOT_SAVE_PAIRING_EDGES:-0}
cell_qc=${CONFIDENCEOT_CELL_QC:-0}
# cuda is right for the gh queue; gg nodes are CPU-only and need cpu here.
device=${CONFIDENCEOT_DEVICE:-cuda}
solver_workers=${CONFIDENCEOT_WORKERS:-1}
# Thresholds are separate from the switch because a run on depth-equalised
# counts needs the QC metrics recorded, since the gate diagnostics read them,
# while filtering again would discard cells that were already selected:
# subsampling lowers the detected-gene count a second pass would test. Set all
# three permissive for that case.
qc_minimum_counts=${CONFIDENCEOT_MINIMUM_TOTAL_COUNTS:-1000}
qc_minimum_genes=${CONFIDENCEOT_MINIMUM_DETECTED_GENES:-500}
qc_maximum_mito=${CONFIDENCEOT_MAXIMUM_MITOCHONDRIAL_PERCENT:-20.0}
calibration_null=${CONFIDENCEOT_CALIBRATION_NULL:-within_side_split}
# Space-separated "LOW HIGH" bounds on each side's rejected fraction. Left
# empty the legacy budgets apply, which give an unconstrained source and a
# target that never rejects. Set a non-zero LOW to cap retention, which makes
# the gate a ranked selection of that size rather than a partition the
# objective chose.
source_bounds=${CONFIDENCEOT_SOURCE_REJECTION_BOUNDS:-}
target_bounds=${CONFIDENCEOT_TARGET_REJECTION_BOUNDS:-}
enforce_budget=${CONFIDENCEOT_ENFORCE_REJECTION_BUDGET:-0}
# An unset path variable used to fall through to mkdir at the filesystem root,
# which failed with a permission error and then a confusing "manifest does not
# exist" for a path missing its prefix. sbatch cannot be submitted from a
# compute node, so variables exported inside an idev session are absent from
# the login shell that actually submits, which is how this happens.
for name in repo result output_root eligible; do
  value=${!name}
  if [[ -z "$value" ]]; then
    echo "$name is empty; export the paths in the submitting shell" >&2
    exit 2
  fi
  if [[ "$value" != /* ]]; then
    echo "$name is not an absolute path: $value" >&2
    exit 2
  fi
done
mkdir -p "$result/logs" "$output_root"
cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
# A Grace-Grace allocation hands over a whole 144-core node, so leaving this at
# a small value wastes most of it. The transport fits are CPU-feasible at the
# scale these pairs actually reach: the median pair is about 1,600 by 1,700
# cells, which a 16-thread CPU fit completes in roughly two minutes.
export OMP_NUM_THREADS=${CONFIDENCEOT_THREADS:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

pair_count=${CONFIDENCEOT_PAIR_COUNT:-}
if [[ ! -f "$eligible" ]]; then
  echo "Manifest does not exist: $eligible" >&2
  exit 2
fi
manifest_count=$(python - "$eligible" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)
if [[ -z "$pair_count" ]]; then
  pair_count=$manifest_count
fi
if ! [[ "$pair_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid CONFIDENCEOT_PAIR_COUNT=$pair_count" >&2
  exit 2
fi
if [[ "$manifest_count" -ne "$pair_count" ]]; then
  echo "Manifest row count ($manifest_count) != CONFIDENCEOT_PAIR_COUNT ($pair_count)" >&2
  exit 2
fi

task_id=${SLURM_ARRAY_TASK_ID:-0}
task_count=${SLURM_ARRAY_TASK_COUNT:-1}
restart_count=${SLURM_RESTART_COUNT:-0}
failure_file="$result/logs/failed_${SLURM_ARRAY_JOB_ID:-manual}_${task_id}_restart${restart_count}.txt"
failures=0

echo "Worker $task_id/$task_count processing $pair_count manifest rows"
echo "Rejection caps: source=$source_budget target=$target_budget"
echo "Maximum observed cells per side: $maximum_observed_cells"
scope_args=(--analysis-scope "$analysis_scope")
pairing_args=()
qc_args=()
if [[ "$save_pairing_edges" == "1" ]]; then
  pairing_args+=(--save-pairing-edges)
fi
if [[ "$cell_qc" == "1" ]]; then
  qc_args+=(--cell-qc
    --minimum-total-counts "$qc_minimum_counts"
    --minimum-detected-genes "$qc_minimum_genes"
    --maximum-mitochondrial-percent "$qc_maximum_mito")
fi
if [[ "$enforce_budget" == "1" ]]; then
  qc_args+=(--enforce-rejection-budget)
fi
if [[ -n "$source_bounds" ]]; then
  qc_args+=(--source-rejection-bounds $source_bounds)
fi
if [[ -n "$target_bounds" ]]; then
  qc_args+=(--target-rejection-bounds $target_bounds)
fi
# Representation the transport cost is computed in. rank_value is the
# Geneformer formulation: it removed most of the depth dependence in
# simulation, taking auc_total_counts from 0.945 to 0.475 at moderate depth
# spread for 0.008 of perturbed F1. rank_top_n has to stay below the
# shallowest cell's detected-gene count or the encoding manufactures its own
# low-content artefact.
if [[ -n "${CONFIDENCEOT_REPRESENTATION:-}" ]]; then
  qc_args+=(--representation "$CONFIDENCEOT_REPRESENTATION")
fi
if [[ -n "${CONFIDENCEOT_RANK_TOP_N:-}" ]]; then
  qc_args+=(--rank-top-n "$CONFIDENCEOT_RANK_TOP_N")
fi
# Cost geometry. `cosine` L2-normalises the joint representation before the
# cost and the median scale, which discards each cell's magnitude and nothing
# else. In the depth screen this is the one change that separated the rate of
# rejection on populations containing nothing to reject, on both simulators and
# at both sizes, with no loss of power. See SEQUENCING_DEPTH_RESOLUTION.md.
if [[ -n "${CONFIDENCEOT_COST:-}" ]]; then
  qc_args+=(--cost "$CONFIDENCEOT_COST")
fi
if [[ -n "${CONFIDENCEOT_MINIMUM_DETECTION_RATE:-}" ]]; then
  qc_args+=(--minimum-detection-rate "$CONFIDENCEOT_MINIMUM_DETECTION_RATE")
fi

if [[ "$analysis_scope" == "malignant" ]]; then
  if [[ -z "$annotation_spec" ]]; then
    echo "CONFIDENCEOT_INCLUDE_ANNOTATIONS is required for malignant scope" >&2
    exit 2
  fi
  IFS='|' read -r -a annotations <<< "$annotation_spec"
  for annotation in "${annotations[@]}"; do
    scope_args+=(--include-annotation "$annotation")
  done
fi
for ((index=task_id; index<pair_count; index+=task_count)); do
  echo "===== START index=$index $(date --iso-8601=seconds) ====="
  if python cancer_metastasis/02_run_pair.py "$eligible" "$output_root" \
    --index "$index" \
    "${scope_args[@]}" \
    "${pairing_args[@]}" \
    "${qc_args[@]}" \
    --calibration-null "$calibration_null" \
    --source-rejection-budget "$source_budget" \
    --target-rejection-budget "$target_budget" \
    --max-observed-cells-per-side "$maximum_observed_cells" \
    --null-calibration-replicates 5 \
    --null-validation-replicates 5 \
    --calibration-grid-size 5 \
    --tolerance 1e-4 \
    --workers "$solver_workers" \
    --device "$device" \
    --skip-completed; then
    echo "===== COMPLETE index=$index $(date --iso-8601=seconds) ====="
  else
    status=$?
    echo "$index,$status" >> "$failure_file"
    failures=$((failures + 1))
    echo "===== FAILED index=$index exit=$status $(date --iso-8601=seconds) =====" >&2
  fi
done

echo "Worker $task_id finished with $failures failed pair(s)"
if (( failures > 0 )); then
  exit 1
fi
