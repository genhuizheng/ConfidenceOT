#!/bin/bash
# Submit the depth-equalised GSE180661 pair array, in one of two rejection
# configurations.
#
# Environment is exported into this shell and inherited by the job, rather than
# passed through sbatch --export. Vista's user guide says to avoid --export
# because it interferes with how the system propagates the inherited
# environment, and passing values through its comma-separated list also cannot
# carry a value containing a space, which the rejection bounds do.
#
# Every path is validated here, before a job number is spent. Two submissions
# were lost to an unset $RESULTS expanding to nothing inside a long --export
# string: the variables arrived, but already stripped of their prefix, so the
# job failed on a mkdir at the filesystem root.
#
#   free     source rejection unconstrained, (0, 1); target never rejects
#   bounded  source rejection held to [0.5, 0.85]; target never rejects
#
# Usage:
#   bash cancer_metastasis/submit_downsampled_gse180661.sh free
#   bash cancer_metastasis/submit_downsampled_gse180661.sh bounded
set -euo pipefail

config=${1:-}
case "$config" in
  free|bounded) ;;
  *) echo "Usage: $0 {free|bounded}" >&2; exit 2 ;;
esac

export CONFIDENCEOT_REPO=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
export CANCER_COT_ROOT=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
downsampled=${CONFIDENCEOT_DOWNSAMPLED_ROOT:-$CANCER_COT_ROOT/downsampled_GSE180661_20260914}

export CONFIDENCEOT_MANIFEST="$downsampled/pair_manifest_downsampled.csv"
export CONFIDENCEOT_OUTPUT_ROOT="$CANCER_COT_ROOT/ot_downsampled_${config}_20260914"
export CONFIDENCEOT_ANALYSIS_SCOPE=malignant
export CONFIDENCEOT_INCLUDE_ANNOTATIONS=Ovarian.cancer.cell
export CONFIDENCEOT_DEVICE=cpu
export CONFIDENCEOT_THREADS=144
export CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE=10000
export CONFIDENCEOT_SAVE_PAIRING_EDGES=1

# The cells were selected by the downsampling step, and subsampling lowers the
# detected-gene count a second QC pass would test, so the metrics are recorded
# without filtering again. The gate diagnostics read those metrics.
export CONFIDENCEOT_CELL_QC=1
export CONFIDENCEOT_MINIMUM_TOTAL_COUNTS=0
export CONFIDENCEOT_MINIMUM_DETECTED_GENES=0
export CONFIDENCEOT_MAXIMUM_MITOCHONDRIAL_PERCENT=100

if [[ "$config" == "free" ]]; then
  # The legacy budgets resolve to an unconstrained source and a target that
  # never rejects. Any bounds left in the shell would silently override that.
  unset CONFIDENCEOT_SOURCE_REJECTION_BOUNDS CONFIDENCEOT_TARGET_REJECTION_BOUNDS
  export CONFIDENCEOT_SOURCE_REJECTION_BUDGET=0.85
  export CONFIDENCEOT_TARGET_REJECTION_BUDGET=0.00
else
  export CONFIDENCEOT_SOURCE_REJECTION_BOUNDS="0.5 0.85"
  export CONFIDENCEOT_TARGET_REJECTION_BOUNDS="0.0 0.0"
fi

for name in CONFIDENCEOT_REPO CANCER_COT_ROOT CONFIDENCEOT_MANIFEST; do
  value=${!name}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done
if [[ ! -f "$CONFIDENCEOT_MANIFEST" ]]; then
  echo "Manifest does not exist: $CONFIDENCEOT_MANIFEST" >&2
  exit 2
fi

rows=$(python - "$CONFIDENCEOT_MANIFEST" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)
# The array script refuses to start when this disagrees, so it is resolved from
# the manifest rather than restated and left to drift.
export CONFIDENCEOT_PAIR_COUNT="$rows"

cd "$CONFIDENCEOT_REPO"
printf 'config      : %s\n' "$config"
printf 'manifest    : %s (%s pairs)\n' "$CONFIDENCEOT_MANIFEST" "$rows"
printf 'output      : %s\n' "$CONFIDENCEOT_OUTPUT_ROOT"
printf 'source      : %s\n' \
  "${CONFIDENCEOT_SOURCE_REJECTION_BOUNDS:-budget ${CONFIDENCEOT_SOURCE_REJECTION_BUDGET:-} unenforced}"
printf 'target      : %s\n' \
  "${CONFIDENCEOT_TARGET_REJECTION_BOUNDS:-budget ${CONFIDENCEOT_TARGET_REJECTION_BUDGET:-}}"
printf 'device      : %s with %s threads\n' "$CONFIDENCEOT_DEVICE" "$CONFIDENCEOT_THREADS"

sbatch -p "${CONFIDENCEOT_QUEUE:-gg}" -t "${CONFIDENCEOT_WALLTIME:-12:00:00}" \
  --array="${CONFIDENCEOT_ARRAY:-0-7}" \
  cancer_metastasis/tacc_full_array.slurm
