#!/bin/bash
# Run the four-state pseudobulk and the paired DEG for every prostate arm, in
# one job, after the transport arrays finish.
#
# The stages are seconds each -- 30s per patient for the pseudobulk and 31s for
# PyDESeq2 on the first run -- so a serial loop inside one allocation is
# simpler and faster than ten chained submissions, and it keeps every arm's
# result written by the same code at the same time.
#
# Arms differ in two ways that matter here. The depth-equalised arms read the
# manifest 27_downsample_counts.py rewrote and can size lesions by its
# analysed_cell_n; the raw-depth arms read the prepared manifest and fall back
# to the manifest's own count. Either is a no-op for this dataset, because
# 34_prepare_prostate_pairs.py already named one lymph node per patient, so
# each primary cell is gated exactly once whichever way the size is read.
#
# No `set -u`: the explicit path check below catches the defined-but-empty
# variable that -u permits, which is the failure that has cost job numbers here.
#
# Usage:
#   bash cancer_metastasis/submit_prostate_deg_arms.sh                 # submit now
#   bash cancer_metastasis/submit_prostate_deg_arms.sh 1000855:1000856 # wait first
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-${REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}}
result=${CANCER_COT_ROOT:-${RESULTS:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
stamp=${PROSTATE_STAMP:-20260916}

for name in repo result env_path; do
  value=${!name}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done

after=${1:-}
dependency=()
if [[ -n "$after" ]]; then
  dependency=(--dependency=afterok:"$after")
fi

body=$(cat <<SCRIPT
source /home1/10119/ghzheng/.bashrc
conda activate $env_path
cd $repo
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16

result=$result
stamp=$stamp
ds_manifest=\$result/downsampled_GSE271675_\$stamp/pair_manifest_downsampled.csv
raw_manifest=\$result/prepared_GSE271675_\$stamp/pair_manifest_eligible.csv
size_csv=\$result/downsampled_GSE271675_\$stamp/downsample_per_sample.csv

# arm_label : ot_root_suffix : manifest : size_csv_or_dash
arms="
logcpm_free_ds:free:\$ds_manifest:\$size_csv
rank256_free_ds:rank256_free_ds:\$ds_manifest:\$size_csv
rank256_free_raw:rank256_free_raw:\$raw_manifest:-
rank256_cap085_ds:rank256_cap085_ds:\$ds_manifest:\$size_csv
rank256_cap085_raw:rank256_cap085_raw:\$raw_manifest:-
"

for row in \$arms; do
  label=\$(echo "\$row" | cut -d: -f1)
  suffix=\$(echo "\$row" | cut -d: -f2)
  manifest=\$(echo "\$row" | cut -d: -f3)
  size=\$(echo "\$row" | cut -d: -f4)
  gate=\$result/ot_GSE271675_\${suffix}_\$stamp
  four=\$result/four_state_\${label}_\$stamp
  deg=\$result/deg_\${label}_\$stamp

  echo "##################################################"
  echo "ARM \$label  gate=\$gate"
  if [[ ! -d "\$gate" ]]; then
    echo "SKIP \$label: no transport output"
    continue
  fi

  extra=(--malignant-annotation Epithelial)
  if [[ "\$size" != "-" && -f "\$size" ]]; then
    extra+=(--metastasis-size-csv "\$size")
  fi

  count=\$(python cancer_metastasis/21_prepare_four_state_malignant_pseudobulk.py \\
    "\$manifest" "\$gate" "\$four" --print-patient-count "\${extra[@]}") || {
      echo "SKIP \$label: patient count failed"; continue; }
  echo "patients: \$count"

  index=0
  while [[ "\$index" -lt "\$count" ]]; do
    python cancer_metastasis/21_prepare_four_state_malignant_pseudobulk.py \\
      "\$manifest" "\$gate" "\$four" --index "\$index" "\${extra[@]}" > /dev/null 2>&1 \\
      || echo "  pseudobulk failed at index \$index"
    index=\$((index + 1))
  done
  ready=\$(ls \$four/patients/*/PSEUDOBULK_READY 2>/dev/null | wc -l)
  echo "pseudobulk ready: \$ready / \$count"
  if [[ "\$ready" -lt 3 ]]; then
    echo "SKIP \$label: fewer than three patients carry a pseudobulk"
    continue
  fi

  python cancer_metastasis/13_run_paired_pydeseq2.py "\$four" "\$deg" \\
    --minimum-cells-per-patient-status 20 --n-cpus 16 2>&1 | tail -40
done
echo "##################################################"
echo "ALL ARMS COMPLETE \$(date --iso-8601=seconds)"
SCRIPT
)

mkdir -p "$result/logs"
cd "$repo"
sbatch --parsable "${dependency[@]}" \
  -p gg -N 1 -n 1 -c 16 -t 04:00:00 -A MCB26031 -J cot_deg_arms \
  -o "$result/logs/deg_arms_%j.out" -e "$result/logs/deg_arms_%j.err" \
  --wrap="$body" | tail -n 1 | tr -dc '0-9'
echo
echo "logs: $result/logs/deg_arms_<jobid>.out"
