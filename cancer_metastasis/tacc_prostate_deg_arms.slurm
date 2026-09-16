#!/bin/bash
#SBATCH -J cot_deg_arms
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/deg_arms_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/deg_arms_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

# Four-state pseudobulk and paired DEG for every prostate arm, in one job.
#
# Written as a file rather than an sbatch --wrap string. The first attempt put
# this script inside --wrap and the arguments arrived misaligned -- argparse saw
# a fourth positional and every arm reported "patient count failed" in 18
# seconds. Passing a sixty-line script through a heredoc, then a shell variable,
# then --wrap, then sh is four layers of quoting for no benefit.
#
# No `set -u`: the explicit path check catches the defined-but-empty variable
# that -u permits, which is the failure that has cost job numbers here.
set -eo pipefail

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
stamp=${PROSTATE_STAMP:-20260916}

for name in repo result; do
  value=${!name}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis:$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p "$result/logs"

ds_manifest="$result/downsampled_GSE271675_$stamp/pair_manifest_downsampled.csv"
raw_manifest="$result/prepared_GSE271675_$stamp/pair_manifest_eligible.csv"
size_csv="$result/downsampled_GSE271675_$stamp/downsample_per_sample.csv"

prepare=cancer_metastasis/21_prepare_four_state_malignant_pseudobulk.py
deseq=cancer_metastasis/13_run_paired_pydeseq2.py

# label, transport-root suffix, manifest, size table ("-" when there is none).
# The depth-equalised arms can size lesions by the analysed malignant count;
# the raw-depth arms fall back to the manifest figure. Either is a no-op here,
# since 34_prepare_prostate_pairs.py already named one node per patient.
run_arm() {
  local label=$1 suffix=$2 manifest=$3 size=$4
  local gate="$result/ot_GSE271675_${suffix}_$stamp"
  local four="$result/four_state_${label}_$stamp"
  local deg="$result/deg_${label}_$stamp"

  echo "##################################################"
  echo "ARM $label"
  echo "  gate     $gate"
  echo "  manifest $manifest"
  if [[ ! -d "$gate" ]]; then
    echo "SKIP $label: no transport output"
    return 0
  fi

  local extra=(--malignant-annotation Epithelial)
  if [[ "$size" != "-" && -f "$size" ]]; then
    extra+=(--metastasis-size-csv "$size")
  fi

  local count
  if ! count=$(python "$prepare" "$manifest" "$gate" "$four" \
      --print-patient-count "${extra[@]}" 2>&1); then
    echo "SKIP $label: patient count failed"
    echo "$count" | tail -3
    return 0
  fi
  echo "  patients $count"

  local index=0
  while [[ "$index" -lt "$count" ]]; do
    if ! python "$prepare" "$manifest" "$gate" "$four" \
        --index "$index" "${extra[@]}" > /dev/null 2>&1; then
      echo "  pseudobulk failed at index $index"
    fi
    index=$((index + 1))
  done

  local ready
  ready=$(find "$four/patients" -name PSEUDOBULK_READY 2>/dev/null | wc -l)
  echo "  pseudobulk ready $ready / $count"
  # Three patients is the floor for a paired design to leave any residual
  # degrees of freedom. The free arms retain about 1% of primary cells, so the
  # twenty-cell minimum drops most patients there; that is the gate's property,
  # not a failure of this job.
  if [[ "$ready" -lt 3 ]]; then
    echo "SKIP $label: fewer than three patients carry a pseudobulk"
    return 0
  fi

  python "$deseq" "$four" "$deg" \
    --minimum-cells-per-patient-status 20 --n-cpus 16 2>&1 | tail -40
}

run_arm logcpm_free_ds     free               "$ds_manifest"  "$size_csv"
run_arm rank256_free_ds    rank256_free_ds    "$ds_manifest"  "$size_csv"
run_arm rank256_free_raw   rank256_free_raw   "$raw_manifest" "-"
run_arm rank256_cap085_ds  rank256_cap085_ds  "$ds_manifest"  "$size_csv"
run_arm rank256_cap085_raw rank256_cap085_raw "$raw_manifest" "-"

echo "##################################################"
echo "ALL ARMS COMPLETE $(date --iso-8601=seconds)"
