#!/bin/bash
#SBATCH -J app_prep_sz
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/app_search_20261117/logs/prepare_sz_%j.out
# PLAN_TACC.md post-hoc 2026-11-27: SPLINTR and zebrafish inputs (06_prepare_tacc.py), then 07_verify_inputs.py on those
# two only (verify/inputs_20261127.tsv; the earlier verify/inputs.tsv is not touched). Time and peak memory per dataset
# in logs/prep_<dataset>_<jobid>.log. Python: the otdb_otbench venv (the conda env lost its interpreter in the purge).
#   sbatch lineage_validation/app_search_20261117/tacc/prepare_sz.slurm            (default: splintr zebrafish)
# No set -e / -u / pipefail (TACC rules). Threads capped at 4.

module load gcc python3/3.11.8 2> /dev/null || module load python3
venv=${V3_VENV:-/scratch/10119/ghzheng/OT_project/envs/otdb_otbench}
export PATH="$venv/bin:$PATH"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
app=/scratch/10119/ghzheng/OT_project/app_search_20261117
cd "$repo" || exit 1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
sets=("$@"); (( ${#sets[@]} )) || sets=(splintr zebrafish)
echo "job ${SLURM_JOB_ID:-local} on $(hostname); commit $(git rev-parse --short HEAD); python $(which python); datasets: ${sets[*]}; $(date -Iseconds)"
for ds in "${sets[@]}"; do
    if [[ -e "$app/$ds/manifest.csv" ]]; then
        echo "=== $ds: $app/$ds/manifest.csv exists, not overwritten; skipped"
        continue
    fi
    echo "=== $ds start $(date -Iseconds)"
    /usr/bin/time -v python -I lineage_validation/app_search_20261117/06_prepare_tacc.py --dataset "$ds" --data "$app/data" --out "$app" \
        > "$app/logs/prep_${ds}_${SLURM_JOB_ID:-local}.log" 2>&1
    echo "=== $ds exit $? $(date -Iseconds)"
    grep -E "Elapsed|Maximum resident|Error|error" "$app/logs/prep_${ds}_${SLURM_JOB_ID:-local}.log" | head -20
done
echo "=== verify $(date -Iseconds)"
python -I lineage_validation/app_search_20261117/07_verify_inputs.py --app "$app" --only "${sets[@]}" --tag _20261127
echo "all done $(date -Iseconds)"
