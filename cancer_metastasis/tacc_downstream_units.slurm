#!/bin/bash
#SBATCH -J cot_units
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/units_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/units_%j.err
#
# Several runs of tacc_factorial_downstream.slurm in one allocation, one after
# another, so a design of many small OT roots costs one job against the
# submission cap instead of one per root.
#
# Each line of the file CONFIDENCEOT_UNITS names is one run: tab-separated
# NAME=VALUE settings, which that run sees as its environment -- exactly what
# an array task of its own would have been given -- and it runs as task 0.
# Every run uses the whole node while it runs, as an array task would have.
# A run that fails does not stop the next; the job exits 1 if any failed.
#
# No set -u, as everywhere in this repo.
set -o pipefail
units=${CONFIDENCEOT_UNITS:-}
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
job=$repo/cancer_metastasis/tacc_factorial_downstream.slurm
if [[ -z "$units" || ! -f "$units" ]]; then
    echo "CONFIDENCEOT_UNITS must name the file of runs, found '$units'" >&2
    exit 2
fi

failed=0
n=0
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    n=$(( n + 1 ))
    IFS=$'\t' read -r -a settings <<< "$line"
    echo "===== UNIT $n START $(date --iso-8601=seconds): ${settings[*]} ====="
    if env "${settings[@]}" SLURM_ARRAY_TASK_ID=0 bash "$job"; then
        echo "===== UNIT $n DONE $(date --iso-8601=seconds) ====="
    else
        failed=1
        echo "===== UNIT $n FAILED $(date --iso-8601=seconds) ====="
    fi
done < "$units"
echo "$n units from $units, any failed: $failed"
exit $failed
