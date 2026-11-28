#!/bin/bash
#SBATCH -J app_score
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/app_search_20261117/logs/score_%j.out
# Scoring of the ot_20261123 rows (PLAN_TACC.md; 09_score_apps.py, LARRY in vivo by larry_20261112/03_score.py with
# the in vivo mature definition). Reads only; writes <dataset>/ot_20261123/score/. No set -e / -u / pipefail.
#   sbatch [--dependency=afterany:IDS] score_apps.slurm [datasets...]   (default: all six)
module load gcc python3/3.11.8 2> /dev/null || module load python3
venv=${V3_VENV:-/scratch/10119/ghzheng/OT_project/envs/otdb_otbench}
export PATH="$venv/bin:$PATH"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
repo=/scratch/10119/ghzheng/OT_project/code/ConfidenceOT
app=/scratch/10119/ghzheng/OT_project/app_search_20261117
gmt=$repo/results/reference/msigdb_human/h.all.v2025.1.Hs.symbols.gmt
[[ -f "$gmt" ]] || gmt=$app/code/h.all.v2025.1.Hs.symbols.gmt
cd "$repo" || exit 1
echo "job ${SLURM_JOB_ID:-local} on $(hostname), commit $(git rev-parse --short HEAD), $(date -Iseconds)"
sets=("$@"); (( ${#sets[@]} )) || sets=(kang sciplex2 schiebinger celegans pancreas larry_invivo)
for ds in "${sets[@]}"; do
    echo "=== $ds $(date -Iseconds)"
    if [[ $ds == larry_invivo ]]; then
        /usr/bin/time -v python -I "$app/code/03_score_larry.py" --v3-root "$app/$ds/ot_20261123" \
            --inputs "$app/$ds/inputs" --out "$app/$ds/ot_20261123/score" \
            --units LARRYvivo_LSK_d2_to_d9 LARRYvivo_LSK_d2_to_d16 LARRYvivo_LK_d2_to_d9 --non-mature Undifferentiated Prog \
            > "$app/logs/score_${ds}_${SLURM_JOB_ID:-local}.log" 2>&1
    elif [[ $ds == splintr || $ds == zebrafish ]]; then
        # added 2026-11-27: the repo copy of 09 (a superset; earlier datasets unchanged) with --data (baseline library, trees)
        /usr/bin/time -v python -I lineage_validation/app_search_20261117/09_score_apps.py --app "$app" --dataset "$ds" --data "$app/data" \
            > "$app/logs/score_${ds}_${SLURM_JOB_ID:-local}.log" 2>&1
    else
        /usr/bin/time -v python -I "$app/code/09_score_apps.py" --app "$app" --dataset "$ds" --gmt "$gmt" \
            > "$app/logs/score_${ds}_${SLURM_JOB_ID:-local}.log" 2>&1
    fi
    echo "=== $ds exit $? $(date -Iseconds)"
    grep -E "Elapsed|Maximum resident|Traceback|Error" "$app/logs/score_${ds}_${SLURM_JOB_ID:-local}.log" | head -5
done
echo "all done $(date -Iseconds)"
