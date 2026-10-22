#!/bin/bash
# Level 2 pilot (PLAN.md): one gg job per row of inputs/pilot_pairs.tsv, run_pairwise.py (four runs each).
#
#   bash lineage_validation/level2_pairwise_ot/submit_pilot.sh --dry-run
#   bash lineage_validation/level2_pairwise_ot/submit_pilot.sh
#
# Output <root>/level2_pairwise_ot/pilot/<pair_id>/. A row whose SUCCESS exists is not resubmitted.
# Reads the KP-Tracer equalised/raw files and the lineage-validation Quinn sides; writes nothing else.
# No set -u, as everywhere in this repo.
set -eo pipefail

dry_run=0
[[ "${1:-}" == --dry-run ]] && dry_run=1
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${LINEAGE_VALIDATION_ROOT:-/scratch/10119/ghzheng/OT_project/lineage_validation}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
out=$root/level2_pairwise_ot/pilot
logs=$root/level2_pairwise_ot/logs
pairs=$repo/lineage_validation/level2_pairwise_ot/inputs/pilot_pairs.tsv
cap=${CONFIDENCEOT_JOB_CAP:-40}
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo"

for path in "$pairs" "$root/inputs/manifests/quinn_D_B_raw.csv" "$root/inputs/h5ad/Quinn2021/ds/pair_manifest_downsampled.csv" \
            /scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot/downsampled/pair_manifest_downsampled.csv; do
    [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done
n=$(( $(wc -l < "$pairs") - 1 ))
queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l) || queued=0
if (( ! dry_run && queued + n > cap )); then echo "$queued queued; $n more would pass the cap of $cap" >&2; exit 3; fi
(( dry_run )) || mkdir -p "$out" "$logs"

ids=""
while IFS=$'\t' read -r index pair_id rest; do
    [[ "$index" == index ]] && continue
    if [[ -f "$out/$pair_id/SUCCESS" ]]; then echo "done: $pair_id" >&2; continue; fi
    cmd="$activate && python lineage_validation/level2_pairwise_ot/run_pairwise.py --index $index --out $out"
    if (( dry_run )); then echo "DRY RUN  sbatch -p gg -N 1 -n 1 -t 12:00:00 -A MCB26031 -J l2_$index --wrap \"... run_pairwise.py --index $index --out $out\"  ($pair_id)" >&2; continue; fi
    id=$(sbatch --parsable -p gg -N 1 -n 1 -t 12:00:00 -A MCB26031 -J "l2_$index" -o "$logs/${pair_id}_%j.out" --wrap "$cmd" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id; stopping (anything above was submitted)" >&2; exit 1; }
    echo "$pair_id job $id" >&2
    ids="$ids $index=$id"
done < "$pairs"
summary="commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)
level2 pilot: arms A_logcpm_BOT B_ranknm256dscos_UOT AxUOT_logcpm_UOT BxBOT_ranknm256dscos_BOT, epsilon 0.1, no rejection
out $out
jobs$ids"
echo "$summary"
(( dry_run )) || echo "$summary" > "$root/level2_pairwise_ot/SUBMITTED_pilot_$(date +%Y%m%d_%H%M%S).txt"
