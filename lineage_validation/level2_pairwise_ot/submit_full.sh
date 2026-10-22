#!/bin/bash
# Full Level 2 run (PLAN.md): every case of inputs/cases.tsv (every eligible family / clone), each with exactly
# the two OT runs of inputs/run_manifest.tsv: A_standard_OT (logcpm_noscale + Balanced OT) and B_our_OT
# (ranknm256_noscale_ds_cos + our UOT, rejection budget 0, KL-relaxed marginals kept).
#
#   bash lineage_validation/level2_pairwise_ot/submit_full.sh --dry-run          # print manifest + sbatch lines
#   bash lineage_validation/level2_pairwise_ot/submit_full.sh                    # submit every case not done
#   bash lineage_validation/level2_pairwise_ot/submit_full.sh --only 4,8         # retry these case indices
#
# SLURM: one array per resource class of cases.tsv (standard 12 h, large 48 h; gg, one node per task, so the
# whole node's memory), level2_task.slurm per task; then one scoring job (afterany: it runs whatever finished
# and lists every failed / missing case in run_status.tsv) that writes scores_<stamp>/ and packs the small files
# into level2_full_small_<stamp>.tar.gz (written as .partial, renamed when complete, md5 beside it).
# A case with SUCCESS is never resubmitted; a failed case keeps its FAILED.json until its retry starts. Nothing
# is re-parameterised on retry. Output <root>/level2_pairwise_ot/full/<case_id>/. No set -u, as everywhere here.
set -eo pipefail

dry_run=0; only=""
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --only) only=$2; shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${LINEAGE_VALIDATION_ROOT:-/scratch/10119/ghzheng/OT_project/lineage_validation}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
kp=${KPTRACER_ROOT:-/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot}
l2=$root/level2_pairwise_ot
out=$l2/full
logs=$l2/logs
here=$repo/lineage_validation/level2_pairwise_ot
cases=$here/inputs/cases.tsv
manifest=$here/inputs/run_manifest.tsv
cap=${CONFIDENCEOT_JOB_CAP:-40}
declare -A hours=([standard]=12:00:00 [large]=48:00:00)

for path in "$cases" "$manifest" "$here/level2_task.slurm" "$root/inputs/manifests/quinn_D_B_raw.csv" \
            "$root/inputs/h5ad/Quinn2021/ds/pair_manifest_downsampled.csv" \
            "$kp/downsampled/pair_manifest_downsampled.csv"; do
    [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done
# Every case must have exactly the two methods, and every case must be in the manifest.
bad=$(awk -F'\t' 'NR > 1 {m[$1] = m[$1] " " $3} END {for (i in m) if (m[i] != " A_standard_OT B_our_OT") print i ":" m[i]}' "$manifest")
[[ -z "$bad" ]] || { echo "run_manifest.tsv: a case does not have exactly A_standard_OT and B_our_OT: $bad" >&2; exit 1; }
[[ $(cut -f1 "$cases" | tail -n +2 | sort -n | tr '\n' ' ') == $(cut -f1 "$manifest" | tail -n +2 | sort -nu | tr '\n' ' ') ]] \
    || { echo "cases.tsv and run_manifest.tsv list different cases" >&2; exit 1; }
if (( dry_run )); then
    echo "== run manifest ($manifest): $(( $(wc -l < "$manifest") - 1 )) OT runs" >&2
    cut -f1-6,13 "$manifest" | column -t -s $'\t' >&2
fi

declare -A todo=([standard]="" [large]="")
n=0
while IFS=$'\t' read -r index case_id dataset mouse source targets families left estimate resource; do
    resource=${resource%$'\r'}  # a CRLF checkout must not change the class name
    [[ "$index" == index ]] && continue
    if [[ -n "$only" && ",$only," != *",$index,"* ]]; then continue; fi
    if [[ -f "$out/$case_id/SUCCESS" ]]; then echo "done, not resubmitted: $index $case_id" >&2; continue; fi
    [[ -f "$out/$case_id/FAILED.json" ]] && echo "earlier failure, resubmitted: $index $case_id" >&2
    [[ -n "${hours[$resource]}" ]] || { echo "case $index: unknown resource class $resource" >&2; exit 1; }
    todo[$resource]="${todo[$resource]:+${todo[$resource]},}$index"
    n=$(( n + 1 ))
done < "$cases"
(( n > 0 )) || { echo "nothing to submit" >&2; exit 0; }
queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l) || queued=0
if (( ! dry_run && queued + n + 1 > cap )); then echo "$queued queued; $n tasks + 1 scoring job would pass the cap of $cap" >&2; exit 3; fi
(( dry_run )) || mkdir -p "$out" "$logs"

stamp=$(date +%Y%m%d_%H%M%S)
ids=""; deps=""
for resource in standard large; do
    list=${todo[$resource]}
    [[ -n "$list" ]] || continue
    args=(--array="$list" -t "${hours[$resource]}" -J "l2_${resource}" -o "$logs/l2_%A_%a.out" -e "$logs/l2_%A_%a.err"
          --export=ALL,L2_OUT="$out",CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_ENV="$env_path" "$here/level2_task.slurm")
    if (( dry_run )); then echo "DRY RUN  sbatch ${args[*]}" >&2; continue; fi
    id=$(sbatch --parsable "${args[@]}" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id; stopping (anything above was submitted)" >&2; exit 1; }
    echo "$resource array $id: cases $list" >&2
    ids="$ids $resource=$id[$list]"; deps="$deps:$id"
done
scores=$l2/scores_$stamp
tarball=$l2/level2_full_small_$stamp.tar.gz
score_cmd="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo; \
python lineage_validation/level2_pairwise_ot/score_pairwise.py --runs $out --out $scores; \
cd $l2 && tar czf $tarball.partial --exclude=coupling.npz --exclude='coordinates_*.npz' full logs $(basename "$scores") \
&& mv $tarball.partial $tarball && md5sum $(basename "$tarball") > $tarball.md5"
if (( dry_run )); then
    echo "DRY RUN  sbatch --dependency=afterany<arrays> -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J l2_score --wrap \"score_pairwise.py --runs $out --out $scores; tar small files -> $tarball\"" >&2
    exit 0
fi
id=$(sbatch --parsable --dependency=afterany$deps -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J l2_score -o "$logs/l2_score_%j.out" --wrap "$score_cmd" | tail -n 1)
[[ "$id" =~ ^[0-9]+$ ]] || { echo "scoring sbatch returned no job id (the arrays above were submitted)" >&2; exit 1; }
summary="commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)
level2 full: A_standard_OT (logcpm_noscale + Balanced OT) and B_our_OT (ranknm256_noscale_ds_cos + UOT, rejection budget 0), epsilon 0.1
cases $n of $(( $(wc -l < "$cases") - 1 ))${only:+ (only $only)}; out $out
arrays$ids; scoring job $id -> $scores, $tarball"
echo "$summary"
echo "$summary" > "$l2/SUBMITTED_full_$stamp.txt"
