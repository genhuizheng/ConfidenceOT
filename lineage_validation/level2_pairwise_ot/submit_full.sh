#!/bin/bash
# Full Level 2 run (PLAN.md): every case of inputs/cases.tsv (every eligible family / clone) x the three strategies
# (rna, cnv, rna_cnv), each with exactly the two OT runs of inputs/run_manifest.tsv: A_standard_OT (Balanced OT)
# and B_our_OT (our UOT, rejection budget 0, KL-relaxed marginals kept). The two-step strategies are not run:
# without rejection they equal rna and cnv exactly (strategy_check/README.md).
#
#   bash lineage_validation/level2_pairwise_ot/submit_full.sh --dry-run                    # manifest + sbatch lines
#   bash lineage_validation/level2_pairwise_ot/submit_full.sh                              # every strategy, every case not done
#   bash lineage_validation/level2_pairwise_ot/submit_full.sh --strategies cnv,rna_cnv     # some strategies only
#   bash lineage_validation/level2_pairwise_ot/submit_full.sh --only 4,8                   # retry these case indices
#   bash lineage_validation/level2_pairwise_ot/submit_full.sh --score-only                 # one scoring job after every queued Level 2 job
#
# SLURM: one array per strategy and resource class of cases.tsv (standard 12 h, large 48 h; gg, one node per task),
# level2_task.slurm per task; then one scoring job (afterany) that scores every strategy found on disk, lists every
# failed / missing case in run_status.tsv, writes scores_<stamp>/ and packs the small files into
# level2_full_small_<stamp>.tar.gz (written as .partial, renamed when complete, md5 beside it).
# A case is not submitted when it has SUCCESS or when its RUNNING names a job still in the queue, and nothing is
# submitted while a Level 2 job of a requested strategy is still queued: no case is ever run twice at once. A failed case keeps
# its FAILED.json until its retry starts; nothing is re-parameterised on retry. Output
# <root>/level2_pairwise_ot/{full (rna), full_cnv, full_rna_cnv}/<case_id>/. No set -u, as everywhere in this repo.
set -eo pipefail

dry_run=0; only=""; strategies="rna,cnv,rna_cnv"; score_only=0
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --score-only) score_only=1 ;;
        --only) only=$2; shift ;;
        --strategies) strategies=$2; shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${LINEAGE_VALIDATION_ROOT:-/scratch/10119/ghzheng/OT_project/lineage_validation}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
kp=${KPTRACER_ROOT:-/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot}
quinn_cnv=${QUINN_CNV_ROOT:-/scratch/10119/ghzheng/OT_project/quinn2021_confidenceot/cnv_profiles/infercnvpy_experiment}
l2=$root/level2_pairwise_ot
logs=$l2/logs
here=$repo/lineage_validation/level2_pairwise_ot
cases=$here/inputs/cases.tsv
manifest=$here/inputs/run_manifest.tsv
cap=${CONFIDENCEOT_JOB_CAP:-40}
declare -A hours=([standard]=12:00:00 [large]=48:00:00)
declare -A folder=([rna]=full [cnv]=full_cnv [rna_cnv]=full_rna_cnv)

for s in ${strategies//,/ }; do [[ -n "${folder[$s]}" ]] || { echo "unknown strategy $s (rna, cnv, rna_cnv)" >&2; exit 2; }; done
for path in "$cases" "$manifest" "$here/level2_task.slurm" "$root/inputs/manifests/quinn_D_B_raw.csv" \
            "$root/inputs/h5ad/Quinn2021/ds/pair_manifest_downsampled.csv" "$kp/downsampled/pair_manifest_downsampled.csv"; do
    [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done
if [[ ",$strategies," == *",cnv,"* || ",$strategies," == *",rna_cnv,"* ]]; then
    for path in "$kp/cnv_profiles/infercnvpy_mouse/KPTracer" "$quinn_cnv/Quinn2021"; do
        [[ -d "$path" ]] || { echo "missing CNV exports: $path" >&2; exit 1; }
    done
fi
# Every case x strategy must have exactly the two methods, and every case must be in the manifest.
bad=$(awk -F'\t' 'NR > 1 {m[$1 "/" $3] = m[$1 "/" $3] " " $4} END {for (i in m) if (m[i] != " A_standard_OT B_our_OT") print i ":" m[i]}' "$manifest")
[[ -z "$bad" ]] || { echo "run_manifest.tsv: a case / strategy does not have exactly A_standard_OT and B_our_OT: $bad" >&2; exit 1; }
[[ $(cut -f1 "$cases" | tail -n +2 | sort -n | tr '\n' ' ') == $(cut -f1 "$manifest" | tail -n +2 | sort -nu | tr '\n' ' ') ]] \
    || { echo "cases.tsv and run_manifest.tsv list different cases" >&2; exit 1; }
queue=$(squeue -u "$USER" -h -r -o "%i %j %T" 2> /dev/null) || queue=""
stamp=$(date +%Y%m%d_%H%M%S)
scores=$l2/scores_$stamp
tarball=$l2/level2_full_small_$stamp.tar.gz
score_cmd="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo; \
python lineage_validation/level2_pairwise_ot/score_pairwise.py --root $l2 --out $scores; \
cd $l2 && tar czf $tarball.partial --exclude=coupling.npz --exclude='coordinates_*.npz' \$(ls -d full full_cnv full_rna_cnv 2> /dev/null) logs $(basename "$scores") \
&& mv $tarball.partial $tarball && md5sum $(basename "$tarball") > $tarball.md5"
if (( score_only )); then
    # One scoring job after every Level 2 OT job still queued (any strategy): the combined comparison.
    deps=$(printf '%s\n' "$queue" | awk '$2 ~ /^l2_/ && $2 != "l2_score" {split($1, a, "_"); print a[1]}' | sort -u | tr '\n' ':' | sed 's/:$//')
    args=(-p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J l2_score -o "$logs/l2_score_%j.out" ${deps:+--dependency=afterany:$deps} --wrap "$score_cmd")
    if (( dry_run )); then echo "DRY RUN  sbatch ${deps:+--dependency=afterany:$deps} -J l2_score (score every strategy -> $scores, $tarball)" >&2; exit 0; fi
    mkdir -p "$logs"
    id=$(sbatch --parsable "${args[@]}" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "scoring sbatch returned no job id" >&2; exit 1; }
    echo "scoring job $id${deps:+ after $deps} -> $scores, $tarball"
    exit 0
fi
if (( dry_run )); then
    echo "== run manifest ($manifest): $(( $(wc -l < "$manifest") - 1 )) OT runs" >&2
    cut -f1-7,15 "$manifest" | column -t -s $'\t' >&2
fi
# A queued Level 2 job of a requested strategy blocks the submission (no case is run twice at once). Its strategy
# is read off its name: l2_standard / l2_large (the first runs) are rna, l2_<strategy>_<class> name theirs,
# l2_score runs no OT.
clash=$(printf '%s\n' "$queue" | awk -v want=",$strategies," '
    $2 == "l2_standard" || $2 == "l2_large" {s = "rna"}
    $2 ~ /^l2_(rna|cnv|rna_cnv)_(standard|large)$/ {s = $2; sub(/^l2_/, "", s); sub(/_(standard|large)$/, "", s)}
    $2 !~ /^l2_/ || $2 == "l2_score" {s = ""}
    s != "" && index(want, "," s ",") {print}')
if [[ -n "$clash" ]]; then
    echo "Level 2 jobs of a requested strategy are still queued; submit after they end, or leave that strategy out:" >&2
    printf '%s\n' "$clash" >&2
    (( dry_run )) || exit 3
fi

declare -A todo
n=0
for strategy in ${strategies//,/ }; do
    out=$l2/${folder[$strategy]}
    while IFS=$'\t' read -r index case_id dataset mouse source targets families left estimate resource; do
        resource=${resource%$'\r'}  # a CRLF checkout must not change the class name
        [[ "$index" == index ]] && continue
        if [[ -n "$only" && ",$only," != *",$index,"* ]]; then continue; fi
        d=$out/$case_id
        if [[ -f "$d/SUCCESS" ]]; then echo "done, not resubmitted: $strategy $index $case_id" >&2; continue; fi
        if [[ -f "$d/RUNNING" ]]; then
            job=$(tr -d '[:space:]' < "$d/RUNNING")
            if printf '%s\n' "$queue" | awk -v j="$job" '$1 == j || index($1, j "_") == 1' | grep -q .; then
                echo "running (job $job), not resubmitted: $strategy $index $case_id" >&2; continue
            fi
            echo "stale RUNNING (job $job no longer queued), resubmitted: $strategy $index $case_id" >&2
        fi
        [[ -f "$d/FAILED.json" ]] && echo "earlier failure, resubmitted: $strategy $index $case_id" >&2
        [[ -n "${hours[$resource]}" ]] || { echo "case $index: unknown resource class $resource" >&2; exit 1; }
        key="$strategy:$resource"
        todo[$key]="${todo[$key]:+${todo[$key]},}$index"
        n=$(( n + 1 ))
    done < "$cases"
done
(( n > 0 )) || { echo "nothing to submit" >&2; exit 0; }
queued=$(printf '%s\n' "$queue" | grep -c . || true)
if (( ! dry_run && queued + n + 1 > cap )); then echo "$queued queued; $n tasks + 1 scoring job would pass the cap of $cap" >&2; exit 3; fi
(( dry_run )) || mkdir -p "$logs"

ids=""; deps=""
for strategy in ${strategies//,/ }; do
    for resource in standard large; do
        list=${todo[$strategy:$resource]}
        [[ -n "$list" ]] || continue
        out=$l2/${folder[$strategy]}
        args=(--array="$list" -t "${hours[$resource]}" -J "l2_${strategy}_${resource}" -o "$logs/l2_${strategy}_%A_%a.out"
              -e "$logs/l2_${strategy}_%A_%a.err"
              --export=ALL,L2_OUT="$out",L2_STRATEGY="$strategy",CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_ENV="$env_path" "$here/level2_task.slurm")
        if (( dry_run )); then echo "DRY RUN  sbatch ${args[*]}" >&2; continue; fi
        mkdir -p "$out"
        id=$(sbatch --parsable "${args[@]}" | tail -n 1)
        [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id; stopping (anything above was submitted)" >&2; exit 1; }
        echo "$strategy $resource array $id: cases $list" >&2
        ids="$ids $strategy/$resource=$id[$list]"; deps="$deps:$id"
    done
done
if (( dry_run )); then
    echo "DRY RUN  sbatch --dependency=afterany<arrays> -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J l2_score --wrap \"score_pairwise.py --root $l2 --out $scores; tar small files -> $tarball\"" >&2
    echo "DRY RUN  $n tasks + 1 scoring job; currently queued: $queued (cap $cap)" >&2
    exit 0
fi
id=$(sbatch --parsable --dependency=afterany$deps -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J l2_score -o "$logs/l2_score_%j.out" --wrap "$score_cmd" | tail -n 1)
[[ "$id" =~ ^[0-9]+$ ]] || { echo "scoring sbatch returned no job id (the arrays above were submitted)" >&2; exit 1; }
summary="commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)
level2 full: strategies $strategies x A_standard_OT (Balanced OT) / B_our_OT (UOT, rejection budget 0), epsilon 0.1
tasks $n${only:+ (only $only)}; arrays$ids; scoring job $id -> $scores, $tarball"
echo "$summary"
echo "$summary" > "$l2/SUBMITTED_full_$stamp.txt"
