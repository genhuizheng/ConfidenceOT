#!/bin/bash
# Submit the MOSTA biological application on TACC (gg, CPU): read-only on the finished runs, no OT is solved.
#
#   bash mouse_embryo/biology/submit_mosta_biology.sh [--dry-run] first|organs
#
# first   audit (31_) -> ds extraction of heart with marker expression -> logcpm extraction of heart with the
#         M4-R couplings rebuilt from potentials -> heart case (33_).  4 jobs, each afterok the previous.
# organs  the same for neural, liver and urogenital (extraction of ds, of logcpm, case). 3 jobs.
#
# Output root MOSTA_BIO_OUT (default .../mouse_embryo_results/mosta_biology_20261030); nothing outside it is
# written, and its writer refuses to replace a file it did not write.  No set -e / -u / pipefail (TACC rules).

dry_run=0; stage=""
for a in "$@"; do
    case "$a" in
        --dry-run) dry_run=1 ;;
        first|organs) stage=$a ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done
[[ -n "$stage" ]] || { echo "usage: bash mouse_embryo/biology/submit_mosta_biology.sh [--dry-run] first|organs" >&2; exit 2; }
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
out=${MOSTA_BIO_OUT:-/scratch/10119/ghzheng/OT_project/mouse_embryo_results/mosta_biology_20261030}
job=$repo/mouse_embryo/biology/tacc_mosta_biology.slurm
cap=${CONFIDENCEOT_JOB_CAP:-40}

submit() {  # $1 = time limit, $2 = name, rest = exports; prints the job id
    local limit=$1 name=$2; shift 2
    local exports="ALL,MOSTA_BIO_OUT=$out$(printf ',%s' "$@")"
    if (( dry_run )); then echo "DRY RUN  sbatch ${dep[*]} -t $limit -J $name --export=$exports $job" >&2; echo DRYRUN; return 0; fi
    local id
    id=$(sbatch --parsable "${dep[@]}" -t "$limit" -J "$name" -o "$out/logs/${name}_%j.out" -e "$out/logs/${name}_%j.err" \
        --export="$exports" "$job" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; return 1; }
    echo "$id"
}

need=3; [[ $stage == first ]] && need=4
queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l)
if (( ! dry_run && queued + need > cap )); then echo "$queued queued; $need more would pass the cap of $cap" >&2; exit 3; fi
(( dry_run )) || mkdir -p "$out/logs" || exit 1

dep=()
if [[ $stage == first ]]; then
    id=$(submit 02:00:00 bio_audit MOSTA_BIO_STEP=audit) || exit 1; echo "audit $id" >&2; dep=(--dependency=afterok:$id)
    id=$(submit 06:00:00 bio_ext_ds MOSTA_BIO_STEP=extract MOSTA_BIO_RUN=ds "MOSTA_BIO_SETS=heart" MOSTA_BIO_EXPRESSION=1) || exit 1
    echo "extract ds heart + expression $id" >&2; dep=(--dependency=afterok:$id)
    id=$(submit 12:00:00 bio_ext_lc MOSTA_BIO_STEP=extract MOSTA_BIO_RUN=logcpm "MOSTA_BIO_SETS=heart" MOSTA_BIO_M4R=1) || exit 1
    echo "extract logcpm heart + M4-R couplings $id" >&2; dep=(--dependency=afterok:$id)
    id=$(submit 04:00:00 bio_case MOSTA_BIO_STEP=case "MOSTA_BIO_SETS=heart") || exit 1
    echo "heart case $id" >&2
else
    id=$(submit 08:00:00 bio_ext_ds MOSTA_BIO_STEP=extract MOSTA_BIO_RUN=ds "MOSTA_BIO_SETS=neural liver urogenital" MOSTA_BIO_EXPRESSION=1) || exit 1
    echo "extract ds organs $id" >&2; dep=(--dependency=afterok:$id)
    id=$(submit 12:00:00 bio_ext_lc MOSTA_BIO_STEP=extract MOSTA_BIO_RUN=logcpm "MOSTA_BIO_SETS=neural liver urogenital") || exit 1
    echo "extract logcpm organs $id" >&2; dep=(--dependency=afterok:$id)
    id=$(submit 08:00:00 bio_case MOSTA_BIO_STEP=case "MOSTA_BIO_SETS=neural liver urogenital") || exit 1
    echo "organ cases $id" >&2
fi
(( dry_run )) || echo "$stage submitted $(date -Iseconds), commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null)" >> "$out/SUBMITTED.txt"
