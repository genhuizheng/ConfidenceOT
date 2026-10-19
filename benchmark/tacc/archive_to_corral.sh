#!/bin/bash
# Copy the native simulation benchmark to corral, as it is, and check the copy.
#
#   bash benchmark/tacc/archive_to_corral.sh --dry-run
#   bash benchmark/tacc/archive_to_corral.sh               # submits one gg job per source (48 h)
#   bash benchmark/tacc/archive_to_corral.sh --without-pcc # do not wait for log1p_noscale_pcc
#
# Destination: /corral/utexas/MCB26031/genhui/Primary_Metastasis_archive/simulation_benchmark/
#   benchmark_native/      copy of /scratch/.../OT_project/benchmark_native     (current benchmark)
#   code_<commit>.tar.gz   git archive of the repository at the commit that ran this
#   ARCHIVE_MANIFEST.md    what each directory holds and which reported result it backs
#   checks/                per source: file list with sizes on both sides, their diff,
#                          md5 of every file under 50 MB on both sides, and STATUS
#
# Nothing on scratch is changed or removed. A destination directory that already
# exists is never written into: the job refuses, so an archive is made once.
# By default the job refuses until both log1p_noscale_pcc scoring folders hold
# SUCCESS, so the archive includes that arm.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

mode=submit
dry_run=0
need_pcc=1
only=""
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --without-pcc) need_pcc=0; shift ;;
        --run) mode=run; shift ;;
        --only) only=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
project=${CONFIDENCEOT_PROJECT:-/scratch/10119/ghzheng/OT_project}
dest=${CONFIDENCEOT_ARCHIVE:-/corral/utexas/MCB26031/genhui/Primary_Metastasis_archive}/simulation_benchmark
sources="benchmark_native"  # the superseded benchmark_technical is not archived (user, 2026-10-19)
logs=$project/archive_logs

preflight() {  # $1: one source
    local name=$1
    test -d "$(dirname "$dest")" || { echo "no $(dirname "$dest")" >&2; exit 2; }
    test -d "$project/$name" || { echo "no $project/$name" >&2; exit 2; }
    test ! -e "$dest/$name" || { echo "$dest/$name already exists; not written into" >&2; exit 2; }
    if [[ "$name" == benchmark_native ]] && (( need_pcc )); then
        for a in 0.99 0.90; do
            test -f "$project/benchmark_native/metrics_log1p_pcc/acceptance_$a/SUCCESS" || {
                echo "log1p_noscale_pcc at $a is not scored yet; archive after it (or --without-pcc)" >&2; exit 2; }
        done
    fi
    test -f "$repo/benchmark/ARCHIVE_MANIFEST.md" || { echo "no ARCHIVE_MANIFEST.md in the repo" >&2; exit 2; }
}

listing() {  # relative path and size of every file, sorted
    (cd "$1" && find . -type f -printf '%P\t%s\n' | LC_ALL=C sort)
}

checksums() {  # md5 of every file under 50 MB, sorted by path
    (cd "$1" && find . -type f -size -50M -printf '%P\0' | LC_ALL=C sort -z | xargs -0 -r md5sum)
}

if [[ "$mode" == submit ]]; then
    # One job per source, so a failure or a time-out in one does not hold the other.
    need_kb=0
    for name in $sources; do
        preflight "$name"
        kb=$(du -sk "$project/$name" | cut -f1)
        need_kb=$(( need_kb + kb ))
        echo "$name: $(( kb / 1024 / 1024 )) GB"
    done
    avail_kb=$(df -Pk "$(dirname "$dest")" | awk 'NR == 2 { print $4 }')
    echo "needed $(( need_kb / 1024 / 1024 )) GB; available on $(dirname "$dest"): $(( avail_kb / 1024 / 1024 )) GB (df; the project quota may be lower)"
    if (( avail_kb < need_kb * 105 / 100 )); then
        echo "not enough room on corral for both copies with 5% to spare; nothing submitted" >&2; exit 3
    fi
    if (( dry_run )); then
        for name in $sources; do echo "DRY RUN  job: cp $project/$name -> $dest/$name"; done
        echo "DRY RUN  code $(git -C "$repo" rev-parse --short HEAD) and the manifest go with the benchmark_native job"
        exit 0
    fi
    mkdir -p "$logs"
    flags=""
    (( need_pcc )) || flags="--without-pcc"
    for name in $sources; do
        sbatch -p gg -N 1 -n 1 -t 48:00:00 -A MCB26031 -J cot_archive \
            -o "$logs/archive_${name}_%j.out" -e "$logs/archive_${name}_%j.err" \
            --wrap "bash $repo/benchmark/tacc/archive_to_corral.sh --run --only $name $flags"
    done
    exit 0
fi

# --- the job: one source -----------------------------------------------------------------
name=$only
[[ " $sources " == *" $name "* ]] || { echo "--run needs --only one of: $sources" >&2; exit 2; }
preflight "$name"
mkdir -p "$dest/checks"
commit=$(git -C "$repo" rev-parse --short HEAD)
echo "archive of $name at commit $commit -> $dest ($(date -Iseconds))"
if [[ "$name" == benchmark_native ]]; then
    git -C "$repo" archive --format=tar.gz -o "$dest/code_${commit}.tar.gz" HEAD
    cp "$repo/benchmark/ARCHIVE_MANIFEST.md" "$dest/ARCHIVE_MANIFEST.md"
fi
echo "== $name: listing and checksums on scratch ($(date -Iseconds))"
listing "$project/$name" > "$dest/checks/${name}.scratch.files.tsv"
checksums "$project/$name" > "$dest/checks/${name}.scratch.md5"
echo "== $name: cp ($(date -Iseconds))"
# Mode and timestamps only: cp -a also carries extended attributes and ACLs,
# which a copy from Lustre to corral can refuse, and that would stop the job.
cp -r --preserve=mode,timestamps "$project/$name" "$dest/$name"
echo "== $name: checking the copy ($(date -Iseconds))"
listing "$dest/$name" > "$dest/checks/${name}.corral.files.tsv"
checksums "$dest/$name" > "$dest/checks/${name}.corral.md5"
files=$(wc -l < "$dest/checks/${name}.scratch.files.tsv")
if diff "$dest/checks/${name}.scratch.files.tsv" "$dest/checks/${name}.corral.files.tsv" > "$dest/checks/${name}.files.diff" \
   && diff "$dest/checks/${name}.scratch.md5" "$dest/checks/${name}.corral.md5" > "$dest/checks/${name}.md5.diff"; then
    echo "OK $name: $files files, names and sizes identical; md5 identical for files under 50 MB; code $commit; $(date -Iseconds)" \
        | tee -a "$dest/checks/STATUS"
else
    echo "FAILED $name: see checks/${name}.*.diff; code $commit; $(date -Iseconds)" | tee -a "$dest/checks/STATUS"
    exit 1
fi
