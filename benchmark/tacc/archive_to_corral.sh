#!/bin/bash
# Copy the simulation benchmark to corral, as it is, and check the copy.
#
#   bash benchmark/tacc/archive_to_corral.sh --dry-run
#   bash benchmark/tacc/archive_to_corral.sh               # submits one gg job
#   bash benchmark/tacc/archive_to_corral.sh --without-pcc # do not wait for log1p_noscale_pcc
#
# Destination: /corral/utexas/MCB26031/genhui/Primary_Metastasis_archive/simulation_benchmark/
#   benchmark_native/      copy of /scratch/.../OT_project/benchmark_native     (current benchmark)
#   benchmark_technical/   copy of /scratch/.../OT_project/benchmark_technical  (superseded thinning version)
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
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=1 ;;
        --without-pcc) need_pcc=0 ;;
        --run) mode=run ;;
        *) echo "unknown option: $argument" >&2; exit 2 ;;
    esac
done

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
project=${CONFIDENCEOT_PROJECT:-/scratch/10119/ghzheng/OT_project}
dest=${CONFIDENCEOT_ARCHIVE:-/corral/utexas/MCB26031/genhui/Primary_Metastasis_archive}/simulation_benchmark
sources="benchmark_native benchmark_technical"
logs=$project/archive_logs

preflight() {
    test -d "$(dirname "$dest")" || { echo "no $(dirname "$dest")" >&2; exit 2; }
    for name in $sources; do
        test -d "$project/$name" || { echo "no $project/$name" >&2; exit 2; }
        test ! -e "$dest/$name" || { echo "$dest/$name already exists; not written into" >&2; exit 2; }
    done
    if (( need_pcc )); then
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
    preflight
    if (( dry_run )); then
        for name in $sources; do echo "DRY RUN  cp $project/$name -> $dest/$name ($(du -sh "$project/$name" | cut -f1))"; done
        echo "DRY RUN  code $(git -C "$repo" rev-parse --short HEAD), manifest, checks -> $dest"
        exit 0
    fi
    mkdir -p "$logs"
    flags=""
    (( need_pcc )) || flags="--without-pcc"
    sbatch -p gg -N 1 -n 1 -t 24:00:00 -A MCB26031 -J cot_archive \
        -o "$logs/archive_%j.out" -e "$logs/archive_%j.err" \
        --wrap "bash $repo/benchmark/tacc/archive_to_corral.sh --run $flags"
    exit 0
fi

# --- the job ------------------------------------------------------------------------------
preflight
mkdir -p "$dest/checks"
commit=$(git -C "$repo" rev-parse --short HEAD)
echo "archive at commit $commit -> $dest ($(date -Iseconds))"
git -C "$repo" archive --format=tar.gz -o "$dest/code_${commit}.tar.gz" HEAD
cp "$repo/benchmark/ARCHIVE_MANIFEST.md" "$dest/ARCHIVE_MANIFEST.md"
status=0
for name in $sources; do
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
        echo "OK $name: $files files, names and sizes identical; md5 identical for files under 50 MB" | tee -a "$dest/checks/STATUS"
    else
        echo "FAILED $name: see checks/${name}.*.diff" | tee -a "$dest/checks/STATUS"
        status=1
    fi
done
echo "code ${commit}; finished $(date -Iseconds)" >> "$dest/checks/STATUS"
exit $status
