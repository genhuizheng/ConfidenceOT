#!/bin/bash
#SBATCH -J cot_score_arms
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 06:00:00
#SBATCH -A MCB26031
#
# Score named configurations of one runs directory into a NEW folder, then pool
# and summarise them. Unlike metrics.slurm, which removes its metrics folder and
# rewrites the pooled table for every configuration, this touches nothing that
# exists: it refuses to start when its output folder already holds SUCCESS, and
# it writes nowhere else.
#
#   CONFIDENCEOT_BENCH_ROOT     benchmark root (conditions/, conditions_ds/, runs*/)
#   CONFIDENCEOT_SCORE_RUNS     runs directory under the root, e.g. runs_acceptance_0.99
#   CONFIDENCEOT_SCORE_ARMS     preprocessing labels to score, joined by + (or spaces)
#   CONFIDENCEOT_SCORE_OUT      output folder (must not hold SUCCESS)
#
# A label containing _ds reads the equalised conditions_ds/ manifests, as the
# runs did; every other label reads conditions/.
#
# No set -u, as everywhere in this repo.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:?set CONFIDENCEOT_BENCH_ROOT}
runs_dir=${CONFIDENCEOT_SCORE_RUNS:?set CONFIDENCEOT_SCORE_RUNS}
arms=${CONFIDENCEOT_SCORE_ARMS:?set CONFIDENCEOT_SCORE_ARMS}
arms=${arms//+/ }  # "+"-joined in --export, which splits on commas and spaces
out=${CONFIDENCEOT_SCORE_OUT:?set CONFIDENCEOT_SCORE_OUT}
parallel=${CONFIDENCEOT_BENCH_SCORE_PARALLEL:-12}

test ! -f "$out/SUCCESS" || { echo "$out/SUCCESS exists; not rescored" >&2; exit 2; }
mkdir -p "$out"
rm -f "$out/FAILED"
echo "scoring [$arms] from $root/$runs_dir into $out"

score_one() {
    local arm=$1 manifest=$2 tag=$3
    if ! OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python "$repo/benchmark/40_score.py" \
            --run-root "$root/$runs_dir/$arm" --preprocessing "$arm" \
            --manifest "$manifest" --scope all \
            --out "$out/${arm}__${tag}.csv" > "$out/${arm}__${tag}.log" 2>&1; then
        echo "${arm}__${tag}" >> "$out/FAILED"
    fi
}

for arm in $arms; do
    test -d "$root/$runs_dir/$arm" || { echo "no runs for $arm under $root/$runs_dir" >&2; exit 2; }
    tree=conditions
    [[ "$arm" == *_ds* ]] && tree=conditions_ds
    for manifest in "$root/$tree"/*/*/*/manifest.csv; do
        level_dir=$(dirname "$manifest")
        replicate_dir=$(dirname "$level_dir")
        tag=$(basename "$(dirname "$replicate_dir")")_$(basename "$replicate_dir")_$(basename "$level_dir")
        score_one "$arm" "$manifest" "$tag" &
        while (( $(jobs -rp | wc -l) >= parallel )); do
            wait -n || true
        done
    done
done
wait || true

if [[ -s "$out/FAILED" ]]; then
    echo "$(wc -l < "$out/FAILED") condition(s) failed to score; logs in $out" >&2
fi
python "$repo/benchmark/58_summarise_arms.py" "$out" --arms $arms
test ! -s "$out/FAILED" && echo "scored $arms" > "$out/SUCCESS"
echo "scoring finished"
