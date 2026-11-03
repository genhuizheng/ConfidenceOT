#!/bin/bash
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -A MCB26031
# One replicate of the N = 5,000 benchmark: every level generated (20_generate_n5000.R in the Splatter container),
# L0-L2 compared with the native generation (25_check_native_identity.py), then the conditions and truth
# (21_build_conditions_n5000.py). Submitted by submit_n5000.sh with RB5_ROOT, RB5_REP, RB5_NATIVE_ROOT.
# Seeds as the native benchmark: base 20260925 + 1000 x replicate + 137 x level index (L0 0, L1 1, L2 2); L3-L5 and
# the ablations take L2's seed, so they are the L2 simulation plus their factors. RB5_PARAM_SET (benchmark | splatter_default,
# default benchmark) is passed to 20_generate_n5000.R --param-set. A level already generated
# (GENERATION_DONE) or built (manifest.csv) is skipped. The levels are independent (each its own Splatter calls, seed,
# native check and conditions), so they run concurrently, one process each (R is single-threaded; about 6-8 GB per
# level at N = 5,000); RB5_SERIAL=1 runs them one after another as before. No set -e / -u / pipefail (TACC rules).

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
cd "$repo" || exit 1
rscript=${CONFIDENCEOT_RSCRIPT:-$repo/benchmark/tacc/rscript.sh}
rep=$RB5_REP
# RB5_LEVELS (space-separated) restricts the levels, e.g. the pilot's "L2_depth_dropout L4_batch_perturb"
levels=(${RB5_LEVELS:-L0_matched L1_depth L2_depth_dropout L3_batch L4_batch_perturb L5_batch_perturb_imbalance A4_perturb_only A5_imbalance_only})
echo "replicate $rep, commit $(git rev-parse --short HEAD), $(date -Iseconds)"
one_level() {  # $1 level; returns non-zero if its generation or conditions failed
    local level=$1 idx seed gen cond
    case $level in
        L0_matched) idx=0 ;;
        L1_depth) idx=1 ;;
        *) idx=2 ;;
    esac
    seed=$(( 20260925 + 1000 * rep + 137 * idx ))
    gen=$RB5_ROOT/generated/N5000/rep$rep/$level
    cond=$RB5_ROOT/conditions/N5000/rep$rep/$level
    if [[ ! -f "$gen/GENERATION_DONE" ]]; then
        echo "== generate $level seed $seed"
        $rscript realdata_benchmark/splatter_n5000/20_generate_n5000.R --out "$gen" --n 5000 --replicate "$rep" --level "$level" --seed "$seed" --param-set "${RB5_PARAM_SET:-benchmark}" \
            || { echo "generation FAILED: $level"; return 1; }
    fi
    source /home1/10119/ghzheng/.bashrc
    conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
    export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
    if [[ $idx -lt 2 || $level == L2_depth_dropout ]] && [[ ! -f "$gen/NATIVE_IDENTITY.json" ]]; then
        python realdata_benchmark/splatter_n5000/25_check_native_identity.py --new "$gen" --native "$RB5_NATIVE_ROOT/generated/N5000/rep$rep/$level" \
            || echo "NOTE $level rep $rep is not identical to the native generation; its rows will be refitted"
    fi
    if [[ ! -f "$cond/manifest.csv" ]]; then
        python realdata_benchmark/splatter_n5000/21_build_conditions_n5000.py --generated "$gen" --out "$cond" --n-cells 5000 --replicate "$rep" --level "$level" \
            || { echo "conditions FAILED: $level"; return 1; }
    fi
    echo "== $level done $(date -Iseconds)"
}
status=0
if [[ ${RB5_SERIAL:-0} == 1 ]]; then
    for level in "${levels[@]}"; do one_level "$level" || status=1; done
else
    pids=()
    for level in "${levels[@]}"; do
        ( one_level "$level" ) > "$RB5_ROOT/logs/generate_rep${rep}_${level}_${SLURM_JOB_ID:-local}.log" 2>&1 &
        pids+=("$!")
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "level ${levels[$i]} FAILED (log generate_rep${rep}_${levels[$i]}_${SLURM_JOB_ID:-local}.log)"; status=1; }
    done
    cat "$RB5_ROOT"/logs/generate_rep${rep}_*_${SLURM_JOB_ID:-local}.log
fi
echo "replicate $rep finished $(date -Iseconds), status $status"
exit $status
