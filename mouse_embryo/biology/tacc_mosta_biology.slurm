#!/bin/bash
#SBATCH -J mosta_bio
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -t 12:00:00
#SBATCH -A MCB26031
#
# One step of the MOSTA biological application (mouse_embryo/biology/31_-33_)
# on one gg node, CPU.  Reads the finished MOSTA runs, writes only under
# MOSTA_BIO_OUT.  Submitted by submit_mosta_biology.sh; MOSTA_BIO_STEP picks:
#   audit                 31_ (all three runs, with the coupling probe), then 34_ (every annotation, ds and logcpm)
#   extract               32_ for MOSTA_BIO_RUN (ds | logcpm | std) and MOSTA_BIO_SETS
#                         (space-separated organ sets); MOSTA_BIO_EXPRESSION=1
#                         adds the marker expression, MOSTA_BIO_M4R=1 the M4-R
#                         couplings rebuilt from potentials (logcpm only)
#   case                  33_ for each organ set in MOSTA_BIO_SETS, runs ds and logcpm
# No set -e, set -u or pipefail: the python step's status is the job's status.
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}" || exit 1
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
cd "$repo" || exit 1
export PYTHONPATH="$repo/src:$repo/mouse_embryo:$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=4
out=${MOSTA_BIO_OUT:?MOSTA_BIO_OUT is not set}
results=/scratch/10119/ghzheng/OT_project/mouse_embryo_results
step=${MOSTA_BIO_STEP:?MOSTA_BIO_STEP is not set}
echo "job=$SLURM_JOB_ID host=$(hostname) commit=$(git -C "$repo" rev-parse HEAD) step=$step run=${MOSTA_BIO_RUN:-} sets=${MOSTA_BIO_SETS:-} start=$(date --iso-8601=seconds)"
case "$step" in
    audit)
        python mouse_embryo/biology/31_audit_runs.py --out "$out" --probe || exit $?
        python mouse_embryo/biology/34_systematic_annotations.py --out "$out" \
            --run ds=$results/mosta_all_stages --run logcpm=$results/mosta_all_stages_logcpm || exit $?
        ;;
    extract)
        run=${MOSTA_BIO_RUN:?MOSTA_BIO_RUN is not set}
        case "$run" in
            ds) root=$results/mosta_all_stages ;;
            logcpm) root=$results/mosta_all_stages_logcpm ;;
            std) root=$results/mosta_all_stages_std ;;
            *) echo "MOSTA_BIO_RUN must be ds, logcpm or std" >&2; exit 2 ;;
        esac
        args=()
        for s in ${MOSTA_BIO_SETS:?MOSTA_BIO_SETS is not set}; do args+=(--organ-set "$s"); done
        [[ "${MOSTA_BIO_EXPRESSION:-0}" == 1 ]] && args+=(--expression)
        [[ "${MOSTA_BIO_M4R:-0}" == 1 ]] && args+=(--m4r-coupling --threads 144)
        python mouse_embryo/biology/32_extract_organ_bins.py --run-name "$run" --run-root "$root" --out "$out" \
            --workers 48 "${args[@]}" || exit $?
        ;;
    case)
        for s in ${MOSTA_BIO_SETS:?MOSTA_BIO_SETS is not set}; do
            python mouse_embryo/biology/33_organ_case.py --out "$out" --organ-set "$s" --runs ds logcpm std \
                --run-root ds=$results/mosta_all_stages --run-root logcpm=$results/mosta_all_stages_logcpm \
                --run-root std=$results/mosta_all_stages_std \
                ${MOSTA_BIO_FOCAL:+--focal-group "$MOSTA_BIO_FOCAL"} || exit $?
        done
        ;;
    *) echo "unknown MOSTA_BIO_STEP $step" >&2; exit 2 ;;
esac
echo "end=$(date --iso-8601=seconds)"
