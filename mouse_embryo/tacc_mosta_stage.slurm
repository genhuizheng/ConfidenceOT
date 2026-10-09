#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A MCB26031
#
# One stage of the all-stage MOSTA run, chosen by MOSTA_STAGE.  Submitted by
# mouse_embryo/submit_mosta_all_stages.sh, which sets the partition, the time,
# the log paths, the stage and the device, and orders the stages:
#
#   prepare            20_ manifest + 21_ common-depth equalisation         gg
#   representation     22_ ranknm256_noscale_ds_cos for every pair           gg
#   equivalence        24_ dense against blockwise on the early pairs        gg (cpu) / gh (cuda)
#                      cpu: one shard of the test pairs (MOSTA_SHARD of MOSTA_N_SHARDS)
#                      cuda: all of them, then, only on PASS, 23_ on the
#                      largest pair as a scale probe (a normal, published result)
#   equivalence_merge  24_ --merge: the shards' reports into one PASS or FAIL  gg
#   ot                 23_ calibration, M4-E, M4-R, balanced OT, summaries   gg (cpu) / gh (cuda)
#   collect            25_ audit, tables, pseudobulks, download list         gg
#
# MOSTA_DEVICE (cuda or cpu, default cuda) selects the device and with it the
# production path of that device: on the CPU, calibration through the NumPy
# reference and the torch solvers in float64, as the cancer runs on gg do; on
# the GPU, the torch CUDA solver in float32.  One run uses one device for every
# pair.
#
# Several representation or ot jobs walk the same pair list; claim files keep
# them off each other's pairs.
#
# No set -u, as everywhere in this repo: TACC's module files and conda's shell
# functions read variables they do not define.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${MOSTA_ROOT:-/scratch/10119/ghzheng/OT_project/mouse_embryo_results/mosta_all_stages}
data=${MOSTA_DATA:-/scratch/10119/ghzheng/OT_project/data}
stage=${MOSTA_STAGE:-}
device=${MOSTA_DEVICE:-cuda}
threads=${MOSTA_THREADS:-16}
export PYTHONPATH="$repo/src:$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads NUMEXPR_NUM_THREADS=$threads
CONFIDENCEOT_COMMIT=$(git -C "$repo" rev-parse HEAD 2> /dev/null || echo unknown)
export CONFIDENCEOT_COMMIT
cd "$repo"
echo "stage=$stage device=$device threads=$threads root=$root commit=$CONFIDENCEOT_COMMIT host=$(hostname) start=$(date --iso-8601=seconds)"
if [[ "$device" != cpu && "$device" != cuda ]]; then
    echo "MOSTA_DEVICE must be cpu or cuda; got '$device'" >&2
    exit 2
fi
if [[ "$device" == cuda ]]; then
    nvidia-smi || true
fi

case "$stage" in
    prepare)
        python mouse_embryo/20_build_mosta_manifest.py "$data" "$root"
        python mouse_embryo/21_equalise_mosta_depth.py "$root" --workers "${MOSTA_WORKERS:-4}"
        ;;
    representation)
        python mouse_embryo/22_prepare_mosta_representation.py "$root" --workers "${MOSTA_WORKERS:-4}"
        ;;
    equivalence)
        if [[ "$device" == cuda ]]; then
            python mouse_embryo/24_check_blockwise_equivalence.py "$root" --device cuda
            # Reached only on PASS: set -e stops the job when 24_ exits 1.
            python mouse_embryo/23_run_mosta_pair.py "$root" --device cuda --largest
        else
            python mouse_embryo/24_check_blockwise_equivalence.py "$root" --device cpu \
                --shard "${MOSTA_SHARD:-0}" --n-shards "${MOSTA_N_SHARDS:-1}"
        fi
        ;;
    equivalence_merge)
        python mouse_embryo/24_check_blockwise_equivalence.py "$root" --merge
        ;;
    ot)
        python mouse_embryo/23_run_mosta_pair.py "$root" --device "$device" \
            --stop-claiming-after-hours "${MOSTA_STOP_HOURS:-40}"
        ;;
    collect)
        python mouse_embryo/25_collect_mosta.py "$root"
        ;;
    *)
        echo "MOSTA_STAGE must be prepare, representation, equivalence, equivalence_merge, ot or collect; got '$stage'" >&2
        exit 2
        ;;
esac
echo "stage=$stage end=$(date --iso-8601=seconds)"
