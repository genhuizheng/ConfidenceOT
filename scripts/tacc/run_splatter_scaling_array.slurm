#!/bin/bash
#SBATCH -J cot_splat_scale
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/benchmark_scaling/logs/fit_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/benchmark_scaling/logs/fit_%A_%a.err

set -euo pipefail

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_SCALING_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_scaling}
n_cells=${CONFIDENCEOT_N_CELLS:?Set CONFIDENCEOT_N_CELLS}
worker_id=${SLURM_ARRAY_TASK_ID:?Submit as a worker array}
worker_count=${CONFIDENCEOT_WORKER_COUNT:-8}

case "$n_cells" in
    1000|5000|10000|20000) ;;
    *) echo "Unsupported benchmark size: $n_cells" >&2; exit 2 ;;
esac
if (( worker_id < 0 || worker_id >= worker_count )); then
    echo "Worker $worker_id is outside [0,$((worker_count - 1))]" >&2
    exit 2
fi

export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

scenarios=(S1_extinction S2_emergence S3_source_outlier S4_bifurcation S5_abundance_shift)
printf -v size_dir "n_%04d" "$n_cells"

cd "$repo"
for ((case_index=worker_id; case_index<20; case_index+=worker_count)); do
    index=$case_index
    scenario=${scenarios[$((index % 5))]}
    index=$((index / 5))
    replicate=$((index % 2 + 1))
    index=$((index / 2))
    if [[ "$index" -eq 0 ]]; then
        batch=none
    else
        batch=mild
    fi

    data_root="$root/data/$batch"
    m4_output="$root/raw/m4/$batch/$scenario/$size_dir/rep_$(printf '%02d' "$replicate")"
    partial_output="$root/raw/partial/$batch/$scenario/$size_dir/rep_$(printf '%02d' "$replicate")"

    if [[ -f "$m4_output/manifest.json" ]]; then
        echo "M4 already complete: $m4_output"
    elif [[ -e "$m4_output" ]]; then
        echo "Incomplete M4 output exists: $m4_output" >&2
        exit 4
    else
        mkdir -p "$m4_output"
        python scripts/run_splatter_dual_backbone_calibrated_case.py \
            "$data_root" "$m4_output" \
            --scenario "$scenario" \
            --batch-condition "$batch" \
            --n-cells "$n_cells" \
            --replicate "$replicate" \
            --gene-fractions 0 0.05 0.10 0.20 \
            --fold-change 2.0 \
            --n-hvg 500 --n-pcs 20 \
            --epsilon 0.1 --lambda-a 1.0 --lambda-b 1.0 \
            --source-budget 0.15 --target-budget 0.15 \
            --solver-tolerance 1e-3 --outer-cap 30 \
            --hard-grid-size 7 \
            --calibration-null-replicates 2 \
            --validation-null-replicates 2 \
            --null-escape-fraction 0.10 \
            --fixed-binary-c 0.5 \
            --workers 1 \
            --calibration-workers 1 \
            --calibration-backbone-workers 1 \
            --device cuda --cuda-dtype float32 \
            --skip-soft-gate \
            --skip-coupling-stability
    fi

    if (( n_cells > 1000 )); then
        echo "Partial OT is N/A at N=$n_cells: exact augmented Hungarian comparator is cubic"
    elif [[ -f "$partial_output/manifest.json" ]]; then
        echo "Partial OT already complete: $partial_output"
    elif [[ -e "$partial_output" ]]; then
        echo "Incomplete Partial OT output exists: $partial_output" >&2
        exit 5
    else
        mkdir -p "$partial_output"
        python scripts/run_splatter_partial_ot_comparator_case.py \
            "$data_root" "$partial_output" \
            --scenario "$scenario" \
            --batch-condition "$batch" \
            --n-cells "$n_cells" \
            --replicate "$replicate" \
            --gene-fractions 0 0.05 0.10 0.20 \
            --fold-change 2.0 \
            --n-hvg 500 --n-pcs 20 \
            --source-budget 0.15 --target-budget 0.15
    fi
done
