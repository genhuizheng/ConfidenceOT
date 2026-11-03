#!/bin/bash
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
#SBATCH -J rb5_batch_audit
# Batch-effect audit of the N = 5,000 benchmark (read only): Splatter defaults and batch / dropout code from the same
# container (28_splatter_defaults.R), then 27_batch_effect_audit.py on the generated data and the all_shared
# conditions. Output BATCH_EFFECT_AUDIT_<date> under the root (both scripts refuse to overwrite), then a tarball.
#   sbatch -o ROOT/logs/batch_audit_%j.out realdata_benchmark/splatter_n5000/tacc/batch_effect_audit.slurm
# No set -e / -u / pipefail (TACC rules).

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${RB5_ROOT:-/scratch/10119/ghzheng/OT_project/splatter_n5000_benchmark_20261101}
out=$root/BATCH_EFFECT_AUDIT_$(date +%Y%m%d)
cd "$repo" || exit 1
[[ -e "$out" ]] && { echo "$out exists; not redone"; exit 2; }
echo "batch audit, commit $(git rev-parse --short HEAD), $(date -Iseconds) -> $out"
${CONFIDENCEOT_RSCRIPT:-$repo/benchmark/tacc/rscript.sh} realdata_benchmark/splatter_n5000/28_splatter_defaults.R --out "$out/splatter" || echo "Splatter defaults FAILED"
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
python realdata_benchmark/splatter_n5000/27_batch_effect_audit.py --root "$root" --out "$out/tables" || echo "audit FAILED"
cd "$root" && tar -cf "batch_effect_audit_$(date +%Y%m%d).tar" "$(basename "$out")"
echo "finished $(date -Iseconds)"
