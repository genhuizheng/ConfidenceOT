#!/bin/bash
#SBATCH -J cot_device_check
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/devcheck_%j.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/devcheck_%j.err
#
# One condition, solved twice: once on the CPU and once on the GPU, and then
# compared.
#
# The question is not whether the GPU is faster. It is whether the two agree,
# because api.py names the CPU implementation the numerical reference and
# anything the GPU changes about the answer is a change to the benchmark
# rather than to its runtime. The cost matrix is built by the same NumPy code
# on both paths, so cost_scale, cost_median and cost_max must come back
# **identical**, not merely close -- a difference there means the geometry
# moved and nothing downstream of it is comparable. The gate and the decision
# costs are Sinkhorn output and are allowed a tolerance.
#
# Submit with the partition and any GPU resource syntax on the command line,
# because those are site configuration and this file should not guess them:
#
#   sbatch -p <gpu partition> [--gres=...] benchmark/tacc/device_check.slurm
#
# No set -u. TACC's module files and conda's shell functions read variables
# they do not define, and under set -u that kills the job in seconds with
# nothing in the log.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_technical}
label=${CONFIDENCEOT_DEVICE_CHECK_LABEL:-logcpm}
out=${CONFIDENCEOT_DEVICE_CHECK_ROOT:-$root/device_check}
index=${CONFIDENCEOT_DEVICE_CHECK_INDEX:-0}
workers=${CONFIDENCEOT_WORKERS:-8}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis:$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$root/logs" "$out"

# Which condition. The largest size is where a GPU would earn its place, so
# it is the default; a smaller one is faster to iterate on.
manifest=${CONFIDENCEOT_DEVICE_CHECK_MANIFEST:-}
if [[ -z "$manifest" ]]; then
    shopt -s nullglob
    candidates=("$root"/conditions/N10000/rep1/L2_depth_dropout/manifest.csv
                "$root"/conditions/*/*/*/manifest.csv)
    manifest=${candidates[0]}
fi
if [[ ! -f "$manifest" ]]; then
    echo "no condition manifest found; pass CONFIDENCEOT_DEVICE_CHECK_MANIFEST" >&2
    exit 2
fi
echo "manifest    $manifest"
echo "label       $label"
echo "index       $index"

echo
echo "=== is CUDA actually here ==="
python - <<'PY'
import torch

print(f"torch            {torch.__version__}")
print(f"torch.version.cuda {torch.version.cuda}")
print(f"cuda available   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device           {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"memory           {free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
else:
    raise SystemExit("no CUDA on this node: the comparison has nothing to compare")
PY

allow=()
[[ "$label" == raw* ]] && allow=(--allow-precomputed)

for device in cpu cuda; do
    echo
    echo "===== $device $(date --iso-8601=seconds) ====="
    start=$(date +%s)
    /usr/bin/time -v python cancer_metastasis/02_run_pair.py \
        "$manifest" "$out/$device" \
        --index "$index" \
        --analysis-scope all \
        "${allow[@]}" \
        --preprocessing "$label" \
        --calibration-null within_side_split \
        --null-calibration-replicates 5 \
        --null-validation-replicates 5 \
        --calibration-grid-size 5 \
        --tolerance 1e-4 \
        --workers "$workers" \
        --device "$device" 2> "$out/${device}_time.txt" || {
            echo "$device failed" >&2; tail -20 "$out/${device}_time.txt" >&2; exit 1; }
    echo "$device wall $(( $(date +%s) - start ))s"
    grep -E "Maximum resident set size|Elapsed \(wall" "$out/${device}_time.txt" || true
done

echo
echo "===== do they agree ====="
python "$repo/benchmark/60_compare_devices.py" "$out/cpu" "$out/cuda" \
    --out "$out/device_agreement.json"
