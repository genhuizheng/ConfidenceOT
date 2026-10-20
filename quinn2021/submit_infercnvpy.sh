#!/bin/bash
# infercnvpy on Quinn et al. 2021 (GSE161363), one unit per experiment, no reference.
#
#   bash quinn2021/submit_infercnvpy.sh --dry-run
#   bash quinn2021/submit_infercnvpy.sh
#
# 1. One job builds the units (quinn2021/02_build_infercnvpy_units.py) from the converted
#    H5ADs, gene positions from GENCODE v28 (GRCh38) or v19 (GRCh37) by each matrix's reference.
# 2. An array runs cancer_metastasis/49_infercnvpy_unit.py --unit-h5ad --no-reference:
#    infercnvpy at its package defaults, the reference being its default, the mean of all
#    cells (every cell is an A549 tumour cell). A unit with SUCCESS is skipped.
# 3. A summary job writes summary.tsv.
# Output: <root>/cnv_profiles/infercnvpy_experiment/Quinn2021/<experiment>/, the same files
# as the other datasets (full CNV matrix of every cell, the 02_run_pair.py export).
# No set -u, as everywhere in this repo.
set -eo pipefail

dry_run=0
tasks=5
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --tasks) tasks=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
data=${QUINN_DATA:-/scratch/10119/ghzheng/OT_project/data/quinn2021_GSE161363}
root=${QUINN_ROOT:-/scratch/10119/ghzheng/OT_project/quinn2021_confidenceot}
env_path=${INFERCNVPY_ENV:-/scratch/10119/ghzheng/conda_envs/infercnvpy}
reference=/scratch/10119/ghzheng/OT_project/data/reference
gtf38=$reference/gencode_v28/gencode.v28.annotation.gtf.gz
gtf37=$reference/gencode_v19/gencode.v19.annotation.gtf.gz
converted=$data/converted
units=$root/cnv_units/infercnvpy
out=$root/cnv_profiles/infercnvpy_experiment
logs=$out/logs

for path in "$gtf38" "$gtf37" "$converted/conversion_report.json"; do
    [[ -f "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done
[[ -x "$env_path/bin/python" ]] || { echo "no environment at $env_path" >&2; exit 1; }
build=1
[[ -f "$units/units.tsv" ]] && { echo "$units/units.tsv exists; the units are not rebuilt" >&2; build=0; }
echo "converted $converted ($(ls $converted/h5ad | wc -l) H5ADs)" >&2
if (( dry_run )); then
    echo "DRY RUN  build units: $build; array of $tasks tasks, --no-reference; summary. Nothing submitted"
    exit 0
fi
mkdir -p "$logs" "$units"

activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1"
wait_build=()
if (( build )); then
    id=$(sbatch --parsable -p gg -N 1 -n 1 -t 01:00:00 -A MCB26031 -J qn_cnvunits -o "$logs/units_%j.out" \
        --wrap "$activate && python quinn2021/02_build_infercnvpy_units.py --converted $converted --gtf-grch38 $gtf38 --gtf-grch37 $gtf37 --out $units" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; exit 1; }
    echo "units job $id" >&2
    wait_build=(--dependency=afterok:$id)
fi
loop="k=0; failed=0; while IFS=\$'\\t' read -r d p u; do if (( k % $tasks == SLURM_ARRAY_TASK_ID )); then if [[ -f $out/\$d/\$p/SUCCESS ]]; then echo SKIP \$d \$p; else echo START \$d \$p; python cancer_metastasis/49_infercnvpy_unit.py --unit-h5ad \$u --no-reference --dataset \$d --patient \$p --out $out && echo DONE \$d \$p || { echo FAILED \$d \$p; failed=1; }; fi; fi; k=\$((k+1)); done < $units/units.tsv; exit \$failed"
array=$(sbatch --parsable -p gg -N 1 -n 1 -t 04:00:00 -A MCB26031 -J qn_icnvpy --array=0-$((tasks - 1)) "${wait_build[@]}" \
    -o "$logs/unit_%A_%a.out" -e "$logs/unit_%A_%a.err" --wrap "$activate && $loop" | tail -n 1)
[[ "$array" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; exit 1; }
summary="import json, pathlib, csv
root = pathlib.Path('$out'); rows = []
for line in open('$units/units.tsv'):
    d, p, u = line.rstrip('\\n').split('\\t'); f = root / d / p / 'diagnostics.json'
    if f.exists():
        j = json.loads(f.read_text()); rows.append([d, p, 'done', j['cells'], j['observation_cells'], j['windows'], round(j['signal_median']['observation'], 4)])
    else:
        rows.append([d, p, 'missing'] + [''] * 4)
w = csv.writer(open(root / 'summary.tsv', 'w', newline=''), delimiter='\\t')
w.writerow(['dataset', 'experiment', 'status', 'cells', 'observation', 'windows', 'signal_obs'])
w.writerows(rows); print(sum(r[2] == 'done' for r in rows), 'of', len(rows), 'experiments done')"
sbatch --parsable -p gg -N 1 -n 1 -t 00:30:00 -A MCB26031 -J qn_icnvpy_sum --dependency=afterany:$array \
    -o "$logs/summary_%j.out" --wrap "$activate && python -c \"$summary\"" > /dev/null
echo "array $array ($tasks tasks), then the summary"
echo "units $units"
echo "out $out"
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)"
