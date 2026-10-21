#!/bin/bash
# infercnvpy on KP-Tracer, one unit per mouse, as for the three human cancers.
#
#   bash kp_lineage/cnv/submit_infercnvpy_kp.sh --dry-run
#   bash kp_lineage/cnv/submit_infercnvpy_kp.sh
#
# 1. One job builds the units (kp_lineage/cnv/01_build_infercnvpy_units.py): per mouse with
#    an eligible pair, all its tumour H5ADs from prepared/h5ad, reference = the authors'
#    normal cells, gene positions from GENCODE vM23 (GRCm38/mm10).
# 2. An array runs cancer_metastasis/49_infercnvpy_unit.py --unit-h5ad on every unit,
#    infercnvpy 0.6.1 with its infercnv parameters at their defaults, the reference being the
#    unit's ref_normal cells (not the package default, the mean of all cells), input
#    normalize_total(1e4) + log1p; a unit with SUCCESS is skipped.
# 3. A summary job writes summary.tsv over all mice.
# Output: <root>/cnv_profiles/infercnvpy_mouse/KPTracer/<mouse>/, the same files as for the
# human units (full CNV matrix of every cell, the 02_run_pair.py export, diagnostics).
# Reads prepared/ only; the OT runs, readouts and validation outputs are not touched.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

dry_run=0
tasks=4
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --tasks) tasks=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${KPTRACER_ROOT:-/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot}
env_path=${INFERCNVPY_ENV:-/scratch/10119/ghzheng/conda_envs/infercnvpy}
gtf=${GENCODE_GTF:-/scratch/10119/ghzheng/OT_project/data/reference/gencode_vM23/gencode.vM23.annotation.gtf.gz}
prepared=$root/prepared
units=$root/cnv_units/infercnvpy
out=$root/cnv_profiles/infercnvpy_mouse
logs=$out/logs

for path in "$gtf" "$prepared/pair_manifest_eligible.csv" "$prepared/kptracer_tumours.csv"; do
    [[ -f "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done
[[ -d "$prepared/h5ad" ]] || { echo "missing: $prepared/h5ad" >&2; exit 1; }
[[ -x "$env_path/bin/python" ]] || { echo "no environment at $env_path" >&2; exit 1; }
if [[ -f "$units/units.tsv" ]]; then
    echo "$units/units.tsv exists; the units are not rebuilt" >&2
    build=0
else
    build=1
fi
echo "prepared $prepared ($(ls $prepared/h5ad | wc -l) tumour H5ADs), gtf $gtf" >&2
if (( dry_run )); then
    echo "DRY RUN  build units: $build; array of $tasks tasks; summary. Nothing submitted"
    exit 0
fi
mkdir -p "$logs" "$units"

activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1"
wait_build=()
if (( build )); then
    id=$(sbatch --parsable -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J kp_cnvunits -o "$logs/units_%j.out" \
        --wrap "$activate && python kp_lineage/cnv/01_build_infercnvpy_units.py --prepared $prepared --gtf $gtf --out $units" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; exit 1; }
    echo "units job $id" >&2
    wait_build=(--dependency=afterok:$id)
fi
loop="k=0; failed=0; while IFS=\$'\\t' read -r d p u; do if (( k % $tasks == SLURM_ARRAY_TASK_ID )); then if [[ -f $out/\$d/\$p/SUCCESS ]]; then echo SKIP \$d \$p; else echo START \$d \$p; python cancer_metastasis/49_infercnvpy_unit.py --unit-h5ad \$u --dataset \$d --patient \$p --out $out && echo DONE \$d \$p || { echo FAILED \$d \$p; failed=1; }; fi; fi; k=\$((k+1)); done < $units/units.tsv; exit \$failed"
array=$(sbatch --parsable -p gg -N 1 -n 1 -t 06:00:00 -A MCB26031 -J kp_icnvpy --array=0-$((tasks - 1)) "${wait_build[@]}" \
    -o "$logs/unit_%A_%a.out" -e "$logs/unit_%A_%a.err" --wrap "$activate && $loop" | tail -n 1)
[[ "$array" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; exit 1; }
summary="import json, pathlib, csv
root = pathlib.Path('$out'); rows = []
for line in open('$units/units.tsv'):
    d, p, u = line.rstrip('\\n').split('\\t'); f = root / d / p / 'diagnostics.json'
    if f.exists():
        j = json.loads(f.read_text()); rows.append([d, p, 'done', j['cells'], j['reference_cells'], j['observation_cells'], j['windows'], round(j['signal_median']['reference'], 4), round(j['signal_median']['observation'], 4)])
    else:
        rows.append([d, p, 'missing'] + [''] * 6)
w = csv.writer(open(root / 'summary.tsv', 'w', newline=''), delimiter='\\t')
w.writerow(['dataset', 'mouse', 'status', 'cells', 'reference', 'observation', 'windows', 'signal_ref', 'signal_obs'])
w.writerows(rows); print(sum(r[2] == 'done' for r in rows), 'of', len(rows), 'mice done')"
sbatch --parsable -p gg -N 1 -n 1 -t 00:30:00 -A MCB26031 -J kp_icnvpy_sum --dependency=afterany:$array \
    -o "$logs/summary_%j.out" --wrap "$activate && python -c \"$summary\"" > /dev/null
echo "array $array ($tasks tasks), then the summary"
echo "units $units"
echo "out $out"
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)"
