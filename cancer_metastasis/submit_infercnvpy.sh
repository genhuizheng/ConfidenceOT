#!/bin/bash
# infercnvpy on every patient unit of the three cancers, replacing R inferCNV.
#
#   bash cancer_metastasis/submit_infercnvpy.sh --dry-run
#   bash cancer_metastasis/submit_infercnvpy.sh
#   bash cancer_metastasis/submit_infercnvpy.sh --all-uniform   # every patient of the uniform manifest
#
# Units, read only:
#   ovarian GSE180661 and colorectal GSE225857: every (dataset, patient) of the uniform
#     manifest, unit <base>/infercnv_r/runs_patient/<dataset>/<patient>/
#   --all-uniform: every (dataset, patient) of the uniform manifest, whatever the dataset
#     (added 2026-10-21: the uniform block's other 8 datasets, 17 patients, had no profile,
#     so 21 of its 110 pairs could not run the CNV strategies). Same units, same method.
#   prostate GSE271675: /scratch/10119/ghzheng/checks/prostate/pilot/Patient{1,2,4,5}/
# Each unit's own counts, reference groups and gene order are used
# (49_infercnvpy_unit.py); only the method changes. The R inferCNV outputs and the
# 44_ export (cnv_profiles/infercnv_r_patient) are not touched.
#
# Output: <result>/cnv_profiles/infercnvpy_patient/<dataset>/<patient>/ with the full CNV
# matrix of every cell (cnv_matrix.npz, cnv_cells.tsv, cnv_windows.tsv), the 02_run_pair.py
# export (cnv.f32, cells.tsv, genes.tsv), diagnostics.json, provenance.json, SUCCESS.
# A unit with SUCCESS is skipped, so a resubmission only redoes the failed ones. After the
# array (whatever its state), one job writes summary.tsv over all units.
#
# Needs the environment made once on the login node (see the end of this file).
# No set -u, as everywhere in this repo.
set -eo pipefail

dry_run=0
tasks=8
all_uniform=0
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --tasks) tasks=$2; shift 2 ;;
        --all-uniform) all_uniform=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
base=${CNV_BASE:-/scratch/10119/ghzheng/primary_metastatic_cancer}
result=${CANCER_COT_ROOT:-$base/confidenceot_results}
env_path=${INFERCNVPY_ENV:-/scratch/10119/ghzheng/conda_envs/infercnvpy}
uniform_manifest=$result/manifest/factorial_20260929/uniform.csv
prostate_pilot=${PROSTATE_PILOT:-/scratch/10119/ghzheng/checks/prostate/pilot}
out=$result/cnv_profiles/infercnvpy_patient
logs=$out/logs
units=$out/units.tsv

[[ -f "$uniform_manifest" ]] || { echo "missing: $uniform_manifest" >&2; exit 1; }
[[ -x "$env_path/bin/python" ]] || { echo "no environment at $env_path (make it first, see the end of this file)" >&2; exit 1; }
"$env_path/bin/python" -c "import infercnvpy" 2> /dev/null || { echo "infercnvpy is not importable in $env_path" >&2; exit 1; }

# The unit list: dataset, patient, unit directory. Prostate units first, so the
# round robin puts each large prostate unit in a different task.
list=$(mktemp)
for p in Patient1 Patient2 Patient4 Patient5; do
    printf 'GSE271675\t%s\t%s\n' "$p" "$prostate_pilot/$p" >> "$list"
done
python3 - "$uniform_manifest" "$base/infercnv_r/runs_patient" "$all_uniform" >> "$list" <<'PY'
import csv, sys
seen = []
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        key = (row["dataset_id"], row["patient_id"])
        if (sys.argv[3] == "1" or key[0] in ("GSE180661", "GSE225857")) and key not in seen:
            seen.append(key)
for d, p in seen:
    print(f"{d}\t{p}\t{sys.argv[2]}/{d}/{p}")
PY
missing=0
while IFS=$'\t' read -r d p u; do
    for f in counts.tsv.gz annotations.tsv gene_order.tsv cell_meta.tsv.gz; do
        [[ -f "$u/$f" ]] || { echo "missing: $u/$f" >&2; missing=1; }
    done
done < "$list"
n=$(wc -l < "$list")
echo "$n units: $(cut -f1 "$list" | sort | uniq -c | tr '\n' ' ')" >&2
if (( missing )); then rm -f "$list"; echo "inputs missing; nothing submitted" >&2; exit 1; fi

if (( dry_run )); then
    cat "$list" >&2
    rm -f "$list"
    echo "DRY RUN  array of $tasks tasks over $n units, then the summary; nothing submitted"
    exit 0
fi
mkdir -p "$logs"
cp "$list" "$units"
rm -f "$list"

activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1"
loop="k=0; failed=0; while IFS=\$'\\t' read -r d p u; do if (( k % $tasks == SLURM_ARRAY_TASK_ID )); then if [[ -f $out/\$d/\$p/SUCCESS ]]; then echo SKIP \$d \$p; else echo START \$d \$p; python cancer_metastasis/49_infercnvpy_unit.py --unit \$u --dataset \$d --patient \$p --out $out && echo DONE \$d \$p || { echo FAILED \$d \$p; failed=1; }; fi; fi; k=\$((k+1)); done < $units; exit \$failed"
array=$(sbatch --parsable -p gg -N 1 -n 1 -t 08:00:00 -A MCB26031 -J icnvpy --array=0-$((tasks - 1)) \
    -o "$logs/unit_%A_%a.out" -e "$logs/unit_%A_%a.err" --wrap "$activate && $loop" | tail -n 1)
[[ "$array" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; exit 1; }
summary="import json, pathlib, csv, sys
root = pathlib.Path('$out'); rows = []
for line in open(root / 'units.tsv'):
    d, p, u = line.rstrip('\\n').split('\\t'); f = root / d / p / 'diagnostics.json'
    if f.exists():
        j = json.loads(f.read_text()); rows.append([d, p, 'done', j['cells'], j['reference_cells'], j['observation_cells'], j['windows'], round(j['signal_median']['reference'], 4), round(j['signal_median']['observation'], 4)])
    else:
        rows.append([d, p, 'missing'] + [''] * 6)
w = csv.writer(open(root / 'summary.tsv', 'w', newline=''), delimiter='\\t')
w.writerow(['dataset', 'patient', 'status', 'cells', 'reference', 'observation', 'windows', 'signal_ref', 'signal_obs'])
w.writerows(rows); print(sum(r[2] == 'done' for r in rows), 'of', len(rows), 'units done')"
sbatch --parsable -p gg -N 1 -n 1 -t 00:30:00 -A MCB26031 -J icnvpy_sum --dependency=afterany:$array \
    -o "$logs/summary_%j.out" --wrap "$activate && python -c \"$summary\"" > /dev/null
echo "array $array ($tasks tasks over $n units), then the summary"
echo "out $out"
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)"

# The environment, once, on the login node:
#   source /home1/10119/ghzheng/.bashrc
#   conda create -y -p /scratch/10119/ghzheng/conda_envs/infercnvpy python=3.11
#   conda activate /scratch/10119/ghzheng/conda_envs/infercnvpy
#   pip install infercnvpy==0.6.1
