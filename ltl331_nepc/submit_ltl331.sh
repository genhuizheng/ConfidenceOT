#!/bin/bash
# GSE297328 (LTL331 PDX neuroendocrine transdifferentiation time course) through the project's pipeline.
#
#   bash ltl331_nepc/submit_ltl331.sh [--dry-run] convert|prepare|run|all
#
# convert  01_convert_geo.py on a compute node: one raw-count H5AD per time point (the authors' QC cells and labels),
#          cross-checked against the per-sample CellRanger matrices of GSE297328_RAW.tar
# prepare  02_build_manifest.py (7 adjacent time-point pairs), 27_downsample_counts.py (project QC, depth equalised to
#          the pooled 10th percentile), 03_raw_on_ds_cells.py (original counts of exactly the equalised cells)
# run      tacc_preprocessing_factorial.slurm, one array task per label: ranknm256_noscale_ds_cos (equalised counts)
#          and logcpm (original counts, the same cells); ConfidenceOT M4-E / M4-R, within-side split null,
#          acceptance 0.99, no cell cap, couplings saved
# all      convert -> prepare -> run, each afterok the previous
# Download first, once, on the login node (commands at the end). Nothing outside $data and $root is written; a stage
# whose output exists is refused, never overwritten. No set -e / -u / pipefail (TACC rules); each step checks itself.

dry_run=0; stage=""
for a in "$@"; do
    case "$a" in
        --dry-run) dry_run=1 ;;
        convert|prepare|run|all) stage=$a ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done
[[ -n "$stage" ]] || { echo "usage: bash ltl331_nepc/submit_ltl331.sh [--dry-run] convert|prepare|run|all" >&2; exit 2; }
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
data=${LTL331_DATA:-/scratch/10119/ghzheng/OT_project/data/ltl331_GSE297328}
root=${LTL331_ROOT:-/scratch/10119/ghzheng/OT_project/ltl331_confidenceot}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
cap=${CONFIDENCEOT_JOB_CAP:-40}
labels="ranknm256_noscale_ds_cos logcpm"
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo"

annotated=$data/GSE297328_ltl331_annotated.h5ad
converted=$data/converted
inputs=$root/inputs
logs=$root/logs

submit() {  # prints the job id; a failure stops the caller through its own || exit
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-500 >&2; echo DRYRUN; return 0; fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; return 1; }
    echo "$id"
}

queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l)
need=0
[[ $stage == convert || $stage == all ]] && need=$(( need + 1 ))
[[ $stage == prepare || $stage == all ]] && need=$(( need + 1 ))
[[ $stage == run || $stage == all ]] && need=$(( need + 2 ))
if (( ! dry_run && queued + need > cap )); then echo "$queued queued; $need more would pass the cap of $cap" >&2; exit 3; fi
(( dry_run )) || mkdir -p "$logs" "$data/logs" || exit 1

dep=()
if [[ $stage == convert || $stage == all ]]; then
    [[ -f "$annotated" ]] || { echo "missing $annotated (download first)" >&2; exit 1; }
    [[ -d "$data/raw" ]] || { echo "missing $data/raw (extract GSE297328_RAW.tar first)" >&2; exit 1; }
    [[ -f "$converted/conversion_report.json" ]] && { echo "$converted/conversion_report.json exists; not converted again" >&2; exit 2; }
    id=$(submit -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J ltl_conv -o "$data/logs/convert_%j.out" \
        --wrap "$activate && python ltl331_nepc/01_convert_geo.py --annotated $annotated --raw-dir $data/raw --out $converted") || exit 1
    echo "convert job $id" >&2
    dep=(--dependency=afterok:$id)
fi
if [[ $stage == prepare || $stage == all ]]; then
    [[ $stage == all || -f "$converted/samples.tsv" ]] || { echo "missing $converted/samples.tsv (convert first)" >&2; exit 1; }
    [[ -f "$inputs/rawqc/pair_manifest_rawqc.csv" ]] && { echo "$inputs/rawqc/pair_manifest_rawqc.csv exists; not prepared again" >&2; exit 2; }
    cmd="$activate && python ltl331_nepc/02_build_manifest.py --converted $converted --out $inputs/manifest_raw.csv \
&& python cancer_metastasis/27_downsample_counts.py $inputs/manifest_raw.csv $inputs/ds --malignant-column malignant \
&& python ltl331_nepc/03_raw_on_ds_cells.py --raw-manifest $inputs/manifest_raw.csv --ds-manifest $inputs/ds/pair_manifest_downsampled.csv --out $inputs/rawqc"
    id=$(submit "${dep[@]}" -p gg -N 1 -n 1 -t 03:00:00 -A MCB26031 -J ltl_prep -o "$logs/prepare_%j.out" --wrap "$cmd") || exit 1
    echo "prepare job $id" >&2
    dep=(--dependency=afterok:$id)
fi
if [[ $stage == run || $stage == all ]]; then
    if [[ $stage == run ]]; then
        for p in "$inputs/rawqc/pair_manifest_rawqc.csv" "$inputs/ds/pair_manifest_downsampled.csv" "$inputs/ds/predownsample_depth.csv.gz"; do
            [[ -f "$p" ]] || { echo "missing $p (prepare first)" >&2; exit 1; }
        done
    fi
    id=$(submit "${dep[@]}" --array=0-1 -J ltl_ot -o "$logs/run_%A_%a.out" -e "$logs/run_%A_%a.err" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",CANCER_COT_ROOT="$root",CONFIDENCEOT_FACTORIAL_MANIFEST="$inputs/rawqc/pair_manifest_rawqc.csv",CONFIDENCEOT_FACTORIAL_MANIFEST_DS="$inputs/ds/pair_manifest_downsampled.csv",CONFIDENCEOT_PREDOWNSAMPLE_DEPTH="$inputs/ds/predownsample_depth.csv.gz",CONFIDENCEOT_FACTORIAL_LABELS="$labels",CONFIDENCEOT_FACTORIAL_WORKERS=1,CONFIDENCEOT_FACTORIAL_ROOT="$root/runs/acc0.99",CONFIDENCEOT_FACTORIAL_BLOCK=ltl331,CONFIDENCEOT_ANALYSIS_SCOPE=malignant,CONFIDENCEOT_MALIGNANT_COLUMN=malignant,CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE=0,CONFIDENCEOT_MINIMUM_SCOPE_CELLS=4,CONFIDENCEOT_FACTORIAL_EXTRA_ARGS="--within-side-acceptance-minimum 0.99 --save-coupling" \
        "$repo/cancer_metastasis/tacc_preprocessing_factorial.slurm") || exit 1
    echo "run array $id: task 0 ranknm256_noscale_ds_cos, task 1 logcpm; out $root/runs/acc0.99/ltl331/<label>/" >&2
fi
(( dry_run )) || echo "$stage submitted $(date -Iseconds), commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null)" >> "$root/SUBMITTED.txt"

# Download, once, on the login node (about 3.3 GB):
#   D=/scratch/10119/ghzheng/OT_project/data/ltl331_GSE297328 && mkdir -p $D/raw && cd $D
#   U=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE297nnn/GSE297328/suppl
#   wget -q $U/GSE297328_README.txt $U/filelist.txt $U/GSE297328_RAW.tar $U/GSE297328_ltl331_annotated.h5ad
#   test $(stat -c %s GSE297328_RAW.tar) -eq 694087680 && tar -xf GSE297328_RAW.tar -C raw
#   test $(stat -c %s GSE297328_ltl331_annotated.h5ad) -eq $(curl -sI $U/GSE297328_ltl331_annotated.h5ad | awk 'tolower($1)=="content-length:"{print $2+0}')
