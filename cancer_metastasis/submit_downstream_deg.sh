#!/bin/bash
# Downstream of a completed gate: pseudobulk, paired DEG, GSEA, disqualifiers,
# and the leave-one-patient-out refit. One dataset per invocation.
#
#   bash cancer_metastasis/submit_downstream_deg.sh ovarian
#   DRY_RUN=1 bash cancer_metastasis/submit_downstream_deg.sh prostate
#
# WHY THE MANIFEST AND THE GATE ROOT DISAGREE ON PURPOSE.
# The gate was computed on read-equalised counts; the expression stages read
# the ORIGINAL matrices, because equalisation is justified for the
# representation and the gate and was never priced for what it costs the
# differential expression. 21_prepare_four_state_malignant_pseudobulk.py takes
# the manifest and the gate root as separate arguments and joins cells by
# identifier, so this is a supported invocation rather than a trick -- and it
# raises loudly if any gate id is absent from the original matrices, which is
# the one way the split can break. See section 2e of
# EXPERIMENT_2026-09-23_PREPROCESSING_ARMS.md.
#
# WHY THE MANIFEST IS TRIMMED FIRST.
# The original manifest holds pairs the equalisation and the OT array dropped:
# GSE180661 finishes 92 gates against 94 rows. 21_ raises for the WHOLE
# PATIENT when one of that patient's pairs has no gate directory, so two absent
# pairs can delete two entire patients and fail the array, which through
# afterok leaves every later stage PENDING. J1 turns that into a counted,
# reported trim.
#
# WHY --metastasis-size-csv STILL POINTS INSIDE THE EQUALISED ROOT.
# It supplies the malignant QC-passing cell count per sample, which is what the
# prespecification's "largest lesion per patient" rule selects on. Without it
# 21_ falls back silently to the manifest's all-cell-type target_n, which is a
# different number and picks a different lesion. That file exists only in the
# equalised root, so the raw-count arm depends on it and that is correct.
#
# NO DEPTH-BASED PAIR FILTER. Every gated pair enters. The per-pair depth and
# detected-gene statistics are diagnostics; the result-level disqualifiers do
# that work. See "Inclusion is closed" in DEG_PRESPECIFICATION_2026-09-15.md.
#
# No `set -u`. It fires on an empty array expansion while permitting the
# defined-but-empty path variable that has actually cost job numbers here; the
# explicit checks below are the protection that matters.
set -eo pipefail

dataset=${1:-}
# Set per arm below, and emptied here so the branch that chooses between
# them cannot read a value left over from the caller's environment.
malignant_column=
annotations=
case "$dataset" in
  ovarian|prostate|colorectal|headneck) ;;
  *)
    echo "Usage: $0 {ovarian|prostate|colorectal|headneck}" >&2
    exit 2
    ;;
esac

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
preprocessing=${PREPROCESSING:-rank256_ds_cos}
stamp=${GATE_STAMP:-20260921}
minimum_cells=${MINIMUM_CELLS_PER_STATUS:-10}

# Per dataset: accession, the malignant labels the OT used, the ORIGINAL
# manifest, and the equalised root that carries the malignant cell counts.
# Every one of these is set explicitly because the downstream slurm files all
# default to a GSE180661 path that EXISTS on disk -- a forgotten export runs
# the ovarian analysis and writes it under another dataset's name, with no
# error anywhere.
case "$dataset" in
  ovarian)
    accession=GSE180661
    malignant_column=malignant
    source_manifest=$result/manifest/pair_manifest_eligible.csv
    equalised=$result/downsampled_GSE180661_20260914
    patient_array=${PATIENT_ARRAY:-0-7}
    ;;
  prostate)
    accession=GSE271675
    # The only dataset that cannot use the uniform call: these h5ads come from
    # a different source and predate it. Its compartment is therefore defined
    # differently in kind from the other three -- a Methods statement, not a
    # preference.
    annotations="Epithelial|Basal Epithelial|Neuroendocrine"
    source_manifest=$result/prepared_GSE271675_20260916/pair_manifest_eligible.csv
    equalised=$result/downsampled_GSE271675_20260916
    patient_array=${PATIENT_ARRAY:-0-3}
    ;;
  colorectal)
    accession=GSE225857
    malignant_column=malignant
    source_manifest=$result/author_labeled_replication_GSE181919_GSE225857_rerun_20260912/GSE225857/manifest/pair_manifest_malignant_eligible.csv
    equalised=$result/downsampled_GSE225857_$stamp
    patient_array=${PATIENT_ARRAY:-0-1}
    ;;
  headneck)
    accession=GSE181919
    malignant_column=malignant
    source_manifest=$result/author_labeled_replication_GSE181919_GSE225857_rerun_20260912/GSE181919/manifest/pair_manifest_malignant_eligible.csv
    equalised=$result/downsampled_GSE181919_$stamp
    patient_array=${PATIENT_ARRAY:-0-1}
    ;;
esac

gate=$result/ot_${accession}_${preprocessing}_$stamp
trimmed=$result/manifest_trimmed/${accession}_${preprocessing}_$stamp.csv
four_state=$result/four_state_${accession}_${preprocessing}_$stamp
deg=$result/deg_${accession}_${preprocessing}_$stamp
gsea=$result/gsea_${accession}_${preprocessing}_$stamp
audit=$result/deg_audit_${accession}_${preprocessing}_$stamp
loo=$result/deg_loo_${accession}_${preprocessing}_$stamp
size_csv=$equalised/downsample_per_sample.csv

for name in repo result env_path gate trimmed four_state deg gsea audit loo; do
  value=${!name}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done
for name in source_manifest size_csv; do
  value=${!name}
  if [[ ! -f "$value" ]]; then
    echo "$name does not exist: $value" >&2
    exit 2
  fi
done
if [[ ! -d "$gate" ]]; then
  echo "Gate root does not exist: $gate" >&2
  echo "Run submit_rank_cosine_chain.sh for $dataset first." >&2
  exit 2
fi

submit() {
  if [[ -n "${DRY_RUN:-}" ]]; then
    printf 'DRY_RUN sbatch %s\n' "$*" >&2
    echo "000000"
    return
  fi
  # Vista's sbatch prints a welcome banner on stdout, so --parsable alone does
  # not yield an id: take the last line, digits only.
  sbatch --parsable "$@" | tail -n 1 | tr -dc '0-9_'
}

wrap() {
  printf 'source /home1/10119/ghzheng/.bashrc; conda activate %s; cd %s; export PYTHONPATH=%s/src:%s/cancer_metastasis:%s; %s' \
    "$env_path" "$repo" "$repo" "$repo" "$repo" "$1"
}

cd "$repo"
mkdir -p "$result/logs" "$(dirname "$trimmed")"

# J1. Trim. Small and fast, but a job rather than a login-node command so the
# array can depend on it and so its report lands beside the other logs.
trim_job=$(submit \
  -p gg -N 1 -n 1 -t 00:30:00 -A MCB26031 -J "cot_trim_$dataset" \
  -o "$result/logs/trim_${dataset}_%j.out" \
  -e "$result/logs/trim_${dataset}_%j.err" \
  --wrap="$(wrap "python cancer_metastasis/35_trim_manifest_to_gate.py $source_manifest $gate $trimmed")")
echo "J1 trim        $trim_job -> $trimmed"

# J2. Pseudobulk, arrayed over patients. Every variable set explicitly.
export CONFIDENCEOT_REPO="$repo"
export CANCER_COT_ROOT="$result"
export CONFIDENCEOT_ENV="$env_path"
export CONFIDENCEOT_MANIFEST="$trimmed"
export CONFIDENCEOT_GATE_ROOT="$gate"
export CONFIDENCEOT_FOUR_STATE_ROOT="$four_state"
export CONFIDENCEOT_METASTASIS_SIZE_CSV="$size_csv"
if [[ -n "${malignant_column:-}" ]]; then
  export CONFIDENCEOT_MALIGNANT_COLUMN="$malignant_column"
  unset CONFIDENCEOT_MALIGNANT_ANNOTATIONS
else
  export CONFIDENCEOT_MALIGNANT_ANNOTATIONS="$annotations"
  unset CONFIDENCEOT_MALIGNANT_COLUMN
fi
# Cap-robustness and origin ranking stay off: the budget is reported rather
# than enforced, so the cap no longer decides the gate.
unset CONFIDENCEOT_ROBUSTNESS_CSV CONFIDENCEOT_SENSITIVITY_ROOT
unset CONFIDENCEOT_MALIGNANT_ANNOTATION
# Counted from the trimmed manifest at run time, which does not exist yet.
unset CONFIDENCEOT_FOUR_STATE_PATIENT_COUNT

pseudobulk_job=$(submit --dependency=afterok:"$trim_job" \
  --array="$patient_array" \
  cancer_metastasis/tacc_four_state_malignant_array.slurm)
echo "J2 pseudobulk  $pseudobulk_job -> $four_state"

# J3. Paired DEG. Depends on the whole array: a missing patient does not make
# 13_ fail, it is silently absent from the fit, so the dependency is what
# prevents a partial result that looks complete.
export CONFIDENCEOT_DEG_ROOT="$four_state"
export CONFIDENCEOT_PYDESEQ2_ROOT="$deg"
export CONFIDENCEOT_MINIMUM_CELLS_PER_STATUS="$minimum_cells"

deg_job=$(submit --dependency=afterok:"${pseudobulk_job%%_*}" \
  cancer_metastasis/tacc_paired_pydeseq2.slurm)
echo "J3 pydeseq2    $deg_job -> $deg  (floor $minimum_cells cells/status)"

# J4. GSEA. The gene sets are dataset-independent and reused.
export CONFIDENCEOT_GSEA_ROOT="$gsea"
gsea_job=$(submit --dependency=afterok:"$deg_job" \
  cancer_metastasis/tacc_pydeseq2_gsea.slurm)
echo "J4 gsea        $gsea_job -> $gsea"

# J5. The three disqualifiers. 31_ has no slurm of its own, so it is wrapped
# here. It raises below six patients carrying both sides, which colorectal and
# head and neck are not expected to clear -- that is a property of those
# datasets and the job failing says so rather than hiding it.
audit_job=$(submit --dependency=afterok:"$deg_job" \
  -p gg -N 1 -n 1 -t 01:00:00 -A MCB26031 -J "cot_degaudit_$dataset" \
  -o "$result/logs/degaudit_${dataset}_%j.out" \
  -e "$result/logs/degaudit_${dataset}_%j.err" \
  --wrap="$(wrap "python cancer_metastasis/31_audit_deg_disqualifiers.py $deg $four_state $audit")")
echo "J5 disqualify  $audit_job -> $audit"

# J6. Leave-one-patient-out, the literal third disqualifier. One PyDESeq2 fit
# per patient plus one, run in sequence inside a single job: the folds are
# independent but collecting P output directories is the part that goes wrong,
# and a fold that fails is recorded rather than failing the job.
loo_job=$(submit --dependency=afterok:"$deg_job" \
  -p gg -N 1 -n 1 -c 16 -t 08:00:00 -A MCB26031 -J "cot_loo_$dataset" \
  -o "$result/logs/loo_${dataset}_%j.out" \
  -e "$result/logs/loo_${dataset}_%j.err" \
  --wrap="$(wrap "python cancer_metastasis/34_leave_one_patient_out.py $four_state $loo --minimum-cells-per-patient-status $minimum_cells --skip-completed")")
echo "J6 leave-1-out $loo_job -> $loo"

echo
printf 'dataset        %s (%s)\n' "$dataset" "$accession"
printf 'preprocessing  %s\n' "$preprocessing"
printf 'gate           %s\n' "$gate"
printf 'manifest       %s (original)\n' "$source_manifest"
printf 'trimmed        %s\n' "$trimmed"
printf 'lesion size    %s\n' "$size_csv"
if [[ -n "$malignant_column" ]]; then
  printf 'malignant      %s == malignant   (uniform inferCNV)
' "$malignant_column"
else
  printf 'malignant      cell_type in: %s   (deposit labels)
' "$annotations"
fi
printf 'four_state     %s\n' "$four_state"
printf 'deg            %s\n' "$deg"
printf 'gsea           %s\n' "$gsea"
printf 'audit          %s\n' "$audit"
printf 'leave-one-out  %s\n' "$loo"
echo
# The gg queue allows 40 submitted jobs per user and counts each array task
# separately, so the array is the whole cost and the rest is five.
array_n=$(python - "$patient_array" <<'PY'
import sys
lo, _, hi = sys.argv[1].partition("-")
print(int(hi or lo) - int(lo) + 1)
PY
)
printf 'slurm budget   %s jobs (1 trim + %s array + 4)\n' "$((array_n + 5))" "$array_n"
echo
if [[ -z "${DRY_RUN:-}" ]]; then
  squeue -u "$USER" -o "%.12i %.16j %.10T %.28E"
fi
