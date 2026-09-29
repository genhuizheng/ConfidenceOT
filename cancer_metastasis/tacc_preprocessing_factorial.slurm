#!/bin/bash
#SBATCH -J cot_factorial
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/fact_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/fact_%A_%a.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 24:00:00
#SBATCH -A MCB26031
#
# The sixteen-cell preprocessing factorial on the real pairs.
#
# The same four binary choices the technical benchmark varies -- library-size
# normalisation, rank encoding, gene-wise scaling, angular geometry -- run
# here against data whose answer nobody knows. That is the whole point of
# running both: the benchmark says which configuration recovers a known
# answer, and this says which one the real gate agrees with. A configuration
# that wins on simulated data and disagrees with every other arm here has not
# been validated by the benchmark; it has been contradicted by the data.
#
# Twelve arms, not sixteen. Library-size normalisation and rank encoding are
# one field in the production code, so the four cells that turn both on have
# no name -- and no content: ranking is per-cell and library-size
# normalisation is a strictly increasing per-cell map, so it cannot reorder a
# cell's genes. tests/test_benchmark_identity.py asserts that bit for bit.
#
# EVERYTHING ELSE IS HELD FIXED. One manifest, one scope, one calibration
# null, one budget, one solver, one seed rule. An arm that differed in a
# second thing would not be a factorial cell, and the comparison would be
# uninterpretable in exactly the way that is hardest to notice afterwards.
#
# The array index decodes to (label, worker) and each worker strides its
# label's manifest rather than taking a contiguous block, which is the same
# trick tacc_full_array.slurm uses: array tasks count one by one against the
# submission cap, so twelve labels times a handful of workers has to stay
# small, and striding keeps every worker's share the same size when the
# manifest is ordered by dataset.
#
# No set -u. TACC's module files and conda's shell functions read variables
# they do not define -- module load sources a completion script that reads
# BASH_COMPLETION_DEBUG -- and under set -u that kills the job in seconds
# with nothing in the log. Every expansion below carries its own default.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_FACTORIAL_MANIFEST:-$result/manifest/pancancer_20260924/pair_manifest_eligible.csv}
out=${CONFIDENCEOT_FACTORIAL_ROOT:-$result/preprocessing_factorial}
# One block is one (manifest, compartment definition). The name goes into the
# path, so the diagnostics can find every block that ran and carry which is
# which into the comparison rather than pooling two definitions of malignant
# into one median without saying so.
block=${CONFIDENCEOT_FACTORIAL_BLOCK:-uniform}
labels=(${CONFIDENCEOT_FACTORIAL_LABELS:-\
raw raw_noscale raw_cos raw_noscale_cos \
raw_ds raw_noscale_ds raw_ds_cos raw_noscale_ds_cos \
logcpm logcpm_noscale logcpm_cos logcpm_noscale_cos \
logcpm_ds logcpm_noscale_ds logcpm_ds_cos logcpm_noscale_ds_cos \
ranknm256 ranknm256_noscale ranknm256_cos ranknm256_noscale_cos \
ranknm256_ds ranknm256_noscale_ds ranknm256_ds_cos ranknm256_noscale_ds_cos})
# The equalised counts. Read equalisation is the fifth axis of the factorial
# and it is the only one that happens to the input rather than inside the
# representation: 27_downsample_counts.py rewrites the h5ads and the label
# only records that it did. So a _ds arm is the same twelve transforms on a
# different matrix, and which matrix is decided per label below.
manifest_ds=${CONFIDENCEOT_FACTORIAL_MANIFEST_DS:-}
workers=${CONFIDENCEOT_FACTORIAL_WORKERS:-3}
scope=${CONFIDENCEOT_ANALYSIS_SCOPE:-malignant}
malignant_column=${CONFIDENCEOT_MALIGNANT_COLUMN:-malignant}
malignant_value=${CONFIDENCEOT_MALIGNANT_VALUE:-malignant}
# The other way of naming the malignant compartment, pipe separated. Each
# deposit uses its own best available call: the uniform inferCNV labels where
# they exist, and Faming Zhao's annotation of the prostate object, which is
# the authoritative labelling for that deposit -- the GEO conversion of the
# same accession carries no call at all, since its paper used NUMBAT and that
# needs BAMs behind EGA access.
#
# Both are correct for the data they cover. The block name and the recorded
# definition exist so the comparison can say which pairs came from which, not
# because either is provisional.
annotation_spec=${CONFIDENCEOT_INCLUDE_ANNOTATIONS:-}
calibration_null=${CONFIDENCEOT_CALIBRATION_NULL:-within_side_split}
maximum_observed_cells=${CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE:-10000}
# Must match what the manifest was trimmed at. Left at the runner's
# default of 20 against a manifest trimmed at 1, the small pairs are
# submitted and refused, which is the failure the trim existed to
# remove.
minimum_scope_cells=${CONFIDENCEOT_MINIMUM_SCOPE_CELLS:-1}
# Cell QC on, thresholds wide open. The switch and the thresholds are
# separate for exactly this case: 02_run_pair.py records total_counts,
# n_genes_by_counts and pct_counts_mitochondrial into cell_confidence.csv
# only when --cell-qc is given -- without it the else branch writes qc_pass
# and nothing else -- and those three columns are what the gate diagnostics
# read. The first 1,320 runs of this factorial finished with no depth column
# at all, so every depth AUC came back NaN and the readout the factorial
# exists to produce was absent from its own summary.
#
# Wide open because nothing here should be filtered. The pairs were already
# chosen by 40_trim_manifest_to_evaluable.py, and a second QC pass would
# discard cells that had been selected, changing the comparison between arms
# into a comparison between cell sets. A cell with zero counts passes >= 0
# and a cell at 100% mitochondrial passes <= 100, so the thresholds record
# without removing.
qc_minimum_counts=${CONFIDENCEOT_MINIMUM_TOTAL_COUNTS:-0}
qc_minimum_genes=${CONFIDENCEOT_MINIMUM_DETECTED_GENES:-0}
qc_maximum_mito=${CONFIDENCEOT_MAXIMUM_MITOCHONDRIAL_PERCENT:-100.0}
# CPU because gg has no GPU, and because the solve runs on cores anyway. The
# null calibration fits several matrices per cost across the grid, which is
# the expensive part and the part --workers parallelises, so the core count
# and the worker count go together.
device=${CONFIDENCEOT_DEVICE:-cpu}
solver_workers=${CONFIDENCEOT_WORKERS:-8}
force=${CONFIDENCEOT_FACTORIAL_FORCE:-0}

for name in repo result manifest out; do
    value=${!name}
    if [[ -z "$value" ]]; then
        echo "$name is empty; export the paths in the submitting shell" >&2
        exit 2
    fi
    if [[ "$value" != /* ]]; then
        echo "$name is not an absolute path: $value" >&2
        exit 2
    fi
done
if [[ ! -f "$manifest" ]]; then
    echo "manifest does not exist: $manifest" >&2
    exit 2
fi

cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${CONFIDENCEOT_THREADS:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

pairs=$(python - "$manifest" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)

task=${SLURM_ARRAY_TASK_ID:?Submit as an array over labels x workers}
label_index=$(( task / workers ))
worker=$(( task % workers ))
if (( label_index >= ${#labels[@]} )); then
    echo "task $task is past the ${#labels[@]} labels; nothing to do"
    exit 0
fi
label=${labels[$label_index]}

# Which input this label is about. Asked of the class that owns the label
# rather than matched against its text: a shell that guessed wrong would run
# equalised counts under a name that does not say so, or the reverse, and
# nothing downstream could detect it -- equalisation is recorded by the run,
# not applied by it, so the label would still verify either way.
wants_ds=$(python - "$label" <<'PYDS'
import sys

from confidenceot.preprocessing import Preprocessing

print("1" if Preprocessing.from_label(sys.argv[1]).equalise_depth else "0")
PYDS
) || { echo "could not read $label with confidenceot.Preprocessing" >&2; exit 2; }

if [[ "$wants_ds" == "1" ]]; then
    if [[ -z "$manifest_ds" ]]; then
        echo "$label needs equalised counts but no manifest was given." >&2
        echo "  run 27_downsample_counts.py and pass --manifest-ds" >&2
        exit 2
    fi
    if [[ ! -f "$manifest_ds" ]]; then
        echo "equalised manifest does not exist: $manifest_ds" >&2
        exit 2
    fi
    manifest=$manifest_ds
    echo "input       equalised counts"
else
    echo "input       original counts"
fi

# The two manifests have to cover the same pairs, or the _ds arms and the
# rest are not two halves of one factorial: a difference between them would
# be partly a difference in which patients were measured, and no comparison
# could separate the two.
if [[ -n "$manifest_ds" && -f "$manifest_ds" ]]; then
    python - "$manifest" "$manifest_ds" <<'PYSAME' || exit 2
import sys

import pandas as pd

left, right = (set(pd.read_csv(path)["pair_id"].astype(str))
               for path in sys.argv[1:3])
if left != right:
    only_left = sorted(left - right)[:6]
    only_right = sorted(right - left)[:6]
    raise SystemExit(
        f"the two manifests cover different pairs: {len(left)} against "
        f"{len(right)}; only in the first {only_left}, only in the second "
        f"{only_right}")
PYSAME
fi

output_root=$out/$block/$label
mkdir -p "$result/logs" "$output_root"
# Written beside the results, so the diagnostics do not have to be told again
# what this block was. A record that travels with the data cannot fall out of
# step with it the way a second argument can.
echo "$manifest" > "$out/$block/MANIFEST_PATH"
if [[ -n "$annotation_spec" ]]; then definition=author_labels; else definition=uniform_infercnv; fi
echo "$definition" > "$out/$block/COMPARTMENT_DEFINITION"

# The raw arms resolve to the 'precomputed' transform, which the pair runner
# refuses unless asked, because an unrecognised stem resolves to it too and a
# typo must not silently run the counts as the representation. Asking here is
# deliberate and narrow; a typo in any other arm still fails.
allow=()
[[ "$label" == raw* ]] && allow=(--allow-precomputed)

echo "block       $block"
echo "label       $label  (arm $(( label_index + 1 )) of ${#labels[@]})"
echo "worker      $worker of $workers, striding $pairs manifest rows"
echo "manifest    $manifest"
echo "output      $output_root"
echo "scope       $scope, null $calibration_null, device $device"
echo "minimum     $minimum_scope_cells cells per side"
echo "cell qc     recorded, thresholds $qc_minimum_counts/$qc_minimum_genes/$qc_maximum_mito (filters nothing)"

skip=(--skip-completed)
[[ "$force" == "1" ]] && skip=()

scope_args=(--analysis-scope "$scope")
if [[ "$scope" == "malignant" ]]; then
    if [[ -n "$annotation_spec" ]]; then
        IFS='|' read -r -a annotations <<< "$annotation_spec"
        for annotation in "${annotations[@]}"; do
            scope_args+=(--include-annotation "$annotation")
        done
        echo "compartment author labels: ${annotations[*]}"
    else
        scope_args+=(--malignant-column "$malignant_column"
                     --malignant-value "$malignant_value")
        echo "compartment uniform call: $malignant_column==$malignant_value"
    fi
fi

failures=0
failed_rows=()
for (( index = worker; index < pairs; index += workers )); do
    echo "===== START $label index=$index $(date --iso-8601=seconds) ====="
    if python cancer_metastasis/02_run_pair.py "$manifest" "$output_root" \
        --index "$index" \
        "${scope_args[@]}" \
        "${allow[@]}" \
        "${skip[@]}" \
        --preprocessing "$label" \
        --calibration-null "$calibration_null" \
        --max-observed-cells-per-side "$maximum_observed_cells" \
        --minimum-scope-cells "$minimum_scope_cells" \
        --cell-qc \
        --minimum-total-counts "$qc_minimum_counts" \
        --minimum-detected-genes "$qc_minimum_genes" \
        --maximum-mitochondrial-percent "$qc_maximum_mito" \
        --null-calibration-replicates 5 \
        --null-validation-replicates 5 \
        --calibration-grid-size 5 \
        --tolerance 1e-4 \
        --workers "$solver_workers" \
        --device "$device"; then
        echo "===== COMPLETE $label index=$index $(date --iso-8601=seconds) ====="
    else
        status=$?
        failures=$(( failures + 1 ))
        failed_rows+=("$index:$status")
        # To stdout as well as stderr. A failure written only to stderr sits
        # in a different file from the unit lines, and an array whose tasks
        # all "finished" is how twelve labels of nothing got mistaken for
        # twelve labels of results once already.
        echo "===== FAILED $label index=$index exit=$status $(date --iso-8601=seconds) ====="
        echo "===== FAILED $label index=$index exit=$status =====" >&2
    fi
done

echo "$label worker $worker finished with $failures failed pair(s)"
if (( failures > 0 )); then
    echo "failed rows: ${failed_rows[*]}"
    exit 1
fi
