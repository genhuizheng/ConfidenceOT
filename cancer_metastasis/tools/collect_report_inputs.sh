#!/bin/bash
# Gather every summary this project's figures need into one small archive.
#
# The results live on the cluster and the figures are drawn on a laptop, so the
# tables travel rather than the matrices. Only summaries are copied: per-pair
# and per-patient tables, the arm summaries, the GSEA results and the run
# reports. Nothing here is larger than a few megabytes.
#
# Names are flattened and prefixed by what produced them, because a directory
# tree of twenty `gate_covariate_dataset_summary.csv` files tells the reader
# nothing about which arm each one came from.
#
# No `set -u`: the path check below catches the defined-but-empty variable that
# -u permits, which is the failure that has cost job numbers here.
#
# Usage:
#   bash cancer_metastasis/tools/collect_report_inputs.sh
set -eo pipefail

result=${CANCER_COT_ROOT:-${RESULTS:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}}
downsampled=${DOWNSAMPLED:-$result/downsampled_GSE180661_20260914}
stage=${REPORT_STAGE:-$result/report_inputs_20260916}

for name in result downsampled; do
  value=${!name}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done

rm -rf "$stage"
mkdir -p "$stage"

take() {
  # take <source> <destination-name>; missing sources are reported, not fatal,
  # because which arms exist depends on how far the runs have got.
  if [[ -f "$1" ]]; then
    cp "$1" "$stage/$2"
    printf '  %-56s %s\n' "$2" "$(du -h "$1" | cut -f1)"
  else
    printf '  %-56s MISSING\n' "$2"
  fi
}

echo "== ovarian gate diagnostics"
take "$result/gate_cov_free/gate_covariate_dataset_summary.csv"        ov_gate_logcpm_summary.csv
take "$result/gate_cov_free/gate_covariate_pair_statistics.csv"        ov_gate_logcpm_pairs.csv
take "$result/gate_cov_bounded/gate_covariate_dataset_summary.csv"     ov_gate_bounded_summary.csv
take "$result/gate_cov_mismatched/gate_covariate_dataset_summary.csv"  ov_gate_mismatched_summary.csv
for d in "$result"/gc_ov_*; do
  [[ -d "$d" ]] || continue
  take "$d/gate_covariate_dataset_summary.csv" "ov_gate_$(basename "$d" | sed 's/^gc_ov_//')_summary.csv"
  take "$d/gate_covariate_pair_statistics.csv" "ov_gate_$(basename "$d" | sed 's/^gc_ov_//')_pairs.csv"
done

echo "== ovarian controls and similarity"
take "$result/matched_vs_mismatched/matched_vs_mismatched_pairs.csv"   ov_matched_vs_mismatched_pairs.csv
take "$result/matched_vs_mismatched/matched_vs_mismatched_report.json" ov_matched_vs_mismatched_report.json
take "$result/similarity_readout/similarity_readout_pairs.csv"         ov_similarity_pairs.csv
take "$result/similarity_readout/similarity_readout_report.json"       ov_similarity_report.json
take "$result/cell_identity/gate_by_author_subtype.csv.gz"             ov_gate_by_subtype.csv.gz

echo "== ovarian per-patient retained fractions and DEG"
if [[ -d "$result/four_state_free/patients" ]]; then
  python - "$result/four_state_free/patients" "$stage/ov_patient_states.csv" <<'PY'
import glob, json, os, sys
import pandas as pd
rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*", "diagnostics.json"))):
    record = json.load(open(path, encoding="utf-8"))
    counts = record.get("state_cell_n", {})
    rows.append({
        "patient_id": record.get("patient_id"),
        "lesion": record.get("designated_metastasis"),
        "primary_retained": counts.get("primary_retained", 0),
        "primary_rejected": counts.get("primary_rejected", 0),
        "metastasis_n": counts.get("metastasis_retained", 0),
    })
frame = pd.DataFrame(rows)
total = frame["primary_retained"] + frame["primary_rejected"]
frame["retained_fraction"] = frame["primary_retained"] / total.where(total > 0)
frame.to_csv(sys.argv[2], index=False)
print(f"  ov_patient_states.csv                                    {len(frame)} patients")
PY
else
  echo "  ov_patient_states.csv                                    MISSING"
fi
take "$result/deg_free/contrasts/primary_rejected_vs_primary_retained/pydeseq2_all_gene_discovery.csv" ov_deg_discovery.csv
for f in "$result"/deg_free/*.json; do
  [[ -f "$f" ]] && take "$f" "ov_deg_report.json"
done

echo "== representation benchmark on simulated ground truth"
for tag in log_cpm rank_value rank256 rank128 rank_no_median pearson_residuals; do
  take "$result/sim_${tag}_20260916/depth_null_arm_summary.csv" "sim_${tag}_arms.csv"
done

echo "== prostate preparation"
take "$result/prepared_GSE271675_20260916/preparation_report.json"   pca_preparation_report.json
take "$result/downsampled_GSE271675_20260916/downsample_report.json" pca_downsample_report.json

echo "== prostate gate diagnostics"
for d in "$result"/gcf_*; do
  [[ -d "$d" ]] || continue
  take "$d/gate_covariate_dataset_summary.csv" "pca_gate_$(basename "$d" | sed 's/^gcf_//')_summary.csv"
done

echo "== prostate DEG and GSEA"
for arm in logcpm_free_ds rank256_free_ds rank256_free_raw rank256_cap085_ds rank256_cap085_raw; do
  for f in "$result/deg_${arm}_20260916"/*.json; do
    [[ -f "$f" ]] && take "$f" "pca_deg_${arm}_report.json"
  done
  take "$result/deg_${arm}_20260916/contrasts/primary_rejected_vs_primary_retained/pydeseq2_all_gene_discovery.csv" \
       "pca_deg_${arm}_discovery.csv"
  for f in "$result/gsea_${arm}_20260916/contrasts"/*/all_gene_discovery_gsea_results.csv; do
    [[ -f "$f" ]] && take "$f" "pca_gsea_${arm}.csv"
  done
done

echo "== per-cell overlays for the UMAP figures"
take "$result/umap_overlay_ovarian.csv.gz"  umap_ovarian.csv.gz
take "$result/umap_overlay_prostate.csv.gz" umap_prostate.csv.gz

echo "== prostate lymphoid contamination"
for s in Pat2_Tu5 Pat4_LN1; do
  take "$result/contam3_$s/lineage_contamination_report.json" "pca_contamination_$s.json"
done

archive="$result/report_inputs_20260916.tar.gz"
tar -czf "$archive" -C "$(dirname "$stage")" "$(basename "$stage")"
echo
echo "archive  $archive"
du -h "$archive"
echo
echo "copy it to the laptop with:"
echo "  scp ghzheng@vista.tacc.utexas.edu:$archive ."
