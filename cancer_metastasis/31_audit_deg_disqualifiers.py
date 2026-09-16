"""Check the prespecified disqualifiers against a completed DEG.

``DEG_PRESPECIFICATION_2026-09-15.md`` names three conditions that make the
gene list uninterpretable whatever it contains.  The first, keratin/SPRR
domination, is read off the ranked table directly.  The other two need the
per-patient effects behind the pooled result, which is what this computes.

*Similarity tracking.*  The retained fraction was shown to be a readout of
primary--metastasis transcriptional similarity, ranging 0.000 to 0.879 across
patients.  If a gene's per-patient effect tracks that fraction, the pooled
result is describing how alike the two tumours are rather than what separates
the two groups of cells.  This is the specific way this analysis can fool
itself, so it is tested gene by gene rather than in aggregate.

*Patient dominance.*  A pooled effect can rest on a few patients.  Counting how
many patients carry the pooled sign is a cheap and direct form of the
leave-one-out check: a gene that only a third of patients agree on is not a
programme, whatever its FDR.

Genes used to build the transport are **not** excluded.  The joint PCA is built
on highly variable genes, so removing them would discard most of the expressed
biology and leave a list selected for being uninformative.  Their circularity
is a caveat to state, not a filter to apply: being in the HVG set means a gene
helped define the geometry, not that it must separate the two groups within it.

Per-patient effects are computed from the pseudobulk CPM directly rather than
by refitting, so this runs in seconds and adds no modelling assumption of its
own.  It cannot replace a true leave-one-out refit, and does not claim to; it
answers whether the pooled sign is shared.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ARTEFACT_FAMILIES = r"^(KRT\d|SPRR|S100A\d)"


def read_metadata(path: Path) -> pd.DataFrame:
    """Read the pseudobulk metadata, whose sample_id may be the index column."""
    table = pd.read_csv(path)
    if "sample_id" not in table.columns:
        table = table.rename(columns={table.columns[0]: "sample_id"})
    return table


def retained_fractions(four_state_root: Path) -> dict[str, float]:
    fractions: dict[str, float] = {}
    for path in sorted(glob.glob(str(four_state_root / "patients" / "*" / "diagnostics.json"))):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        counts = record.get("state_cell_n", {})
        retained = counts.get("primary_retained", 0)
        rejected = counts.get("primary_rejected", 0)
        if retained + rejected:
            fractions[str(record["patient_id"])] = retained / (retained + rejected)
    return fractions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deg_dir", type=Path, help="Output of 13_run_paired_pydeseq2.py")
    parser.add_argument("four_state_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--contrast", default="primary_rejected_vs_primary_retained")
    parser.add_argument("--maximum-fdr", type=float, default=0.05)
    parser.add_argument("--minimum-absolute-log2-fold-change", type=float, default=1.0)
    args = parser.parse_args()

    contrast_dir = args.deg_dir / "contrasts" / args.contrast
    pooled = pd.read_csv(contrast_dir / "pydeseq2_all_gene_discovery.csv")
    significant = pooled[
        pooled["fdr"].lt(args.maximum_fdr)
        & pooled["log2_fold_change"].abs().ge(args.minimum_absolute_log2_fold_change)
    ].copy().sort_values("fdr")
    if significant.empty:
        raise RuntimeError("No gene passed the prespecified thresholds")

    counts = pd.read_csv(args.deg_dir / "pseudobulk_raw_counts.csv.gz", index_col=0)
    metadata = read_metadata(args.deg_dir / "pseudobulk_sample_metadata.csv")
    metadata = metadata[metadata["contrast"].eq(args.contrast)]
    cpm = counts.div(counts.sum(axis=1), axis=0) * 1e6
    fractions = retained_fractions(args.four_state_root)

    genes = [gene for gene in significant["gene"] if gene in cpm.columns]
    missing = sorted(set(significant["gene"]) - set(genes))
    rows = []
    for patient, group in metadata.groupby("patient_id"):
        sides = dict(zip(group["comparison_status"], group["sample_id"]))
        if not {"case", "reference"} <= set(sides):
            continue
        case = cpm.loc[sides["case"], genes]
        reference = cpm.loc[sides["reference"], genes]
        # A pseudocount on CPM keeps a gene absent from one side finite without
        # letting it dominate; the pooled fit handles those genes properly and
        # this only asks whether patients agree with its sign.
        effect = np.log2((case + 1.0) / (reference + 1.0))
        rows.append({
            "patient_id": str(patient),
            "retained_fraction": fractions.get(str(patient), np.nan),
            **effect.to_dict(),
        })
    per_patient = pd.DataFrame(rows)
    if len(per_patient) < 6:
        raise RuntimeError(f"Only {len(per_patient)} patients carry both sides")

    pooled_sign = significant.set_index("gene")["log2_fold_change"].reindex(genes).gt(0)
    usable = per_patient.dropna(subset=["retained_fraction"])
    records = []
    for gene in genes:
        values = per_patient[gene].to_numpy(float)
        finite = np.isfinite(values)
        agreeing = int(
            ((values[finite] > 0) == bool(pooled_sign[gene])).sum()
        )
        paired = usable[["retained_fraction", gene]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if len(paired) >= 6 and paired[gene].nunique() >= 3:
            similarity = stats.spearmanr(
                paired["retained_fraction"].to_numpy(), paired[gene].to_numpy()
            )
            rho, pvalue = float(similarity.statistic), float(similarity.pvalue)
        else:
            rho, pvalue = float("nan"), float("nan")
        records.append({
            "gene": gene,
            "pooled_log2_fold_change": float(
                significant.set_index("gene").loc[gene, "log2_fold_change"]
            ),
            "pooled_fdr": float(significant.set_index("gene").loc[gene, "fdr"]),
            "used_for_ot_anywhere": bool(
                significant.set_index("gene").loc[gene, "used_for_ot_anywhere"]
            ),
            "patients_agreeing": agreeing,
            "patient_n": int(finite.sum()),
            "agreement_fraction": agreeing / int(finite.sum()) if finite.sum() else np.nan,
            "median_patient_log2": float(np.nanmedian(values)),
            "spearman_effect_vs_retained_fraction": rho,
            "spearman_p": pvalue,
        })
    audit = pd.DataFrame(records)

    args.output_root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_root / "deg_disqualifier_audit.csv", index=False)
    per_patient.to_csv(args.output_root / "deg_per_patient_effects.csv", index=False)

    top = significant.head(30)
    artefact = top["gene"].str.match(ARTEFACT_FAMILIES, case=False, na=False)
    absolute = audit["spearman_effect_vs_retained_fraction"].abs()
    report = {
        "contrast": args.contrast,
        "patient_n": int(len(per_patient)),
        "significant_gene_n": int(len(significant)),
        "genes_absent_from_pseudobulk": missing,
        "keratin_sprr_s100a_in_top_30": int(artefact.sum()),
        "keratin_sprr_s100a_names": sorted(top.loc[artefact, "gene"]),
        "agreement_fraction_median": float(audit["agreement_fraction"].median()),
        "genes_below_two_thirds_agreement": sorted(
            audit.loc[audit["agreement_fraction"].lt(2 / 3), "gene"]
        ),
        "similarity_abs_rho_median": float(absolute.median()),
        "similarity_abs_rho_max": float(absolute.max()),
        "genes_tracking_similarity_p05": sorted(
            audit.loc[audit["spearman_p"].lt(0.05), "gene"]
        ),
        "reading": {
            "keratin": "Domination of the top 30 means the residual depth signal "
                       "still drives the result.",
            "agreement": "A gene the patients do not share is not a programme, "
                         "whatever its FDR.",
            "similarity": "Effects tracking the retained fraction mean the DEG "
                          "reads how alike the two tumours are.",
        },
    }
    (args.output_root / "deg_disqualifier_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    columns = ["gene", "pooled_log2_fold_change", "pooled_fdr", "used_for_ot_anywhere",
               "patients_agreeing", "patient_n",
               "spearman_effect_vs_retained_fraction", "spearman_p"]
    pd.set_option("display.width", 200)
    print(audit[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
