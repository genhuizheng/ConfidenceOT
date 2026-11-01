"""PREPROCESSING_CONFIG_REGISTRY.tsv: every preprocessing configuration this project has executed, each described from
the code that builds it (confidenceot.Preprocessing.from_label + Preprocessing.representation +
cancer_metastasis/common.prepare_joint_representation + 27_downsample_counts.py + 02_run_pair.py defaults), never
from its name.

    python realdata_benchmark/01_config_registry.py --out OUT_DIR

Columns describe, per label: input counts, library-size normalisation, rank transformation, gene-median division,
downsampling target, tie-breaking, HVG selection, gene scaling, PCA dimension, cost geometry, median cost scaling,
entropic regularisation and calibration, plus where the label was executed (from the repository's run records:
benchmark/ARCHIVE_MANIFEST.md, the submit scripts and the LTL331 / KP / Quinn / MOSTA run roots as recorded). Labels
are kept whatever their results were. The executed-in column records what the repository documents; a run that left
no record here is not claimed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from confidenceot import Preprocessing

GRID = [f"{stem}{scale}{ds}{cos}" for stem in ("raw", "logcpm", "rank256", "ranknm256")
        for scale in ("", "_noscale") for ds in ("", "_ds") for cos in ("", "_cos")]
EXTRA = ["log1p_noscale_pcc", "log1p_noscale", "log1p_allgenes_noscale", "log1p_nopca_noscale", "log1p_allgenes_nopca_noscale"]

# Where each label ran, as recorded in this repository (source in brackets).
EXECUTED = {
    **{label: ["synthetic native benchmark, acceptance 0.90 (benchmark/ARCHIVE_MANIFEST.md, 32 arms)",
               "real-data pan-cancer preprocessing factorial (cancer_metastasis/submit_preprocessing_factorial.sh default grid)"]
       for label in GRID},
}
for label in ("ranknm256_noscale_ds_cos", "ranknm256_noscale_cos", "raw"):
    EXECUTED[label] = EXECUTED[label] + ["synthetic benchmark, acceptance 0.95 and 0.99 (ARCHIVE_MANIFEST.md)"]
EXECUTED["ranknm256_noscale_ds_cos"] += [
    "CNV/RNA strategy comparison, three cancers (cancer_metastasis/submit_cnv_rna_strategies.sh)",
    "KP-Tracer lineage runs and Level 1-3 validation (kp_lineage/, lineage_validation/)",
    "Quinn A549 lineage runs (lineage_validation/)",
    "MOSTA all stages (mouse_embryo/, MOSTA_PREPROCESSING default)",
    "LTL331 GSE297328 (ltl331_nepc/submit_ltl331.sh)"]
EXECUTED["raw"] += ["KP-Tracer second arm (kp_lineage/submit_kptracer_chain.sh)", "cycling x CNV factorial (submit_cycling_cnv_factorial.sh)"]
EXECUTED["logcpm"] += ["MOSTA logcpm arm (mosta_all_stages_logcpm)", "LTL331 GSE297328 (ltl331_nepc/submit_ltl331.sh)"]
EXECUTED["log1p_noscale_pcc"] = ["synthetic benchmark, acceptance 0.99 / 0.95 / 0.90 (benchmark/EXPERIMENT_2026-10-19_LOG1P_PEARSON.md)"]
for label in EXTRA[1:]:
    EXECUTED[label] = ["synthetic benchmark log1p dimension arms: scripted (benchmark/tacc/submit_log1p_dimensions.sh); "
                       "no run record in the repository"]

LIBRARY = {
    "precomputed": "none: the count matrix is used as supplied",
    "log1p": "none: log1p of the counts, no library-size division",
    "log_cpm": "each cell scaled to 1e4 total, then log1p",
    "cpm": "each cell scaled to 1e4 total",
    "rank_value": "each cell divided by its total (does not change the within-cell order)",
    "rank_no_median": "each cell divided by its total (does not change the within-cell order)",
    "pearson_residuals": "analytic Pearson residuals",
}


def describe(label: str) -> dict:
    c = Preprocessing.from_label(label)
    rank = c.normalisation in ("rank_value", "rank_no_median")
    row = {"label": label, "parsed_label_roundtrip": c.label() == label, "normalisation": c.normalisation}
    row["input_counts"] = ("depth-equalised counts written by 27_downsample_counts.py (exact multivariate hypergeometric "
                           "subsampling without replacement; cells at or below the target untouched)") if c.equalise_depth \
        else "original counts (after the project's QC)"
    row["downsampling_target"] = (f"pooled quantile {c.equalise_quantile:g} of the analysed cells' total counts, per run "
                                  "(LTL331: 2,817)") if c.equalise_depth else "none"
    row["library_size_normalisation"] = LIBRARY[c.normalisation]
    row["rank_transformation"] = (f"per cell: top {c.rank_top_n} genes by value, value 1 - r/{c.rank_top_n} for rank r = 0..{c.rank_top_n - 1}, "
                                  "every other gene 0") if rank else "none"
    row["gene_median_normalisation"] = ("each gene divided by its nonzero median over both sides before ranking"
                                        if c.normalisation == "rank_value" else
                                        "none (ranks the cell's own values directly)" if c.normalisation == "rank_no_median" else "not applicable")
    row["tie_breaking"] = ("numpy argsort(-value, kind='stable') over the cell's nonzero entries: equal values keep column "
                           "order, and columns are the common gene keys sorted as strings (alphabetical by symbol); with "
                           "integer counts and rank_no_median, ties are frequent" if c.normalisation == "rank_no_median" else
                           "as rank_no_median, but on count / total / gene median, where exact ties are rarer" if c.normalisation == "rank_value"
                           else "not applicable (no ranking); HVG variance ties also break by column order")
    row["hvg_selection"] = ("every common gene" if c.n_hvg == 0 else
                            f"top {c.n_hvg} genes by variance over both sides stacked, computed on the transformed matrix "
                            "(stable argsort: equal variances keep column order)")
    row["gene_scaling"] = ("genes centred and divided by their standard deviation over both sides" if c.scale_genes
                           else "none (PCA centres internally)")
    row["pca_dimension"] = "none (genes are the coordinates)" if c.n_pcs == 0 else f"{c.n_pcs}, sklearn PCA on both sides stacked, random_state = run seed"
    row["cost_geometry"] = {"cosine": "PCA coordinates L2-normalised per cell, then squared Euclidean = 2 - 2 cos",
                            "squared_euclidean": "squared Euclidean on the PCA coordinates",
                            "pearson": "per cell centred then L2-normalised, squared Euclidean = 2 - 2 r"}[c.cost]
    row["median_cost_scaling"] = ("cost divided by the median of up to 1,000,000 sampled cross-side squared distances, zeros excluded"
                                  if c.scale == "median_sampled_pair" else "none")
    row["entropic_regularisation"] = "epsilon 0.1, lambda_a = lambda_b = 1.0, tolerance 1e-4 (02_run_pair.py defaults; LTL331 passes tolerance 1e-4)"
    row["calibration"] = ("rejection cost calibrated on the within-side split null (5 calibration + 5 validation replicates, grid 5); "
                          "the within-side acceptance minimum is set per run (0.90 synthetic main grid, 0.99 real data)")
    row["executed_in"] = " | ".join(EXECUTED.get(label, ["no run record in the repository"]))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    target = a.out / "PREPROCESSING_CONFIG_REGISTRY.tsv"
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [describe(label) for label in GRID + EXTRA]
    pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
    print(f"{len(rows)} configurations -> {target}")


if __name__ == "__main__":
    main()
