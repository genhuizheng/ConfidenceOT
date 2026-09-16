"""Flag lymphoid contamination among cells labelled malignant.

Lymph-node metastases carry B and plasma cells into any epithelial gate, and a
tumour cell that shares a droplet with one is not a tumour cell.  This scores
each cell and separates two things a single score cannot:

*A doublet or a genuinely contaminated cell* expresses B-lineage **structural**
genes -- MS4A1, CD79A, CD79B, BANK1 -- alongside immunoglobulin, and keeps a
high epithelial score, because two cells are being read as one.

*Ambient immunoglobulin* gives a high IGKC or IGHG1 with CD79A and MS4A1 at
zero.  Immunoglobulin transcripts are the most abundant free RNA in lymphoid
tissue, so a threshold on immunoglobulin alone removes the cells sitting in the
most contaminated part of the dissociation rather than the cells that are
actually two cells.  In a lymph node that is most of the sample, and the filter
would discard a large share of real tumour cells while leaving the doublets it
was meant to catch.

The flag therefore requires the structural score, and immunoglobulin is
reported beside it rather than used to decide.

**Shape, not mean.**  Contamination is a small strongly-positive minority; a
biological programme is graded across the population.  The report gives the
fraction above threshold and the score quantiles so the two are
distinguishable, because a difference in means alone cannot tell them apart.
That lesson cost this project two wrong calls on 2026-09-15, both from reading
pooled composition instead of distributions.

An androgen-response set is scored too, since reduced AR signalling is the
headline of the metastasis comparison this feeds, and a contamination filter
must not be the thing that produces it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


MARKER_SETS = {
    # Structural B-lineage genes. A cell is only flagged on these.
    "b_structural": [
        "MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "TNFRSF13C", "CD22",
        "FCRL1", "PAX5", "BLK",
    ],
    # Plasma-cell genes that are restricted to the lineage. An earlier version
    # of this set included XBP1, SEC11C, SSR4, FKBP11 and SDC1, which flagged
    # 98.5% of a prostate primary tumour. Those are general endoplasmic
    # reticulum and secretory machinery, and SDC1 is a classic epithelial
    # syndecan, so in a secretory gland they are high in the tissue itself. The
    # tell was that the flagged cells scored *higher* on epithelial identity
    # and on androgen response than the unflagged ones, which is the opposite
    # of what a lymphoid contaminant looks like.
    "plasma_structural": ["MZB1", "TNFRSF17", "DERL3", "PRDM1", "POU2AF1"],
    # Reported, never used to flag: the dominant ambient species in lymphoid
    # tissue.
    "immunoglobulin": [
        "IGKC", "IGHG1", "IGHG3", "IGHA1", "IGHM", "IGLC1", "IGLC2", "JCHAIN",
    ],
    "t_cell": ["CD3D", "CD3E", "CD2", "TRAC", "IL7R", "CD8A"],
    "myeloid": ["LYZ", "AIF1", "TYROBP", "FCER1G", "CD68", "C1QA"],
    "epithelial": [
        "EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "CLDN4", "KLK3", "NKX3-1",
    ],
    # Low androgen response is the result this filter must not manufacture.
    "androgen_response": [
        "KLK3", "KLK2", "NKX3-1", "TMPRSS2", "FKBP5", "PMEPA1", "STEAP4",
        "ABCC4", "MAF", "ELL2",
    ],
}


def resolve_symbols(data: ad.AnnData) -> np.ndarray:
    symbols = np.asarray(data.var_names.astype(str), dtype=str)
    for column in ("gene_symbol", "gene_symbols", "features", "SYMBOL"):
        if column in data.var:
            candidate = data.var[column].astype(str).str.strip().to_numpy(dtype=str)
            valid = ~pd.Series(candidate).str.lower().isin(
                {"", "na", "n/a", "nan", "none", "null", "<na>"}
            ).to_numpy()
            symbols = np.where(valid, candidate, symbols)
            break
    return np.asarray([str(value).strip() for value in symbols], dtype=str)


def log_cpm(data: ad.AnnData) -> tuple[sparse.csr_matrix, str]:
    """Return log-CPM, taking raw counts when the object still carries them.

    A Seurat object converted to H5AD often has normalised values in X, which
    are already a per-cell scaling and are usable for scoring as they stand.
    Re-normalising them would be harmless but silent, so which one was used is
    reported.
    """
    source = data.layers["counts"] if "counts" in data.layers else data.X
    matrix = sparse.csr_matrix(source, dtype=np.float64)
    integral = matrix.data.size == 0 or (
        matrix.data.min() >= 0 and np.allclose(matrix.data, np.round(matrix.data))
    )
    if not integral:
        return matrix, "pre-normalised values used as given"
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    scaled = (sparse.diags(1e6 / totals) @ matrix).tocsr()
    scaled.data = np.log1p(scaled.data)
    return scaled, "log1p CPM from raw counts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--flag-threshold", type=float, default=0.5,
        help="A cell is flagged when a structural score exceeds this. Choose it "
             "from the reported quantiles rather than accepting the default.",
    )
    parser.add_argument(
        "--group-by", action="append", dest="groups", default=None,
        help="obs column to break the summary down by; may be repeated",
    )
    args = parser.parse_args()

    data = ad.read_h5ad(args.h5ad)
    symbols = resolve_symbols(data)
    index: dict[str, int] = {}
    for position, gene in enumerate(symbols):
        index.setdefault(gene, position)
    scaled, normalisation = log_cpm(data)

    scores = pd.DataFrame(index=pd.Index(data.obs_names.astype(str), name="cell"))
    found: dict[str, list[str]] = {}
    for name, markers in MARKER_SETS.items():
        present = [gene for gene in markers if gene in index]
        found[name] = present
        scores[name] = (
            np.asarray(scaled[:, [index[gene] for gene in present]].mean(axis=1)).ravel()
            if present else np.nan
        )
    if not found["b_structural"] and not found["plasma_structural"]:
        raise RuntimeError(
            "No structural B or plasma marker was found. First gene names: "
            f"{sorted(index)[:8]}"
        )

    # Flagged on structure only. Immunoglobulin is reported beside the flag so
    # that an ambient-driven cut can be recognised rather than adopted.
    flagged = (
        scores["b_structural"].fillna(0).gt(args.flag_threshold)
        | scores["plasma_structural"].fillna(0).gt(args.flag_threshold)
    )
    scores["lymphoid_flagged"] = flagged.to_numpy()
    scores["immunoglobulin_only"] = (
        scores["immunoglobulin"].fillna(0).gt(args.flag_threshold) & ~flagged
    ).to_numpy()

    carry = [column for column in (args.groups or []) if column in data.obs]
    for column in carry:
        scores[column] = data.obs[column].astype(str).to_numpy()

    args.output_root.mkdir(parents=True, exist_ok=True)
    scores.to_csv(
        args.output_root / "lineage_contamination_scores.csv.gz",
        index=True, compression="gzip",
    )
    scores.loc[~scores["lymphoid_flagged"]].index.to_series().to_csv(
        args.output_root / "cells_passing_lymphoid_filter.csv", index=False, header=["cell"]
    )

    quantiles = scores[list(MARKER_SETS)].quantile([0.5, 0.9, 0.95, 0.99]).round(4)

    # A lymphoid contaminant is less epithelial and, here, less androgen
    # responsive than the tumour cells around it. If the flagged cells score
    # higher on those, the marker set is tracking the tissue rather than the
    # contaminant, and the flag is worse than no filter. This check exists
    # because that is exactly what happened with an earlier plasma set.
    warnings: list[str] = []
    if bool(flagged.any()) and bool((~flagged).any()):
        for lineage in ("epithelial", "androgen_response"):
            if lineage not in scores:
                continue
            inside = float(scores.loc[flagged, lineage].median())
            outside = float(scores.loc[~flagged, lineage].median())
            if inside > outside:
                warnings.append(
                    f"flagged cells have a higher median {lineage} score "
                    f"({inside:.3f} vs {outside:.3f}); the marker sets are "
                    "tracking the tissue, not a contaminant"
                )
    if flagged.mean() > 0.5:
        warnings.append(
            f"{flagged.mean():.1%} of cells flagged; a contaminant is a "
            "minority by definition, so the threshold or the marker sets are wrong"
        )

    report = {
        "h5ad": str(args.h5ad),
        "cell_n": int(data.n_obs),
        "normalisation": normalisation,
        "markers_found": {name: len(genes) for name, genes in found.items()},
        "markers_missing": {
            name: sorted(set(MARKER_SETS[name]) - set(genes))
            for name, genes in found.items()
            if set(MARKER_SETS[name]) - set(genes)
        },
        "flag_threshold": args.flag_threshold,
        "flagged_n": int(flagged.sum()),
        "flagged_fraction": float(flagged.mean()),
        "immunoglobulin_only_n": int(scores["immunoglobulin_only"].sum()),
        "immunoglobulin_only_fraction": float(scores["immunoglobulin_only"].mean()),
        "score_quantiles": json.loads(quantiles.to_json()),
        "warnings": warnings,
        "obs_columns_carried": carry,
        "reading": (
            "flagged_fraction small with a high immunoglobulin_only_fraction is "
            "the expected picture in a lymph node: few real doublets, much "
            "ambient immunoglobulin. If a filter on immunoglobulin alone would "
            "have removed the immunoglobulin_only cells too, it would discard "
            "real tumour cells. Compare androgen_response between flagged and "
            "passing cells before using the filter, because the metastasis "
            "result this feeds is a loss of androgen signalling and the filter "
            "must not be what produces it."
        ),
    }
    (args.output_root / "lineage_contamination_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)

    pd.set_option("display.width", 200)
    print("\nmedian score by flag:", flush=True)
    print(
        scores.groupby("lymphoid_flagged")[list(MARKER_SETS)].median().T.to_string(),
        flush=True,
    )
    for column in carry:
        print(f"\nflagged fraction by {column}:", flush=True)
        print(scores.groupby(column)["lymphoid_flagged"].agg(["mean", "size"]).to_string(), flush=True)


if __name__ == "__main__":
    main()
