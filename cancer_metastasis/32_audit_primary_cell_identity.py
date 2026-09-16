"""Ask what the gate's rejected primary cells actually are.

GSE180661 labels major cell types with CellAssign over nine categories:
B.cell, Dendritic.cell, Endothelial.cell, Fibroblast, Myeloid.cell,
Ovarian.cancer.cell, Plasma.cell, T.cell and Mast.cell.  **There is no normal
epithelial category.**  CellAssign returns the most probable of the categories
it is given, and the Ovarian.cancer.cell markers are epithelial rather than
malignant -- WFDC2, CD24, CLDN3, KRT7/8/17/18/19, EPCAM, WT1, CLDN4, MSLN,
FOLR1, MUC1 -- so any epithelial cell in the sample receives that label.

The primary sites here are adnexal, which GEO records as "adnexa (ovary and
fallopian tube)", and fallopian tube epithelium includes ciliated cells, a
non-malignant lineage.  The authors' own sub-clustering of these cells
(Supplementary Table 4) contains **Ciliated.cell.1** and **Ciliated.cell.2**
alongside Cancer.cell.1-6, which is direct evidence that ciliated cells carry
the Ovarian.cancer.cell label in the released data.  This analysis uses that
label, so they are inside its malignant set.

That matters because the gate's rejected primary cells carry DNAH11, RSPH9 and
CCDC96 -- motile cilia genes, none of them a highly variable gene used to build
the transport.  If the rejected set is substantially ciliated epithelium, the
gate is performing cell-type quality control and its differential expression
describes a contaminant rather than a malignant state.

**The published clusters are the right yardstick**, not a marker list invented
here.  Scoring against Cancer.cell.1-6, the two cycling clusters and the two
ciliated clusters places every cell on the same map the original study used, so
the answer is comparable to their figures rather than to our own construction.
Three of those clusters are the ones our differential expression recovered:
Cancer.cell.3 leads on CXCL10, ISG15 and IFIT3, Cancer.cell.2 on S100A9, KRT17
and LCN2, Cancer.cell.6 on VEGFA, SLC2A1 and NDRG1.

Cluster scores are standardised across cells before a cell is assigned to its
best cluster, because the marker sets differ in size and in baseline
expression, and an unstandardised argmax would simply pick the most highly
expressed set every time.

Immune and stromal sets are scored too.  The series was flow-sorted into CD45+
and CD45- fractions and sequenced separately, so immune signal inside the
CD45- fraction is ambient carry-over or a sorting escape, and either way is not
a malignant state.

Scores are mean log-CPM on the depth-equalised counts, so no comparison here
can be a depth artefact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from common import expression_matrix, load_exact_side


# Not from the published table: these ask whether a cell is epithelial at all.
CONTAMINATION_SETS = {
    "immune": ["PTPRC", "CD3E", "CD2", "LYZ", "AIF1", "TYROBP", "FCER1G", "CD68"],
    "stromal": ["COL1A1", "COL1A2", "DCN", "LUM", "PECAM1", "VWF"],
    # The CellAssign definition of Ovarian.cancer.cell, which is what every
    # cell in this analysis satisfied to be here.
    "epithelial": [
        "WFDC2", "CD24", "CLDN3", "KRT7", "KRT8", "KRT17", "KRT18", "KRT19",
        "EPCAM", "WT1", "CLDN4", "MSLN", "FOLR1", "MUC1",
    ],
}


def load_cluster_markers(path: Path, top_n: int) -> dict[str, list[str]]:
    table = pd.read_csv(path)
    missing = [name for name in ("cluster", "rank", "gene") if name not in table.columns]
    if missing:
        raise RuntimeError(f"{path}: missing columns {missing}")
    table = table[table["rank"].le(top_n)]
    return {
        str(cluster): [str(gene) for gene in group.sort_values("rank")["gene"]]
        for cluster, group in table.groupby("cluster", sort=True)
    }


def gene_symbols(data) -> np.ndarray:
    """Resolve HGNC symbols for matching published marker lists.

    ``common.gene_keys`` prefers ``gene_id`` and so returns Ensembl
    identifiers when the H5AD carries them, which match no marker list. The
    symbol column is what the published tables are written in.
    """
    symbols = np.asarray(data.var_names.astype(str), dtype=str)
    if "gene_symbol" in data.var:
        candidate = data.var["gene_symbol"].astype(str).str.strip().to_numpy(dtype=str)
        valid = ~pd.Series(candidate).str.lower().isin(
            {"", "na", "n/a", "nan", "none", "null", "<na>"}
        ).to_numpy()
        symbols = np.where(valid, candidate, symbols)
    return np.asarray([str(value).strip() for value in symbols], dtype=str)


def log_cpm(data) -> tuple[sparse.csr_matrix, dict[str, int]]:
    matrix = sparse.csr_matrix(expression_matrix(data), dtype=np.float64)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    scaled = (sparse.diags(1e6 / totals) @ matrix).tocsr()
    scaled.data = np.log1p(scaled.data)
    index: dict[str, int] = {}
    for position, gene in enumerate(gene_symbols(data)):
        index.setdefault(gene, position)
    return scaled, index


def set_scores(
    scaled: sparse.csr_matrix, index: dict[str, int], sets: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict[str, int]]:
    scores, present_n = {}, {}
    for name, markers in sets.items():
        columns = [index[gene] for gene in markers if gene in index]
        present_n[name] = len(columns)
        scores[name] = (
            np.asarray(scaled[:, columns].mean(axis=1)).ravel()
            if columns else np.full(scaled.shape[0], np.nan)
        )
    return pd.DataFrame(scores), present_n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("gate_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--cluster-markers", type=Path,
        default=Path(__file__).resolve().parent / "reference"
        / "spectrum_epithelial_cluster_markers.csv",
    )
    parser.add_argument("--top-markers", type=int, default=50)
    parser.add_argument("--scope", default="scope_malignant")
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--side", default="source", choices=("source", "target"))
    args = parser.parse_args()

    clusters = load_cluster_markers(args.cluster_markers, args.top_markers)
    manifest = pd.read_csv(args.manifest_csv)
    side = args.side
    frames, seen, marker_n = [], set(), {}
    symbol_sample: list[str] = []
    for row in manifest.to_dict("records"):
        sample = str(row[f"{side}_sample"])
        key = (str(row["patient_id"]), sample)
        if key in seen:
            continue
        matches = sorted(
            args.gate_root.glob(f"{row['pair_id']}/{args.scope}/*/cell_confidence.csv")
        )
        if not matches:
            continue
        gate = pd.read_csv(matches[0], usecols=["method", "side", "observation_id", "retained"])
        gate = gate.loc[gate["method"].eq(args.method) & gate["side"].eq(side)]
        if gate.empty:
            continue
        seen.add(key)

        data = load_exact_side(
            [str(value) for value in json.loads(str(row[f"{side}_h5ads_json"]))], sample
        )
        scaled, index = log_cpm(data)
        if not symbol_sample:
            symbol_sample = sorted(index)[:8]
        cluster_scores, cluster_n = set_scores(scaled, index, clusters)
        other_scores, other_n = set_scores(scaled, index, CONTAMINATION_SETS)
        marker_n = {**cluster_n, **other_n}
        scores = pd.concat([cluster_scores, other_scores], axis=1)
        scores.insert(0, "observation_id", np.asarray(data.obs_names.astype(str)))

        joined = gate.assign(observation_id=gate["observation_id"].astype(str)).merge(
            scores, on="observation_id", how="inner"
        )
        if joined.empty:
            continue
        joined.insert(0, "sample", sample)
        joined.insert(0, "patient_id", str(row["patient_id"]))
        frames.append(joined.drop(columns=["method", "side"]))
        del data, scaled

    if not frames:
        raise RuntimeError("No sample produced both a gate and an expression matrix")
    cells = pd.concat(frames, ignore_index=True)
    cells["retained"] = cells["retained"].astype(bool)

    # A marker set that matched no gene scores NaN for every cell, and the
    # assignment must not silently proceed on what is left. When the symbols do
    # not match at all the cause is the identifier space, not the biology, so
    # say which one the matrix is in.
    names = sorted(name for name in clusters if marker_n.get(name, 0) > 0)
    empty = sorted(name for name in clusters if marker_n.get(name, 0) == 0)
    if len(names) < 2:
        raise RuntimeError(
            f"Only {len(names)} of {len(clusters)} marker sets matched any gene. "
            f"The matrix appears not to be indexed by HGNC symbol; first names "
            f"seen: {symbol_sample}"
        )

    # Standardised within this cohort so the argmax is not decided by which
    # marker set happens to sit highest on the expression scale.
    block = cells[names].to_numpy(float)
    centred = (block - np.nanmean(block, axis=0)) / np.where(
        np.nanstd(block, axis=0) > 0, np.nanstd(block, axis=0), 1.0
    )
    cells["assigned_cluster"] = [names[position] for position in np.nanargmax(centred, axis=1)]
    for position, name in enumerate(names):
        cells[f"z_{name}"] = centred[:, position]

    args.output_root.mkdir(parents=True, exist_ok=True)
    cells.to_csv(
        args.output_root / "primary_cell_identity_scores.csv.gz",
        index=False, compression="gzip",
    )

    composition = (
        pd.crosstab(cells["assigned_cluster"], cells["retained"], normalize="columns")
        .rename(columns={True: "retained_fraction", False: "rejected_fraction"})
    )
    counts = pd.crosstab(cells["assigned_cluster"], cells["retained"]).rename(
        columns={True: "retained_n", False: "rejected_n"}
    )
    summary = counts.join(composition).reset_index()
    summary["rejected_over_retained"] = (
        summary["rejected_fraction"] / summary["retained_fraction"].replace(0, np.nan)
    )
    summary = summary.sort_values("rejected_fraction", ascending=False)
    summary.to_csv(args.output_root / "primary_cluster_composition.csv", index=False)

    ciliated = [name for name in names if name.startswith("Ciliated")]
    is_ciliated = cells["assigned_cluster"].isin(ciliated)
    report = {
        "gate_root": str(args.gate_root),
        "side": side,
        "cell_n": int(len(cells)),
        "retained_n": int(cells["retained"].sum()),
        "rejected_n": int((~cells["retained"]).sum()),
        "cluster_markers": str(args.cluster_markers),
        "markers_found_per_set": marker_n,
        "marker_sets_with_no_match": empty,
        "gene_symbol_sample": symbol_sample,
        "ciliated_share_of_retained": float(is_ciliated[cells["retained"]].mean()),
        "ciliated_share_of_rejected": float(is_ciliated[~cells["retained"]].mean()),
        "reading": (
            "A large ciliated share of the rejected set means the gate is "
            "removing non-malignant fallopian tube epithelium that CellAssign "
            "could not label, because its nine categories include no normal "
            "epithelium. A ciliated share near equal in both groups clears that "
            "explanation and leaves the Cancer.cell clusters as the difference."
        ),
    }
    (args.output_root / "primary_cell_identity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False), flush=True)
    print("\nmedian score by gate status:", flush=True)
    columns = [*names, *CONTAMINATION_SETS]
    print(
        cells.groupby("retained")[columns].median().T.rename(
            columns={True: "retained", False: "rejected"}
        ).to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
