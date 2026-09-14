"""Plot joint primary--metastasis malignant-cell UMAPs for replication datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import cell_qc_table, load_exact_side, prepare_joint_representation


DATASET_LABELS = {
    "GSE181919": ["Malignant.cells"],
    "GSE225857": [
        "Tu01_AREG", "Tu02_DEFA5", "Tu03_SRRM2", "Tu04_RGMB",
        "Tu05_PCNA", "Tu06_NKD1", "Tu07_MKI67", "Tu08_GNG13",
        "Tu09_MUC2", "Tu10_COL3A1", "Tu11_PLA2G2A",
    ],
}


def annotation_column(data: ad.AnnData) -> str:
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            return column
    raise KeyError("H5AD has no supported author cell-type column")


def paths_from_row(row: pd.Series, side: str) -> list[str]:
    column = f"{side}_h5ads_json"
    if column in row and pd.notna(row[column]):
        return [str(value) for value in json.loads(str(row[column]))]
    return [str(row[f"{side}_h5ad"])]


def load_side(
    manifest: pd.DataFrame,
    side: str,
    malignant_labels: list[str],
    *,
    minimum_counts: int,
    minimum_features: int,
    maximum_mitochondrial_percent: float,
) -> ad.AnnData:
    sample_column = f"{side}_sample"
    path_column = f"{side}_h5ads_json"
    fallback_column = f"{side}_h5ad"
    records = manifest[["patient_id", sample_column, path_column, fallback_column]].drop_duplicates()
    pieces = []
    for row in records.to_dict("records"):
        series = pd.Series(row)
        sample = str(series[sample_column])
        data = load_exact_side(paths_from_row(series, side), sample)
        data.obs_names = data.obs_names.astype(str)
        column = annotation_column(data)
        labels = data.obs[column].astype(str)
        data = data[labels.isin(malignant_labels).to_numpy()].copy()
        if not data.n_obs:
            continue
        qc = cell_qc_table(
            data,
            minimum_total_counts=minimum_counts,
            minimum_detected_genes=minimum_features,
            maximum_mitochondrial_percent=maximum_mitochondrial_percent,
        )
        data = data[qc["qc_pass"].to_numpy()].copy()
        if not data.n_obs:
            continue
        data.obs["observation_id"] = data.obs_names.astype(str)
        data.obs["patient_id"] = str(series["patient_id"])
        data.obs["sample_id_exact"] = sample
        data.obs["side"] = "primary" if side == "source" else "metastasis"
        data.obs["author_subtype"] = data.obs[column].astype(str).to_numpy()
        data.obs_names = pd.Index(
            [f"{series['patient_id']}::{sample}::{value}" for value in data.obs_names],
            dtype=str,
        )
        pieces.append(data)
    if not pieces:
        raise RuntimeError(f"No post-QC malignant cells were loaded for side={side}")
    return ad.concat(pieces, join="inner", merge="same", index_unique=None)


def one_confidence_file(ot_root: Path, pair_id: str) -> Path:
    matches = sorted((ot_root / pair_id).glob("scope_malignant/*/cell_confidence.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one cell_confidence.csv for {pair_id}; found {len(matches)}"
        )
    return matches[0]


def primary_gate_table(manifest: pd.DataFrame, ot_root: Path) -> pd.DataFrame:
    rows = []
    for pair in manifest.itertuples(index=False):
        table = pd.read_csv(one_confidence_file(ot_root, str(pair.pair_id)))
        table = table.loc[table["method"].eq("M4-E") & table["side"].eq("source")].copy()
        table["patient_id"] = str(pair.patient_id)
        table["sample_id_exact"] = str(pair.source_sample)
        rows.append(table[[
            "patient_id", "sample_id_exact", "observation_id", "retained",
            "normalized_rejection_score",
        ]])
    long = pd.concat(rows, ignore_index=True)
    keys = ["patient_id", "sample_id_exact", "observation_id"]
    result = []
    for values, group in long.groupby(keys, sort=False):
        states = set(group["retained"].astype(bool))
        state = "retained" if states == {True} else "rejected" if states == {False} else "discordant"
        record = dict(zip(keys, values))
        record.update({
            "primary_gate": state,
            "mean_rejection_score": float(group["normalized_rejection_score"].mean()),
            "pair_occurrence_n": int(len(group)),
        })
        result.append(record)
    return pd.DataFrame(result)


def embed(source: ad.AnnData, target: ad.AnnData, *, n_hvg: int, n_pcs: int,
          neighbors: int, minimum_distance: float, seed: int) -> tuple[np.ndarray, list[str]]:
    source_pca, target_pca, hvg, _ = prepare_joint_representation(
        source, target, n_hvg=n_hvg, n_pcs=n_pcs, seed=seed
    )
    try:
        import umap
    except ImportError as error:
        raise RuntimeError("umap-learn>=0.5 is required") from error
    pca = np.vstack([source_pca, target_pca]).astype(np.float32)
    model = umap.UMAP(
        n_neighbors=min(neighbors, len(pca) - 1),
        min_dist=minimum_distance,
        metric="euclidean",
        random_state=seed,
    )
    return model.fit_transform(pca), hvg


def scatter_categories(axis, table: pd.DataFrame, column: str, title: str) -> None:
    categories = sorted(table[column].astype(str).unique())
    palette = plt.get_cmap("tab20", max(1, len(categories)))
    for index, category in enumerate(categories):
        use = table.loc[table[column].astype(str).eq(category)]
        axis.scatter(
            use["UMAP1"], use["UMAP2"], s=7, alpha=0.68, linewidths=0,
            color=palette(index), label=category, rasterized=True,
        )
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=7, markerscale=2, loc="best")


def clean_axis(axis) -> None:
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_xticks([])
    axis.set_yticks([])


def make_figure(table: pd.DataFrame, dataset: str, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 12))
    scatter_categories(axes[0, 0], table, "author_subtype", "Author malignant-cell subtype")
    scatter_categories(axes[0, 1], table, "side", "Tissue origin")
    scatter_categories(axes[1, 0], table, "patient_id", "Patient")

    axis = axes[1, 1]
    metastasis = table.loc[table["side"].eq("metastasis")]
    axis.scatter(
        metastasis["UMAP1"], metastasis["UMAP2"], s=5, alpha=0.2,
        linewidths=0, color="#bdbdbd", label="Metastasis", rasterized=True,
    )
    colors = {"retained": "#2b8cbe", "rejected": "#d7301f", "discordant": "#969696"}
    primary = table.loc[table["side"].eq("primary")]
    for state in ("retained", "rejected", "discordant"):
        use = primary.loc[primary["primary_gate"].eq(state)]
        if len(use):
            axis.scatter(
                use["UMAP1"], use["UMAP2"], s=8, alpha=0.72, linewidths=0,
                color=colors[state], label=f"Primary {state}", rasterized=True,
            )
    axis.set_title("Primary M4-E gate; metastasis shown as context")
    axis.legend(frameon=False, fontsize=8, markerscale=2)
    for panel in axes.flat:
        clean_axis(panel)
    figure.suptitle(f"{dataset}: joint primary–metastasis malignant-cell UMAP", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output / f"{dataset}_primary_metastasis_subtype_umap.png", dpi=300)
    figure.savefig(output / f"{dataset}_primary_metastasis_subtype_umap.pdf")
    plt.close(figure)


def make_primary_gate_context_figure(
    table: pd.DataFrame, dataset: str, output: Path
) -> None:
    """Show primary M4-E states relative to metastatic malignant cells."""
    figure, axis = plt.subplots(figsize=(8.4, 7.2))
    styles = (
        ("metastasis", None, "#bdbdbd", "Metastatic malignant cells", 6, 0.32),
        ("primary", "retained", "#2b8cbe", "Primary retained", 9, 0.78),
        ("primary", "rejected", "#d7301f", "Primary rejected", 9, 0.78),
        ("primary", "discordant", "#636363", "Primary discordant", 9, 0.78),
    )
    for side, gate, color, label, size, alpha in styles:
        use = table.loc[table["side"].eq(side)]
        if gate is not None:
            use = use.loc[use["primary_gate"].eq(gate)]
        if not len(use):
            continue
        axis.scatter(
            use["UMAP1"], use["UMAP2"], s=size, alpha=alpha, linewidths=0,
            color=color, label=f"{label} (n={len(use):,})", rasterized=True,
        )
    clean_axis(axis)
    axis.set_title(
        f"{dataset}: primary M4-E gate states relative to metastatic malignant cells"
    )
    axis.legend(frameon=False, fontsize=9, markerscale=2, loc="best")
    figure.tight_layout()
    stem = f"{dataset}_primary_retained_rejected_vs_metastasis_umap"
    figure.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def run_dataset(args: argparse.Namespace, dataset: str) -> dict[str, object]:
    manifest_path = args.replication_root / dataset / "manifest" / "pair_manifest_malignant_eligible.csv"
    ot_root = args.replication_root / dataset / "ot"
    manifest = pd.read_csv(manifest_path)
    labels = DATASET_LABELS[dataset]
    source = load_side(
        manifest, "source", labels, minimum_counts=args.minimum_counts,
        minimum_features=args.minimum_features,
        maximum_mitochondrial_percent=args.maximum_mitochondrial_percent,
    )
    target = load_side(
        manifest, "target", labels, minimum_counts=args.minimum_counts,
        minimum_features=args.minimum_features,
        maximum_mitochondrial_percent=args.maximum_mitochondrial_percent,
    )
    coordinates, hvg = embed(
        source, target, n_hvg=args.n_hvg, n_pcs=args.n_pcs,
        neighbors=args.neighbors, minimum_distance=args.minimum_distance, seed=args.seed,
    )
    obs = pd.concat([source.obs, target.obs], axis=0)
    table = obs[[
        "observation_id", "patient_id", "sample_id_exact", "side", "author_subtype"
    ]].reset_index(drop=True)
    table[["UMAP1", "UMAP2"]] = coordinates
    gates = primary_gate_table(manifest, ot_root)
    table = table.merge(
        gates,
        on=["patient_id", "sample_id_exact", "observation_id"],
        how="left", validate="many_to_one",
    )
    invalid = table["side"].eq("primary") & table["primary_gate"].isna()
    if invalid.any():
        raise RuntimeError(f"{invalid.sum()} post-QC primary cells lack an M4-E gate label")
    destination = args.output_root / dataset
    destination.mkdir(parents=True, exist_ok=False)
    table.to_csv(destination / "umap_coordinates_and_labels.csv.gz", index=False)
    pd.Series(hvg, name="gene").to_csv(destination / "umap_hvg.csv", index=False)
    subtype_counts = (
        table.groupby(["side", "author_subtype"], observed=True).size()
        .rename("cell_n").reset_index()
    )
    subtype_counts.to_csv(destination / "cell_subtype_counts.csv", index=False)
    make_figure(table, dataset, destination)
    make_primary_gate_context_figure(table, dataset, destination)
    primary = table.loc[table["side"].eq("primary")]
    report = {
        "dataset": dataset,
        "patient_n": int(table["patient_id"].nunique()),
        "primary_post_qc_malignant_cell_n": int(table["side"].eq("primary").sum()),
        "metastasis_post_qc_malignant_cell_n": int(table["side"].eq("metastasis").sum()),
        "author_subtype_n": int(table["author_subtype"].nunique()),
        "author_subtypes": sorted(table["author_subtype"].astype(str).unique()),
        "primary_gate_counts": primary["primary_gate"].value_counts().to_dict(),
        "embedding": "joint primary-plus-metastasis UMAP from 2000-HVG, 30-PC representation",
        "gate_interpretation": "M4-E retained/rejected is shown only for primary malignant cells",
    }
    (destination / "umap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replication_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--minimum-distance", type=float, default=0.25)
    parser.add_argument("--minimum-counts", type=int, default=1000)
    parser.add_argument("--minimum-features", type=int, default=500)
    parser.add_argument("--maximum-mitochondrial-percent", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    reports = [run_dataset(args, dataset) for dataset in DATASET_LABELS]
    (args.output_root / "umap_summary.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
