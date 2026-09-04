"""Build an expression UMAP and overlay four-state ConfidenceOT cell scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


STATUS_ORDER = ["retained", "rejected", "site_or_cap_discordant"]
STATUS_LABEL = {
    "retained": "Retained",
    "rejected": "Rejected",
    "site_or_cap_discordant": "Discordant",
}
STATUS_COLOR = {
    "retained": "#2b8cbe",
    "rejected": "#d7301f",
    "site_or_cap_discordant": "#969696",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("four_state_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--malignant-annotation", default="Ovarian.cancer.cell")
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=40)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--maximum-cells", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_classification(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("patients/*/four_state_cell_classification.csv.gz")):
        table = pd.read_csv(path)
        table["patient_id"] = path.parent.name.split("_", 1)[-1]
        rows.append(table)
    if not rows:
        raise FileNotFoundError(f"No four-state classifications under {root}")
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["side", "observation_id"]).any():
        raise RuntimeError("Four-state classification contains duplicate side/cell records")
    return result


def annotation_column(data: ad.AnnData) -> str:
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            return column
    raise KeyError("H5AD has no supported cell annotation column")


def selected_h5ad_records(manifest: pd.DataFrame, patients: set[str]) -> pd.DataFrame:
    table = manifest.loc[manifest["patient_id"].astype(str).isin(patients)]
    rows = []
    for side, column in (("primary", "source_h5ad"), ("metastasis", "target_h5ad")):
        use = table[["patient_id", column]].dropna().drop_duplicates().rename(
            columns={column: "h5ad_path"}
        )
        use["side"] = side
        rows.append(use)
    result = pd.concat(rows, ignore_index=True).drop_duplicates()
    conflicts = result.groupby("h5ad_path").agg(
        side_n=("side", "nunique"), patient_n=("patient_id", "nunique")
    )
    if ((conflicts["side_n"] > 1) | (conflicts["patient_n"] > 1)).any():
        raise RuntimeError("An H5AD path maps to multiple patients or biological sides")
    return result.sort_values(["patient_id", "side", "h5ad_path"], kind="stable")


def load_expression(
    records: pd.DataFrame, classification: pd.DataFrame, malignant_annotation: str
) -> ad.AnnData:
    required = set(classification["observation_id"].astype(str))
    found: set[str] = set()
    pieces = []
    for record in records.itertuples(index=False):
        path = Path(record.h5ad_path)
        data = ad.read_h5ad(path)
        data.obs_names = data.obs_names.astype(str)
        column = annotation_column(data)
        data.obs["cell_type_annotation"] = data.obs[column].astype(str).to_numpy()
        data.obs["confidence_side"] = str(record.side)
        data.obs["confidence_patient_id"] = str(record.patient_id)
        classified = data.obs_names.isin(required)
        if np.any(classified):
            observed = data.obs.loc[classified, "cell_type_annotation"]
            invalid = ~observed.eq(malignant_annotation)
            if invalid.any():
                labels = observed.loc[invalid].value_counts().to_dict()
                raise ValueError(f"Selected cells are not {malignant_annotation}: {labels}")
            found.update(data.obs_names[classified])
        pieces.append(data)
    missing = required - found
    if missing:
        preview = sorted(missing)[:5]
        raise KeyError(f"{len(missing)} classified cell IDs were absent from H5AD files: {preview}")
    if not pieces:
        raise RuntimeError("No classified cells were loaded")
    return ad.concat(pieces, join="inner", merge="same", index_unique=None)


def deterministic_subsample(
    data: ad.AnnData, classification: pd.DataFrame, maximum: int, seed: int
) -> tuple[ad.AnnData, pd.DataFrame]:
    if len(data) <= maximum:
        return data, classification
    rng = np.random.default_rng(seed)
    classified_ids = set(classification["observation_id"].astype(str))
    mandatory = np.flatnonzero(data.obs_names.astype(str).isin(classified_ids))
    if len(mandatory) > maximum:
        raise ValueError(
            f"maximum-cells={maximum} is below the {len(mandatory)} classified malignant cells"
        )
    background = np.flatnonzero(~data.obs_names.astype(str).isin(classified_ids))
    background_n = maximum - len(mandatory)
    sampled_background = rng.choice(background, background_n, replace=False)
    chosen = np.sort(np.concatenate([mandatory, sampled_background]))
    keep_ids = set(data.obs_names[chosen].astype(str))
    classification = classification.loc[
        classification["observation_id"].astype(str).isin(keep_ids)
    ].copy()
    return data[chosen].copy(), classification


def compute_umap(data: ad.AnnData, args: argparse.Namespace) -> None:
    try:
        import scanpy as sc
    except ImportError as error:
        raise RuntimeError("scanpy>=1.10,<2 is required") from error
    data.X = sparse.csr_matrix(data.X, dtype=np.float32)
    if data.X.data.size and data.X.data.min() < 0:
        raise ValueError("Expression matrix contains negative values")
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, n_top_genes=args.n_hvg, flavor="seurat", subset=True)
    sc.pp.scale(data, max_value=10)
    n_pcs = min(args.n_pcs, data.n_vars - 1, data.n_obs - 1)
    sc.tl.pca(data, n_comps=n_pcs, svd_solver="arpack", random_state=args.seed)
    sc.pp.neighbors(data, n_neighbors=args.n_neighbors, n_pcs=n_pcs, random_state=args.seed)
    sc.tl.umap(data, random_state=args.seed)


def coordinate_table(data: ad.AnnData, classification: pd.DataFrame) -> pd.DataFrame:
    coordinates = pd.DataFrame({
        "observation_id": data.obs_names.astype(str),
        "side": data.obs["confidence_side"].astype(str).to_numpy(),
        "patient_id": data.obs["confidence_patient_id"].astype(str).to_numpy(),
        "cell_type_annotation": data.obs["cell_type_annotation"].astype(str).to_numpy(),
        "umap_1": data.obsm["X_umap"][:, 0],
        "umap_2": data.obsm["X_umap"][:, 1],
    })
    annotation = classification.drop(columns="patient_id").copy()
    return coordinates.merge(
        annotation, on=["side", "observation_id"], how="left", validate="one_to_one"
    )


def score_axis(ax, table: pd.DataFrame, side: str) -> None:
    all_side = table.loc[table["side"].eq(side)]
    use = all_side.loc[all_side["mean_baseline_rejection_score"].notna()].sort_values(
        "mean_baseline_rejection_score"
    )
    background = all_side.loc[all_side["mean_baseline_rejection_score"].isna()]
    ax.scatter(
        background["umap_1"], background["umap_2"], color="#d9d9d9",
        s=1.3, alpha=0.28, linewidth=0, rasterized=True,
    )
    points = ax.scatter(
        use["umap_1"], use["umap_2"], c=use["mean_baseline_rejection_score"],
        cmap="magma", vmin=0, vmax=1, s=2.2, linewidth=0, rasterized=True,
    )
    ax.set_title(side.capitalize())
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    return points


def status_axis(ax, table: pd.DataFrame, side: str) -> None:
    use = table.loc[table["side"].eq(side)]
    background = use.loc[use["consensus_status"].isna()]
    ax.scatter(
        background["umap_1"], background["umap_2"], color="#d9d9d9",
        s=1.3, alpha=0.28, linewidth=0, label="Other annotated cells", rasterized=True,
    )
    for status in STATUS_ORDER:
        subset = use.loc[use["consensus_status"].eq(status)]
        ax.scatter(
            subset["umap_1"], subset["umap_2"], s=2.2, linewidth=0,
            color=STATUS_COLOR[status], alpha=0.62, label=STATUS_LABEL[status],
            rasterized=True,
        )
    ax.set_title(side.capitalize())
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, markerscale=3, fontsize=8)


def cell_type_axis(ax, table: pd.DataFrame, side: str, color_lookup: dict[str, object]) -> None:
    use = table.loc[table["side"].eq(side)]
    for annotation in sorted(use["cell_type_annotation"].unique()):
        subset = use.loc[use["cell_type_annotation"].eq(annotation)]
        ax.scatter(
            subset["umap_1"], subset["umap_2"], s=1.8, linewidth=0,
            color=color_lookup[annotation], alpha=0.55, label=annotation, rasterized=True,
        )
    ax.set_title(side.capitalize())
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])


def make_figures(table: pd.DataFrame, output: Path) -> None:
    annotations = sorted(table["cell_type_annotation"].unique())
    palette = plt.get_cmap("tab20", max(1, len(annotations)))
    color_lookup = {value: palette(index) for index, value in enumerate(annotations)}
    for side in ("primary", "metastasis"):
        fig, ax = plt.subplots(figsize=(7.8, 5.5))
        cell_type_axis(ax, table, side, color_lookup)
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left",
                  markerscale=4, fontsize=7)
        fig.tight_layout()
        save(fig, output, f"09_{side}_author_cell_type_umap")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, side in zip(axes, ("primary", "metastasis")):
        cell_type_axis(ax, table, side, color_lookup)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, bbox_to_anchor=(0.995, 0.5),
               loc="center left", markerscale=4, fontsize=7)
    fig.suptitle("All-cell expression UMAP colored by author annotation", fontsize=15)
    fig.subplots_adjust(right=0.83, top=0.88, wspace=0.12)
    save(fig, output, "09_author_cell_type_umap_combined")

    for side in ("primary", "metastasis"):
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        points = score_axis(ax, table, side)
        colorbar = fig.colorbar(points, ax=ax, fraction=0.04, pad=0.02)
        colorbar.set_label("Mean ConfidenceOT rejection score")
        fig.tight_layout()
        save(fig, output, f"10_{side}_confidence_umap")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, side in zip(axes, ("primary", "metastasis")):
        points = score_axis(ax, table, side)
    colorbar = fig.colorbar(points, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Mean ConfidenceOT rejection score")
    fig.suptitle("Expression UMAP with cell-level ConfidenceOT scores", fontsize=15)
    save(fig, output, "10_confidence_umap_combined")

    for side in ("primary", "metastasis"):
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        status_axis(ax, table, side)
        fig.tight_layout()
        save(fig, output, f"11_{side}_consensus_state_umap")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, side in zip(axes, ("primary", "metastasis")):
        status_axis(ax, table, side)
    fig.suptitle("Expression UMAP with ConfidenceOT consensus states", fontsize=15)
    fig.tight_layout()
    save(fig, output, "11_consensus_state_umap_combined")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    classification = read_classification(args.four_state_root)
    manifest = pd.read_csv(args.manifest_csv)
    records = selected_h5ad_records(manifest, set(classification["patient_id"].astype(str)))
    data = load_expression(records, classification, args.malignant_annotation)
    data, classification = deterministic_subsample(
        data, classification, args.maximum_cells, args.seed
    )
    compute_umap(data, args)
    table = coordinate_table(data, classification)
    table.to_csv(args.output_root / "cell_confidence_umap_coordinates.csv.gz", index=False)
    make_figures(table, args.output_root)
    print({
        "cell_n": len(table),
        "classified_malignant_cell_n": int(table["consensus_status"].notna().sum()),
        "patient_n": table["patient_id"].nunique(),
        "h5ad_file_n": len(records),
        "author_cell_type_n": table["cell_type_annotation"].nunique(),
        "annotation": args.malignant_annotation,
        "n_hvg": args.n_hvg,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
    })
    print(f"Figures: {args.output_root}")


if __name__ == "__main__":
    main()
