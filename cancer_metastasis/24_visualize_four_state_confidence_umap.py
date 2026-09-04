"""Build an expression UMAP and overlay four-state ConfidenceOT cell scores."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from common import load_exact_side


STATUS_ORDER = ["rejected", "site_or_cap_discordant", "retained"]
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
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--sensitivity-root", type=Path)
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
    keys = ["side", "patient_id", "sample", "observation_id"]
    if result.duplicated(keys).any():
        raise RuntimeError("Four-state classification contains duplicate biological cell records")
    return result


def annotation_column(data: ad.AnnData) -> str:
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            return column
    raise KeyError("H5AD has no supported cell annotation column")


def selected_h5ad_records(manifest: pd.DataFrame, patients: set[str]) -> pd.DataFrame:
    table = manifest.loc[manifest["patient_id"].astype(str).isin(patients)]
    rows = []
    for side, sample_column, paths_column in (
        ("primary", "source_sample", "source_h5ads_json"),
        ("metastasis", "target_sample", "target_h5ads_json"),
    ):
        use = table[["patient_id", sample_column, paths_column]].dropna().drop_duplicates().rename(
            columns={sample_column: "sample", paths_column: "h5ads_json"}
        )
        use["side"] = side
        rows.append(use)
    result = pd.concat(rows, ignore_index=True).drop_duplicates()
    return result.sort_values(["patient_id", "side", "sample"], kind="stable")


def load_expression(
    records: pd.DataFrame, classification: pd.DataFrame, malignant_annotation: str
) -> ad.AnnData:
    required = set(zip(
        classification["patient_id"].astype(str),
        classification["sample"].astype(str),
        classification["observation_id"].astype(str),
    ))
    found: set[tuple[str, str, str]] = set()
    pieces = []
    for record in records.itertuples(index=False):
        paths = json.loads(str(record.h5ads_json))
        data = load_exact_side(paths, str(record.sample))
        data.obs_names = data.obs_names.astype(str)
        column = annotation_column(data)
        data.obs["cell_type_annotation"] = data.obs[column].astype(str).to_numpy()
        data.obs["confidence_side"] = str(record.side)
        data.obs["confidence_patient_id"] = str(record.patient_id)
        data.obs["confidence_sample"] = str(record.sample)
        required_ids = classification.loc[
            classification["patient_id"].astype(str).eq(str(record.patient_id))
            & classification["sample"].astype(str).eq(str(record.sample)),
            "observation_id",
        ].astype(str)
        classified = data.obs_names.isin(set(required_ids))
        if np.any(classified):
            observed = data.obs.loc[classified, "cell_type_annotation"]
            invalid = ~observed.eq(malignant_annotation)
            if invalid.any():
                labels = observed.loc[invalid].value_counts().to_dict()
                raise ValueError(f"Selected cells are not {malignant_annotation}: {labels}")
            found.update(
                (str(record.patient_id), str(record.sample), str(value))
                for value in data.obs_names[classified]
            )
        pieces.append(data)
    missing = required - found
    if missing:
        preview = sorted(missing)[:5]
        raise KeyError(f"{len(missing)} classified cell IDs were absent from H5AD files: {preview}")
    if not pieces:
        raise RuntimeError("No classified cells were loaded")
    return ad.concat(pieces, join="inner", merge="same", index_unique=None)


def deterministic_subsample(
    data: ad.AnnData,
    classification: pd.DataFrame,
    maximum: int,
    seed: int,
    malignant_annotation: str,
) -> tuple[ad.AnnData, pd.DataFrame]:
    if len(data) <= maximum:
        return data, classification
    rng = np.random.default_rng(seed)
    classified_ids = set(classification["observation_id"].astype(str))
    mandatory_mask = data.obs_names.astype(str).isin(classified_ids) | data.obs[
        "cell_type_annotation"
    ].astype(str).eq(malignant_annotation).to_numpy()
    mandatory = np.flatnonzero(mandatory_mask)
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
        "sample": data.obs["confidence_sample"].astype(str).to_numpy(),
        "cell_type_annotation": data.obs["cell_type_annotation"].astype(str).to_numpy(),
        "umap_1": data.obsm["X_umap"][:, 0],
        "umap_2": data.obsm["X_umap"][:, 1],
    })
    return coordinates.merge(
        classification,
        on=["side", "patient_id", "sample", "observation_id"],
        how="left",
        validate="one_to_one",
    )


def score_axis(ax, table: pd.DataFrame, side: str, title: str | None = None) -> None:
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
        cmap="RdBu_r", vmin=0, vmax=1, s=2.2, linewidth=0, rasterized=True,
    )
    ax.set_title(title or side.capitalize())
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    return points


def status_axis(ax, table: pd.DataFrame, side: str, title: str | None = None) -> None:
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
    ax.set_title(title or side.capitalize())
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, markerscale=3, fontsize=8)


def cell_type_axis(
    ax, table: pd.DataFrame, side: str, color_lookup: dict[str, object], title: str | None = None
) -> None:
    use = table.loc[table["side"].eq(side)]
    for annotation in sorted(use["cell_type_annotation"].unique()):
        subset = use.loc[use["cell_type_annotation"].eq(annotation)]
        ax.scatter(
            subset["umap_1"], subset["umap_2"], s=1.8, linewidth=0,
            color=color_lookup[annotation], alpha=0.55, label=annotation, rasterized=True,
        )
    ax.set_title(title or side.capitalize())
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])


def make_overview_figures(table: pd.DataFrame, output: Path, malignant_annotation: str) -> None:
    annotations = sorted(table["cell_type_annotation"].unique())
    palette = plt.get_cmap("tab20", max(1, len(annotations)))
    color_lookup = {value: palette(index) for index, value in enumerate(annotations)}
    color_lookup[malignant_annotation] = "#542788"
    for side in ("primary", "metastasis"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
        cell_type_axis(axes[0], table, side, color_lookup, "Author annotation")
        axes[0].legend(
            frameon=False, bbox_to_anchor=(0.5, -0.04), loc="upper center",
            ncol=3, markerscale=4, fontsize=6,
        )
        points = score_axis(axes[1], table, side, "Cell-level rejection confidence")
        colorbar = fig.colorbar(points, ax=axes[1], fraction=0.045, pad=0.02)
        colorbar.set_label("Mean ConfidenceOT rejection score")
        status_axis(axes[2], table, side, "Cross-pair/cap consensus state")
        fig.suptitle(f"{side.capitalize()} cells only", fontsize=16)
        fig.subplots_adjust(top=0.86, bottom=0.2, wspace=0.2)
        save(fig, output, f"09_{side}_overall_umap")


def one_result_file(root: Path, pair_id: str) -> Path:
    matches = sorted((root / pair_id).glob("scope_malignant/*/cell_confidence.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one cell_confidence.csv for {pair_id} under {root}; found {len(matches)}"
        )
    return matches[0]


def read_pair_gate(root: Path, pair_id: str, side: str, prefix: str) -> pd.DataFrame:
    table = pd.read_csv(
        one_result_file(root, pair_id),
        usecols=["method", "side", "observation_id", "rejected", "normalized_rejection_score"],
    )
    table = table.loc[table["method"].eq("M4-E") & table["side"].eq(side)].copy()
    return table[["observation_id", "rejected", "normalized_rejection_score"]].rename(
        columns={
            "rejected": f"{prefix}_rejected",
            "normalized_rejection_score": f"{prefix}_rejection_score",
        }
    )


def all_pair_consensus(
    manifest: pd.DataFrame, baseline_root: Path, sensitivity_root: Path
) -> pd.DataFrame:
    rows = []
    for pair in manifest.itertuples(index=False):
        for plot_side, result_side, sample in (
            ("primary", "source", str(pair.source_sample)),
            ("metastasis", "target", str(pair.target_sample)),
        ):
            baseline = read_pair_gate(baseline_root, str(pair.pair_id), result_side, "baseline")
            sensitivity = read_pair_gate(
                sensitivity_root, str(pair.pair_id), result_side, "sensitivity"
            )
            gate = baseline.merge(sensitivity, on="observation_id", validate="one_to_one")
            gate["pair_status"] = np.select(
                [
                    gate["baseline_rejected"] & gate["sensitivity_rejected"],
                    ~gate["baseline_rejected"] & ~gate["sensitivity_rejected"],
                ],
                ["rejected", "retained"],
                default="site_or_cap_discordant",
            )
            gate["side"] = plot_side
            gate["patient_id"] = str(pair.patient_id)
            gate["sample"] = sample
            rows.append(gate)
    long = pd.concat(rows, ignore_index=True)
    result = []
    keys = ["side", "patient_id", "sample", "observation_id"]
    for values, table in long.groupby(keys, sort=False):
        statuses = set(table["pair_status"])
        if statuses == {"retained"}:
            status = "retained"
        elif statuses == {"rejected"}:
            status = "rejected"
        else:
            status = "site_or_cap_discordant"
        record = dict(zip(keys, values))
        record.update({
            "consensus_status": status,
            "observed_pair_n": int(len(table)),
            "expected_pair_n": int(len(table)),
            "mean_baseline_rejection_score": float(
                table["baseline_rejection_score"].mean()
            ),
        })
        result.append(record)
    return pd.DataFrame(result)


def pair_cell_table(
    coordinates: pd.DataFrame,
    row: pd.Series,
    baseline_root: Path,
    sensitivity_root: Path,
) -> pd.DataFrame:
    pieces = []
    for plot_side, result_side, sample_column in (
        ("primary", "source", "source_sample"),
        ("metastasis", "target", "target_sample"),
    ):
        sample = str(row[sample_column])
        use = coordinates.loc[
            coordinates["side"].eq(plot_side)
            & coordinates["patient_id"].astype(str).eq(str(row["patient_id"]))
            & coordinates["sample"].astype(str).eq(sample)
        ].copy()
        baseline = read_pair_gate(baseline_root, str(row["pair_id"]), result_side, "baseline")
        sensitivity = read_pair_gate(
            sensitivity_root, str(row["pair_id"]), result_side, "sensitivity"
        )
        gate = baseline.merge(sensitivity, on="observation_id", validate="one_to_one")
        gate["consensus_status"] = np.select(
            [
                gate["baseline_rejected"] & gate["sensitivity_rejected"],
                ~gate["baseline_rejected"] & ~gate["sensitivity_rejected"],
            ],
            ["rejected", "retained"],
            default="site_or_cap_discordant",
        )
        gate["mean_baseline_rejection_score"] = gate["baseline_rejection_score"]
        use = use.drop(
            columns=["consensus_status", "mean_baseline_rejection_score"], errors="ignore"
        ).merge(
            gate[["observation_id", "consensus_status", "mean_baseline_rejection_score"]],
            on="observation_id", how="left", validate="one_to_one",
        )
        pieces.append(use)
    return pd.concat(pieces, ignore_index=True)


def save_png(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_pair_figures(
    coordinates: pd.DataFrame,
    manifest: pd.DataFrame,
    baseline_root: Path,
    sensitivity_root: Path,
    output: Path,
) -> int:
    pair_root = output / "patient_pair_umaps"
    pair_root.mkdir(parents=True, exist_ok=True)
    pairs = manifest.drop_duplicates("pair_id").copy()
    for index, row in pairs.reset_index(drop=True).iterrows():
        pair_id = str(row["pair_id"])
        use = pair_cell_table(coordinates, row, baseline_root, sensitivity_root)
        destination = pair_root / f"{index:03d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', pair_id)}"
        destination.mkdir(parents=True, exist_ok=False)
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for row_index, side in enumerate(("primary", "metastasis")):
            score_axis(axes[row_index, 0], use, side, f"{side.capitalize()} confidence")
            status_axis(axes[row_index, 1], use, side, f"{side.capitalize()} pair-robust state")
        fig.suptitle(
            f"{row['patient_id']}: {row['source_sample']} → {row['target_sample']}", fontsize=14
        )
        fig.tight_layout()
        save_png(fig, destination / "pair_confidence_and_state.png")
        for side in ("primary", "metastasis"):
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            points = score_axis(ax, use, side, f"{side.capitalize()}: {row['source_sample'] if side == 'primary' else row['target_sample']}")
            colorbar = fig.colorbar(points, ax=ax, fraction=0.04, pad=0.02)
            colorbar.set_label("M4-E rejection score")
            fig.tight_layout()
            save_png(fig, destination / f"{side}_confidence.png")
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            status_axis(ax, use, side, f"{side.capitalize()}: {row['source_sample'] if side == 'primary' else row['target_sample']}")
            fig.tight_layout()
            save_png(fig, destination / f"{side}_consensus_state.png")
        use.to_csv(destination / "pair_umap_coordinates_and_scores.csv.gz", index=False)
    pairs.to_csv(pair_root / "pair_figure_manifest.csv", index=False)
    return len(pairs)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = pd.read_csv(args.manifest_csv)
    if (args.baseline_root is None) != (args.sensitivity_root is None):
        raise ValueError("--baseline-root and --sensitivity-root must be supplied together")
    if args.baseline_root is not None:
        classification = all_pair_consensus(
            manifest, args.baseline_root, args.sensitivity_root
        )
    else:
        classification = read_classification(args.four_state_root)
    records = selected_h5ad_records(manifest, set(classification["patient_id"].astype(str)))
    coordinate_tables = []
    for side in ("primary", "metastasis"):
        side_records = records.loc[records["side"].eq(side)].copy()
        side_classification = classification.loc[classification["side"].eq(side)].copy()
        data = load_expression(side_records, side_classification, args.malignant_annotation)
        data, side_classification = deterministic_subsample(
            data,
            side_classification,
            args.maximum_cells,
            args.seed,
            args.malignant_annotation,
        )
        compute_umap(data, args)
        coordinate_tables.append(coordinate_table(data, side_classification))
    table = pd.concat(coordinate_tables, ignore_index=True)
    table.to_csv(args.output_root / "cell_confidence_umap_coordinates.csv.gz", index=False)
    make_overview_figures(table, args.output_root, args.malignant_annotation)
    pair_n = 0
    if args.baseline_root is not None:
        pair_n = make_pair_figures(
            table,
            manifest,
            args.baseline_root,
            args.sensitivity_root,
            args.output_root,
        )
    print({
        "cell_n": len(table),
        "classified_malignant_cell_n": int(table["consensus_status"].notna().sum()),
        "patient_n": table["patient_id"].nunique(),
        "exact_sample_record_n": len(records),
        "author_cell_type_n": table["cell_type_annotation"].nunique(),
        "annotation": args.malignant_annotation,
        "primary_and_metastasis_embedded_separately": True,
        "patient_pair_figure_n": pair_n,
        "n_hvg": args.n_hvg,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
    })
    print(f"Figures: {args.output_root}")


if __name__ == "__main__":
    main()
