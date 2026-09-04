"""Plot one primary--metastasis malignant-cell pair in a shared UMAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


def shared_umap(pca: pd.DataFrame, *, neighbors: int, min_dist: float, seed: int):
    try:
        import umap
    except ImportError as error:
        raise RuntimeError("umap-learn>=0.5 is required") from error
    pc_columns = [column for column in pca if column.startswith("PC")]
    if len(pc_columns) < 2:
        raise RuntimeError("Joint PCA table contains fewer than two components")
    model = umap.UMAP(
        n_neighbors=min(neighbors, max(2, len(pca) - 1)),
        min_dist=min_dist, metric="euclidean", random_state=seed,
    )
    return model.fit_transform(pca[pc_columns].to_numpy(dtype=np.float32))


def save_panel(figure, output: Path, stem: str):
    figure.tight_layout()
    figure.savefig(output / f"{stem}.png", dpi=300)
    figure.savefig(output / f"{stem}.pdf")
    plt.close(figure)


def base_axis(axis, title: str):
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")


def origin_panel(axis, table: pd.DataFrame):
    styles = {
        "primary": ("#2474B5", "o"),
        "metastasis": ("#D64B40", "^"),
    }
    for side, (color, marker) in styles.items():
        use = table[table["side"].eq(side)]
        axis.scatter(use["UMAP1"], use["UMAP2"], s=7, c=color, marker=marker,
                     alpha=0.65, linewidths=0, label=side.title(), rasterized=True)
    axis.legend(frameon=False, markerscale=2)
    base_axis(axis, "Sample origin")


def mapping_panel(axis, table: pd.DataFrame, edges: pd.DataFrame):
    if len(edges):
        segments = np.stack([
            edges[["source_UMAP1", "source_UMAP2"]].to_numpy(),
            edges[["target_UMAP1", "target_UMAP2"]].to_numpy(),
        ], axis=1)
        mass = edges["transport_mass"].to_numpy(float)
        scaled = (mass - mass.min()) / max(mass.max() - mass.min(), 1e-12)
        collection = LineCollection(
            segments, colors="#4A4A4A", linewidths=0.25 + 1.5 * scaled,
            alpha=0.12 + 0.58 * scaled, rasterized=True,
        )
        axis.add_collection(collection)
    for side, color, marker in (
        ("primary", "#2474B5", "o"), ("metastasis", "#D64B40", "^")
    ):
        use = table[table["side"].eq(side)]
        axis.scatter(use["UMAP1"], use["UMAP2"], s=5, c=color, marker=marker,
                     alpha=0.55, linewidths=0, rasterized=True)
    base_axis(axis, f"Reciprocal one-to-one OT mappings (n={len(edges):,})")


def confidence_panel(axis, table: pd.DataFrame):
    points = axis.scatter(
        table["UMAP1"], table["UMAP2"],
        c=table["normalized_rejection_score"], cmap="coolwarm", vmin=0, vmax=1,
        s=7, alpha=0.75, linewidths=0, rasterized=True,
    )
    plt.colorbar(points, ax=axis, fraction=0.045, pad=0.02,
                 label="Rejection confidence")
    base_axis(axis, "Cell-level rejection confidence")


def gate_panel(axis, table: pd.DataFrame):
    table = table.copy()
    table["display_state"] = np.select(
        [
            table["side"].eq("primary") & table["retained"],
            table["side"].eq("primary") & ~table["retained"],
            table["side"].eq("metastasis") & table["retained"],
            table["side"].eq("metastasis") & ~table["retained"],
        ],
        ["Primary compatible", "Primary restricted",
         "Metastasis retained", "Metastasis rejected"],
    )
    styles = {
        "Primary compatible": ("#1B9E77", "o"),
        "Primary restricted": ("#7570B3", "o"),
        "Metastasis retained": ("#E6AB02", "^"),
        "Metastasis rejected": ("#D95F02", "^"),
    }
    for state, (color, marker) in styles.items():
        use = table[table["display_state"].eq(state)]
        axis.scatter(use["UMAP1"], use["UMAP2"], s=7, c=color, marker=marker,
                     alpha=0.7, linewidths=0, label=state, rasterized=True)
    axis.legend(frameon=False, fontsize=7, markerscale=1.7)
    base_axis(axis, "ConfidenceOT state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_result_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--minimum-distance", type=float, default=0.25)
    parser.add_argument("--maximum-displayed-edges", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pca = pd.read_csv(args.pair_result_dir / "joint_pca_coordinates.csv.gz")
    pca[["UMAP1", "UMAP2"]] = shared_umap(
        pca, neighbors=args.neighbors, min_dist=args.minimum_distance, seed=args.seed
    )
    confidence = pd.read_csv(args.pair_result_dir / "cell_confidence.csv")
    confidence = confidence[confidence["method"].eq(args.method)].copy()
    confidence["side"] = confidence["side"].map(
        {"source": "primary", "target": "metastasis"}
    )
    table = pca.merge(
        confidence[["side", "observation_id", "retained", "rejected",
                    "normalized_rejection_score"]],
        on=["side", "observation_id"], how="left", validate="one_to_one",
    )
    if table["retained"].isna().any():
        raise RuntimeError("Joint PCA and confidence tables do not identify the same cells")
    table["retained"] = table["retained"].astype(bool)

    edges = pd.read_csv(args.pair_result_dir / "reciprocal_cell_pairs.csv.gz")
    edges = edges[edges["method"].eq(args.method)].copy()
    source_lookup = table[table["side"].eq("primary")].set_index("observation_id")
    target_lookup = table[table["side"].eq("metastasis")].set_index("observation_id")
    for prefix, lookup, identifier in (
        ("source", source_lookup, "source_observation_id"),
        ("target", target_lookup, "target_observation_id"),
    ):
        edges[f"{prefix}_UMAP1"] = edges[identifier].map(lookup["UMAP1"])
        edges[f"{prefix}_UMAP2"] = edges[identifier].map(lookup["UMAP2"])
    if edges.filter(regex="_UMAP[12]$").isna().any().any():
        raise RuntimeError("Pairing edges contain cells absent from the joint UMAP")
    edges = edges.sort_values("transport_mass", ascending=False, kind="stable")
    displayed = edges.head(args.maximum_displayed_edges).copy()

    run = json.loads((args.pair_result_dir / "run.json").read_text(encoding="utf-8"))
    title = f"{run['patient_id']} · {run['pair_id']} · malignant cells only"
    figure, axes = plt.subplots(2, 2, figsize=(15, 12))
    origin_panel(axes[0, 0], table)
    mapping_panel(axes[0, 1], table, displayed)
    confidence_panel(axes[1, 0], table)
    gate_panel(axes[1, 1], table)
    figure.suptitle(title, fontsize=14)
    save_panel(figure, args.output_dir, "00_joint_umap_mapping_overview")

    for stem, function in (
        ("01_sample_origin", lambda axis: origin_panel(axis, table)),
        ("02_reciprocal_mapping", lambda axis: mapping_panel(axis, table, displayed)),
        ("03_rejection_confidence", lambda axis: confidence_panel(axis, table)),
        ("04_confidenceot_state", lambda axis: gate_panel(axis, table)),
    ):
        figure, axis = plt.subplots(figsize=(8, 7))
        function(axis)
        figure.suptitle(title, fontsize=11)
        save_panel(figure, args.output_dir, stem)
    table.to_csv(args.output_dir / "joint_umap_coordinates.csv.gz", index=False,
                 compression="gzip")
    displayed.to_csv(args.output_dir / "displayed_reciprocal_pairs.csv.gz", index=False,
                     compression="gzip")
    report = {
        "pair_id": run["pair_id"], "patient_id": run["patient_id"],
        "method": args.method, "primary_cell_n": int(table["side"].eq("primary").sum()),
        "metastatic_cell_n": int(table["side"].eq("metastasis").sum()),
        "reciprocal_pair_n": len(edges), "displayed_pair_n": len(displayed),
        "embedding": "joint pair-specific UMAP from saved joint PCA coordinates",
    }
    (args.output_dir / "mapping_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
