"""Test whether the transport geometry pairs cells by sequencing quality.

The gate diagnostic in ``25_diagnose_gate_covariates.py`` shows that retained
primary cells are the deeply sequenced ones.  That leaves a separate question
open: within the joint representation, are low-quality primary cells paired
with low-quality metastatic cells, or are they simply far from everything?

``spearman_nn_partner_depth`` answers the pairing question directly: a
positive value means shallow primary cells sit beside shallow metastatic
cells, so the transported mass structure is quality-stratified even where the
gate is not.  It does **not** identify the mechanism.  Synthetic fixtures show
it is positive under both candidate mechanisms, because a tight core plus a
dispersed shell is itself depth-graded: 0.51 for pure depth-graded dispersion
and 0.68 for a shared depth axis.

Two other statistics separate the mechanisms, and they matter because the
remedies differ.

* A shared depth axis places shallow cells on both sides at the same end of
  one component.  ``max_abs_spearman_pc_depth`` is then high (0.86 on the
  fixture) while ``spearman_depth_nn_distance`` stays weak (-0.17), because
  shallow cells still have close neighbours.  Regressing out that one
  component is enough.
* Independent dropout noise adds variance rather than a shift, so squared
  distances between two noisy cells grow by both noise terms.  Depth then
  loads on no single component (0.09 on the fixture) yet strongly predicts
  distance to the closest metastatic cell (-0.80).  No component can be
  removed; the normalization itself has to change.

The script reports all of them, over all primary cells and over rejected
primary cells alone.

Inputs are the stored ``joint_pca_coordinates.csv.gz`` (written only when the
OT run used ``--save-pairing-edges``), ``cell_confidence.csv`` for per-cell
depth and gate state, and the optional ``reciprocal_cell_pairs.csv.gz`` for the
transport plan's dominant partner edges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


SIDE_FROM_PANEL = {"primary": "source", "metastasis": "target"}
LEADING_PC_N = 5


def dataset_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--dataset expects LABEL=PATH, found {value!r}"
        )
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("--dataset label must be non-empty")
    return label.strip(), Path(path.strip())


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return Spearman rho, or NaN when either side is short or constant."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 10 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def nearest_partner(
    source: np.ndarray, target: np.ndarray, chunk: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return each source row's nearest target index and squared distance.

    Distances are computed in chunks so a 10,000 by 10,000 pair never
    materialises a full matrix.
    """
    target_norm = np.einsum("ij,ij->i", target, target)
    indices = np.empty(source.shape[0], dtype=np.int64)
    distances = np.empty(source.shape[0], dtype=np.float64)
    for start in range(0, source.shape[0], chunk):
        block = source[start : start + chunk]
        squared = (
            np.einsum("ij,ij->i", block, block)[:, None]
            + target_norm[None, :]
            - 2.0 * block @ target.T
        )
        np.maximum(squared, 0.0, out=squared)
        nearest = np.argmin(squared, axis=1)
        indices[start : start + chunk] = nearest
        distances[start : start + chunk] = squared[
            np.arange(block.shape[0]), nearest
        ]
    return indices, distances


def load_panels(
    result_dir: Path, depth_column: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    coordinates = pd.read_csv(result_dir / "joint_pca_coordinates.csv.gz")
    confidence = pd.read_csv(result_dir / "cell_confidence.csv")
    confidence = confidence.loc[confidence["method"].eq("M4-E")]
    keep = ["side", "observation_id", "retained", depth_column]
    missing = [column for column in keep if column not in confidence]
    if missing:
        raise RuntimeError(f"{result_dir}: cell_confidence.csv lacks {missing}")
    confidence = confidence[keep].copy()
    confidence["observation_id"] = confidence["observation_id"].astype(str)
    coordinates["observation_id"] = coordinates["observation_id"].astype(str)
    coordinates["side"] = coordinates["side"].map(SIDE_FROM_PANEL)
    if coordinates["side"].isna().any():
        raise RuntimeError(f"{result_dir}: unexpected side label in PCA coordinates")
    merged = coordinates.merge(
        confidence, on=["side", "observation_id"], how="left", validate="one_to_one"
    )
    if merged[depth_column].isna().any():
        raise RuntimeError(f"{result_dir}: PCA cells missing from cell_confidence")
    pcs = [column for column in coordinates.columns if column.startswith("PC")]
    pcs.sort(key=lambda name: int(name[2:]))
    primary = merged.loc[merged["side"].eq("source")].reset_index(drop=True)
    metastasis = merged.loc[merged["side"].eq("target")].reset_index(drop=True)
    return primary, metastasis, pcs


def reciprocal_edge_correlation(
    result_dir: Path, primary: pd.DataFrame, metastasis: pd.DataFrame,
    depth_column: str,
) -> tuple[float, int]:
    path = result_dir / "reciprocal_cell_pairs.csv.gz"
    if not path.is_file():
        return float("nan"), 0
    edges = pd.read_csv(path)
    edges = edges.loc[edges["method"].eq("M4-E")]
    if edges.empty:
        return float("nan"), 0
    source_depth = primary.set_index("observation_id")[depth_column]
    target_depth = metastasis.set_index("observation_id")[depth_column]
    left = source_depth.reindex(edges["source_observation_id"].astype(str)).to_numpy()
    right = target_depth.reindex(edges["target_observation_id"].astype(str)).to_numpy()
    return rank_correlation(left, right), int(len(edges))


def pair_record(
    result_dir: Path, dataset: str, depth_column: str, chunk: int
) -> dict[str, object]:
    primary, metastasis, pcs = load_panels(result_dir, depth_column)
    source_pcs = primary[pcs].to_numpy(dtype=np.float64)
    target_pcs = metastasis[pcs].to_numpy(dtype=np.float64)
    depth = primary[depth_column].to_numpy(dtype=np.float64)
    target_depth = metastasis[depth_column].to_numpy(dtype=np.float64)
    retained = primary["retained"].astype(bool).to_numpy()

    indices, distances = nearest_partner(source_pcs, target_pcs, chunk)
    partner_depth = target_depth[indices]
    edge_rho, edge_n = reciprocal_edge_correlation(
        result_dir, primary, metastasis, depth_column
    )
    record: dict[str, object] = {
        "dataset": dataset,
        "pair_id": result_dir.parents[1].name,
        "budget_tag": result_dir.name,
        "primary_n": int(primary.shape[0]),
        "metastasis_n": int(metastasis.shape[0]),
        "pc_n": len(pcs),
        "retained_fraction": float(retained.mean()),
        # Does a shallow primary cell sit next to a shallow metastatic cell?
        "spearman_nn_partner_depth": rank_correlation(depth, partner_depth),
        "spearman_nn_partner_depth_rejected": rank_correlation(
            depth[~retained], partner_depth[~retained]
        ),
        "spearman_nn_partner_depth_retained": rank_correlation(
            depth[retained], partner_depth[retained]
        ),
        # Is a shallow primary cell far from even its closest metastatic cell?
        "spearman_depth_nn_distance": rank_correlation(depth, distances),
        "median_nn_distance_retained": float(np.median(distances[retained]))
        if retained.any() else float("nan"),
        "median_nn_distance_rejected": float(np.median(distances[~retained]))
        if (~retained).any() else float("nan"),
        "reciprocal_edge_n": edge_n,
        "spearman_reciprocal_edge_depth": edge_rho,
    }
    # One strongly depth-correlated component can be regressed out; depth
    # spread thinly across many components cannot.  Principal component signs
    # are arbitrary and flip between pairs, so the per-component correlations
    # below are only readable through max_abs_spearman_pc_depth.
    absolute = []
    for index, column in enumerate(pcs, start=1):
        rho = rank_correlation(primary[column].to_numpy(dtype=np.float64), depth)
        absolute.append(abs(rho) if np.isfinite(rho) else 0.0)
        if index <= LEADING_PC_N:
            record[f"spearman_pc{index}_depth"] = rho
    record["max_abs_spearman_pc_depth"] = float(max(absolute)) if absolute else float("nan")
    record["argmax_pc_depth"] = int(np.argmax(absolute) + 1) if absolute else 0
    # Rank-regressing depth out of every component at once bounds how much of
    # the embedding is depth.
    ranked = np.column_stack([
        np.ones(depth.size), rankdata(depth)
    ])
    explained = []
    for column in pcs:
        response = rankdata(primary[column].to_numpy(dtype=np.float64))
        coefficients, *_ = np.linalg.lstsq(ranked, response, rcond=None)
        residual = response - ranked @ coefficients
        total = float(np.var(response))
        explained.append(
            1.0 - float(np.var(residual)) / total if total > 0 else 0.0
        )
    record["mean_pc_rank_variance_from_depth"] = float(np.mean(explained))
    return record


def summarize(pairs: pd.DataFrame) -> pd.DataFrame:
    skip = {"dataset", "pair_id", "budget_tag", "argmax_pc_depth"}
    columns = [
        column for column in pairs.columns
        if column not in skip
        and pd.api.types.is_numeric_dtype(pairs[column])
        and pairs[column].notna().any()
    ]
    rows = []
    for dataset, table in pairs.groupby("dataset", sort=True):
        for column in columns:
            values = pd.to_numeric(table[column], errors="coerce").dropna()
            rows.append({
                "dataset": dataset,
                "statistic": column,
                "pair_n": int(len(values)),
                "median": float(values.median()) if len(values) else float("nan"),
                "q25": float(values.quantile(0.25)) if len(values) else float("nan"),
                "q75": float(values.quantile(0.75)) if len(values) else float("nan"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--dataset", type=dataset_argument, action="append", required=True,
        metavar="LABEL=OT_ROOT",
    )
    parser.add_argument("--scope", default="scope_malignant")
    parser.add_argument("--budget-tag", default=None)
    parser.add_argument(
        "--depth-column", default="total_counts",
        choices=("total_counts", "n_genes_by_counts"),
    )
    parser.add_argument(
        "--distance-chunk", type=int, default=1000,
        help="Source rows per distance block; lower it to reduce peak memory",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    records = []
    inventory = []
    for dataset, root in args.dataset:
        paths = sorted(
            root.glob(f"*/{args.scope}/{args.budget_tag or '*'}/joint_pca_coordinates.csv.gz")
        )
        inventory.append({
            "dataset": dataset, "ot_root": str(root), "pca_file_n": len(paths),
        })
        if not paths:
            raise RuntimeError(
                f"{dataset}: no joint_pca_coordinates.csv.gz under {root}. "
                "Only OT runs launched with --save-pairing-edges store them."
            )
        for path in paths:
            records.append(
                pair_record(
                    path.parent, dataset, args.depth_column, args.distance_chunk
                )
            )

    pairs = pd.DataFrame(records).sort_values(["dataset", "pair_id"], kind="stable")
    summary = summarize(pairs)
    pairs.to_csv(args.output_root / "pairing_quality_pair_statistics.csv", index=False)
    summary.to_csv(args.output_root / "pairing_quality_dataset_summary.csv", index=False)
    report = {
        "depth_column": args.depth_column,
        "scope": args.scope,
        "budget_tag": args.budget_tag or "any_completed",
        "inventory": inventory,
        "pair_n": int(len(pairs)),
        "interpretation": {
            "spearman_nn_partner_depth": (
                "Positive means shallow primary cells sit next to shallow "
                "metastatic cells, so the transport plan is quality-stratified. "
                "It does not identify the mechanism: fixtures give 0.51 for "
                "depth-graded dispersion and 0.68 for a shared depth axis."
            ),
            "mechanism_discriminator": (
                "Read max_abs_spearman_pc_depth together with "
                "spearman_depth_nn_distance. High and weak (fixture 0.86, "
                "-0.17) is a shared depth axis, so regressing out that one "
                "component suffices. Low and strong (fixture 0.09, -0.80) is "
                "dropout dispersion, so the normalization must change."
            ),
            "spearman_depth_nn_distance": (
                "Strongly negative means shallow cells are far even from their "
                "closest metastatic cell."
            ),
            "max_abs_spearman_pc_depth": (
                "Largest per-component rank correlation with depth."
            ),
            "mean_pc_rank_variance_from_depth": (
                "Mean fraction of each component's rank variance explained by "
                "depth alone, averaged over components."
            ),
        },
    }
    (args.output_root / "pairing_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
