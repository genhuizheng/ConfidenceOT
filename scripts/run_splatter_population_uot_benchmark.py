"""Repeated S0--S5 Splatter benchmark for traditional OT and three UOTs.

Methods:
  1. traditional balanced entropic OT;
  2. vanilla entropic KL-UOT;
  3. finite-c cost-gated bidirectional UOT (reversible M4-R);
  4. direct-box soft-gated bidirectional UOT with hard readout/refit.

The benchmark uses one joint PCA and one scaled cost per simulated pair. PCA
dimension is fixed across cell counts. Method timing excludes preprocessing and
UMAP; preprocessing time is reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

from traditional_ot import (
    balanced_ot,
    confidence_filtered_bidirectional_uot,
    soft_gated_uot,
    unbalanced_ot,
)


SCENARIOS = (
    "S0_clean_movement",
    "S1_extinction",
    "S2_emergence",
    "S3_source_outlier",
    "S4_bifurcation",
    "S5_abundance_shift",
)
METHODS = (
    "traditional_OT",
    "vanilla_UOT",
    "cost_gate_UOT_M4R",
    "soft_gate_UOT",
)
METHOD_LABELS = {
    "traditional_OT": "Traditional OT",
    "vanilla_UOT": "Vanilla UOT",
    "cost_gate_UOT_M4R": "Cost-gated UOT (M4-R)",
    "soft_gate_UOT": "Soft-gated UOT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=(100, 500, 1000))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-hvg", type=int, default=500)
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.50)
    parser.add_argument("--source-budget", type=float, default=0.15)
    parser.add_argument("--target-budget", type=float, default=0.15)
    parser.add_argument("--inner-tolerance", type=float, default=1e-4)
    parser.add_argument("--cost-gate-outer-cap", type=int, default=30)
    parser.add_argument("--soft-outer-cap", type=int, default=50)
    parser.add_argument("--soft-gate-tolerance", type=float, default=1e-4)
    parser.add_argument("--soft-block-tolerance", type=float, default=1e-4)
    parser.add_argument("--umap-size", type=int, default=500)
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip representative UMAP embeddings; OT/UOT tables and other figures are unchanged.",
    )
    return parser.parse_args()


def load_pair(directory: Path) -> tuple[sp.csr_matrix, pd.DataFrame, list[str]]:
    counts = sp.csr_matrix(mmread(directory / "counts.mtx"), dtype=np.float64).T
    cells = pd.read_csv(directory / "cells.csv")
    genes = pd.read_csv(directory / "genes.tsv", sep="\t")["gene_id"].astype(str).tolist()
    if counts.shape != (len(cells), len(genes)):
        raise ValueError(f"Shape mismatch in {directory}: {counts.shape}.")
    cells["expected_rejection"] = cells["expected_rejection"].astype(bool)
    return counts, cells, genes


def preprocess(
    counts: sp.csr_matrix,
    *,
    n_hvg: int,
    n_pcs: int,
    seed: int,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    library = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(library <= 0.0):
        raise ValueError("All cells must have positive library size.")
    normalized = counts.multiply((10_000.0 / library)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.mean(axis=0)).ravel()
    second = np.asarray(normalized.multiply(normalized).mean(axis=0)).ravel()
    variance = np.maximum(second - mean * mean, 0.0)
    hvg_count = min(int(n_hvg), normalized.shape[1])
    hvg = np.argsort(variance, kind="stable")[-hvg_count:]
    scaled = StandardScaler().fit_transform(normalized[:, hvg].toarray())
    components = min(int(n_pcs), scaled.shape[0] - 1, scaled.shape[1])
    if components != int(n_pcs):
        raise ValueError(
            f"Fixed PCA dimension {n_pcs} is unavailable for shape {scaled.shape}."
        )
    coordinates = PCA(
        n_components=components, svd_solver="full", random_state=seed
    ).fit_transform(scaled)
    return normalized, hvg, coordinates, time.perf_counter() - start


def scaled_cost(
    coordinates: np.ndarray, cells: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    source = np.flatnonzero(cells["condition"].to_numpy() == "source")
    target = np.flatnonzero(cells["condition"].to_numpy() == "target")
    start = time.perf_counter()
    cost = pairwise_distances(
        coordinates[source], coordinates[target], metric="sqeuclidean"
    )
    positive = cost[cost > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    cost = cost / scale
    return cost, source, target, scale, time.perf_counter() - start


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def rejection_metrics(gate: np.ndarray, truth: np.ndarray, prefix: str) -> dict[str, float]:
    rejected = ~gate
    tp = int(np.sum(rejected & truth))
    fp = int(np.sum(rejected & ~truth))
    fn = int(np.sum(~rejected & truth))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        f"{prefix}_rejected": int(rejected.sum()),
        f"{prefix}_rejection_rate": float(rejected.mean()),
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_clean_false_rejection_rate": safe_div(
            np.sum(rejected & ~truth), np.sum(~truth)
        ),
    }


def expected_pairs(scenario: str) -> set[tuple[str, str]]:
    pairs = {(letter, letter) for letter in "ABCDEF"}
    if scenario == "S4_bifurcation":
        pairs.remove(("B", "B"))
        pairs.update({("B", "B1"), ("B", "B2")})
    return pairs


def population_transition(
    coupling: np.ndarray,
    source_population: np.ndarray,
    target_population: np.ndarray,
) -> pd.DataFrame:
    source_levels = sorted(np.unique(source_population))
    target_levels = sorted(np.unique(target_population))
    matrix = np.zeros((len(source_levels), len(target_levels)), dtype=np.float64)
    for i, source_label in enumerate(source_levels):
        source_mask = source_population == source_label
        for j, target_label in enumerate(target_levels):
            target_mask = target_population == target_label
            matrix[i, j] = float(coupling[np.ix_(source_mask, target_mask)].sum())
        total = matrix[i].sum()
        if total > 0.0:
            matrix[i] /= total
    return pd.DataFrame(matrix, index=source_levels, columns=target_levels)


def transition_alignment(
    coupling: np.ndarray,
    source_population: np.ndarray,
    target_population: np.ndarray,
    scenario: str,
) -> float:
    total = float(coupling.sum())
    if total <= 0.0:
        return math.nan
    valid = 0.0
    for source_label, target_label in expected_pairs(scenario):
        valid += float(
            coupling[np.ix_(source_population == source_label, target_population == target_label)].sum()
        )
    return valid / total


def population_filter_rows(
    *,
    scenario: str,
    n_cells: int,
    replicate: int,
    method: str,
    source_gate: np.ndarray,
    target_gate: np.ndarray,
    source_population: np.ndarray,
    target_population: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side, gate, population in (
        ("source", source_gate, source_population),
        ("target", target_gate, target_population),
    ):
        for label in sorted(np.unique(population)):
            mask = population == label
            rows.append({
                "scenario": scenario,
                "n_cells": n_cells,
                "replicate": replicate,
                "method": method,
                "side": side,
                "population": label,
                "n_population": int(mask.sum()),
                "rejected_fraction": float(np.mean(~gate[mask])),
            })
    return rows


def fit_methods(cost: np.ndarray, args: argparse.Namespace) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}

    start = time.perf_counter()
    balanced = balanced_ot(
        cost, epsilon=args.epsilon,
        threshold=args.inner_tolerance, max_iterations=20_000,
    )
    results["traditional_OT"] = {
        "coupling": balanced.coupling,
        "source_gate": np.ones(cost.shape[0], dtype=bool),
        "target_gate": np.ones(cost.shape[1], dtype=bool),
        "status": "converged" if balanced.converged else "iteration-capped-warning",
        "terminal_result_retained": True,
        "numerical_warning": not balanced.converged,
        "fit_seconds": time.perf_counter() - start,
        "n_transport_solves": 1,
        "total_inner_iterations": balanced.n_iterations,
        "cycle_detected": False,
    }

    start = time.perf_counter()
    vanilla = unbalanced_ot(
        cost, epsilon=args.epsilon,
        lambda_a=args.lambda_a, lambda_b=args.lambda_b,
        threshold=args.inner_tolerance, max_iterations=20_000,
    )
    results["vanilla_UOT"] = {
        "coupling": vanilla.coupling,
        "source_gate": np.ones(cost.shape[0], dtype=bool),
        "target_gate": np.ones(cost.shape[1], dtype=bool),
        "status": "converged" if vanilla.converged else "iteration-capped-warning",
        "terminal_result_retained": True,
        "numerical_warning": not vanilla.converged,
        "fit_seconds": time.perf_counter() - start,
        "n_transport_solves": 1,
        "total_inner_iterations": vanilla.n_iterations,
        "cycle_detected": False,
    }

    start = time.perf_counter()
    cost_gate = confidence_filtered_bidirectional_uot(
        cost,
        rejection_cost=args.kappa,
        epsilon=args.epsilon,
        lambda_a=args.lambda_a,
        lambda_b=args.lambda_b,
        variant="reversible",
        source_rejection_budget=args.source_budget,
        target_rejection_budget=args.target_budget,
        tau_s=0.0,
        threshold=args.inner_tolerance,
        max_iterations=20_000,
        max_outer_iterations=args.cost_gate_outer_cap,
    )
    results["cost_gate_UOT_M4R"] = {
        "coupling": cost_gate.coupling,
        "source_gate": cost_gate.source_gate,
        "target_gate": cost_gate.target_gate,
        "status": cost_gate.status,
        "fit_seconds": time.perf_counter() - start,
        "n_transport_solves": cost_gate.n_transport_solves,
        "total_inner_iterations": cost_gate.total_inner_iterations,
        "cycle_detected": cost_gate.cycle_detected,
        "terminal_result_retained": True,
        "numerical_warning": bool(
            not cost_gate.outer_converged
            or not cost_gate.inner_converged
            or cost_gate.cycle_detected
        ),
    }

    start = time.perf_counter()
    soft = soft_gated_uot(
        cost,
        epsilon=args.epsilon,
        lambda_a=args.lambda_a,
        lambda_b=args.lambda_b,
        c_s=args.kappa / cost.shape[0],
        c_t=args.kappa / cost.shape[1],
        source_rejection_budget=args.source_budget,
        target_rejection_budget=args.target_budget,
        gate_floor=1e-3,
        readout_threshold=0.5,
        initial_step=float(cost.shape[0]),
        reference_step=float(cost.shape[0]),
        gate_tolerance=args.soft_gate_tolerance,
        block_tolerance=args.soft_block_tolerance,
        gap_tolerance=args.inner_tolerance,
        threshold=args.inner_tolerance,
        max_iterations=20_000,
        max_outer_iterations=args.soft_outer_cap,
    )
    results["soft_gate_UOT"] = {
        "coupling": soft.coupling,
        "source_gate": soft.source_gate,
        "target_gate": soft.target_gate,
        "source_soft_gate": soft.source_soft_gate,
        "target_soft_gate": soft.target_soft_gate,
        "status": soft.status,
        "fit_seconds": time.perf_counter() - start,
        "n_transport_solves": soft.n_transport_solves,
        "total_inner_iterations": soft.total_inner_iterations,
        "cycle_detected": False,
        "terminal_result_retained": soft.status != "inner-solver-failure",
        "numerical_warning": soft.status != "numerically-soft-stationary",
        "source_fractional_suppression": soft.source_fractional_suppression,
        "target_fractional_suppression": soft.target_fractional_suppression,
        "primal_dual_gap": soft.primal_dual_gap,
    }
    return results


def plot_runtime(results: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for method, marker in zip(METHODS, ("o", "s", "^", "D")):
        subset = results[results.method == method]
        summary = subset.groupby("n_cells")["fit_seconds"].agg(["mean", "std"]).reset_index()
        axis.errorbar(
            summary.n_cells, summary["mean"], yerr=summary["std"].fillna(0.0),
            marker=marker, linewidth=1.8, capsize=3, label=METHOD_LABELS[method],
        )
    axis.set_xlabel("Cells per condition")
    axis.set_ylabel("OT fit time (seconds; PCA/UMAP excluded)")
    axis.set_yscale("log")
    axis.set_title("Runtime scaling across S0–S5 and repeated Splatter draws")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_accuracy(results: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    x = np.arange(len(METHODS))
    width = 0.22
    for panel_index, (axis, scenario) in enumerate(zip(axes.ravel(), SCENARIOS)):
        subset = results[results.scenario == scenario]
        for offset, n_cells in enumerate(sorted(subset.n_cells.unique())):
            values = [
                subset[(subset.method == method) & (subset.n_cells == n_cells)]["macro_f1"].mean()
                for method in METHODS
            ]
            axis.bar(x + (offset - 1) * width, values, width=width, label=f"N={n_cells}")
        axis.set_title(scenario.replace("_", " "))
        axis.set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Rejection macro-F1")
    axes[1, 0].set_ylabel("Rejection macro-F1")
    axes[0, 2].legend(frameon=False)
    figure.suptitle("Population-level endpoint filtering (truth exists only in S1–S3)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_population_filtering(filters: pd.DataFrame, n_cells: int, output: Path) -> None:
    subset = filters[filters.n_cells == n_cells]
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for panel_index, (axis, scenario) in enumerate(zip(axes.ravel(), SCENARIOS)):
        data = subset[subset.scenario == scenario].copy()
        data["endpoint"] = data.side.str[0].str.upper() + ":" + data.population
        endpoints = sorted(data.endpoint.unique(), key=lambda x: (x[0], x[2:]))
        matrix = np.zeros((len(METHODS), len(endpoints)))
        for i, method in enumerate(METHODS):
            means = data[data.method == method].groupby("endpoint").rejected_fraction.mean()
            matrix[i] = [means.get(endpoint, 0.0) for endpoint in endpoints]
        image = axis.imshow(matrix, vmin=0, vmax=max(args_global.source_budget, args_global.target_budget),
                            cmap="magma", aspect="auto")
        axis.set_title(scenario.replace("_", " "))
        axis.set_xticks(range(len(endpoints)), endpoints, rotation=45, ha="right")
        axis.set_yticks(range(len(METHODS)))
        if panel_index % 3 == 0:
            axis.set_yticklabels([METHOD_LABELS[m] for m in METHODS])
        else:
            axis.set_yticklabels([])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0:
                    axis.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7,
                              color="white" if matrix[i,j] > 0.07 else "black")
    color_axis = figure.add_axes((0.945, 0.18, 0.012, 0.64))
    figure.colorbar(image, cax=color_axis, label="Mean rejected fraction")
    figure.suptitle(f"Which populations were filtered (N={n_cells}, repeated draws)")
    figure.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.90, wspace=0.35, hspace=0.45)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_transition_matrices(
    transition_tables: dict[tuple[str, str], pd.DataFrame],
    output_dir: Path,
) -> None:
    for scenario in SCENARIOS:
        tables = [transition_tables[(scenario, method)] for method in METHODS]
        figure, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        for axis, method, table in zip(axes, METHODS, tables):
            image = axis.imshow(table.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
            axis.set_title(METHOD_LABELS[method])
            axis.set_xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
            axis.set_yticks(range(len(table.index)), table.index)
            axis.set_xlabel("Target population")
            if axis is axes[0]:
                axis.set_ylabel("Source population")
        color_axis = figure.add_axes((0.945, 0.20, 0.012, 0.60))
        figure.colorbar(image, cax=color_axis, label="Row-normalized mass")
        figure.suptitle(f"{scenario.replace('_', ' ')}: population transition matrix (N=500, replicate 1)")
        figure.subplots_adjust(left=0.06, right=0.92, bottom=0.20, top=0.82, wspace=0.35)
        figure.savefig(output_dir / f"transition_{scenario}.png", dpi=180)
        plt.close(figure)


def plot_population_umap(
    representatives: dict[tuple[str, int], tuple[np.ndarray, pd.DataFrame]],
    output: Path,
) -> None:
    # UMAP/numba is expensive to import and is irrelevant to result-only
    # replotting, so keep it out of the module import path.
    import umap

    sizes = sorted({key[1] for key in representatives})
    figure, axes = plt.subplots(len(SCENARIOS), len(sizes), figsize=(13, 20))
    axes = np.asarray(axes, dtype=object).reshape(len(SCENARIOS), len(sizes))
    population_names = sorted({
        population
        for _, cells in representatives.values()
        for population in cells.population.unique()
    })
    palette = {name: plt.cm.tab20(i / max(1, len(population_names) - 1))
               for i, name in enumerate(population_names)}
    for row, scenario in enumerate(SCENARIOS):
        for column, n_cells in enumerate(sizes):
            axis = axes[row, column]
            embedding, cells = representatives[(scenario, n_cells)]
            for population in sorted(cells.population.unique()):
                for condition, marker in (("source", "o"), ("target", "^")):
                    mask = ((cells.population.to_numpy() == population)
                            & (cells.condition.to_numpy() == condition))
                    if np.any(mask):
                        axis.scatter(
                            embedding[mask, 0], embedding[mask, 1], s=5,
                            color=palette[population], marker=marker, alpha=0.65,
                            linewidths=0,
                        )
            if row == 0:
                axis.set_title(f"N={n_cells} per condition")
            if column == 0:
                axis.set_ylabel(scenario.replace("_", " "))
            axis.set_xticks([]); axis.set_yticks([])
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=palette[p], label=p, markersize=5)
        for p in population_names
    ]
    handles.extend([
        plt.Line2D([0], [0], marker="o", linestyle="", color="gray", label="Source", markersize=5),
        plt.Line2D([0], [0], marker="^", linestyle="", color="gray", label="Target", markersize=5),
    ])
    figure.legend(handles=handles, loc="lower center", ncol=min(10, len(handles)), frameon=False)
    figure.suptitle("Joint-PCA UMAP of fresh Splatter populations (replicate 1)")
    figure.tight_layout(rect=(0, 0.04, 1, 0.98))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(output: Path, results: pd.DataFrame, parameters: dict[str, object]) -> None:
    runtime = results.groupby(["method", "n_cells"]).fit_seconds.agg(["mean", "std"]).reset_index()
    runtime["method"] = runtime.method.map(METHOD_LABELS)
    status = results.groupby(["method", "status"]).size().rename("runs").reset_index()
    status["method"] = status.method.map(METHOD_LABELS)
    key = results[results.scenario.isin(["S1_extinction", "S2_emergence", "S3_source_outlier"])]
    filtering = key.groupby("method")[["macro_f1", "transition_alignment"]].mean().reset_index()
    filtering["method"] = filtering.method.map(METHOD_LABELS)
    directional_rows = []
    for scenario, metric, label in (
        ("S1_extinction", "source_recall", "S1 source A recall"),
        ("S2_emergence", "target_recall", "S2 target G recall"),
        ("S3_source_outlier", "source_recall", "S3 source O recall"),
    ):
        table = results[results.scenario == scenario].pivot_table(
            index="method", columns="n_cells", values=metric, aggfunc="mean"
        )
        for method in ("cost_gate_UOT_M4R", "soft_gate_UOT"):
            directional_rows.append({
                "scenario / expected rejection": label,
                "method": METHOD_LABELS[method],
                **{f"N={n}": table.loc[method, n] for n in sorted(table.columns)},
            })
    directional = pd.DataFrame(directional_rows)
    no_truth = results[results.scenario.isin(
        ["S0_clean_movement", "S4_bifurcation", "S5_abundance_shift"]
    )]
    false_rejection = no_truth.groupby(["method", "n_cells"])[
        ["source_rejection_rate", "target_rejection_rate"]
    ].mean().reset_index()
    false_rejection["method"] = false_rejection.method.map(METHOD_LABELS)
    cost_cycles = int(results.loc[results.method == "cost_gate_UOT_M4R", "cycle_detected"].sum())
    soft_stationary = int((results.loc[results.method == "soft_gate_UOT", "status"]
                           == "numerically-soft-stationary").sum())
    lines = [
        "# Splatter population benchmark: traditional OT and three UOT formulations",
        "",
        "Fresh Splatter pools were generated independently for each cell count and repetition. "
        "All four methods use the same fixed-dimensional joint PCA and scaled cost within a run.",
        "",
        "## Parameters",
        "",
        "```json",
        json.dumps(parameters, indent=2),
        "```",
        "",
        "## Runtime (seconds; PCA and UMAP excluded)",
        "",
        runtime.round(4).to_markdown(index=False),
        "",
        "## Fit statuses",
        "",
        status.to_markdown(index=False),
        "",
        "## Mean filtering performance on S1–S3",
        "",
        filtering.round(4).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "Traditional OT and vanilla UOT have no binary endpoint gate and therefore reject no cells. "
        "Cost-gated and soft-gated results are reported only under their actual terminal statuses. "
        "S0, S4 and S5 contain no endpoints that should be rejected; their rejection F1 is therefore "
        "descriptive zero and false rejection is the relevant metric.",
        "",
        "The macro-F1 averages source and target endpoints although the truth is one-sided in S1–S3. "
        "The direction-specific recall is therefore the more interpretable filtering check:",
        "",
        directional.round(3).to_markdown(index=False),
        "",
        "## False rejection on S0, S4 and S5",
        "",
        false_rejection.round(3).to_markdown(index=False),
        "",
        "## Bottom line at this operating point",
        "",
        f"Cost-gated M4-R has population-rejection signal, especially at N=500 and N=1000, "
        f"but {cost_cycles} of its 54 runs entered a cycle and hit the outer-iteration cap. It also "
        "falsely filters matched populations and is much slower than vanilla UOT. "
        f"Soft-gated UOT reached a numerical soft stationary point in {soft_stationary} of 54 runs "
        "but behaves as an all-accept solution at this kappa. These results evaluate one "
        "prespecified operating point; calibration or sensitivity analysis is required before a "
        "parameter-wide claim.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


args_global: argparse.Namespace


def main() -> None:
    global args_global
    args = parse_args()
    args_global = args
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    transition_dir = args.output_dir / "transitions"
    figure_dir.mkdir(exist_ok=True)
    transition_dir.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    filter_rows: list[dict[str, object]] = []
    transitions: dict[tuple[str, str], pd.DataFrame] = {}
    representatives: dict[tuple[str, int], tuple[np.ndarray, pd.DataFrame]] = {}

    for n_cells in args.sizes:
        for replicate in range(1, args.repeats + 1):
            for scenario in SCENARIOS:
                directory = (
                    args.data_root / scenario / f"n_{n_cells:04d}" / f"rep_{replicate:02d}"
                )
                counts, cells, _ = load_pair(directory)
                seed = int(cells.seed.iloc[0])
                normalized, hvg, coordinates, pca_seconds = preprocess(
                    counts, n_hvg=args.n_hvg, n_pcs=args.n_pcs, seed=seed
                )
                cost, source_index, target_index, cost_scale, cost_seconds = scaled_cost(
                    coordinates, cells
                )
                preprocessing_seconds = pca_seconds + cost_seconds
                source_frame = cells.iloc[source_index]
                target_frame = cells.iloc[target_index]
                source_population = source_frame.population.astype(str).to_numpy()
                target_population = target_frame.population.astype(str).to_numpy()
                source_truth = source_frame.expected_rejection.to_numpy(bool)
                target_truth = target_frame.expected_rejection.to_numpy(bool)
                fitted = fit_methods(cost, args)
                for method in METHODS:
                    fit = fitted[method]
                    source_gate = np.asarray(fit["source_gate"], dtype=bool)
                    target_gate = np.asarray(fit["target_gate"], dtype=bool)
                    row: dict[str, object] = {
                        "scenario": scenario,
                        "n_cells": n_cells,
                        "replicate": replicate,
                        "seed": seed,
                        "method": method,
                        "status": fit["status"],
                        "fit_seconds": fit["fit_seconds"],
                        "preprocessing_seconds": preprocessing_seconds,
                        "total_seconds_excluding_umap": preprocessing_seconds + float(fit["fit_seconds"]),
                        "cost_scale": cost_scale,
                        "n_transport_solves": fit["n_transport_solves"],
                        "total_inner_iterations": fit["total_inner_iterations"],
                        "cycle_detected": fit["cycle_detected"],
                        "terminal_result_retained": fit.get(
                            "terminal_result_retained", True
                        ),
                        "numerical_warning": fit.get("numerical_warning", False),
                        "transition_alignment": transition_alignment(
                            np.asarray(fit["coupling"]), source_population, target_population, scenario
                        ),
                        "source_fractional_suppression": fit.get("source_fractional_suppression", math.nan),
                        "target_fractional_suppression": fit.get("target_fractional_suppression", math.nan),
                        "primal_dual_gap": fit.get("primal_dual_gap", math.nan),
                    }
                    row.update(rejection_metrics(source_gate, source_truth, "source"))
                    row.update(rejection_metrics(target_gate, target_truth, "target"))
                    row["macro_f1"] = 0.5 * (float(row["source_f1"]) + float(row["target_f1"]))
                    rows.append(row)
                    filter_rows.extend(population_filter_rows(
                        scenario=scenario, n_cells=n_cells, replicate=replicate,
                        method=method, source_gate=source_gate, target_gate=target_gate,
                        source_population=source_population, target_population=target_population,
                    ))
                    if n_cells == args.umap_size and replicate == 1:
                        table = population_transition(
                            np.asarray(fit["coupling"]), source_population, target_population
                        )
                        transitions[(scenario, method)] = table
                        table.to_csv(transition_dir / f"{scenario}_{method}.csv")
                if replicate == 1 and not args.skip_umap:
                    import umap

                    embedding = umap.UMAP(
                        n_neighbors=min(20, len(coordinates) - 1),
                        min_dist=0.30,
                        n_components=2,
                        metric="euclidean",
                        random_state=seed,
                        n_jobs=1,
                    ).fit_transform(coordinates)
                    representatives[(scenario, n_cells)] = (embedding, cells.copy())
                print(f"{scenario} N={n_cells} rep={replicate}: complete", flush=True)

    results = pd.DataFrame(rows)
    filters = pd.DataFrame(filter_rows)
    results.to_csv(args.output_dir / "runs.csv", index=False)
    filters.to_csv(args.output_dir / "population_filtering.csv", index=False)
    plot_runtime(results, figure_dir / "runtime_scaling.png")
    plot_accuracy(results, figure_dir / "rejection_f1.png")
    plot_population_filtering(filters, args.umap_size, figure_dir / "population_filtering.png")
    plot_transition_matrices(transitions, figure_dir)
    if not args.skip_umap:
        plot_population_umap(representatives, figure_dir / "population_umap.png")
    parameters = {
        "data_root": str(args.data_root.resolve()),
        "sizes_per_condition": list(args.sizes),
        "repeats": args.repeats,
        "scenarios": list(SCENARIOS),
        "methods": list(METHODS),
        "fixed_pca_dimension": args.n_pcs,
        "n_hvg": args.n_hvg,
        "epsilon": args.epsilon,
        "lambda_a": args.lambda_a,
        "lambda_b": args.lambda_b,
        "kappa": args.kappa,
        "cost_gate_rejection_cost": args.kappa,
        "soft_gate_per_endpoint_price": "kappa / N_side",
        "source_rejection_budget": args.source_budget,
        "target_rejection_budget": args.target_budget,
        "inner_tolerance": args.inner_tolerance,
        "cost_gate_outer_cap": args.cost_gate_outer_cap,
        "soft_gate_outer_cap": args.soft_outer_cap,
        "umap_skipped": args.skip_umap,
        "timing_scope": "fit_seconds excludes normalization, HVG, PCA, cost construction, UMAP, and plotting",
    }
    (args.output_dir / "parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "REPORT.md", results, parameters)
    print(f"Wrote benchmark to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
