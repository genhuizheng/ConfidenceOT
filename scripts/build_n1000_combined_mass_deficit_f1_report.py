"""N=1000 report combining fixed and calibrated gates with Vanilla-UOT mass loss."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from cellot import unbalanced_ot
from run_splatter_dual_backbone_calibrated_case import vanilla_uot_mass_diagnostics
from run_splatter_gene_perturbation_benchmark import FrozenJointPCA, nested_gene_order, perturb_target_genes
from run_splatter_population_uot_benchmark import load_pair


SCENARIOS = {
    "S1_extinction": "S1: Extinction",
    "S2_emergence": "S2: Emergence",
    "S3_source_outlier": "S3: Source outlier",
    "S4_bifurcation": "S4: Bifurcation",
    "S5_abundance_shift": "S5: Abundance shift",
}
EXPECTED = {
    "S1_extinction": ("source", "A"),
    "S2_emergence": ("target", "G"),
    "S3_source_outlier": ("source", "O"),
}
M4_LABELS = {
    "Cost-matrix gate / Balanced / Exact (M4-E)": "M4-E / Balanced",
    "Cost-matrix gate / Balanced / Reversible (M4-R)": "M4-R / Balanced",
    "Cost-matrix gate / UOT / Exact (M4-E)": "M4-E / UOT",
    "Cost-matrix gate / UOT / Reversible (M4-R)": "M4-R / UOT",
}
DISPLAY_ORDER = [
    "Traditional OT",
    "Vanilla UOT†",
    "Fixed | M4-E / Balanced",
    "Fixed | M4-R / Balanced",
    "Fixed | M4-E / UOT",
    "Fixed | M4-R / UOT",
    "Calibrated | M4-E / Balanced",
    "Calibrated | M4-R / Balanced",
    "Calibrated | M4-E / UOT",
    "Calibrated | M4-R / UOT",
    "Partial OT",
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("combined_root", type=Path)
    p.add_argument("none_data_root", type=Path)
    p.add_argument("mild_data_root", type=Path)
    p.add_argument("output_root", type=Path)
    p.add_argument("--n-cells", type=int, default=1000)
    p.add_argument("--replicates", nargs="+", type=int, default=(1, 2))
    p.add_argument("--gene-fractions", nargs="+", type=float, default=(0.0, 0.05, 0.10, 0.20))
    p.add_argument("--n-hvg", type=int, default=500)
    p.add_argument("--n-pcs", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--lambda-a", type=float, default=1.0)
    p.add_argument("--lambda-b", type=float, default=1.0)
    p.add_argument("--solver-tolerance", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=314159)
    return p.parse_args()


def recompute_vanilla_mass(cfg: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    populations: list[dict[str, object]] = []
    for batch, data_root in (("none", cfg.none_data_root), ("mild", cfg.mild_data_root)):
        for scenario in SCENARIOS:
            for replicate in cfg.replicates:
                directory = data_root / scenario / f"n_{cfg.n_cells:04d}" / f"rep_{replicate:02d}"
                counts, cells, _ = load_pair(directory)
                condition = cells.condition.astype(str).to_numpy()
                source_index = np.flatnonzero(condition == "source")
                target_index = np.flatnonzero(condition == "target")
                source_population = cells.iloc[source_index].population.astype(str).to_numpy()
                target_population = cells.iloc[target_index].population.astype(str).to_numpy()
                source_truth = cells.iloc[source_index].expected_rejection.to_numpy(bool)
                target_truth = cells.iloc[target_index].expected_rejection.to_numpy(bool)
                frozen = FrozenJointPCA(n_hvg=cfg.n_hvg, n_pcs=cfg.n_pcs, seed=cfg.seed).fit(counts)
                clean_coordinates = frozen.transform(counts)
                raw = pairwise_distances(clean_coordinates[source_index], clean_coordinates[target_index], metric="sqeuclidean")
                positive = raw[raw > 0]
                scale = float(np.median(positive)) if positive.size else 1.0
                gene_order = nested_gene_order(counts[target_index], seed=cfg.seed)
                for fraction in cfg.gene_fractions:
                    perturbed, _, _ = perturb_target_genes(
                        counts, target_index, gene_order, fraction=fraction, fold_change=2.0,
                        seed=cfg.seed + int(round(10_000 * fraction)),
                    )
                    coordinates = frozen.transform(perturbed)
                    cost = pairwise_distances(coordinates[source_index], coordinates[target_index], metric="sqeuclidean") / scale
                    start = time.thread_time()
                    fit = unbalanced_ot(
                        cost, epsilon=cfg.epsilon, lambda_a=cfg.lambda_a, lambda_b=cfg.lambda_b,
                        threshold=cfg.solver_tolerance, max_iterations=20_000,
                    )
                    fit_cpu = time.thread_time() - start
                    metadata = {
                        "batch_condition": batch, "scenario": scenario, "n_cells": cfg.n_cells,
                        "replicate": replicate, "gene_fraction": fraction,
                        "parameter_mode": "shared", "method": "Vanilla UOT",
                    }
                    summary, rows = vanilla_uot_mass_diagnostics(
                        coupling=fit.coupling, source_population=source_population,
                        target_population=target_population, source_truth=source_truth,
                        target_truth=target_truth, metadata=metadata,
                    )
                    summaries.append({**metadata, **summary, "fit_cpu_seconds": fit_cpu, "converged": fit.converged})
                    populations.extend(rows)
    return pd.DataFrame(summaries), pd.DataFrame(populations)


def display_metadata(method: str, mode: str) -> tuple[str, str, str]:
    if method == "Traditional OT":
        return "Traditional OT", "shared", "binary gate rejection (identically zero)"
    if method == "Vanilla UOT":
        return "Vanilla UOT†", "shared", "positive marginal mass deficit"
    if method == "Partial OT (fixed transported mass)":
        return "Partial OT", "shared", "binary native partial-OT rejection"
    prefix = "Fixed" if mode == "fixed" else "Calibrated"
    return f"{prefix} | {M4_LABELS[method]}", mode, "binary gate rejection"


def combined_rejection_table(native: pd.DataFrame, vanilla: pd.DataFrame, n_cells: int) -> pd.DataFrame:
    native = native[native.n_cells == n_cells].copy()
    keep = (
        ((native.method.isin(["Traditional OT"])) & (native.parameter_mode == "fixed"))
        | ((native.method.isin(M4_LABELS)) & native.parameter_mode.isin(["fixed", "null_calibrated"]))
        | ((native.method == "Partial OT (fixed transported mass)") & (native.parameter_mode == "fixed_mass_budget"))
    )
    native = native[keep]
    native[["display_method", "regime", "measure_type"]] = native.apply(
        lambda r: pd.Series(display_metadata(r.method, r.parameter_mode)), axis=1
    )
    native = native.rename(columns={"rejection_rate": "rejection_signal"})
    mass = vanilla.copy()
    mass["display_method"] = "Vanilla UOT†"
    mass["regime"] = "shared"
    mass["measure_type"] = "positive marginal mass deficit"
    mass = mass.rename(columns={"mass_deficit_rate": "rejection_signal"})
    columns = [
        "batch_condition", "gene_fraction", "scenario", "n_cells", "replicate", "display_method",
        "regime", "measure_type", "side", "population", "population_n", "rejection_signal",
    ]
    result = pd.concat([native[columns], mass[columns]], ignore_index=True)
    result["method_order"] = result.display_method.map({m: i for i, m in enumerate(DISPLAY_ORDER)})
    return result.sort_values(["batch_condition", "gene_fraction", "scenario", "method_order", "side", "population", "replicate"])


def combined_f1_table(runs: pd.DataFrame, vanilla: pd.DataFrame, n_cells: int) -> pd.DataFrame:
    runs = runs[runs.n_cells == n_cells].copy()
    keep = (
        ((runs.method == "Traditional OT") & (runs.parameter_mode == "fixed"))
        | (runs.method.isin(M4_LABELS) & runs.parameter_mode.isin(["fixed", "null_calibrated"]))
        | ((runs.method == "Partial OT (fixed transported mass)") & (runs.parameter_mode == "fixed_mass_budget"))
    )
    runs = runs[keep]
    rows: list[dict[str, object]] = []
    for r in runs.itertuples(index=False):
        display, regime, _ = display_metadata(r.method, r.parameter_mode)
        applicable = r.scenario in EXPECTED
        rows.append({
            "batch_condition": r.batch_condition, "gene_fraction": r.gene_fraction,
            "scenario": r.scenario, "n_cells": r.n_cells, "replicate": r.replicate,
            "display_method": display, "regime": regime,
            "f1_measure_type": "binary gate F1" if applicable else "not applicable (no planted rejection)",
            "source_f1": r.source_f1 if applicable else np.nan,
            "target_f1": r.target_f1 if applicable else np.nan,
            "directional_f1": r.directional_f1 if applicable else np.nan,
            "macro_f1": r.macro_f1 if applicable else np.nan,
        })
    for r in vanilla.itertuples(index=False):
        applicable = r.scenario in EXPECTED
        rows.append({
            "batch_condition": r.batch_condition, "gene_fraction": r.gene_fraction,
            "scenario": r.scenario, "n_cells": r.n_cells, "replicate": r.replicate,
            "display_method": "Vanilla UOT†", "regime": "shared",
            "f1_measure_type": "threshold-free mass-deficit soft F1" if applicable else "not applicable (no planted rejection)",
            "source_f1": r.source_mass_deficit_soft_f1 if applicable else np.nan,
            "target_f1": r.target_mass_deficit_soft_f1 if applicable else np.nan,
            "directional_f1": r.directional_mass_deficit_soft_f1 if applicable else np.nan,
            "macro_f1": r.macro_mass_deficit_soft_f1 if applicable else np.nan,
        })
    out = pd.DataFrame(rows)
    out["method_order"] = out.display_method.map({m: i for i, m in enumerate(DISPLAY_ORDER)})
    return out.sort_values(["batch_condition", "gene_fraction", "scenario", "method_order", "replicate"])


def summarize(
    frame: pd.DataFrame, values: list[str], extra_keys: tuple[str, ...] = ()
) -> pd.DataFrame:
    keys = [
        "batch_condition", "gene_fraction", "scenario", "n_cells",
        "display_method", "regime", *extra_keys,
    ]
    named = {"replicate_count": ("replicate", "nunique")}
    for value in values:
        named[f"{value}_mean"] = (value, "mean")
        named[f"{value}_sd"] = (value, "std")
    result = frame.groupby(keys, dropna=False).agg(**named).reset_index()
    result["method_order"] = result.display_method.map({m: i for i, m in enumerate(DISPLAY_ORDER)})
    return result.sort_values(["batch_condition", "gene_fraction", "scenario", "method_order"])


def rejection_figures(summary: pd.DataFrame, out: Path, n_cells: int) -> list[dict[str, object]]:
    index: list[dict[str, object]] = []
    compact_method_labels = [
        "OT", "Vanilla UOT†",
        "Fixed M4-E / Bal.", "Fixed M4-R / Bal.",
        "Fixed M4-E / UOT", "Fixed M4-R / UOT",
        "Cal. M4-E / Bal.", "Cal. M4-R / Bal.",
        "Cal. M4-E / UOT", "Cal. M4-R / UOT",
        "Partial OT",
    ]
    for batch in ("none", "mild"):
        for dose in (0.0, 0.05, 0.10, 0.20):
            selected = summary[(summary.batch_condition == batch) & np.isclose(summary.gene_fraction, dose)]
            # PPT-friendly 16:9 layout: three panels above and two wider panels
            # below.  The earlier 2x3 grid left one entire panel blank.
            fig = plt.figure(figsize=(16, 9))
            grid = fig.add_gridspec(2, 6)
            axes = [
                fig.add_subplot(grid[0, 0:2]),
                fig.add_subplot(grid[0, 2:4]),
                fig.add_subplot(grid[0, 4:6]),
                fig.add_subplot(grid[1, 0:3]),
                fig.add_subplot(grid[1, 3:6]),
            ]
            image = None
            for panel_index, (axis, scenario) in enumerate(zip(axes, SCENARIOS)):
                data = selected[selected.scenario == scenario]
                source = sorted(data.loc[data.side == "source", "population"].astype(str).unique())
                target = sorted(data.loc[data.side == "target", "population"].astype(str).unique())
                columns = [("source", p) for p in source] + [("target", p) for p in target]
                labels = [f"Source: {p}" for p in source] + [f"Target: {p}" for p in target]
                values = np.full((len(DISPLAY_ORDER), len(columns)), np.nan)
                lookup = data.set_index(["display_method", "side", "population"])["rejection_signal_mean"]
                for i, method in enumerate(DISPLAY_ORDER):
                    for j, key in enumerate(columns):
                        if (method, *key) in lookup.index:
                            values[i, j] = float(lookup.loc[(method, *key)])
                image = axis.imshow(values, cmap="magma", vmin=0, vmax=1, aspect="auto")
                if scenario in EXPECTED:
                    col = columns.index(EXPECTED[scenario]); labels[col] += "*"
                    axis.add_patch(Rectangle((col - .5, -.5), 1, len(DISPLAY_ORDER), fill=False, edgecolor="#E31A1C", linewidth=2.4))
                axis.axvline(len(source) - .5, color="white", alpha=.75, linewidth=1.25)
                axis.set_xticks(range(len(labels)), labels, rotation=40, ha="right", rotation_mode="anchor", fontsize=9)
                axis.set_yticks(range(len(DISPLAY_ORDER)))
                # Repeat the method labels at the start of each row so that the
                # lower row remains readable when cropped or pasted into PPT.
                axis.set_yticklabels(compact_method_labels if panel_index in (0, 3) else [], fontsize=9.5)
                axis.set_title(SCENARIOS[scenario], loc="left", fontsize=14, pad=7)
                axis.tick_params(length=0, pad=3)
                for i in range(values.shape[0]):
                    for j in range(values.shape[1]):
                        if np.isfinite(values[i, j]):
                            is_expected = scenario in EXPECTED and j == columns.index(EXPECTED[scenario])
                            axis.text(
                                j, i, f"{values[i,j]:.2f}", ha="center", va="center",
                                fontsize=8.0, fontweight="bold" if is_expected else "normal",
                                color="black" if values[i,j] >= .72 else "white",
                            )
                        elif compact_method_labels[i] == "Partial OT":
                            axis.text(
                                j, i, "N/A", ha="center", va="center",
                                fontsize=7.0, color="#777777",
                            )
            cax = fig.add_axes((.942, .18, .014, .64))
            colorbar = fig.colorbar(image, cax=cax)
            colorbar.set_label("Rejection signal", fontsize=11)
            colorbar.ax.tick_params(labelsize=10)
            fig.suptitle(
                f"Population rejection by scenario  |  batch: {batch}  |  perturbation: {dose:.0%}  |  N={n_cells}",
                fontsize=18, y=.975,
            )
            fig.text(
                .5, .025,
                "Bal. = balanced backbone; †Vanilla UOT = positive marginal mass deficit; all other rows = rejected-cell fraction.  Red/* = planted anomaly.",
                ha="center", fontsize=10,
            )
            fig.subplots_adjust(left=.12, right=.925, bottom=.14, top=.90, hspace=.48, wspace=.32)
            stem = out / f"rejection_combined_batch_{batch}_perturb_{int(round(dose*100)):02d}_n{n_cells}_ppt_compact"
            fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"); plt.close(fig)
            index.append({"figure_type":"rejection", "batch_condition":batch, "gene_fraction":dose, "png":str(stem.with_suffix('.png')), "pdf":str(stem.with_suffix('.pdf'))})
    return index


def f1_figure(summary: pd.DataFrame, out: Path, n_cells: int) -> dict[str, object]:
    fig, axes = plt.subplots(2, 4, figsize=(19.2, 10.5), sharex=True, sharey=True)
    image = None
    for row, batch in enumerate(("none", "mild")):
        for col, dose in enumerate((0.0, 0.05, 0.10, 0.20)):
            axis = axes[row, col]
            data = summary[(summary.batch_condition == batch) & np.isclose(summary.gene_fraction, dose)]
            lookup = data.set_index(["display_method", "scenario"])["directional_f1_mean"]
            values = np.full((len(DISPLAY_ORDER), len(SCENARIOS)), np.nan)
            for i, method in enumerate(DISPLAY_ORDER):
                for j, scenario in enumerate(SCENARIOS):
                    if (method, scenario) in lookup.index: values[i,j] = float(lookup.loc[(method,scenario)])
            image = axis.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
            axis.set_xticks(range(len(SCENARIOS)), [s.split(":")[0] for s in SCENARIOS.values()], fontsize=8)
            axis.set_yticks(range(len(DISPLAY_ORDER)), DISPLAY_ORDER, fontsize=8)
            axis.set_title(f"batch={batch}; perturbation={dose:.0%}", fontsize=11)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    if np.isfinite(values[i,j]):
                        axis.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", fontsize=6, color="white" if values[i,j] < .55 else "black")
                    else:
                        axis.text(j, i, "N/A", ha="center", va="center", fontsize=6, color="#666666")
    cax = fig.add_axes((.925, .18, .012, .66)); fig.colorbar(image, cax=cax, label="Directional F1")
    fig.suptitle(f"Directional F1 — fixed and calibrated together; N={n_cells}; mean of 2 replicates", fontsize=16, y=.985)
    note = "† Vanilla UOT uses threshold-free mass-deficit soft F1; gated methods use binary gate F1. S4/S5 are N/A because no rejection is planted."
    if n_cells > 1000:
        note += " Partial OT is N/A because the historical exact Hungarian comparator is cubic."
    fig.text(.5, .012, note, ha="center", fontsize=9)
    fig.subplots_adjust(left=.14, right=.90, bottom=.08, top=.92, hspace=.22, wspace=.15)
    stem = out / f"directional_f1_combined_n{n_cells}"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"); plt.close(fig)
    return {"figure_type":"directional_f1", "batch_condition":"all", "gene_fraction":"all", "png":str(stem.with_suffix('.png')), "pdf":str(stem.with_suffix('.pdf'))}


def runtime_table(runs: pd.DataFrame, n_cells: int) -> pd.DataFrame:
    data = runs[runs.n_cells == n_cells].copy()
    keep = (
        ((data.method.isin(["Traditional OT", "Vanilla UOT"])) & (data.parameter_mode == "fixed"))
        | (data.method.isin(M4_LABELS) & data.parameter_mode.isin(["fixed", "null_calibrated"]))
        | ((data.method == "Partial OT (fixed transported mass)") & (data.parameter_mode == "fixed_mass_budget"))
    )
    data = data[keep]
    data[["display_method", "regime", "measure_type"]] = data.apply(
        lambda r: pd.Series(display_metadata(r.method, r.parameter_mode)), axis=1
    )
    result = data.groupby(
        ["batch_condition", "gene_fraction", "n_cells", "display_method", "regime"],
        dropna=False,
    ).agg(
        method_evaluations=("replicate", "size"),
        single_fit_seconds_mean=("fit_seconds", "mean"),
        single_fit_seconds_median=("fit_seconds", "median"),
        single_fit_seconds_sd=("fit_seconds", "std"),
        timing_basis=("fit_time_basis", lambda values: ";".join(sorted(set(values)))),
    ).reset_index()
    result["method_order"] = result.display_method.map({m: i for i, m in enumerate(DISPLAY_ORDER)})
    return result.sort_values(["batch_condition", "gene_fraction", "method_order"])


def deployment_runtime_figure(summary: pd.DataFrame, out: Path, n_cells: int) -> dict[str, object]:
    fig, axes = plt.subplots(2, 4, figsize=(19.2, 10.5), sharex=False, sharey=True)
    for row, batch in enumerate(("none", "mild")):
        for col, dose in enumerate((0.0, 0.05, 0.10, 0.20)):
            axis = axes[row, col]
            data = summary[(summary.batch_condition == batch) & np.isclose(summary.gene_fraction, dose)].sort_values("method_order")
            values = data.single_fit_seconds_mean.to_numpy(float)
            colors = data.regime.map({"shared":"#7F7F7F", "fixed":"#4C78A8", "null_calibrated":"#F58518"}).tolist()
            axis.barh(data.display_method, values, color=colors)
            axis.set_xscale("log")
            axis.set_xlim(max(float(np.min(values)) * .70, .01), float(np.max(values)) * 1.55)
            axis.invert_yaxis()
            axis.set_title(f"batch={batch}; perturbation={dose:.0%}", fontsize=11)
            axis.set_xlabel("Mean seconds per single OT fit (log scale)")
            axis.grid(axis="x", alpha=.2)
            axis.tick_params(axis="y", labelsize=8)
            for i, value in enumerate(values):
                axis.text(value * 1.08, i, f"{value:.2f}", va="center", fontsize=6)
    fig.suptitle(f"Mean time per single OT fit (calibration excluded) — N={n_cells}", fontsize=16, y=.985)
    fig.legend(
        handles=[
            Patch(facecolor="#4C78A8", label="Fixed c=0.5"),
            Patch(facecolor="#F58518", label="Null-calibrated deployment c*"),
            Patch(facecolor="#7F7F7F", label="Shared baseline / comparator"),
        ],
        loc="upper center", bbox_to_anchor=(.5, .955), ncol=3,
        frameon=False, fontsize=10,
    )
    note = "Blue=fixed; orange=calibrated deployment; gray=shared comparator. Do not interpret orange bars as end-to-end calibrated runtime; shared calibration is only in the case-total figure."
    if n_cells > 1000:
        note += " Exact Partial OT is omitted above N=1000 because its Hungarian solver is cubic."
    fig.text(.5, .012, note, ha="center", fontsize=9)
    fig.subplots_adjust(left=.14, right=.98, bottom=.08, top=.90, hspace=.28, wspace=.22)
    stem = out / f"deployment_runtime_combined_n{n_cells}"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"); plt.close(fig)
    return {"figure_type":"deployment_runtime", "batch_condition":"all", "gene_fraction":"all", "png":str(stem.with_suffix('.png')), "pdf":str(stem.with_suffix('.pdf'))}


def case_runtime_table(combined_root: Path, n_cells: int) -> pd.DataFrame:
    benchmark_root = combined_root.parent
    rows: list[dict[str, object]] = []
    for component, batch in (("m4_none", "none"), ("m4_mild", "mild")):
        for manifest in (benchmark_root / component).rglob("manifest.json"):
            payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
            if int(payload["n_cells"]) == n_cells:
                rows.append({"component":"M4 benchmark", "batch_condition":batch, "scenario":payload["scenario"], "replicate":payload["replicate"], "case_cpu_seconds":payload["actual_cpu_seconds"], "case_wall_seconds":payload["actual_wall_seconds"], "includes_calibration":True})
    for manifest in (benchmark_root / "partial_ot" / "raw").rglob("manifest.json"):
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if int(payload["n_cells"]) == n_cells:
            rows.append({"component":"Partial OT", "batch_condition":payload["batch_condition"], "scenario":payload["scenario"], "replicate":payload["replicate"], "case_cpu_seconds":payload["actual_cpu_seconds"], "case_wall_seconds":payload["actual_wall_seconds"], "includes_calibration":False})
    return pd.DataFrame(rows)


def case_runtime_figure(cases: pd.DataFrame, out: Path, n_cells: int) -> dict[str, object]:
    summary = cases.groupby(["component", "batch_condition", "includes_calibration"], dropna=False).agg(
        case_count=("replicate", "size"), total_cpu_seconds=("case_cpu_seconds", "sum"),
        mean_case_wall_seconds=("case_wall_seconds", "mean"), max_case_wall_seconds=("case_wall_seconds", "max"),
    ).reset_index()
    labels = [f"{r.component}\nbatch={r.batch_condition}" for r in summary.itertuples(index=False)]
    values = summary.total_cpu_seconds.to_numpy(float)
    fig, axis = plt.subplots(figsize=(10.5, 6.5))
    bars = axis.bar(labels, values, color=["#4C78A8" if c == "M4 benchmark" else "#F58518" for c in summary.component])
    axis.set_ylabel("Total case CPU seconds")
    axis.set_title(f"N={n_cells} case runtime including shared workflow costs")
    axis.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width()/2, value, f"{value:,.1f} s\n({value/3600:.2f} h)", ha="center", va="bottom", fontsize=9)
    fig.text(.5, .015, "M4 totals include PCA, perturbation preparation, null calibration, fixed fits, and calibrated fits. Partial OT has no null calibration.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    stem = out / f"case_runtime_including_calibration_n{n_cells}"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"); plt.close(fig)
    summary.to_csv(out.parent / "tables" / "case_runtime_n1000_summary.csv", index=False)
    return {"figure_type":"case_runtime_including_calibration", "batch_condition":"all", "gene_fraction":"all", "png":str(stem.with_suffix('.png')), "pdf":str(stem.with_suffix('.pdf'))}


def main() -> None:
    cfg = args(); tables = cfg.output_root / "tables"; figures = cfg.output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(cfg.combined_root / "runs_combined.csv")
    population = pd.read_csv(cfg.combined_root / "population_rejection_rates_combined.csv")
    cached_summary = tables / "vanilla_uot_mass_deficit_n1000_by_replicate.csv"
    cached_population = tables / "vanilla_uot_population_mass_deficit_n1000.csv"
    if cached_summary.exists() and cached_population.exists():
        vanilla_summary = pd.read_csv(cached_summary)
        vanilla_population = pd.read_csv(cached_population)
    else:
        vanilla_summary, vanilla_population = recompute_vanilla_mass(cfg)
    rejection = combined_rejection_table(population, vanilla_population, cfg.n_cells)
    f1 = combined_f1_table(runs, vanilla_summary, cfg.n_cells)
    rejection_summary = summarize(
        rejection, ["population_n", "rejection_signal"],
        ("measure_type", "side", "population"),
    )
    f1_summary = summarize(
        f1, ["source_f1", "target_f1", "directional_f1", "macro_f1"],
        ("f1_measure_type",),
    )
    rejection.to_csv(tables / "population_rejection_signal_n1000_by_replicate.csv", index=False)
    rejection_summary.to_csv(tables / "population_rejection_signal_n1000_summary.csv", index=False)
    f1.to_csv(tables / "f1_n1000_by_replicate.csv", index=False)
    f1_summary.to_csv(tables / "f1_n1000_summary.csv", index=False)
    vanilla_summary.to_csv(tables / "vanilla_uot_mass_deficit_n1000_by_replicate.csv", index=False)
    vanilla_population.to_csv(tables / "vanilla_uot_population_mass_deficit_n1000.csv", index=False)
    runtime = runtime_table(runs, cfg.n_cells)
    runtime.to_csv(tables / "deployment_runtime_n1000_summary.csv", index=False)
    cases = case_runtime_table(cfg.combined_root, cfg.n_cells)
    cases.to_csv(tables / "case_runtime_n1000_by_case.csv", index=False)
    figure_index = rejection_figures(rejection_summary, figures, cfg.n_cells)
    figure_index.append(f1_figure(f1_summary, figures, cfg.n_cells))
    figure_index.append(deployment_runtime_figure(runtime, figures, cfg.n_cells))
    figure_index.append(case_runtime_figure(cases, figures, cfg.n_cells))
    pd.DataFrame(figure_index).to_csv(tables / "figure_index.csv", index=False)
    audit = {"n_cells":cfg.n_cells, "replicates":list(cfg.replicates), "fixed_and_calibrated_same_figure":True, "population_pooling":False, "vanilla_uot_refit_count":int(len(vanilla_summary)), "reused_other_method_runs":True, "rejection_figures":8, "f1_figures":1, "runtime_figures":2}
    (cfg.output_root / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
