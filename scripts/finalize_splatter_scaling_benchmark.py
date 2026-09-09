"""Aggregate and plot the complete large-N Splatter benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_n1000_combined_mass_deficit_f1_report import (
    combined_f1_table,
    combined_rejection_table,
    deployment_runtime_figure,
    f1_figure,
    rejection_figures,
    runtime_table,
    summarize,
)


SIZES = (1000, 5000, 10000, 20000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("m4_root", type=Path)
    parser.add_argument("partial_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--replicates", type=int, default=2)
    return parser.parse_args()


def read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def manifest_count(root: Path) -> int:
    return sum(1 for path in root.rglob("manifest.json") if path.parent != root)


def main() -> None:
    args = parse_args()
    sizes = tuple(sorted(set(args.sizes)))
    expected_m4_cases = 2 * 5 * len(sizes) * args.replicates
    partial_sizes = tuple(size for size in sizes if size <= 1000)
    expected_partial_cases = 2 * 5 * len(partial_sizes) * args.replicates
    counts = {
        "m4": manifest_count(args.m4_root),
        "partial_ot": manifest_count(args.partial_root),
    }
    expected_counts = {
        "m4": expected_m4_cases,
        "partial_ot": expected_partial_cases,
    }
    if counts != expected_counts:
        raise RuntimeError(
            f"Incomplete benchmark: expected {expected_counts}, found {counts}."
        )

    m4_runs = read(args.m4_root / "runs_combined.csv")
    partial_runs = read(args.partial_root / "runs_combined.csv")
    runs = pd.concat((m4_runs, partial_runs), ignore_index=True, sort=False)
    populations = pd.concat(
        (
            read(args.m4_root / "population_rejection_rates_combined.csv"),
            read(args.partial_root / "population_rejection_rates_combined.csv"),
        ),
        ignore_index=True,
        sort=False,
    )
    mass = read(args.m4_root / "population_mass_diagnostics_combined.csv")
    vanilla_runs = m4_runs[
        m4_runs["method"].eq("Vanilla UOT")
        & m4_runs["parameter_mode"].eq("fixed")
    ].copy()
    vanilla_mass = mass[mass["parameter_mode"].eq("fixed")].copy()

    args.output_root.mkdir(parents=True, exist_ok=False)
    timing_columns = [
        "batch_condition",
        "gene_fraction",
        "scenario",
        "n_cells",
        "replicate",
        "parameter_mode",
        "method",
        "fit_seconds",
        "fit_time_basis",
        "compute_device",
        "compute_backend",
        "status",
    ]
    available_timing_columns = [column for column in timing_columns if column in runs]
    runs.loc[:, available_timing_columns].to_csv(
        args.output_root / "single_ot_fit_times_all_runs.csv.gz",
        index=False,
        compression="gzip",
    )
    all_figure_rows: list[dict[str, object]] = []
    size_summary_rows: list[dict[str, object]] = []
    all_rejection_summaries: list[pd.DataFrame] = []
    all_f1_summaries: list[pd.DataFrame] = []
    all_runtime_summaries: list[pd.DataFrame] = []
    for size in sizes:
        size_root = args.output_root / f"n_{size}"
        tables = size_root / "tables"
        figures = size_root / "figures"
        tables.mkdir(parents=True)
        figures.mkdir()

        size_vanilla_mass = vanilla_mass[vanilla_mass["n_cells"].eq(size)]
        size_vanilla_runs = vanilla_runs[vanilla_runs["n_cells"].eq(size)]
        rejection = combined_rejection_table(populations, size_vanilla_mass, size)
        f1 = combined_f1_table(runs, size_vanilla_runs, size)
        rejection_summary = summarize(
            rejection,
            ["population_n", "rejection_signal"],
            ("measure_type", "side", "population"),
        )
        f1_summary = summarize(
            f1,
            ["source_f1", "target_f1", "directional_f1", "macro_f1"],
            ("f1_measure_type",),
        )
        runtime = runtime_table(runs, size)
        all_rejection_summaries.append(rejection_summary)
        all_f1_summaries.append(f1_summary)
        all_runtime_summaries.append(runtime)

        rejection.to_csv(tables / f"population_rejection_n{size}_by_replicate.csv", index=False)
        rejection_summary.to_csv(tables / f"population_rejection_n{size}_summary.csv", index=False)
        f1.to_csv(tables / f"directional_f1_n{size}_by_replicate.csv", index=False)
        f1_summary.to_csv(tables / f"directional_f1_n{size}_summary.csv", index=False)
        runtime.to_csv(tables / f"deployment_runtime_n{size}_summary.csv", index=False)

        figure_rows = rejection_figures(rejection_summary, figures, size)
        figure_rows.append(f1_figure(f1_summary, figures, size))
        figure_rows.append(deployment_runtime_figure(runtime, figures, size))
        for row in figure_rows:
            row["n_cells_per_side"] = size
        all_figure_rows.extend(figure_rows)
        size_summary_rows.append(
            {
                "n_cells_per_side": size,
                "run_rows": int((runs["n_cells"] == size).sum()),
                "population_rows": int((populations["n_cells"] == size).sum()),
                "figure_n": len(figure_rows),
            }
        )

    pd.DataFrame(all_figure_rows).to_csv(args.output_root / "figure_index.csv", index=False)
    pd.DataFrame(size_summary_rows).to_csv(args.output_root / "size_summary.csv", index=False)
    pd.concat(all_rejection_summaries, ignore_index=True).to_csv(
        args.output_root / "population_rejection_all_sizes_summary.csv", index=False
    )
    pd.concat(all_f1_summaries, ignore_index=True).to_csv(
        args.output_root / "directional_f1_all_sizes_summary.csv", index=False
    )
    pd.concat(all_runtime_summaries, ignore_index=True).to_csv(
        args.output_root / "deployment_runtime_all_sizes_summary.csv", index=False
    )
    audit = {
        "complete": True,
        "n_is_cells_per_side": True,
        "sizes": list(sizes),
        "replicates": args.replicates,
        "batch_conditions": ["none", "mild"],
        "mild_splatter_batch_fac_loc": 0.1,
        "mild_splatter_batch_fac_scale": 0.1,
        "splatter_gene_n": 1000,
        "splatter_base_seed": 7300,
        "legacy_n1000_seed_schedule_preserved": True,
        "target_gene_perturbation_fractions": [0.0, 0.05, 0.10, 0.20],
        "target_gene_perturbation_fold_change": 2.0,
        "scenarios": [
            "S1_extinction",
            "S2_emergence",
            "S3_source_outlier",
            "S4_bifurcation",
            "S5_abundance_shift",
        ],
        "expected_cases": expected_counts,
        "completed_cases": counts,
        "partial_ot_sizes": list(partial_sizes),
        "partial_ot_larger_sizes_status": (
            "N/A: the historical exact augmented-Hungarian comparator is O(N^3)"
        ),
        "coupling_stability_metric_skipped_for_memory": True,
        "runtime_definition": (
            "Every raw record is one method fit for one scenario, replicate, "
            "batch condition, perturbation dose, and parameter mode. Figure bars "
            "show the arithmetic mean of these single-fit durations, never their sum."
        ),
        "runtime_excludes": "PCA, cost construction, null calibration, file I/O, and scheduler wait",
        "figures": {
            "population_rejection_per_size": 8,
            "directional_f1_per_size": 1,
            "deployment_runtime_per_size": 1,
        },
    }
    (args.output_root / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
