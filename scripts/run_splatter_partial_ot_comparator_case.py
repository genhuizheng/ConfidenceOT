"""Run fixed-mass partial OT on one existing Splatter benchmark case.

The data loading, frozen joint PCA, cost normalization, perturbation nesting,
and seeds mirror ``run_splatter_dual_backbone_calibrated_case.py``.  This
script only computes the new partial-W comparator and never reruns M4 or soft
gate methods.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from benchmark_logging import configure_run_logger, run_with_exception_logging
from cellot import partial_wasserstein_uniform
from run_splatter_gene_perturbation_benchmark import (
    FrozenJointPCA,
    SCENARIOS,
    alignment,
    nested_gene_order,
    perturb_target_genes,
)
from run_splatter_population_uot_benchmark import (
    load_pair,
    population_transition,
    rejection_metrics,
)


METHOD = "Partial OT (fixed transported mass)"


def combined_micro_f1(
    source_gate: np.ndarray,
    target_gate: np.ndarray,
    source_truth: np.ndarray,
    target_truth: np.ndarray,
) -> float:
    tp = int(np.sum((~source_gate) & source_truth) + np.sum((~target_gate) & target_truth))
    fp = int(np.sum((~source_gate) & (~source_truth)) + np.sum((~target_gate) & (~target_truth)))
    fn = int(np.sum(source_gate & source_truth) + np.sum(target_gate & target_truth))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def population_rejection_rows(
    *,
    source_gate: np.ndarray,
    target_gate: np.ndarray,
    source_population: np.ndarray,
    target_population: np.ndarray,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side, gate, populations in (
        ("source", source_gate, source_population),
        ("target", target_gate, target_population),
    ):
        for population in sorted(np.unique(populations)):
            mask = populations == population
            rejected = int(np.sum(~gate[mask]))
            rows.append({
                **metadata,
                "side": side,
                "population": str(population),
                "population_n": int(np.sum(mask)),
                "rejected_n": rejected,
                "rejection_rate": float(rejected / np.sum(mask)),
            })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--batch-condition", choices=("none", "mild"), required=True)
    parser.add_argument("--n-cells", type=int, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument(
        "--gene-fractions", nargs="+", type=float, default=(0.0, 0.05, 0.10, 0.20)
    )
    parser.add_argument("--fold-change", type=float, default=2.0)
    parser.add_argument("--n-hvg", type=int, default=500)
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--source-budget", type=float, default=0.15)
    parser.add_argument("--target-budget", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=314159)
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype=str)
    if not {"key", "value"}.issubset(table.columns):
        return {}
    return dict(zip(table["key"], table["value"]))


def main() -> None:
    workflow_wall_start = time.perf_counter()
    workflow_cpu_start = time.process_time()
    args = parse_args()
    if args.n_cells <= 0 or args.replicate <= 0:
        raise ValueError("Cell count and replicate must be positive.")
    if not np.isclose(args.source_budget, args.target_budget, atol=1e-12, rtol=0.0):
        raise ValueError(
            "Fixed-mass partial OT rejects the same total mass on both sides; "
            "source and target budgets must therefore be equal."
        )
    if not 0.0 <= args.source_budget < 1.0:
        raise ValueError("Rejection budgets must lie in [0,1).")
    transported_mass = 1.0 - args.source_budget
    fractions = tuple(sorted(set(float(value) for value in args.gene_fractions)))
    if not fractions or fractions[0] < 0.0 or fractions[-1] > 1.0:
        raise ValueError("Gene fractions must lie in [0,1].")
    if 0.0 not in fractions:
        fractions = (0.0, *fractions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_run_logger(
        args.output_dir, "splatter_partial_ot", log_filename="run.log"
    )
    logger.info("parameters | %s", json.dumps(vars(args), default=str, sort_keys=True))

    directory = (
        args.data_root
        / args.scenario
        / f"n_{args.n_cells:04d}"
        / f"rep_{args.replicate:02d}"
    )
    source_manifest_path = directory / "manifest.csv"
    source_manifest = read_manifest(source_manifest_path)
    declared_batch = source_manifest.get("batch_condition", "none")
    if declared_batch != args.batch_condition:
        raise ValueError(
            f"Requested batch_condition={args.batch_condition!r}, but the data "
            f"declare {declared_batch!r} in {source_manifest_path}."
        )

    counts, cells, genes = load_pair(directory)
    condition = cells.condition.astype(str).to_numpy()
    source_index = np.flatnonzero(condition == "source")
    target_index = np.flatnonzero(condition == "target")
    if source_index.size != target_index.size:
        raise ValueError("The exact uniform partial-W comparator requires equal support sizes.")
    source_population = cells.iloc[source_index].population.astype(str).to_numpy()
    target_population = cells.iloc[target_index].population.astype(str).to_numpy()
    source_truth = cells.iloc[source_index].expected_rejection.to_numpy(bool)
    target_truth = cells.iloc[target_index].expected_rejection.to_numpy(bool)

    preprocessing_cpu_start = time.process_time()
    frozen = FrozenJointPCA(n_hvg=args.n_hvg, n_pcs=args.n_pcs, seed=args.seed).fit(counts)
    clean_coordinates = frozen.transform(counts)
    raw_clean_cost = pairwise_distances(
        clean_coordinates[source_index],
        clean_coordinates[target_index],
        metric="sqeuclidean",
    )
    positive = raw_clean_cost[raw_clean_cost > 0.0]
    cost_scale = float(np.median(positive)) if positive.size else 1.0
    preprocessing_cpu_seconds = time.process_time() - preprocessing_cpu_start

    rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    population_rows: list[dict[str, object]] = []
    gate_index_rows: list[dict[str, object]] = []
    gate_arrays: dict[str, np.ndarray] = {}
    intervention_rows: list[dict[str, object]] = []
    gene_order = nested_gene_order(counts[target_index], seed=args.seed)
    clean_coupling: np.ndarray | None = None

    for fraction in fractions:
        dose_cpu_start = time.process_time()
        perturbed, up, down = perturb_target_genes(
            counts,
            target_index,
            gene_order,
            fraction=fraction,
            fold_change=args.fold_change,
            seed=args.seed + int(round(10_000 * fraction)),
        )
        coordinates = frozen.transform(perturbed)
        cost = pairwise_distances(
            coordinates[source_index], coordinates[target_index], metric="sqeuclidean"
        ) / cost_scale
        dose_preparation_cpu_seconds = time.process_time() - dose_cpu_start
        intervention_rows.extend(
            {
                "scenario": args.scenario,
                "n_cells": args.n_cells,
                "replicate": args.replicate,
                "batch_condition": args.batch_condition,
                "gene_fraction": fraction,
                "direction": direction,
                "gene_index": int(index),
                "gene_id": str(genes[index]),
            }
            for direction, indices in (("up", up), ("down", down))
            for index in indices
        )

        fit_cpu_start = time.process_time()
        fit = partial_wasserstein_uniform(cost, transported_mass=transported_mass)
        fit_cpu_seconds = time.process_time() - fit_cpu_start
        coupling = fit.coupling
        if fraction == 0.0:
            clean_coupling = coupling.copy()
        if clean_coupling is None:
            raise RuntimeError("The zero-perturbation dose must be evaluated first.")

        row: dict[str, object] = {
            "scenario": args.scenario,
            "n_cells": args.n_cells,
            "replicate": args.replicate,
            "gene_fraction": fraction,
            "batch_condition": args.batch_condition,
            "parameter_mode": "fixed_mass_budget",
            "method": METHOD,
            "method_family": "classical partial transport",
            "backbone": "partial",
            "algorithm_variant": "exact-cardinality",
            "status": fit.status,
            "numerical_warning": False,
            "fit_seconds": fit_cpu_seconds,
            "fit_cpu_seconds": fit_cpu_seconds,
            "fit_time_basis": "process_cpu_seconds",
            "preprocessing_cpu_seconds": preprocessing_cpu_seconds,
            "dose_preparation_cpu_seconds": dose_preparation_cpu_seconds,
            "transition_alignment": alignment(
                coupling, source_population, target_population
            ),
            "coupling_l1_from_clean": float(np.sum(np.abs(coupling - clean_coupling))),
            "source_rejection_rate": float(np.mean(~fit.source_gate)),
            "target_rejection_rate": float(np.mean(~fit.target_gate)),
            "source_budget": args.source_budget,
            "target_budget": args.target_budget,
            "requested_transported_mass": transported_mass,
            "transported_mass": fit.transported_mass,
            "transported_count": fit.transported_count,
            "partial_objective": fit.objective,
            "binary_c": np.nan,
            "soft_c_s": np.nan,
            "soft_c_t": np.nan,
        }
        row.update(rejection_metrics(fit.source_gate, source_truth, "source"))
        row.update(rejection_metrics(fit.target_gate, target_truth, "target"))
        row["directional_f1"] = (
            row["target_f1"] if args.scenario == "S2_emergence" else row["source_f1"]
        )
        row["macro_f1"] = 0.5 * (float(row["source_f1"]) + float(row["target_f1"]))
        row["micro_f1"] = combined_micro_f1(
            fit.source_gate, fit.target_gate, source_truth, target_truth
        )
        rows.append(row)
        metadata = {
            "scenario": args.scenario,
            "n_cells": args.n_cells,
            "replicate": args.replicate,
            "gene_fraction": fraction,
            "batch_condition": args.batch_condition,
            "parameter_mode": "fixed_mass_budget",
            "method": METHOD,
        }
        population_rows.extend(population_rejection_rows(
            source_gate=fit.source_gate,
            target_gate=fit.target_gate,
            source_population=source_population,
            target_population=target_population,
            metadata=metadata,
        ))
        gate_key = f"gate_{len(gate_index_rows):04d}"
        gate_arrays[f"{gate_key}_source"] = np.asarray(fit.source_gate, dtype=np.uint8)
        gate_arrays[f"{gate_key}_target"] = np.asarray(fit.target_gate, dtype=np.uint8)
        gate_index_rows.append({
            "gate_key": gate_key,
            **metadata,
            "source_array": f"{gate_key}_source",
            "target_array": f"{gate_key}_target",
        })

        transition = population_transition(coupling, source_population, target_population)
        for source_label in transition.index:
            for target_label in transition.columns:
                transition_rows.append(
                    {
                        "scenario": args.scenario,
                        "n_cells": args.n_cells,
                        "replicate": args.replicate,
                        "gene_fraction": fraction,
                        "batch_condition": args.batch_condition,
                        "parameter_mode": "fixed_mass_budget",
                        "method": METHOD,
                        "source_population": source_label,
                        "target_population": target_label,
                        "transition_probability": float(
                            transition.loc[source_label, target_label]
                        ),
                    }
                )
        logger.info(
            "dose-complete | gene_fraction=%g | fit_cpu_seconds=%.6f | objective=%.8g",
            fraction,
            fit_cpu_seconds,
            fit.objective,
        )

    pd.DataFrame(rows).to_csv(args.output_dir / "runs.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(
        args.output_dir / "population_transitions.csv", index=False
    )
    pd.DataFrame(population_rows).to_csv(
        args.output_dir / "population_rejection_rates.csv", index=False
    )
    pd.DataFrame(gate_index_rows).to_csv(args.output_dir / "gate_index.csv", index=False)
    np.savez_compressed(args.output_dir / "cell_gates.npz", **gate_arrays)
    pd.DataFrame(intervention_rows).to_csv(
        args.output_dir / "perturbed_genes.csv", index=False
    )
    manifest = {
        "scenario": args.scenario,
        "n_cells": args.n_cells,
        "replicate": args.replicate,
        "batch_condition": args.batch_condition,
        "splatter_batch_fac_loc": source_manifest.get("batch_fac_loc", "0"),
        "splatter_batch_fac_scale": source_manifest.get("batch_fac_scale", "0"),
        "gene_fractions": fractions,
        "cost_scale": cost_scale,
        "method": METHOD,
        "partial_ot_definition": "exact fixed-mass partial-W on uniform empirical measures",
        "requested_transported_mass": transported_mass,
        "rejection_budget_each_side": args.source_budget,
        "same_frozen_pca_and_perturbation_seed_as_dual_backbone": True,
        "calibration_applicable": False,
        "population_rejection_rates_saved": True,
        "cell_gates_saved": True,
        "actual_wall_seconds": time.perf_counter() - workflow_wall_start,
        "actual_cpu_seconds": time.process_time() - workflow_cpu_start,
        "primary_timing_metric": "actual_cpu_seconds",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info(
        "run-complete | rows=%d | cpu_seconds=%.6f | wall_seconds=%.6f",
        len(rows), manifest["actual_cpu_seconds"], manifest["actual_wall_seconds"],
    )


if __name__ == "__main__":
    run_with_exception_logging(main, "splatter_partial_ot")
