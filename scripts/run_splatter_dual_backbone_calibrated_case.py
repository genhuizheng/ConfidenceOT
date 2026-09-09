"""One leakage-free Splatter case for fixed and null-calibrated UOT/BOT gates.

Calibration is performed once on the unperturbed representation and frozen for
all target-gene perturbation doses.  Binary gates use either prespecified
``c=0.5`` or backbone-specific geometric-null calibration.  Soft gates use a
prespecified escape multiplier for the fixed regime and side-specific Q90
geometric-null anchors for the calibrated regime.  The complete terminal soft
grid is diagnostic; planted labels are never used to choose a price.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from benchmark_logging import configure_run_logger, run_with_exception_logging
from cellot import balanced_ot, multi_start_soft_gate_balanced, multi_start_soft_gate_uot, unbalanced_ot
from confidenceot import calibrate_confidence_cost, rotation_null_costs
from run_splatter_gene_perturbation_benchmark import (
    FrozenJointPCA,
    METHOD_TAXONOMY,
    SCENARIOS,
    alignment,
    fit_methods,
    nested_gene_order,
    perturb_target_genes,
)
from run_splatter_population_uot_benchmark import load_pair, population_transition, rejection_metrics


SOFT_GRID = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--batch-condition", choices=("none", "mild"), default="none")
    parser.add_argument("--n-cells", type=int, required=True)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--gene-fractions", nargs="+", type=float, default=(0.0, 0.05, 0.10, 0.20))
    parser.add_argument("--fold-change", type=float, default=2.0)
    parser.add_argument("--n-hvg", type=int, default=500)
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument("--source-budget", type=float, default=0.15)
    parser.add_argument("--target-budget", type=float, default=0.15)
    parser.add_argument("--solver-tolerance", type=float, default=1e-3)
    parser.add_argument("--outer-cap", type=int, default=30)
    parser.add_argument("--soft-outer-cap", type=int, default=50)
    parser.add_argument("--hard-grid-size", type=int, default=7)
    parser.add_argument("--calibration-null-replicates", type=int, default=2)
    parser.add_argument("--validation-null-replicates", type=int, default=2)
    parser.add_argument("--null-escape-fraction", type=float, default=0.10)
    parser.add_argument("--fixed-binary-c", type=float, default=0.5)
    parser.add_argument("--fixed-soft-alpha", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--calibration-workers", type=int, default=2,
        help="Independent null-replicate workers within each backbone calibration.",
    )
    parser.add_argument(
        "--calibration-backbone-workers", type=int, choices=(1, 2), default=2,
        help="Run balanced and UOT calibration serially (1) or concurrently (2).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cuda-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--cuda-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--skip-soft-gate",
        action="store_true",
        help="Run only Traditional OT, Vanilla UOT, and balanced/UOT M4-E/M4-R.",
    )
    parser.add_argument(
        "--skip-coupling-stability",
        action="store_true",
        help=(
            "Do not retain clean-dose dense couplings for L1 stability. This "
            "reduces host memory for large-N scaling runs; the corresponding "
            "metric is recorded as NaN."
        ),
    )
    return parser.parse_args()


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


def vanilla_uot_mass_diagnostics(
    *,
    coupling: np.ndarray,
    source_population: np.ndarray,
    target_population: np.ndarray,
    source_truth: np.ndarray,
    target_truth: np.ndarray,
    metadata: dict[str, object],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Threshold-free marginal mass loss for the ungated Vanilla UOT baseline.

    Binary gate rejection is undefined for Vanilla UOT.  We therefore retain
    its positive marginal deficit and excess separately.  ``soft_f1`` treats
    the clipped per-cell deficit fraction as a soft anomaly prediction; it is
    deliberately named separately from the binary gate F1.
    """
    source_reference = np.full(coupling.shape[0], 1.0 / coupling.shape[0])
    target_reference = np.full(coupling.shape[1], 1.0 / coupling.shape[1])
    source_mass = coupling.sum(axis=1)
    target_mass = coupling.sum(axis=0)

    def side(
        reference: np.ndarray,
        transported: np.ndarray,
        truth: np.ndarray,
        populations: np.ndarray,
        side_name: str,
    ) -> tuple[dict[str, float], list[dict[str, object]]]:
        deficit_mass = np.maximum(reference - transported, 0.0)
        excess_mass = np.maximum(transported - reference, 0.0)
        deficit_fraction = np.clip(deficit_mass / reference, 0.0, 1.0)
        truth_float = truth.astype(float)
        tp = float(np.sum(truth_float * deficit_fraction))
        fp = float(np.sum((1.0 - truth_float) * deficit_fraction))
        fn = float(np.sum(truth_float * (1.0 - deficit_fraction)))
        soft_f1 = float(2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) else 0.0
        summary = {
            f"{side_name}_mass_deficit_rate": float(deficit_mass.sum() / reference.sum()),
            f"{side_name}_mass_excess_rate": float(excess_mass.sum() / reference.sum()),
            f"{side_name}_mass_deficit_soft_f1": soft_f1,
        }
        rows: list[dict[str, object]] = []
        for population in sorted(np.unique(populations)):
            mask = populations == population
            reference_total = float(reference[mask].sum())
            rows.append({
                **metadata,
                "side": side_name,
                "population": str(population),
                "population_n": int(mask.sum()),
                "reference_mass": reference_total,
                "transported_mass": float(transported[mask].sum()),
                "mass_deficit_rate": float(deficit_mass[mask].sum() / reference_total),
                "mass_excess_rate": float(excess_mass[mask].sum() / reference_total),
            })
        return summary, rows

    source_summary, source_rows = side(
        source_reference, source_mass, source_truth, source_population, "source"
    )
    target_summary, target_rows = side(
        target_reference, target_mass, target_truth, target_population, "target"
    )
    summary = {**source_summary, **target_summary}
    summary["directional_mass_deficit_soft_f1"] = (
        summary["target_mass_deficit_soft_f1"]
        if metadata["scenario"] == "S2_emergence"
        else summary["source_mass_deficit_soft_f1"]
    )
    summary["macro_mass_deficit_soft_f1"] = 0.5 * (
        summary["source_mass_deficit_soft_f1"] + summary["target_mass_deficit_soft_f1"]
    )
    return summary, [*source_rows, *target_rows]


def higher_quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability, method="higher"))


def uot_escape(cost: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    fit = unbalanced_ot(
        cost, epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
        threshold=args.solver_tolerance, max_iterations=20_000,
    )
    mass = float(fit.coupling.sum())
    source = (args.epsilon + args.lambda_a) * (mass * fit.source_marginal - fit.source_mass)
    target = (args.epsilon + args.lambda_b) * (mass * fit.target_marginal - fit.target_mass)
    return source, target


def balanced_escape(cost: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    fit = balanced_ot(cost, epsilon=args.epsilon, threshold=args.solver_tolerance, max_iterations=20_000)
    source_potential = args.epsilon * fit.log_source_scaling
    target_potential = args.epsilon * fit.log_target_scaling
    source = fit.source_marginal * (source_potential - np.dot(source_potential, fit.source_marginal))
    target = fit.target_marginal * (target_potential - np.dot(target_potential, fit.target_marginal))
    return source, target


def positive_max(values: np.ndarray, *, name: str) -> float:
    value = float(np.max(values))
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{name} has no positive escape scale.")
    return value


def soft_null_anchor(
    source_nulls: list[np.ndarray],
    target_nulls: list[np.ndarray],
    escape_function,
    args: argparse.Namespace,
) -> tuple[float, float, list[float], list[float]]:
    probability = 1.0 - args.null_escape_fraction
    source_candidates = [higher_quantile(escape_function(cost, args)[0], probability) for cost in source_nulls]
    target_candidates = [higher_quantile(escape_function(cost, args)[1], probability) for cost in target_nulls]
    source = float(np.median(source_candidates))
    target = float(np.median(target_candidates))
    if source <= 0.0 or target <= 0.0:
        raise RuntimeError("Soft geometric-null calibration produced a non-positive side-specific anchor.")
    return source, target, source_candidates, target_candidates


def soft_terminal_scan(
    observed: np.ndarray,
    nulls: list[np.ndarray],
    *,
    backbone: str,
    c_source: float,
    c_target: float,
    args: argparse.Namespace,
    multipliers: tuple[float, ...] | None = None,
    include_observed: bool = True,
    null_role_prefix: str = "calibration_null",
) -> list[dict[str, object]]:
    grid = multipliers if multipliers is not None else ((1.0,) if args.smoke_test else SOFT_GRID)
    datasets = ([("observed", observed)] if include_observed else [])
    datasets.extend((f"{null_role_prefix}_{i + 1}", cost) for i, cost in enumerate(nulls))
    jobs = [(index, multiplier, role, cost) for index, (multiplier, (role, cost)) in enumerate(
        (multiplier, dataset) for multiplier in grid for dataset in datasets
    )]

    def fit_job(job: tuple[int, float, str, np.ndarray]) -> tuple[int, dict[str, object]]:
        index, multiplier, role, cost = job
        cpu_start = time.thread_time()
        if backbone == "uot":
            multi = multi_start_soft_gate_uot(
                cost, epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
                c_s=multiplier * c_source, c_t=multiplier * c_target,
                source_rejection_budget=args.source_budget, target_rejection_budget=args.target_budget,
                random_seeds=() if args.smoke_test else (17,), gate_tolerance=args.solver_tolerance,
                block_tolerance=args.solver_tolerance, gap_tolerance=args.solver_tolerance,
                threshold=args.solver_tolerance, max_iterations=2_000 if args.smoke_test else 20_000,
                max_outer_iterations=2 if args.smoke_test else args.soft_outer_cap,
            )
        else:
            multi = multi_start_soft_gate_balanced(
                cost, epsilon=args.epsilon, c_s=multiplier * c_source, c_t=multiplier * c_target,
                source_rejection_budget=args.source_budget, target_rejection_budget=args.target_budget,
                random_seeds=() if args.smoke_test else (17,), gate_tolerance=args.solver_tolerance,
                block_tolerance=args.solver_tolerance, gap_tolerance=args.solver_tolerance,
                threshold=args.solver_tolerance, max_iterations=2_000 if args.smoke_test else 20_000,
                max_outer_iterations=2 if args.smoke_test else args.soft_outer_cap,
            )
        selected = next((run for run in multi.runs if run.initialization == "escape-score"), multi.runs[0])
        fit = selected.result
        cpu_seconds = time.thread_time() - cpu_start
        return index, {
            "backbone": backbone, "calibration_component": "soft_terminal_grid",
            "data_role": role, "grid_value": multiplier,
            "soft_price_source": multiplier * c_source, "soft_price_target": multiplier * c_target,
            "source_rejection_rate": float(np.mean(~fit.source_gate)),
            "target_rejection_rate": float(np.mean(~fit.target_gate)),
            "status": fit.status, "fit_seconds": cpu_seconds,
            "fit_cpu_seconds": cpu_seconds, "fit_time_basis": "thread_cpu_seconds",
        }

    rows_by_index: dict[int, dict[str, object]] = {}
    worker_count = min(args.workers, len(jobs))
    if worker_count <= 1:
        for job in jobs:
            index, row = fit_job(job)
            rows_by_index[index] = row
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"soft-cal-{backbone}") as executor:
            futures = [executor.submit(fit_job, job) for job in jobs]
            for future in as_completed(futures):
                index, row = future.result()
                rows_by_index[index] = row
    return [rows_by_index[index] for index in range(len(jobs))]


def main() -> None:
    workflow_start = time.perf_counter()
    workflow_cpu_start = time.process_time()
    args = parse_args()
    args.method_set = "all"
    if args.n_cells <= 0 or args.replicate <= 0:
        raise ValueError("Cell count and replicate must be positive.")
    if not 0.0 < args.null_escape_fraction < 1.0:
        raise ValueError("--null-escape-fraction must lie in (0,1).")
    fractions = tuple(sorted(set(float(value) for value in args.gene_fractions)))
    if not fractions or fractions[0] < 0.0 or fractions[-1] > 1.0:
        raise ValueError("Gene fractions must lie in [0,1].")
    if 0.0 not in fractions:
        fractions = (0.0, *fractions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_run_logger(args.output_dir, "dual_backbone_case", log_filename="run.log")
    logger.info("parameters | %s", json.dumps(vars(args), default=str, sort_keys=True))

    directory = args.data_root / args.scenario / f"n_{args.n_cells:04d}" / f"rep_{args.replicate:02d}"
    source_manifest_path = directory / "manifest.csv"
    source_manifest: dict[str, str] = {}
    if source_manifest_path.exists():
        manifest_table = pd.read_csv(source_manifest_path, dtype=str)
        if {"key", "value"}.issubset(manifest_table.columns):
            source_manifest = dict(zip(manifest_table["key"], manifest_table["value"]))
    source_batch_condition = source_manifest.get("batch_condition", "none")
    if source_batch_condition != args.batch_condition:
        raise ValueError(
            f"Requested batch_condition={args.batch_condition!r}, but source data declare "
            f"batch_condition={source_batch_condition!r} in {source_manifest_path}."
        )
    counts, cells, genes = load_pair(directory)
    condition = cells.condition.astype(str).to_numpy()
    source_index = np.flatnonzero(condition == "source")
    target_index = np.flatnonzero(condition == "target")
    source_population = cells.iloc[source_index].population.astype(str).to_numpy()
    target_population = cells.iloc[target_index].population.astype(str).to_numpy()
    source_truth = cells.iloc[source_index].expected_rejection.to_numpy(bool)
    target_truth = cells.iloc[target_index].expected_rejection.to_numpy(bool)

    frozen = FrozenJointPCA(n_hvg=args.n_hvg, n_pcs=args.n_pcs, seed=args.seed).fit(counts)
    clean_coordinates = frozen.transform(counts)
    source_coordinates = clean_coordinates[source_index]
    target_coordinates = clean_coordinates[target_index]
    raw_cost = pairwise_distances(source_coordinates, target_coordinates, metric="sqeuclidean")
    positive = raw_cost[raw_cost > 0.0]
    cost_scale = float(np.median(positive)) if positive.size else 1.0
    clean_cost = raw_cost / cost_scale

    total_nulls = args.calibration_null_replicates + args.validation_null_replicates
    source_nulls, target_nulls = rotation_null_costs(
        source_coordinates, target_coordinates, observed_scale=cost_scale,
        seed=args.seed, n_replicates=total_nulls,
    )
    split = args.calibration_null_replicates
    calibration_source = source_nulls[:split]
    calibration_target = target_nulls[:split]
    validation_source = source_nulls[split:]
    validation_target = target_nulls[split:]
    calibration_nulls = [*calibration_source, *calibration_target]
    validation_nulls = [*validation_source, *validation_target]

    fixed_prices = {
        "binary_balanced_c": args.fixed_binary_c,
        "binary_uot_c": args.fixed_binary_c,
        "soft_balanced_c_s": np.nan,
        "soft_balanced_c_t": np.nan,
        "soft_uot_c_s": np.nan,
        "soft_uot_c_t": np.nan,
    }
    if not args.skip_soft_gate:
        observed_uot_source, observed_uot_target = uot_escape(clean_cost, args)
        observed_bot_source, observed_bot_target = balanced_escape(clean_cost, args)
        fixed_prices.update({
            "soft_balanced_c_s": args.fixed_soft_alpha * positive_max(observed_bot_source, name="balanced source escape"),
            "soft_balanced_c_t": args.fixed_soft_alpha * positive_max(observed_bot_target, name="balanced target escape"),
            "soft_uot_c_s": args.fixed_soft_alpha * positive_max(observed_uot_source, name="UOT source escape"),
            "soft_uot_c_t": args.fixed_soft_alpha * positive_max(observed_uot_target, name="UOT target escape"),
        })

    calibration_rows: list[dict[str, object]] = []
    calibration_common = dict(
        epsilon=args.epsilon,
        lambda_a=args.lambda_a,
        lambda_b=args.lambda_b,
        source_raw_acceptance_target=args.null_escape_fraction,
        target_raw_acceptance_target=args.null_escape_fraction,
        source_rejection_budget=args.source_budget,
        target_rejection_budget=args.target_budget,
        tolerance=args.solver_tolerance,
        max_iterations=2_000 if args.smoke_test else 20_000,
        max_outer_iterations=3 if args.smoke_test else args.outer_cap,
        grid_size=max(3, args.hard_grid_size),
        refinement_relative_tolerance=args.solver_tolerance,
        device=args.device,
        cuda_dtype=args.cuda_dtype,
        fallback_to_cpu=args.cuda_fallback,
        workers=args.calibration_workers,
        emit_warnings=False,
    )
    calibration_jobs = {
        "uot": lambda: calibrate_confidence_cost(
            calibration_nulls, validation_nulls, backbone="uot", **calibration_common
        ),
        "balanced": lambda: calibrate_confidence_cost(
            calibration_nulls, validation_nulls, backbone="balanced", **calibration_common
        ),
    }
    if args.calibration_backbone_workers == 1:
        hard_results = {name: job() for name, job in calibration_jobs.items()}
    else:
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="confidenceot-calibration-backbone"
        ) as executor:
            futures = {executor.submit(job): name for name, job in calibration_jobs.items()}
            hard_results = {futures[future]: future.result() for future in as_completed(futures)}
    uot_hard = hard_results["uot"]
    bot_hard = hard_results["balanced"]
    # Emit warnings after joining in a fixed order so parallel scheduling does
    # not make logs nondeterministic.
    for calibrated in (uot_hard, bot_hard):
        for message in calibrated.warning_messages:
            warnings.warn(
                f"{calibrated.backbone} calibration: {message}",
                RuntimeWarning,
                stacklevel=2,
            )
    for calibrated in (uot_hard, bot_hard):
        calibration_rows.extend({
            "backbone": calibrated.backbone,
            "calibration_component": "hard_grid",
            "grid_value": float(c),
            "source_raw_acceptance": float(s),
            "target_raw_acceptance": float(t),
            "feasible": bool(s <= args.null_escape_fraction and t <= args.null_escape_fraction),
        } for c, s, t in zip(
            calibrated.curve_costs,
            calibrated.source_raw_acceptance_curve,
            calibrated.target_raw_acceptance_curve,
        ))
        calibration_rows.extend({
            "backbone": calibrated.backbone,
            "calibration_component": "hard_validation",
            "null_index": record.null_index,
            "grid_value": float(calibrated.rejection_cost),
            "source_raw_acceptance": record.source_raw_acceptance,
            "target_raw_acceptance": record.target_raw_acceptance,
            "status": "converged" if record.inner_converged and record.outer_converged and not record.cycle_detected else "terminal-warning",
        } for record in calibrated.validation)

    calibrated_prices = {
        "binary_balanced_c": float(bot_hard.rejection_cost),
        "binary_uot_c": float(uot_hard.rejection_cost),
        "soft_balanced_c_s": np.nan,
        "soft_balanced_c_t": np.nan,
        "soft_uot_c_s": np.nan,
        "soft_uot_c_t": np.nan,
    }
    uot_source_candidates: list[float] = []
    uot_target_candidates: list[float] = []
    bot_source_candidates: list[float] = []
    bot_target_candidates: list[float] = []
    if not args.skip_soft_gate:
        uot_soft_s, uot_soft_t, uot_source_candidates, uot_target_candidates = soft_null_anchor(
            calibration_source, calibration_target, uot_escape, args
        )
        bot_soft_s, bot_soft_t, bot_source_candidates, bot_target_candidates = soft_null_anchor(
            calibration_source, calibration_target, balanced_escape, args
        )
        calibrated_prices.update({
            "soft_balanced_c_s": bot_soft_s,
            "soft_balanced_c_t": bot_soft_t,
            "soft_uot_c_s": uot_soft_s,
            "soft_uot_c_t": uot_soft_t,
        })
        calibration_rows.extend(soft_terminal_scan(
            clean_cost, calibration_nulls, backbone="uot", c_source=uot_soft_s,
            c_target=uot_soft_t, args=args,
        ))
        calibration_rows.extend(soft_terminal_scan(
            clean_cost, calibration_nulls, backbone="balanced", c_source=bot_soft_s,
            c_target=bot_soft_t, args=args,
        ))
        calibration_rows.extend(soft_terminal_scan(
            clean_cost, validation_nulls, backbone="uot", c_source=uot_soft_s,
            c_target=uot_soft_t, args=args, multipliers=(1.0,), include_observed=False,
            null_role_prefix="validation_null",
        ))
        calibration_rows.extend(soft_terminal_scan(
            clean_cost, validation_nulls, backbone="balanced", c_source=bot_soft_s,
            c_target=bot_soft_t, args=args, multipliers=(1.0,), include_observed=False,
            null_role_prefix="validation_null",
        ))

    rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    population_rows: list[dict[str, object]] = []
    mass_diagnostic_rows: list[dict[str, object]] = []
    gate_index_rows: list[dict[str, object]] = []
    gate_arrays: dict[str, np.ndarray] = {}
    intervention_rows: list[dict[str, object]] = []
    gene_order = nested_gene_order(counts[target_index], seed=args.seed)
    clean_couplings: dict[tuple[str, str], np.ndarray] = {}
    for fraction in fractions:
        perturbed, up, down = perturb_target_genes(
            counts, target_index, gene_order, fraction=fraction, fold_change=args.fold_change,
            seed=args.seed + int(round(10_000 * fraction)),
        )
        coordinates = frozen.transform(perturbed)
        cost = pairwise_distances(coordinates[source_index], coordinates[target_index], metric="sqeuclidean") / cost_scale
        intervention_rows.extend({
            "gene_fraction": fraction, "direction": direction, "gene_index": int(index),
            "gene_id": str(genes[index]),
        } for direction, indices in (("up", up), ("down", down)) for index in indices)
        for mode, prices in (("fixed", fixed_prices), ("null_calibrated", calibrated_prices)):
            method_records, parallel_seconds, parallel_cpu_seconds, summed_seconds = fit_methods(cost, prices, args)
            for record in method_records:
                method = str(record["method"])
                coupling = np.asarray(record["coupling"])
                source_gate = np.asarray(record["source_gate"], dtype=bool)
                target_gate = np.asarray(record["target_gate"], dtype=bool)
                key = (mode, method)
                if fraction == 0.0 and not args.skip_coupling_stability:
                    clean_couplings[key] = coupling.copy()
                method_family, backbone, variant = METHOD_TAXONOMY[method]
                row: dict[str, object] = {
                    "scenario": args.scenario, "n_cells": args.n_cells,
                    "replicate": args.replicate, "gene_fraction": fraction,
                    "batch_condition": args.batch_condition,
                    "parameter_mode": mode, "method": method, "method_family": method_family,
                    "backbone": backbone, "algorithm_variant": variant,
                    "status": record["status"], "numerical_warning": record["numerical_warning"],
                    "fit_seconds": record["fit_seconds"], "fit_cpu_seconds": record["fit_cpu_seconds"],
                    "fit_accelerator_seconds": record["fit_accelerator_seconds"],
                    "fit_time_basis": record["fit_time_basis"],
                    "compute_device": record["compute_device"],
                    "compute_backend": record["compute_backend"],
                    "parallel_method_wall_seconds": parallel_seconds,
                    "parallel_method_cpu_seconds": parallel_cpu_seconds,
                    "summed_method_fit_seconds": summed_seconds,
                    "transition_alignment": alignment(coupling, source_population, target_population),
                    "coupling_l1_from_clean": (
                        float(np.sum(np.abs(coupling - clean_couplings[key])))
                        if not args.skip_coupling_stability
                        else float("nan")
                    ),
                    "source_rejection_rate": float(np.mean(~source_gate)),
                    "target_rejection_rate": float(np.mean(~target_gate)),
                    "binary_c": prices["binary_balanced_c" if backbone == "balanced" else "binary_uot_c"],
                    "soft_c_s": prices["soft_balanced_c_s" if backbone == "balanced" else "soft_uot_c_s"],
                    "soft_c_t": prices["soft_balanced_c_t" if backbone == "balanced" else "soft_uot_c_t"],
                }
                row.update(rejection_metrics(source_gate, source_truth, "source"))
                row.update(rejection_metrics(target_gate, target_truth, "target"))
                row["directional_f1"] = row["target_f1"] if args.scenario == "S2_emergence" else row["source_f1"]
                row["macro_f1"] = 0.5 * (float(row["source_f1"]) + float(row["target_f1"]))
                row["micro_f1"] = combined_micro_f1(
                    source_gate, target_gate, source_truth, target_truth
                )
                if method == "Vanilla UOT":
                    mass_summary, mass_rows = vanilla_uot_mass_diagnostics(
                        coupling=coupling,
                        source_population=source_population,
                        target_population=target_population,
                        source_truth=source_truth,
                        target_truth=target_truth,
                        metadata={
                            "scenario": args.scenario,
                            "n_cells": args.n_cells,
                            "replicate": args.replicate,
                            "gene_fraction": fraction,
                            "batch_condition": args.batch_condition,
                            "parameter_mode": mode,
                            "method": method,
                        },
                    )
                    row.update(mass_summary)
                    mass_diagnostic_rows.extend(mass_rows)
                rows.append(row)
                population_rows.extend(population_rejection_rows(
                    source_gate=source_gate,
                    target_gate=target_gate,
                    source_population=source_population,
                    target_population=target_population,
                    metadata={
                        "scenario": args.scenario,
                        "n_cells": args.n_cells,
                        "replicate": args.replicate,
                        "gene_fraction": fraction,
                        "batch_condition": args.batch_condition,
                        "parameter_mode": mode,
                        "method": method,
                    },
                ))
                gate_key = f"gate_{len(gate_index_rows):04d}"
                gate_arrays[f"{gate_key}_source"] = source_gate.astype(np.uint8)
                gate_arrays[f"{gate_key}_target"] = target_gate.astype(np.uint8)
                gate_index_rows.append({
                    "gate_key": gate_key,
                    "scenario": args.scenario,
                    "n_cells": args.n_cells,
                    "replicate": args.replicate,
                    "gene_fraction": fraction,
                    "batch_condition": args.batch_condition,
                    "parameter_mode": mode,
                    "method": method,
                    "source_array": f"{gate_key}_source",
                    "target_array": f"{gate_key}_target",
                })
                transition = population_transition(coupling, source_population, target_population)
                for source_label in transition.index:
                    for target_label in transition.columns:
                        transition_rows.append({
                            "scenario": args.scenario, "n_cells": args.n_cells,
                            "replicate": args.replicate, "gene_fraction": fraction,
                            "batch_condition": args.batch_condition,
                            "parameter_mode": mode, "method": method,
                            "source_population": source_label, "target_population": target_label,
                            "transition_probability": float(transition.loc[source_label, target_label]),
                        })

    for row in calibration_rows:
        row["batch_condition"] = args.batch_condition
    for row in intervention_rows:
        row["batch_condition"] = args.batch_condition
    pd.DataFrame(rows).to_csv(args.output_dir / "runs.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(args.output_dir / "calibration_diagnostics.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(args.output_dir / "population_transitions.csv", index=False)
    pd.DataFrame(population_rows).to_csv(args.output_dir / "population_rejection_rates.csv", index=False)
    pd.DataFrame(mass_diagnostic_rows).to_csv(
        args.output_dir / "population_mass_diagnostics.csv", index=False
    )
    pd.DataFrame(gate_index_rows).to_csv(args.output_dir / "gate_index.csv", index=False)
    np.savez_compressed(args.output_dir / "cell_gates.npz", **gate_arrays)
    pd.DataFrame(
        intervention_rows,
        columns=("gene_fraction", "direction", "gene_index", "gene_id", "batch_condition"),
    ).to_csv(args.output_dir / "perturbed_genes.csv", index=False)
    price_payload = {
        "fixed": fixed_prices, "null_calibrated": calibrated_prices,
        "uot_hard": {
            "selection": uot_hard.selection_status,
            "calibration_valid": uot_hard.calibration_valid,
            "warning_messages": list(uot_hard.warning_messages),
        },
        "balanced_hard": {
            "selection": bot_hard.selection_status,
            "calibration_valid": bot_hard.calibration_valid,
            "warning_messages": list(bot_hard.warning_messages),
        },
        "soft_null_candidates": {
            "uot_source": uot_source_candidates, "uot_target": uot_target_candidates,
            "balanced_source": bot_source_candidates, "balanced_target": bot_target_candidates,
        },
        "soft_deployment_multiplier": 1.0,
        "soft_grid": list(SOFT_GRID),
    }
    (args.output_dir / "price_configurations.json").write_text(json.dumps(price_payload, indent=2), encoding="utf-8")
    manifest = {
        "scenario": args.scenario, "n_cells": args.n_cells, "replicate": args.replicate,
        "batch_condition": args.batch_condition,
        "splatter_batch_fac_loc": source_manifest.get("batch_fac_loc", "0"),
        "splatter_batch_fac_scale": source_manifest.get("batch_fac_scale", "0"),
        "gene_fractions": fractions, "cost_scale": cost_scale,
        "calibration_frozen_before_perturbation": True,
        "calibration_uses_planted_truth": False,
        "fixed_binary_c": args.fixed_binary_c, "fixed_soft_alpha": args.fixed_soft_alpha,
        "confidenceot_device": args.device,
        "confidenceot_cuda_dtype": args.cuda_dtype,
        "method_workers": args.workers,
        "calibration_workers_per_backbone": args.calibration_workers,
        "calibration_backbone_workers": args.calibration_backbone_workers,
        "soft_calibrated_deployment": "g=1 geometric-null Q90 anchor; full grid retained as diagnostic",
        "soft_gate_skipped": args.skip_soft_gate,
        "coupling_stability_skipped": args.skip_coupling_stability,
        "population_rejection_rates_saved": True,
        "population_mass_diagnostics_saved": True,
        "cell_gates_saved": True,
        "actual_wall_seconds": time.perf_counter() - workflow_start,
        "actual_cpu_seconds": time.process_time() - workflow_cpu_start,
        "primary_timing_metric": "actual_cpu_seconds",
        "smoke_test": args.smoke_test,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "run-complete | rows=%d | cpu_seconds=%.6f | wall_seconds=%.6f",
        len(rows), manifest["actual_cpu_seconds"], manifest["actual_wall_seconds"],
    )


if __name__ == "__main__":
    run_with_exception_logging(main, "dual_backbone_case")
