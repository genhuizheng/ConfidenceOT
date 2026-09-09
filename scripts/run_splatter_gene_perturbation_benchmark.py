"""Splatter target-gene perturbation benchmark with a frozen clean PCA map.

The selected gene sets are nested across perturbation fractions.  Selected
target genes are split deterministically between up-regulation and down-
regulation.  Source counts are never changed.  HVGs, standardization, PCA and
the cost scale are fitted once on the clean pair and frozen across doses.

The built-in prices are deliberately labelled exploratory.  Production runs
should pass a frozen JSON price configuration obtained without planted truth.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import json
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from benchmark_logging import configure_run_logger, log_terminal_warning, run_with_exception_logging
from run_splatter_population_uot_benchmark import load_pair, population_transition, rejection_metrics
from cellot import (
    balanced_ot,
    multi_start_soft_gate_balanced,
    multi_start_soft_gate_uot,
    unbalanced_ot,
)
from confidenceot import ConfidenceOT


DEFAULT_SCENARIO = "S0_clean_movement"
SCENARIOS = (
    "S0_clean_movement",
    "S1_extinction",
    "S2_emergence",
    "S3_source_outlier",
    "S4_bifurcation",
    "S5_abundance_shift",
)
DEFAULT_FRACTIONS = (0.0, 0.05, 0.10, 0.20)
METHOD_TAXONOMY = {
    "Traditional OT": ("ungated baseline", "balanced", "not-applicable"),
    "Vanilla UOT": ("ungated baseline", "KL-unbalanced", "not-applicable"),
    "Cost-matrix gate / Balanced / Exact (M4-E)": ("binary cost-matrix gate", "balanced", "exact"),
    "Cost-matrix gate / Balanced / Reversible (M4-R)": ("binary cost-matrix gate", "balanced", "reversible"),
    "Soft gate / Balanced": ("continuous soft gate", "balanced", "projected-gradient"),
    "Cost-matrix gate / UOT / Exact (M4-E)": ("binary cost-matrix gate", "KL-unbalanced", "exact"),
    "Cost-matrix gate / UOT / Reversible (M4-R)": ("binary cost-matrix gate", "KL-unbalanced", "reversible"),
    "Soft gate / UOT": ("continuous soft gate", "KL-unbalanced", "projected-gradient"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-cells", type=int, default=500)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--scenario", choices=SCENARIOS, default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--method-set",
        choices=("all", "balanced", "uot"),
        default="all",
        help="Run both backbones or only the requested method family.",
    )
    parser.add_argument("--gene-fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
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
    parser.add_argument("--price-config", type=Path)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Parallel method-family workers per perturbation dose.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cuda-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--cuda-fallback", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use short iteration caps and one soft initialization; never use for reported results.",
    )
    return parser.parse_args()


def validate_fractions(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    fractions = tuple(sorted(set(float(value) for value in values)))
    if not fractions or fractions[0] < 0.0 or fractions[-1] > 1.0:
        raise ValueError("Every gene perturbation fraction must lie in [0, 1].")
    if 0.0 not in fractions:
        fractions = (0.0, *fractions)
    return fractions


def normalize_log1p(counts: sp.csr_matrix) -> np.ndarray:
    library = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(library <= 0.0):
        raise ValueError("All cells must have positive library size.")
    normalized = counts.multiply((10_000.0 / library)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized.toarray()


class FrozenJointPCA:
    def __init__(self, *, n_hvg: int, n_pcs: int, seed: int) -> None:
        self.n_hvg = int(n_hvg)
        self.n_pcs = int(n_pcs)
        self.seed = int(seed)
        self.hvg: np.ndarray | None = None
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None

    def fit(self, clean_counts: sp.csr_matrix) -> "FrozenJointPCA":
        matrix = normalize_log1p(clean_counts)
        variance = np.var(matrix, axis=0)
        count = min(self.n_hvg, matrix.shape[1])
        self.hvg = np.argsort(variance, kind="stable")[-count:]
        self.scaler = StandardScaler().fit(matrix[:, self.hvg])
        scaled = self.scaler.transform(matrix[:, self.hvg])
        components = min(self.n_pcs, scaled.shape[0] - 1, scaled.shape[1])
        if components != self.n_pcs:
            raise ValueError(f"Fixed PCA dimension {self.n_pcs} is unavailable for {scaled.shape}.")
        self.pca = PCA(n_components=components, svd_solver="full", random_state=self.seed).fit(scaled)
        return self

    def transform(self, counts: sp.csr_matrix) -> np.ndarray:
        if self.hvg is None or self.scaler is None or self.pca is None:
            raise RuntimeError("FrozenJointPCA must be fitted before transform.")
        matrix = normalize_log1p(counts)
        return self.pca.transform(self.scaler.transform(matrix[:, self.hvg]))


def nested_gene_order(counts: sp.csr_matrix, *, seed: int) -> np.ndarray:
    """Return a reproducible order over genes expressed in target cells."""
    expressed = np.flatnonzero(np.asarray(counts.sum(axis=0)).ravel() > 0.0)
    if expressed.size == 0:
        raise ValueError("No expressed genes are available for perturbation.")
    return np.random.default_rng(seed).permutation(expressed)


def perturb_target_genes(
    counts: sp.csr_matrix,
    target_index: np.ndarray,
    gene_order: np.ndarray,
    *,
    fraction: float,
    fold_change: float,
    seed: int,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Apply nested, signed count perturbations to all target cells.

    Up-regulated counts receive Poisson extra counts with mean
    ``(fold_change-1)*count``.  Down-regulated counts are binomially thinned
    with retention probability ``1/fold_change``.  The returned up/down gene
    indices make the intervention fully auditable.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("`fraction` must lie in [0, 1].")
    if not np.isfinite(fold_change) or fold_change <= 1.0:
        raise ValueError("`fold_change` must be finite and greater than 1.")
    selected_count = int(np.ceil(fraction * counts.shape[1] - 1e-12))
    selected_count = min(selected_count, gene_order.size)
    selected = np.asarray(gene_order[:selected_count], dtype=np.int64)
    up = selected[::2]
    down = selected[1::2]
    if selected.size == 0:
        return counts.copy().tocsr(), up, down
    result = counts.tolil(copy=True)
    target_block = counts[target_index].tocsc()
    rng = np.random.default_rng(seed)
    if up.size:
        dense = target_block[:, up].toarray().astype(np.int64, copy=False)
        dense += rng.poisson((fold_change - 1.0) * dense)
        for local, gene in enumerate(up):
            result[target_index, int(gene)] = dense[:, local, None]
    if down.size:
        dense = target_block[:, down].toarray().astype(np.int64, copy=False)
        dense = rng.binomial(dense, 1.0 / fold_change)
        for local, gene in enumerate(down):
            result[target_index, int(gene)] = dense[:, local, None]
    return result.tocsr(), up, down


def alignment(coupling: np.ndarray, source_population: np.ndarray, target_population: np.ndarray) -> float:
    total = float(coupling.sum())
    if total <= 0.0:
        return float("nan")
    value = 0.0
    for label in sorted(set(source_population) & set(target_population)):
        value += float(coupling[np.ix_(source_population == label, target_population == label)].sum())
    return value / total


def positive_quantile(values: np.ndarray, quantile: float = 0.8) -> float:
    positive = np.asarray(values, dtype=float)[np.asarray(values) > 0.0]
    if positive.size == 0:
        raise ValueError("A positive escape-score tail is required for exploratory pricing.")
    return float(np.quantile(positive, quantile, method="higher"))


def exploratory_prices(clean_cost: np.ndarray, args: argparse.Namespace) -> dict[str, float]:
    """Build frozen pilot prices from the clean pair without planted labels."""
    prices: dict[str, float] = {}
    if args.method_set in ("all", "balanced"):
        balanced = balanced_ot(clean_cost, epsilon=args.epsilon, threshold=args.solver_tolerance)
        a = balanced.source_marginal
        b = balanced.target_marginal
        f = args.epsilon * balanced.log_source_scaling
        g = args.epsilon * balanced.log_target_scaling
        prices.update({
            "binary_balanced_c": float(max(np.quantile(clean_cost, 0.5), 1e-12)),
            "soft_balanced_c_s": positive_quantile(a * (f - np.dot(f, a))),
            "soft_balanced_c_t": positive_quantile(b * (g - np.dot(g, b))),
        })
    if args.method_set in ("all", "uot"):
        vanilla = unbalanced_ot(
            clean_cost,
            epsilon=args.epsilon,
            lambda_a=args.lambda_a,
            lambda_b=args.lambda_b,
            threshold=args.solver_tolerance,
        )
        mass = float(vanilla.coupling.sum())
        uot_source_score = (args.epsilon + args.lambda_a) * (mass * vanilla.source_marginal - vanilla.source_mass)
        uot_target_score = (args.epsilon + args.lambda_b) * (mass * vanilla.target_marginal - vanilla.target_mass)
        prices.update({
            "binary_uot_c": float(max(np.quantile(clean_cost, 0.5), 1e-12)),
            "soft_uot_c_s": positive_quantile(uot_source_score),
            "soft_uot_c_t": positive_quantile(uot_target_score),
        })
    return prices


def load_prices(path: Path | None, clean_cost: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, float], str]:
    if path is None:
        warnings.warn(
            "No --price-config supplied; using clean-data exploratory prices. "
            "Smoke-test outputs are not calibrated benchmark results.",
            RuntimeWarning,
            stacklevel=2,
        )
        return exploratory_prices(clean_cost, args), "exploratory-clean-positive-tail"
    prices = json.loads(path.read_text(encoding="utf-8"))
    required: set[str] = set()
    if args.method_set in ("all", "balanced"):
        required.update({"binary_balanced_c", "soft_balanced_c_s", "soft_balanced_c_t"})
    if args.method_set in ("all", "uot"):
        required.update({"binary_uot_c", "soft_uot_c_s", "soft_uot_c_t"})
    missing = sorted(required - set(prices))
    if missing:
        raise ValueError(f"Price configuration is missing: {missing}")
    return {key: float(prices[key]) for key in required}, "frozen-price-config"


def method_record(
    name: str,
    fit: object,
    coupling: np.ndarray,
    source_gate: np.ndarray,
    target_gate: np.ndarray,
    elapsed_seconds: float,
    warning: bool,
    status: str,
    *,
    time_basis: str = "thread_cpu_seconds",
    cpu_seconds: float | None = None,
) -> dict[str, object]:
    measured_cpu = elapsed_seconds if cpu_seconds is None else cpu_seconds
    return {
        "method": name,
        "fit": fit,
        "coupling": coupling,
        "source_gate": source_gate,
        "target_gate": target_gate,
        # Keep fit_seconds as a compatibility alias, but its timing basis is
        # now CPU time rather than elapsed wall time.
        "fit_seconds": elapsed_seconds,
        "fit_cpu_seconds": measured_cpu,
        "fit_accelerator_seconds": elapsed_seconds if time_basis == "cuda_synchronized_wall_seconds" else np.nan,
        "fit_time_basis": time_basis,
        "compute_device": getattr(fit, "device", "cpu"),
        "compute_backend": getattr(fit, "backend", "numpy"),
        "numerical_warning": warning,
        "status": status,
    }


def fit_methods(
    cost: np.ndarray,
    prices: dict[str, float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], float, float, float]:
    """Fit independent method families concurrently and retain every timing."""
    inner_cap = 2_000 if args.smoke_test else 20_000
    outer_cap = min(args.outer_cap, 3) if args.smoke_test else args.outer_cap
    soft_cap = min(args.soft_outer_cap, 2) if args.smoke_test else args.soft_outer_cap
    if args.workers <= 0:
        raise ValueError("`--workers` must be a positive integer.")

    def balanced_baseline() -> list[dict[str, object]]:
        start = time.thread_time()
        fit = balanced_ot(cost, epsilon=args.epsilon, threshold=args.solver_tolerance, max_iterations=inner_cap)
        return [method_record("Traditional OT", fit, fit.coupling, np.ones(cost.shape[0], bool), np.ones(cost.shape[1], bool), time.thread_time() - start, not fit.converged, "converged" if fit.converged else "iteration-capped")]

    def uot_baseline() -> list[dict[str, object]]:
        start = time.thread_time()
        fit = unbalanced_ot(cost, epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b, threshold=args.solver_tolerance, max_iterations=inner_cap)
        return [method_record("Vanilla UOT", fit, fit.coupling, np.ones(cost.shape[0], bool), np.ones(cost.shape[1], bool), time.thread_time() - start, not fit.converged, "converged" if fit.converged else "iteration-capped")]

    def balanced_cost_matrix(variant: str, label: str) -> list[dict[str, object]]:
        start = time.thread_time()
        fit = ConfidenceOT(
            backbone="balanced", variant=variant,
            rejection_cost=prices["binary_balanced_c"], epsilon=args.epsilon,
            source_rejection_budget=args.source_budget,
            target_rejection_budget=args.target_budget, tolerance=args.solver_tolerance,
            max_iterations=inner_cap, max_outer_iterations=outer_cap,
            device=getattr(args, "device", "auto"),
            cuda_dtype=getattr(args, "cuda_dtype", "float32"),
            fallback_to_cpu=getattr(args, "cuda_fallback", False),
        ).fit(cost)
        warning = not fit.outer_converged or not fit.inner_converged or fit.cycle_detected
        status = "converged" if fit.outer_converged and fit.inner_converged else "terminal-warning"
        cpu_seconds = time.thread_time() - start
        is_cuda = getattr(fit, "device", "cpu") == "cuda"
        return [method_record(
            label, fit, fit.coupling, fit.source_gate, fit.target_gate,
            fit.fit_seconds if is_cuda else cpu_seconds, warning, status,
            time_basis="cuda_synchronized_wall_seconds" if is_cuda else "thread_cpu_seconds",
            cpu_seconds=cpu_seconds,
        )]

    def balanced_soft() -> list[dict[str, object]]:
        start = time.thread_time()
        multi = multi_start_soft_gate_balanced(
            cost, epsilon=args.epsilon,
            c_s=prices["soft_balanced_c_s"], c_t=prices["soft_balanced_c_t"],
            source_rejection_budget=args.source_budget, target_rejection_budget=args.target_budget,
            random_seeds=() if args.smoke_test else (17,),
            gate_tolerance=args.solver_tolerance, block_tolerance=args.solver_tolerance,
            gap_tolerance=args.solver_tolerance, threshold=args.solver_tolerance,
            max_iterations=inner_cap, max_outer_iterations=soft_cap,
        )
        selected = next((run for run in multi.runs if run.initialization == "escape-score"), multi.runs[0])
        fit = selected.result
        return [method_record("Soft gate / Balanced", fit, fit.coupling, fit.source_gate, fit.target_gate, time.thread_time() - start, fit.status != "numerically-soft-stationary", fit.status)]

    def uot_cost_matrix(variant: str, label: str) -> list[dict[str, object]]:
        start = time.thread_time()
        fit = ConfidenceOT(
            backbone="uot", variant=variant,
            rejection_cost=prices["binary_uot_c"], epsilon=args.epsilon,
            lambda_a=args.lambda_a, lambda_b=args.lambda_b,
            source_rejection_budget=args.source_budget,
            target_rejection_budget=args.target_budget, tolerance=args.solver_tolerance,
            max_iterations=inner_cap, max_outer_iterations=outer_cap,
            device=getattr(args, "device", "auto"),
            cuda_dtype=getattr(args, "cuda_dtype", "float32"),
            fallback_to_cpu=getattr(args, "cuda_fallback", False),
        ).fit(cost)
        warning = not fit.outer_converged or not fit.inner_converged or fit.cycle_detected
        status = "converged" if fit.outer_converged and fit.inner_converged else "terminal-warning"
        cpu_seconds = time.thread_time() - start
        is_cuda = getattr(fit, "device", "cpu") == "cuda"
        return [method_record(
            label, fit, fit.coupling, fit.source_gate, fit.target_gate,
            fit.fit_seconds if is_cuda else cpu_seconds, warning, status,
            time_basis="cuda_synchronized_wall_seconds" if is_cuda else "thread_cpu_seconds",
            cpu_seconds=cpu_seconds,
        )]

    def uot_soft() -> list[dict[str, object]]:
        start = time.thread_time()
        multi = multi_start_soft_gate_uot(
            cost, epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
            c_s=prices["soft_uot_c_s"], c_t=prices["soft_uot_c_t"],
            source_rejection_budget=args.source_budget, target_rejection_budget=args.target_budget,
            random_seeds=() if args.smoke_test else (17,),
            gate_tolerance=args.solver_tolerance, block_tolerance=args.solver_tolerance,
            gap_tolerance=args.solver_tolerance, threshold=args.solver_tolerance,
            max_iterations=inner_cap, max_outer_iterations=soft_cap,
        )
        selected = next((run for run in multi.runs if run.initialization == "escape-score"), multi.runs[0])
        fit = selected.result
        return [method_record("Soft gate / UOT", fit, fit.coupling, fit.source_gate, fit.target_gate, time.thread_time() - start, fit.status != "numerically-soft-stationary", fit.status)]

    skip_soft_gate = bool(getattr(args, "skip_soft_gate", False))
    tasks = []
    if args.method_set in ("all", "balanced"):
        tasks.extend((
            balanced_baseline,
            partial(balanced_cost_matrix, "exact", "Cost-matrix gate / Balanced / Exact (M4-E)"),
            partial(balanced_cost_matrix, "reversible", "Cost-matrix gate / Balanced / Reversible (M4-R)"),
        ))
        if not skip_soft_gate:
            tasks.append(balanced_soft)
    if args.method_set in ("all", "uot"):
        tasks.extend((
            uot_baseline,
            partial(uot_cost_matrix, "exact", "Cost-matrix gate / UOT / Exact (M4-E)"),
            partial(uot_cost_matrix, "reversible", "Cost-matrix gate / UOT / Reversible (M4-R)"),
        ))
        if not skip_soft_gate:
            tasks.append(uot_soft)
    parallel_start = time.perf_counter()
    parallel_cpu_start = time.process_time()
    methods: list[dict[str, object]] = []
    if args.workers == 1:
        for task in tasks:
            methods.extend(task())
    else:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks)), thread_name_prefix="cellot-method") as executor:
            futures = [executor.submit(task) for task in tasks]
            for future in as_completed(futures):
                methods.extend(future.result())
    parallel_wall_seconds = time.perf_counter() - parallel_start
    parallel_cpu_seconds = time.process_time() - parallel_cpu_start
    order = {name: index for index, name in enumerate(METHOD_TAXONOMY)}
    methods.sort(key=lambda record: order[str(record["method"])])
    summed_fit_seconds = float(sum(float(record["fit_seconds"]) for record in methods))
    return methods, parallel_wall_seconds, parallel_cpu_seconds, summed_fit_seconds


def main() -> None:
    workflow_start = time.perf_counter()
    workflow_cpu_start = time.process_time()
    args = parse_args()
    fractions = validate_fractions(args.gene_fractions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_run_logger(args.output_dir, "splatter_gene_perturbation", log_filename="run.log")
    logger.info("parameters | %s", json.dumps(vars(args), default=str, sort_keys=True))
    directory = args.data_root / args.scenario / f"n_{args.n_cells:04d}" / f"rep_{args.replicate:02d}"
    counts, cells, genes = load_pair(directory)
    condition = cells.condition.astype(str).to_numpy()
    source_index = np.flatnonzero(condition == "source")
    target_index = np.flatnonzero(condition == "target")
    source_population = cells.iloc[source_index].population.astype(str).to_numpy()
    target_population = cells.iloc[target_index].population.astype(str).to_numpy()
    source_truth = cells.iloc[source_index].expected_rejection.to_numpy(bool)
    target_truth = cells.iloc[target_index].expected_rejection.to_numpy(bool)
    if source_index.size != args.n_cells or target_index.size != args.n_cells:
        raise ValueError("Loaded Splatter case does not match --n-cells per condition.")

    preprocessing_start = time.perf_counter()
    preprocessing_cpu_start = time.process_time()
    frozen = FrozenJointPCA(n_hvg=args.n_hvg, n_pcs=args.n_pcs, seed=args.seed).fit(counts)
    clean_coordinates = frozen.transform(counts)
    clean_cost = pairwise_distances(clean_coordinates[source_index], clean_coordinates[target_index], metric="sqeuclidean")
    positive = clean_cost[clean_cost > 0.0]
    cost_scale = float(np.median(positive)) if positive.size else 1.0
    clean_cost /= cost_scale
    preprocessing_seconds = time.perf_counter() - preprocessing_start
    preprocessing_cpu_seconds = time.process_time() - preprocessing_cpu_start
    prices, price_status = load_prices(args.price_config, clean_cost, args)
    gene_order = nested_gene_order(counts[target_index], seed=args.seed)
    clean_coupling: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    population_rows: list[dict[str, object]] = []
    intervention_rows: list[dict[str, object]] = []
    dose_preparation_seconds: dict[float, float] = {}
    dose_parallel_wall_seconds: dict[float, float] = {}
    dose_parallel_cpu_seconds: dict[float, float] = {}
    dose_preparation_cpu_seconds: dict[float, float] = {}
    dose_summed_fit_seconds: dict[float, float] = {}

    for fraction in fractions:
        logger.info("dose-start | gene_fraction=%g", fraction)
        perturb_start = time.perf_counter()
        perturb_cpu_start = time.process_time()
        perturbed, up, down = perturb_target_genes(
            counts, target_index, gene_order,
            fraction=fraction, fold_change=args.fold_change,
            seed=args.seed + int(round(10_000 * fraction)),
        )
        coordinates = frozen.transform(perturbed)
        cost = pairwise_distances(coordinates[source_index], coordinates[target_index], metric="sqeuclidean") / cost_scale
        perturb_seconds = time.perf_counter() - perturb_start
        perturb_cpu_seconds = time.process_time() - perturb_cpu_start
        dose_preparation_seconds[fraction] = perturb_seconds
        dose_preparation_cpu_seconds[fraction] = perturb_cpu_seconds
        intervention_rows.extend(
            {"gene_fraction": fraction, "direction": direction, "gene_index": int(index), "gene_id": str(genes[index])}
            for direction, indices in (("up", up), ("down", down)) for index in indices
        )
        method_records, parallel_wall_seconds, parallel_cpu_seconds, summed_fit_seconds = fit_methods(cost, prices, args)
        dose_parallel_wall_seconds[fraction] = parallel_wall_seconds
        dose_parallel_cpu_seconds[fraction] = parallel_cpu_seconds
        dose_summed_fit_seconds[fraction] = summed_fit_seconds
        parallel_speedup = summed_fit_seconds / parallel_wall_seconds if parallel_wall_seconds > 0.0 else float("nan")
        logger.info(
            "dose-methods-complete | gene_fraction=%g | parallel_wall_seconds=%.6f | "
            "summed_fit_seconds=%.6f | fit_parallel_speedup=%.3f",
            fraction, parallel_wall_seconds, summed_fit_seconds, parallel_speedup,
        )
        for record in method_records:
            method = str(record["method"])
            method_family, backbone, algorithm_variant = METHOD_TAXONOMY[method]
            coupling = np.asarray(record["coupling"])
            source_gate = np.asarray(record["source_gate"], dtype=bool)
            target_gate = np.asarray(record["target_gate"], dtype=bool)
            if fraction == 0.0:
                clean_coupling[method] = coupling.copy()
            stability = float(np.sum(np.abs(coupling - clean_coupling[method]))) if method in clean_coupling else float("nan")
            row = {
                "scenario": args.scenario,
                "n_cells_per_condition": args.n_cells,
                "replicate": args.replicate,
                "gene_fraction": fraction,
                "n_perturbed_genes": int(up.size + down.size),
                "fold_change": args.fold_change,
                "method": method,
                "method_family": method_family,
                "backbone": backbone,
                "algorithm_variant": algorithm_variant,
                "status": record["status"],
                "numerical_warning": record["numerical_warning"],
                "terminal_result_retained": True,
                "fit_seconds": record["fit_seconds"],
                "fit_cpu_seconds": record["fit_cpu_seconds"],
                "fit_accelerator_seconds": record["fit_accelerator_seconds"],
                "fit_time_basis": record["fit_time_basis"],
                "compute_device": record["compute_device"],
                "compute_backend": record["compute_backend"],
                "preprocessing_seconds_shared": preprocessing_seconds,
                "preprocessing_cpu_seconds_shared": preprocessing_cpu_seconds,
                "dose_preparation_seconds": perturb_seconds,
                "dose_preparation_cpu_seconds": perturb_cpu_seconds,
                "isolated_workflow_cpu_seconds": preprocessing_cpu_seconds + perturb_cpu_seconds + float(record["fit_cpu_seconds"]),
                "dose_parallel_wall_seconds": parallel_wall_seconds,
                "dose_parallel_cpu_seconds": parallel_cpu_seconds,
                "dose_summed_fit_seconds": summed_fit_seconds,
                "dose_fit_parallel_speedup": parallel_speedup,
                "price_status": price_status,
                "transition_alignment": alignment(coupling, source_population, target_population),
                "coupling_l1_from_clean": stability,
                "source_rejection_rate": float(np.mean(~source_gate)),
                "target_rejection_rate": float(np.mean(~target_gate)),
            }
            row.update(rejection_metrics(source_gate, source_truth, "source"))
            row.update(rejection_metrics(target_gate, target_truth, "target"))
            row["macro_f1"] = 0.5 * (float(row["source_f1"]) + float(row["target_f1"]))
            rows.append(row)
            transition = population_transition(coupling, source_population, target_population)
            for source_label in transition.index:
                for target_label in transition.columns:
                    population_rows.append({
                        "gene_fraction": fraction, "method": method,
                        "source_population": source_label, "target_population": target_label,
                        "transition_probability": float(transition.loc[source_label, target_label]),
                    })
            log_terminal_warning(
                logger, method=method, context=f"fraction={fraction:g}",
                status=str(record["status"]), inner_converged=not bool(record["numerical_warning"]),
            )
        logger.info("dose-complete | gene_fraction=%g", fraction)

    pd.DataFrame(rows).to_csv(args.output_dir / "runs.csv", index=False)
    pd.DataFrame(population_rows).to_csv(args.output_dir / "population_transitions.csv", index=False)
    pd.DataFrame(intervention_rows).to_csv(args.output_dir / "perturbed_genes.csv", index=False)
    run_table = pd.DataFrame(rows)
    total_dose_preparation_seconds = float(sum(dose_preparation_seconds.values()))
    total_dose_preparation_cpu_seconds = float(sum(dose_preparation_cpu_seconds.values()))
    total_parallel_method_wall_seconds = float(sum(dose_parallel_wall_seconds.values()))
    total_parallel_method_cpu_seconds = float(sum(dose_parallel_cpu_seconds.values()))
    summed_all_method_fit_seconds = float(sum(dose_summed_fit_seconds.values()))
    timing_rows = []
    for method, group in run_table.groupby("method", sort=False):
        fit_seconds = float(group.fit_seconds.sum())
        timing_rows.append({
            "method": method,
            "shared_preprocessing_seconds": preprocessing_seconds,
            "shared_preprocessing_cpu_seconds": preprocessing_cpu_seconds,
            "all_dose_preparation_seconds": total_dose_preparation_seconds,
            "all_dose_preparation_cpu_seconds": total_dose_preparation_cpu_seconds,
            "summed_fit_seconds": fit_seconds,
            "isolated_method_workflow_cpu_seconds": preprocessing_cpu_seconds + total_dose_preparation_cpu_seconds + fit_seconds,
            "timing_scope": "hypothetical isolated run: clean preprocessing + every dose preparation + this method's fits",
        })
    pd.DataFrame(timing_rows).to_csv(args.output_dir / "timing_summary.csv", index=False)
    dose_timing_rows = []
    for fraction in fractions:
        method_wall = dose_parallel_wall_seconds[fraction]
        method_sum = dose_summed_fit_seconds[fraction]
        dose_timing_rows.append({
            "gene_fraction": fraction,
            "dose_preparation_seconds": dose_preparation_seconds[fraction],
            "dose_preparation_cpu_seconds": dose_preparation_cpu_seconds[fraction],
            "parallel_method_wall_seconds": method_wall,
            "parallel_method_cpu_seconds": dose_parallel_cpu_seconds[fraction],
            "summed_method_fit_seconds": method_sum,
            "fit_parallel_speedup": method_sum / method_wall if method_wall > 0.0 else float("nan"),
            "dose_end_to_end_wall_seconds": dose_preparation_seconds[fraction] + method_wall,
        })
    pd.DataFrame(dose_timing_rows).to_csv(args.output_dir / "dose_timing.csv", index=False)
    actual_script_wall_seconds = time.perf_counter() - workflow_start
    actual_script_cpu_seconds = time.process_time() - workflow_cpu_start
    backbone_count = 2 if args.method_set == "all" else 1
    parallel_task_count = backbone_count * (3 if args.skip_soft_gate else 4)
    benchmark_timing = {
        "workers": args.workers,
        "parallel_atomic_method_tasks": parallel_task_count,
        "shared_preprocessing_seconds": preprocessing_seconds,
        "shared_preprocessing_cpu_seconds": preprocessing_cpu_seconds,
        "total_dose_preparation_seconds": total_dose_preparation_seconds,
        "total_dose_preparation_cpu_seconds": total_dose_preparation_cpu_seconds,
        "total_parallel_method_wall_seconds": total_parallel_method_wall_seconds,
        "total_parallel_method_cpu_seconds": total_parallel_method_cpu_seconds,
        "summed_all_method_fit_seconds": summed_all_method_fit_seconds,
        "fit_parallel_speedup": (
            summed_all_method_fit_seconds / total_parallel_method_wall_seconds
            if total_parallel_method_wall_seconds > 0.0 else None
        ),
        "accounted_end_to_end_wall_seconds": (
            preprocessing_seconds + total_dose_preparation_seconds + total_parallel_method_wall_seconds
        ),
        "actual_script_wall_seconds": actual_script_wall_seconds,
        "actual_script_cpu_seconds": actual_script_cpu_seconds,
        "primary_timing_metric": "actual_script_cpu_seconds",
        "timing_note": (
            "actual_script_cpu_seconds is the primary compute-time metric; wall time is retained "
            "only to describe parallel scheduling and user-visible elapsed time"
        ),
    }
    (args.output_dir / "benchmark_timing.json").write_text(
        json.dumps(benchmark_timing, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "scenario": args.scenario,
        "source_data": str(directory.resolve()),
        "n_cells_per_condition": args.n_cells,
        "replicate": args.replicate,
        "gene_fractions": fractions,
        "nested_gene_sets": True,
        "intervention": "target-only signed count perturbation: Poisson up / binomial down",
        "fold_change": args.fold_change,
        "frozen_preprocessing": "clean HVG + StandardScaler + joint PCA + clean median-positive cost scale",
        "n_hvg": args.n_hvg,
        "n_pcs": args.n_pcs,
        "hvg_indices": frozen.hvg.tolist() if frozen.hvg is not None else [],
        "cost_scale": cost_scale,
        "prices": prices,
        "price_status": price_status,
        "smoke_test": args.smoke_test,
        "workers": args.workers,
        "parallel_atomic_method_tasks": parallel_task_count,
        "script_wall_seconds": actual_script_wall_seconds,
        "actual_cpu_seconds": actual_script_cpu_seconds,
        "primary_timing_metric": "actual_cpu_seconds",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "run-complete | rows=%d | smoke_test=%s | actual_script_cpu_seconds=%.6f | actual_script_wall_seconds=%.6f | "
        "summed_all_method_fit_seconds=%.6f | fit_parallel_speedup=%s",
        len(rows), args.smoke_test, actual_script_cpu_seconds, actual_script_wall_seconds, summed_all_method_fit_seconds,
        f"{benchmark_timing['fit_parallel_speedup']:.3f}" if benchmark_timing["fit_parallel_speedup"] is not None else "nan",
    )


if __name__ == "__main__":
    run_with_exception_logging(main, "splatter_gene_perturbation")
