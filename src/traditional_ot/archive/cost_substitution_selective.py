"""Archived cost-substitution selective OT implementation.

This is the pre-support-restriction formulation retained only for historical
benchmark reproducibility.  New balanced and unbalanced experiments must use
``traditional_ot.support_restricted.support_restricted_ot``.

This module is the implementation counterpart of the frozen Selective OT
specification.  The binary source/target gate layer and its Gauss--Seidel
outer loop live here exactly once.  Backbone adapters only solve the
continuous transport problem for fixed gates.

The public :func:`selective_ot` function is intentionally strict.  A missing
backbone-specific parameter, an irrelevant parameter, an infeasible initial
gate, or an unsupported algorithmic variant raises instead of silently
selecting a different mathematical problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from traditional_ot.balanced import BalancedOTResult
from traditional_ot.bidirectional import (
    GateUpdateDiagnostics,
    solve_fixed_bidirectional_uot,
)
from traditional_ot.bidirectional_balanced import (
    solve_fixed_bidirectional_balanced_ot,
)
from traditional_ot.entropic_partial import (
    EntropicPartialOTResult,
    solve_fixed_confidence_filtered_partial_ot,
)
from traditional_ot.icpot import ICPOTResult, intent_controlled_partial_ot
from traditional_ot.unbalanced import UOTResult, _cost_matrix


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Backbone = Literal["balanced", "unbalanced", "partial", "entropic_partial"]
SelectiveVariant = Literal["exact", "reversible"]
GateBudgetMode = Literal["inequality", "equality"]
SolverStatus = Literal["converged", "cycle_detected", "iteration_capped"]


class SelectiveOTError(RuntimeError):
    """Base class for unified Selective OT failures."""


class SelectiveInnerSolverError(SelectiveOTError):
    """Raised when a fixed-gate transport solve fails to converge."""


class UnsupportedSelectiveVariantError(SelectiveOTError):
    """Raised when a backbone does not define the requested gate variant."""


@dataclass(frozen=True)
class FixedGateSelectiveOTResult:
    """Backbone-normalized view of one fixed-gate transport solution."""

    backbone: Backbone
    coupling: FloatArray
    transition_probability: FloatArray
    reverse_transition_probability: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    source_marginal: FloatArray
    target_marginal: FloatArray
    source_gate: BoolArray
    target_gate: BoolArray
    optimization_cost: FloatArray
    scoring_cost: FloatArray
    objective: float
    converged: bool
    n_iterations: int
    native_result: BalancedOTResult | UOTResult | ICPOTResult | EntropicPartialOTResult
    warm_start: tuple[FloatArray, FloatArray] | None


@dataclass(frozen=True)
class SelectiveOTResult:
    """Unified result returned by the shared selective outer loop."""

    backbone: Backbone
    variant: SelectiveVariant
    gate_budget_mode: GateBudgetMode
    coupling: FloatArray
    transition_probability: FloatArray
    reverse_transition_probability: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    source_marginal: FloatArray
    target_marginal: FloatArray
    source_gate: BoolArray
    target_gate: BoolArray
    source_raw_gate: BoolArray
    target_raw_gate: BoolArray
    source_conditional_loss: FloatArray
    target_conditional_loss: FloatArray
    source_gate_score: FloatArray
    target_gate_score: FloatArray
    source_gate_coefficient: FloatArray
    target_gate_coefficient: FloatArray
    source_partner_mass: FloatArray
    target_partner_mass: FloatArray
    source_gate_history: tuple[BoolArray, ...]
    target_gate_history: tuple[BoolArray, ...]
    objective: float
    objective_history: tuple[float, ...]
    objective_stage_history: tuple[str, ...]
    rejection_cost: float
    source_rejection_budget: float
    target_rejection_budget: float
    source_min_accepted: int
    target_min_accepted: int
    source_budget_binding: bool
    target_budget_binding: bool
    source_terminal_gate_consistent: bool
    target_terminal_gate_consistent: bool
    inner_converged: bool
    outer_converged: bool
    cycle_detected: bool
    cycle_length: int
    status: SolverStatus
    initialization: str
    n_outer_iterations: int
    n_transport_solves: int
    total_inner_iterations: int
    native_result: BalancedOTResult | UOTResult | ICPOTResult | EntropicPartialOTResult
    lp_selection_rule: str | None

    @property
    def source_acceptance(self) -> float:
        return float(np.mean(self.source_gate))

    @property
    def target_acceptance(self) -> float:
        return float(np.mean(self.target_gate))

    @property
    def source_raw_acceptance(self) -> float:
        return float(np.mean(self.source_raw_gate))

    @property
    def target_raw_acceptance(self) -> float:
        return float(np.mean(self.target_raw_gate))


@dataclass(frozen=True)
class TwoStageSelectiveOTResult:
    """Native inequality ranking, exact-coverage projection, then one re-solve."""

    native_result: SelectiveOTResult
    fixed_gate_result: FixedGateSelectiveOTResult
    source_gate: BoolArray
    target_gate: BoolArray


RefitMarginalMode = Literal["submeasure", "renormalized"]


@dataclass(frozen=True)
class SelectiveRefitResult:
    """Post-selection refit embedded back into the original index space."""

    backbone: Backbone
    marginal_mode: RefitMarginalMode
    fixed_gate_result: FixedGateSelectiveOTResult
    coupling: FloatArray
    source_gate: BoolArray
    target_gate: BoolArray
    source_indices: NDArray[np.int64]
    target_indices: NDArray[np.int64]


@dataclass(frozen=True)
class PosthocSelectiveOTResult:
    """Ungated ranking with exact matched coverage and an optional refit."""

    ungated_result: FixedGateSelectiveOTResult
    source_gate: BoolArray
    target_gate: BoolArray
    source_gate_coefficient: FloatArray
    target_gate_coefficient: FloatArray
    refit_result: SelectiveRefitResult | None


@dataclass(frozen=True)
class SelectiveCalibrationResult:
    """Two-sided geometric-null calibration of one backbone's rejection cost."""

    backbone: Backbone
    rejection_cost: float
    candidate_costs: FloatArray
    mean_source_raw_acceptance: FloatArray
    mean_target_raw_acceptance: FloatArray
    source_monotone: bool
    target_monotone: bool
    feasible: BoolArray
    source_null_replicates: int
    target_null_replicates: int
    observed_result: SelectiveOTResult


def _positive_float(value: object, *, name: str, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`{name}` must be a finite float.") from error
    invalid = result < 0.0 if allow_zero else result <= 0.0
    if not np.isfinite(result) or invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"`{name}` must be {qualifier} and finite.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"`{name}` must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return result


def _budget(value: object, *, name: str) -> float:
    result = _positive_float(value, name=name, allow_zero=True)
    if result >= 1.0:
        raise ValueError(f"`{name}` must be in [0, 1).")
    return result


def _coverage_floor(n: int, rejection_budget: float) -> int:
    return max(1, min(n, int(ceil((1.0 - rejection_budget) * n))))


def _gate(
    value: ArrayLike | None,
    *,
    n: int,
    name: str,
    minimum: int,
    exact_count: bool,
) -> BoolArray:
    if value is None:
        result = np.ones(n, dtype=bool)
    else:
        array = np.asarray(value)
        if array.shape != (n,) or not np.all(np.isin(array, (0, 1, False, True))):
            raise ValueError(f"`{name}` must have shape ({n},) and contain only booleans.")
        result = array.astype(bool, copy=True)
    count = int(np.count_nonzero(result))
    if exact_count and count != minimum:
        raise ValueError(f"`{name}` must accept exactly {minimum} endpoints in equality mode.")
    if not exact_count and count < minimum:
        raise ValueError(f"`{name}` must accept at least {minimum} endpoints.")
    return result


def selective_filtered_cost(
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
    *,
    rejection_cost: float,
) -> FloatArray:
    """Return the shared rank-one gated cost matrix."""
    cost = _cost_matrix(cost_matrix)
    source = np.asarray(source_gate, dtype=bool)
    target = np.asarray(target_gate, dtype=bool)
    if source.shape != (cost.shape[0],) or target.shape != (cost.shape[1],):
        raise ValueError("Gate shapes must match the source and target dimensions.")
    c = _positive_float(rejection_cost, name="rejection_cost")
    return np.where(source[:, None] & target[None, :], cost, c)


def gate_coefficients(
    coupling: ArrayLike,
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
    *,
    rejection_cost: float,
) -> tuple[FloatArray, FloatArray]:
    """Compute the exact source and target gate-block coefficients."""
    cost = _cost_matrix(cost_matrix)
    plan = np.asarray(coupling, dtype=np.float64)
    source = np.asarray(source_gate, dtype=bool)
    target = np.asarray(target_gate, dtype=bool)
    if plan.shape != cost.shape or not np.all(np.isfinite(plan)) or np.any(plan < 0.0):
        raise ValueError("`coupling` must be finite, non-negative, and match `cost_matrix`.")
    if source.shape != (cost.shape[0],) or target.shape != (cost.shape[1],):
        raise ValueError("Gate shapes must match the coupling dimensions.")
    c = _positive_float(rejection_cost, name="rejection_cost")
    return (
        np.sum(plan * target[None, :] * (cost - c), axis=1),
        np.sum(plan * source[:, None] * (cost - c), axis=0),
    )


def partner_restricted_statistics(
    coupling: ArrayLike,
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return partner-restricted source/target losses and transported masses."""
    cost = _cost_matrix(cost_matrix)
    plan = np.asarray(coupling, dtype=np.float64)
    source = np.asarray(source_gate, dtype=bool)
    target = np.asarray(target_gate, dtype=bool)
    if plan.shape != cost.shape or source.shape != (cost.shape[0],) or target.shape != (cost.shape[1],):
        raise ValueError("Coupling and gate shapes must match `cost_matrix`.")
    weighted = plan * cost
    source_mass = plan @ target.astype(np.float64)
    target_mass = plan.T @ source.astype(np.float64)
    source_loss = np.full(cost.shape[0], np.nan, dtype=np.float64)
    target_loss = np.full(cost.shape[1], np.nan, dtype=np.float64)
    np.divide(weighted @ target.astype(np.float64), source_mass, out=source_loss, where=source_mass > 0.0)
    np.divide(weighted.T @ source.astype(np.float64), target_mass, out=target_loss, where=target_mass > 0.0)
    return source_loss, target_loss, source_mass, target_mass


def budgeted_gate_update(
    coefficients: ArrayLike,
    current_gate: ArrayLike,
    *,
    n_accepted: int,
    mode: GateBudgetMode,
    tolerance: float = 0.0,
    tolerance_scale: ArrayLike | None = None,
) -> GateUpdateDiagnostics:
    """Shared deterministic gate minimizer for inequality or equality coverage."""
    score = np.asarray(coefficients, dtype=np.float64)
    old = np.asarray(current_gate)
    if score.ndim != 1 or score.size == 0 or not np.all(np.isfinite(score)):
        raise ValueError("`coefficients` must be a non-empty finite 1D array.")
    if old.shape != score.shape or not np.all(np.isin(old, (0, 1, False, True))):
        raise ValueError("`current_gate` must be a boolean vector matching `coefficients`.")
    old = old.astype(bool, copy=True)
    accepted = _positive_integer(n_accepted, name="n_accepted")
    if accepted > score.size:
        raise ValueError("`n_accepted` cannot exceed the number of endpoints.")
    if mode not in ("inequality", "equality"):
        raise ValueError("`mode` must be 'inequality' or 'equality'.")
    tau = _positive_float(tolerance, name="tolerance", allow_zero=True)
    if tolerance_scale is None:
        scale = np.ones(score.size, dtype=np.float64)
    else:
        scale = np.asarray(tolerance_scale, dtype=np.float64)
        if scale.shape != score.shape or not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
            raise ValueError("`tolerance_scale` must be finite, non-negative, and match the coefficients.")
    boundary = tau * scale
    negative = score < -boundary
    positive = score > boundary
    tie = ~(negative | positive)
    ranking = score.copy()
    ranking[tie] = 0.0

    if mode == "equality":
        order = sorted(range(score.size), key=lambda k: (ranking[k], -int(old[k]), k))
        gate = np.zeros(score.size, dtype=bool)
        gate[np.asarray(order[:accepted], dtype=np.int64)] = True
        tie_fill_count = int(sum(tie[k] and not old[k] for k in order[:accepted]))
        return GateUpdateDiagnostics(
            gate=gate,
            tie_count=int(np.count_nonzero(tie)),
            tie_fill=tie_fill_count > 0,
            constraint_active=True,
            approximate=tau > 0.0,
            min_accepted=accepted,
            accepted_before_projection=int(np.count_nonzero(negative)),
            tie_fill_count=tie_fill_count,
            forced_acceptance_count=max(0, accepted - int(np.count_nonzero(ranking <= 0.0))),
        )

    gate = np.zeros(score.size, dtype=bool)
    gate[negative] = True
    gate[tie] = old[tie]
    before = int(np.count_nonzero(gate))
    tie_fill_count = 0
    forced_count = 0
    if before < accepted:
        candidates = np.flatnonzero(~gate)
        order = sorted(candidates.tolist(), key=lambda k: (score[k], -int(old[k]), k))
        chosen = np.asarray(order[: accepted - before], dtype=np.int64)
        if chosen.size:
            tie_fill_count = int(np.count_nonzero(tie[chosen]))
            forced_count = int(np.count_nonzero(positive[chosen]))
            gate[chosen] = True
    return GateUpdateDiagnostics(
        gate=gate,
        tie_count=int(np.count_nonzero(tie)),
        tie_fill=tie_fill_count > 0,
        constraint_active=forced_count > 0,
        approximate=tau > 0.0,
        min_accepted=accepted,
        accepted_before_projection=before,
        tie_fill_count=tie_fill_count,
        forced_acceptance_count=forced_count,
    )


def _safe_transitions(coupling: FloatArray) -> tuple[FloatArray, FloatArray]:
    source_mass = coupling.sum(axis=1)
    target_mass = coupling.sum(axis=0)
    forward = np.zeros_like(coupling)
    reverse = np.zeros_like(coupling)
    np.divide(coupling, source_mass[:, None], out=forward, where=source_mass[:, None] > 0.0)
    np.divide(coupling, target_mass[None, :], out=reverse, where=target_mass[None, :] > 0.0)
    return forward, reverse


def _validate_backbone_parameters(
    backbone: Backbone,
    *,
    epsilon: float | None,
    lambda_a: float | None,
    lambda_b: float | None,
    source_unmatched_cost: ArrayLike | None,
    target_unmatched_cost: ArrayLike | None,
) -> None:
    if backbone not in ("balanced", "unbalanced", "partial", "entropic_partial"):
        raise ValueError("`backbone` must be 'balanced', 'unbalanced', 'partial', or 'entropic_partial'.")
    if backbone in ("balanced", "unbalanced", "entropic_partial") and epsilon is None:
        raise ValueError(f"`epsilon` is required for the {backbone!r} backbone.")
    if backbone == "partial" and epsilon is not None:
        raise ValueError("`epsilon` must be omitted for exact partial OT.")
    if backbone == "unbalanced":
        if lambda_a is None or lambda_b is None:
            raise ValueError("`lambda_a` and `lambda_b` are required for unbalanced OT.")
    elif lambda_a is not None or lambda_b is not None:
        raise ValueError("`lambda_a` and `lambda_b` are only valid for unbalanced OT.")
    needs_unmatched = backbone in ("partial", "entropic_partial")
    if needs_unmatched and (source_unmatched_cost is None or target_unmatched_cost is None):
        raise ValueError(f"Both unmatched-cost arrays are required for the {backbone!r} backbone.")
    if not needs_unmatched and (source_unmatched_cost is not None or target_unmatched_cost is not None):
        raise ValueError("Unmatched-cost arrays are only valid for partial OT backbones.")


def solve_fixed_gate_selective_ot(
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
    *,
    backbone: Backbone,
    rejection_cost: float,
    epsilon: float | None = None,
    lambda_a: float | None = None,
    lambda_b: float | None = None,
    source_unmatched_cost: ArrayLike | None = None,
    target_unmatched_cost: ArrayLike | None = None,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    warm_start: tuple[ArrayLike, ArrayLike] | None = None,
    solver_options: dict[str, object] | None = None,
    normalize_weights: bool = True,
) -> FixedGateSelectiveOTResult:
    """Solve one of the four backbones for a fixed pair of non-empty gates."""
    cost = _cost_matrix(cost_matrix)
    c = _positive_float(rejection_cost, name="rejection_cost")
    tolerance = _positive_float(threshold, name="threshold")
    iterations = _positive_integer(max_iterations, name="max_iterations")
    _validate_backbone_parameters(
        backbone,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        source_unmatched_cost=source_unmatched_cost,
        target_unmatched_cost=target_unmatched_cost,
    )
    source = _gate(source_gate, n=cost.shape[0], name="source_gate", minimum=1, exact_count=False)
    target = _gate(target_gate, n=cost.shape[1], name="target_gate", minimum=1, exact_count=False)
    optimization_cost = selective_filtered_cost(cost, source, target, rejection_cost=c)

    if backbone == "balanced":
        assert epsilon is not None
        native = solve_fixed_bidirectional_balanced_ot(
            cost,
            source,
            target,
            rejection_cost=c,
            epsilon=epsilon,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=tolerance,
            max_iterations=iterations,
            warm_start=warm_start,
            normalize_weights=normalize_weights,
        )
        converged = native.converged
        n_iterations = native.n_iterations
        next_warm = (native.log_source_scaling.copy(), native.log_target_scaling.copy())
    elif backbone == "unbalanced":
        assert epsilon is not None and lambda_a is not None and lambda_b is not None
        native = solve_fixed_bidirectional_uot(
            cost,
            source,
            target,
            rejection_cost=c,
            epsilon=epsilon,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=tolerance,
            max_iterations=iterations,
            warm_start=warm_start,
            normalize_weights=normalize_weights,
        )
        converged = native.converged
        n_iterations = native.n_iterations
        next_warm = (native.log_source_scaling.copy(), native.log_target_scaling.copy())
    elif backbone == "partial":
        if warm_start is not None:
            raise ValueError("Exact partial OT does not support `warm_start`.")
        assert source_unmatched_cost is not None and target_unmatched_cost is not None
        native = intent_controlled_partial_ot(
            optimization_cost,
            source_unmatched_cost=source_unmatched_cost,
            target_unmatched_cost=target_unmatched_cost,
            source_weights=source_weights,
            target_weights=target_weights,
            solver_options=solver_options,
            normalize_weights=normalize_weights,
        )
        converged = native.success
        n_iterations = native.n_iterations
        next_warm = None
    else:
        assert epsilon is not None
        assert source_unmatched_cost is not None and target_unmatched_cost is not None
        native = solve_fixed_confidence_filtered_partial_ot(
            cost,
            source,
            target,
            rejection_cost=c,
            source_unmatched_cost=source_unmatched_cost,
            target_unmatched_cost=target_unmatched_cost,
            epsilon=epsilon,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=tolerance,
            max_iterations=iterations,
            warm_start=warm_start,
            normalize_weights=normalize_weights,
        )
        converged = native.converged
        n_iterations = native.n_iterations
        next_warm = (native.log_source_scaling.copy(), native.log_target_scaling.copy())

    if not converged:
        raise SelectiveInnerSolverError(
            f"The {backbone} fixed-gate solver did not converge after {n_iterations} iterations."
        )
    coupling = np.asarray(native.coupling, dtype=np.float64)
    forward, reverse = _safe_transitions(coupling)
    return FixedGateSelectiveOTResult(
        backbone=backbone,
        coupling=coupling,
        transition_probability=forward,
        reverse_transition_probability=reverse,
        source_mass=coupling.sum(axis=1),
        target_mass=coupling.sum(axis=0),
        source_marginal=np.asarray(native.source_marginal, dtype=np.float64),
        target_marginal=np.asarray(native.target_marginal, dtype=np.float64),
        source_gate=source,
        target_gate=target,
        optimization_cost=optimization_cost,
        scoring_cost=cost,
        objective=float(native.objective),
        converged=True,
        n_iterations=n_iterations,
        native_result=native,
        warm_start=next_warm,
    )


def _counterfactual_losses(
    fixed: FixedGateSelectiveOTResult,
    cost: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    native = fixed.native_result
    if fixed.backbone == "partial":
        raise UnsupportedSelectiveVariantError(
            "The reversible variant is undefined for exact CF-POT; use variant='exact'."
        )
    assert isinstance(native, (BalancedOTResult, UOTResult, EntropicPartialOTResult))
    if isinstance(native, EntropicPartialOTResult):
        source_log = (
            np.log(native.target_marginal)[None, :]
            + native.log_target_scaling[None, :]
            + native.target_unmatched_cost[None, :] / native.epsilon
            - cost / native.epsilon
        )
        target_log = (
            np.log(native.source_marginal)[:, None]
            + native.log_source_scaling[:, None]
            + native.source_unmatched_cost[:, None] / native.epsilon
            - cost / native.epsilon
        )
    else:
        source_log = (
            np.log(native.target_marginal)[None, :]
            + native.log_target_scaling[None, :]
            - cost / native.epsilon
        )
        target_log = (
            np.log(native.source_marginal)[:, None]
            + native.log_source_scaling[:, None]
            - cost / native.epsilon
        )

    def normalize(log_values: FloatArray, axis: int) -> FloatArray:
        maximum = np.max(log_values, axis=axis, keepdims=True)
        weights = np.exp(log_values - maximum)
        return weights / weights.sum(axis=axis, keepdims=True)

    source_probability = normalize(source_log, 1)
    target_probability = normalize(target_log, 0)
    return np.sum(source_probability * cost, axis=1), np.sum(target_probability * cost, axis=0)


def _fixed_coupling_objective(
    fixed: FixedGateSelectiveOTResult,
    source_gate: BoolArray,
    target_gate: BoolArray,
    rejection_cost: float,
) -> float:
    new_cost = selective_filtered_cost(
        fixed.scoring_cost, source_gate, target_gate, rejection_cost=rejection_cost
    )
    base = fixed.objective - float(np.sum(fixed.coupling * fixed.optimization_cost))
    return base + float(np.sum(fixed.coupling * new_cost))


def _solver_kwargs(
    *,
    backbone: Backbone,
    rejection_cost: float,
    epsilon: float | None,
    lambda_a: float | None,
    lambda_b: float | None,
    source_unmatched_cost: ArrayLike | None,
    target_unmatched_cost: ArrayLike | None,
    source_weights: ArrayLike | None,
    target_weights: ArrayLike | None,
    threshold: float,
    max_iterations: int,
    solver_options: dict[str, object] | None,
    normalize_weights: bool = True,
) -> dict[str, object]:
    return {
        "backbone": backbone,
        "rejection_cost": rejection_cost,
        "epsilon": epsilon,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "source_unmatched_cost": source_unmatched_cost,
        "target_unmatched_cost": target_unmatched_cost,
        "source_weights": source_weights,
        "target_weights": target_weights,
        "threshold": threshold,
        "max_iterations": max_iterations,
        "solver_options": solver_options,
        "normalize_weights": normalize_weights,
    }


def selective_ot(
    cost_matrix: ArrayLike,
    *,
    backbone: Backbone,
    rejection_cost: float,
    source_rejection_budget: float = 0.10,
    target_rejection_budget: float = 0.10,
    variant: SelectiveVariant = "exact",
    gate_budget_mode: GateBudgetMode = "inequality",
    epsilon: float | None = None,
    lambda_a: float | None = None,
    lambda_b: float | None = None,
    source_unmatched_cost: ArrayLike | None = None,
    target_unmatched_cost: ArrayLike | None = None,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    initial_source_gate: ArrayLike | None = None,
    initial_target_gate: ArrayLike | None = None,
    gate_tolerance: float = 0.0,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    max_outer_iterations: int = 100,
    warm_start: bool = True,
    solver_options: dict[str, object] | None = None,
) -> SelectiveOTResult:
    """Run the shared source/target selective outer loop.

    ``variant='exact'`` implements the theorem-bearing sequential block
    minimization and therefore requires ``gate_tolerance=0``.
    ``variant='reversible'`` is a separately reported heuristic and is
    unavailable for the unregularized partial backbone.
    """
    cost = _cost_matrix(cost_matrix)
    c = _positive_float(rejection_cost, name="rejection_cost")
    source_budget = _budget(source_rejection_budget, name="source_rejection_budget")
    target_budget = _budget(target_rejection_budget, name="target_rejection_budget")
    source_min = _coverage_floor(cost.shape[0], source_budget)
    target_min = _coverage_floor(cost.shape[1], target_budget)
    tau = _positive_float(gate_tolerance, name="gate_tolerance", allow_zero=True)
    outer_limit = _positive_integer(max_outer_iterations, name="max_outer_iterations")
    if variant not in ("exact", "reversible"):
        raise ValueError("`variant` must be 'exact' or 'reversible'.")
    if gate_budget_mode not in ("inequality", "equality"):
        raise ValueError("`gate_budget_mode` must be 'inequality' or 'equality'.")
    if variant == "exact" and tau != 0.0:
        raise ValueError("The exact variant requires `gate_tolerance=0`.")
    if variant == "reversible" and backbone == "partial":
        raise UnsupportedSelectiveVariantError(
            "The reversible variant is defined only for entropic backbones; exact CF-POT is unsupported."
        )
    if not isinstance(warm_start, (bool, np.bool_)):
        raise ValueError("`warm_start` must be boolean.")
    _validate_backbone_parameters(
        backbone,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        source_unmatched_cost=source_unmatched_cost,
        target_unmatched_cost=target_unmatched_cost,
    )
    kwargs = _solver_kwargs(
        backbone=backbone,
        rejection_cost=c,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        source_unmatched_cost=source_unmatched_cost,
        target_unmatched_cost=target_unmatched_cost,
        source_weights=source_weights,
        target_weights=target_weights,
        threshold=threshold,
        max_iterations=max_iterations,
        solver_options=solver_options,
        normalize_weights=True,
    )

    initialization = "provided"
    equality = gate_budget_mode == "equality"
    if equality and initial_source_gate is None and initial_target_gate is None:
        all_source = np.ones(cost.shape[0], dtype=bool)
        all_target = np.ones(cost.shape[1], dtype=bool)
        ungated = solve_fixed_gate_selective_ot(cost, all_source, all_target, **kwargs)
        source_coefficient, target_coefficient = gate_coefficients(
            ungated.coupling, cost, all_source, all_target, rejection_cost=c
        )
        source_gate = budgeted_gate_update(
            source_coefficient, all_source, n_accepted=source_min, mode="equality"
        ).gate
        target_gate = budgeted_gate_update(
            target_coefficient, all_target, n_accepted=target_min, mode="equality"
        ).gate
        initialization = "ungated_projection"
    else:
        source_gate = _gate(
            initial_source_gate,
            n=cost.shape[0],
            name="initial_source_gate",
            minimum=source_min,
            exact_count=equality,
        )
        target_gate = _gate(
            initial_target_gate,
            n=cost.shape[1],
            name="initial_target_gate",
            minimum=target_min,
            exact_count=equality,
        )
        if initial_source_gate is None and initial_target_gate is None:
            initialization = "all_accepted"

    source_history: list[BoolArray] = [source_gate.copy()]
    target_history: list[BoolArray] = [target_gate.copy()]
    objective_history: list[float] = []
    stages: list[str] = []
    seen = {(source_gate.tobytes(), target_gate.tobytes()): 0}
    log_warm: tuple[FloatArray, FloatArray] | None = None
    fixed: FixedGateSelectiveOTResult | None = None
    solved_source: BoolArray | None = None
    solved_target: BoolArray | None = None
    total_inner = 0
    transport_solves = 0
    outer_converged = False
    cycle_detected = False
    cycle_length = 0
    completed_outer = 0

    for outer in range(1, outer_limit + 1):
        completed_outer = outer
        solved_source = source_gate.copy()
        solved_target = target_gate.copy()
        fixed = solve_fixed_gate_selective_ot(
            cost,
            source_gate,
            target_gate,
            warm_start=log_warm if warm_start and backbone != "partial" else None,
            **kwargs,
        )
        transport_solves += 1
        total_inner += fixed.n_iterations
        objective_history.append(fixed.objective)
        stages.append("transport")
        previous_source = source_gate.copy()
        previous_target = target_gate.copy()
        source_loss, _, source_partner, _ = partner_restricted_statistics(
            fixed.coupling, cost, source_gate, target_gate
        )
        if variant == "exact":
            source_coefficient, _ = gate_coefficients(
                fixed.coupling, cost, source_gate, target_gate, rejection_cost=c
            )
        else:
            source_cf, target_cf = _counterfactual_losses(fixed, cost)
            source_coefficient = source_partner * (source_cf - c)
        source_gate = budgeted_gate_update(
            source_coefficient,
            source_gate,
            n_accepted=source_min,
            mode=gate_budget_mode,
            tolerance=tau,
            tolerance_scale=source_partner,
        ).gate
        objective_history.append(_fixed_coupling_objective(fixed, source_gate, target_gate, c))
        stages.append("source_gate")

        _, target_loss, _, target_partner = partner_restricted_statistics(
            fixed.coupling, cost, source_gate, target_gate
        )
        if variant == "exact":
            _, target_coefficient = gate_coefficients(
                fixed.coupling, cost, source_gate, target_gate, rejection_cost=c
            )
        else:
            target_coefficient = target_partner * (target_cf - c)
        target_gate = budgeted_gate_update(
            target_coefficient,
            target_gate,
            n_accepted=target_min,
            mode=gate_budget_mode,
            tolerance=tau,
            tolerance_scale=target_partner,
        ).gate
        objective_history.append(_fixed_coupling_objective(fixed, source_gate, target_gate, c))
        stages.append("target_gate")
        source_history.append(source_gate.copy())
        target_history.append(target_gate.copy())

        if np.array_equal(source_gate, previous_source) and np.array_equal(target_gate, previous_target):
            outer_converged = True
            break
        key = (source_gate.tobytes(), target_gate.tobytes())
        if key in seen:
            cycle_detected = True
            cycle_length = outer - seen[key]
            if variant == "exact":
                raise SelectiveOTError(
                    "The exact gate loop revisited a gate pair; this contradicts strict exact-block descent."
                )
            break
        seen[key] = outer
        if warm_start and fixed.warm_start is not None:
            log_warm = fixed.warm_start

    assert fixed is not None and solved_source is not None and solved_target is not None
    if not np.array_equal(source_gate, solved_source) or not np.array_equal(target_gate, solved_target):
        fixed = solve_fixed_gate_selective_ot(
            cost,
            source_gate,
            target_gate,
            warm_start=fixed.warm_start if warm_start and backbone != "partial" else None,
            **kwargs,
        )
        transport_solves += 1
        total_inner += fixed.n_iterations
        objective_history.append(fixed.objective)
        stages.append("terminal_transport")

    source_loss, target_loss, source_partner, target_partner = partner_restricted_statistics(
        fixed.coupling, cost, source_gate, target_gate
    )
    exact_source, exact_target = gate_coefficients(
        fixed.coupling, cost, source_gate, target_gate, rejection_cost=c
    )
    if variant == "exact":
        source_score = source_loss
        target_score = target_loss
        reported_source = exact_source
        reported_target = exact_target
    else:
        source_score, target_score = _counterfactual_losses(fixed, cost)
        reported_source = source_partner * (source_score - c)
        reported_target = target_partner * (target_score - c)
    source_raw = reported_source < 0.0
    target_raw = reported_target < 0.0
    terminal_source = budgeted_gate_update(
        reported_source,
        source_gate,
        n_accepted=source_min,
        mode=gate_budget_mode,
        tolerance=tau,
        tolerance_scale=source_partner,
    ).gate
    terminal_target = budgeted_gate_update(
        reported_target,
        target_gate,
        n_accepted=target_min,
        mode=gate_budget_mode,
        tolerance=tau,
        tolerance_scale=target_partner,
    ).gate
    status: SolverStatus = (
        "converged" if outer_converged else "cycle_detected" if cycle_detected else "iteration_capped"
    )
    return SelectiveOTResult(
        backbone=backbone,
        variant=variant,
        gate_budget_mode=gate_budget_mode,
        coupling=fixed.coupling,
        transition_probability=fixed.transition_probability,
        reverse_transition_probability=fixed.reverse_transition_probability,
        source_mass=fixed.source_mass,
        target_mass=fixed.target_mass,
        source_marginal=fixed.source_marginal,
        target_marginal=fixed.target_marginal,
        source_gate=source_gate,
        target_gate=target_gate,
        source_raw_gate=source_raw,
        target_raw_gate=target_raw,
        source_conditional_loss=source_loss,
        target_conditional_loss=target_loss,
        source_gate_score=source_score,
        target_gate_score=target_score,
        source_gate_coefficient=reported_source,
        target_gate_coefficient=reported_target,
        source_partner_mass=source_partner,
        target_partner_mass=target_partner,
        source_gate_history=tuple(source_history),
        target_gate_history=tuple(target_history),
        objective=fixed.objective,
        objective_history=tuple(objective_history),
        objective_stage_history=tuple(stages),
        rejection_cost=c,
        source_rejection_budget=source_budget,
        target_rejection_budget=target_budget,
        source_min_accepted=source_min,
        target_min_accepted=target_min,
        source_budget_binding=int(np.count_nonzero(source_gate)) == source_min,
        target_budget_binding=int(np.count_nonzero(target_gate)) == target_min,
        source_terminal_gate_consistent=np.array_equal(terminal_source, source_gate),
        target_terminal_gate_consistent=np.array_equal(terminal_target, target_gate),
        inner_converged=True,
        outer_converged=outer_converged,
        cycle_detected=cycle_detected,
        cycle_length=cycle_length,
        status=status,
        initialization=initialization,
        n_outer_iterations=completed_outer,
        n_transport_solves=transport_solves,
        total_inner_iterations=total_inner,
        native_result=fixed.native_result,
        lp_selection_rule=("scipy.optimize.linprog(method='highs') sparse optimum" if backbone == "partial" else None),
    )


def two_stage_selective_ot(
    cost_matrix: ArrayLike,
    **kwargs: object,
) -> TwoStageSelectiveOTResult:
    """Run native inequality Selective OT, project to exact coverage, and re-solve once."""
    if kwargs.get("gate_budget_mode", "inequality") != "inequality":
        raise ValueError("Two-stage Selective OT requires a native inequality-budget first stage.")
    native = selective_ot(cost_matrix, **kwargs)
    source_gate = budgeted_gate_update(
        native.source_gate_coefficient,
        native.source_gate,
        n_accepted=native.source_min_accepted,
        mode="equality",
    ).gate
    target_gate = budgeted_gate_update(
        native.target_gate_coefficient,
        native.target_gate,
        n_accepted=native.target_min_accepted,
        mode="equality",
    ).gate
    fixed_keys = {
        "backbone",
        "rejection_cost",
        "epsilon",
        "lambda_a",
        "lambda_b",
        "source_unmatched_cost",
        "target_unmatched_cost",
        "source_weights",
        "target_weights",
        "threshold",
        "max_iterations",
        "solver_options",
        "normalize_weights",
    }
    fixed_kwargs = {key: value for key, value in kwargs.items() if key in fixed_keys}
    fixed = solve_fixed_gate_selective_ot(cost_matrix, source_gate, target_gate, **fixed_kwargs)
    return TwoStageSelectiveOTResult(
        native_result=native,
        fixed_gate_result=fixed,
        source_gate=source_gate,
        target_gate=target_gate,
    )


def _base_measure(weights: ArrayLike | None, *, n: int, name: str) -> FloatArray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"`{name}` must have shape ({n},) and be strictly positive and finite.")
    return values / values.sum()


def _subset_parameter(values: ArrayLike | None, indices: NDArray[np.int64]) -> ArrayLike | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return array[indices]


def refit_selective_ot(
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
    *,
    backbone: Backbone,
    marginal_mode: RefitMarginalMode,
    rejection_cost: float,
    epsilon: float | None = None,
    lambda_a: float | None = None,
    lambda_b: float | None = None,
    source_unmatched_cost: ArrayLike | None = None,
    target_unmatched_cost: ArrayLike | None = None,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    solver_options: dict[str, object] | None = None,
) -> SelectiveRefitResult:
    """Refit a backbone on accepted endpoints using submeasure or renormalized weights.

    A balanced submeasure refit exists only when the retained source and target
    masses are equal.  The function raises for an infeasible request rather
    than silently renormalizing it into the other comparator.
    """
    cost = _cost_matrix(cost_matrix)
    if marginal_mode not in ("submeasure", "renormalized"):
        raise ValueError("`marginal_mode` must be 'submeasure' or 'renormalized'.")
    source = np.asarray(source_gate)
    target = np.asarray(target_gate)
    if source.shape != (cost.shape[0],) or target.shape != (cost.shape[1],):
        raise ValueError("Gate shapes must match `cost_matrix`.")
    if not np.all(np.isin(source, (0, 1, False, True))) or not np.all(
        np.isin(target, (0, 1, False, True))
    ):
        raise ValueError("Gates must contain only booleans.")
    source = source.astype(bool)
    target = target.astype(bool)
    if not np.any(source) or not np.any(target):
        raise ValueError("A refit requires at least one accepted endpoint on each side.")
    source_index = np.flatnonzero(source)
    target_index = np.flatnonzero(target)
    source_measure = _base_measure(source_weights, n=cost.shape[0], name="source_weights")
    target_measure = _base_measure(target_weights, n=cost.shape[1], name="target_weights")
    source_subset = source_measure[source_index]
    target_subset = target_measure[target_index]
    normalize = marginal_mode == "renormalized"
    if backbone == "balanced" and not normalize and not np.isclose(
        source_subset.sum(), target_subset.sum(), rtol=1e-10, atol=1e-12
    ):
        raise ValueError(
            "Balanced O2-submeasure is infeasible because retained source and target masses differ. "
            "Report it as infeasible; do not silently use O2-renormalized."
        )
    sub_cost = cost[np.ix_(source_index, target_index)]
    sub_source_unmatched = _subset_parameter(source_unmatched_cost, source_index)
    sub_target_unmatched = _subset_parameter(target_unmatched_cost, target_index)
    fixed = solve_fixed_gate_selective_ot(
        sub_cost,
        np.ones(source_index.size, dtype=bool),
        np.ones(target_index.size, dtype=bool),
        backbone=backbone,
        rejection_cost=rejection_cost,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        source_unmatched_cost=sub_source_unmatched,
        target_unmatched_cost=sub_target_unmatched,
        source_weights=source_subset,
        target_weights=target_subset,
        threshold=threshold,
        max_iterations=max_iterations,
        solver_options=solver_options,
        normalize_weights=normalize,
    )
    embedded = np.zeros_like(cost)
    embedded[np.ix_(source_index, target_index)] = fixed.coupling
    return SelectiveRefitResult(
        backbone=backbone,
        marginal_mode=marginal_mode,
        fixed_gate_result=fixed,
        coupling=embedded,
        source_gate=source.copy(),
        target_gate=target.copy(),
        source_indices=source_index,
        target_indices=target_index,
    )


def posthoc_selective_ot(
    cost_matrix: ArrayLike,
    *,
    backbone: Backbone,
    rejection_cost: float,
    source_rejection_budget: float = 0.10,
    target_rejection_budget: float = 0.10,
    refit_marginal_mode: RefitMarginalMode | None = "renormalized",
    epsilon: float | None = None,
    lambda_a: float | None = None,
    lambda_b: float | None = None,
    source_unmatched_cost: ArrayLike | None = None,
    target_unmatched_cost: ArrayLike | None = None,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    solver_options: dict[str, object] | None = None,
) -> PosthocSelectiveOTResult:
    """Strong matched-coverage baseline: ungated fit, rank, optionally refit."""
    cost = _cost_matrix(cost_matrix)
    source_budget = _budget(source_rejection_budget, name="source_rejection_budget")
    target_budget = _budget(target_rejection_budget, name="target_rejection_budget")
    source_min = _coverage_floor(cost.shape[0], source_budget)
    target_min = _coverage_floor(cost.shape[1], target_budget)
    all_source = np.ones(cost.shape[0], dtype=bool)
    all_target = np.ones(cost.shape[1], dtype=bool)
    fixed = solve_fixed_gate_selective_ot(
        cost,
        all_source,
        all_target,
        backbone=backbone,
        rejection_cost=rejection_cost,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        source_unmatched_cost=source_unmatched_cost,
        target_unmatched_cost=target_unmatched_cost,
        source_weights=source_weights,
        target_weights=target_weights,
        threshold=threshold,
        max_iterations=max_iterations,
        solver_options=solver_options,
    )
    source_coefficient, target_coefficient = gate_coefficients(
        fixed.coupling, cost, all_source, all_target, rejection_cost=rejection_cost
    )
    source_gate = budgeted_gate_update(
        source_coefficient, all_source, n_accepted=source_min, mode="equality"
    ).gate
    target_gate = budgeted_gate_update(
        target_coefficient, all_target, n_accepted=target_min, mode="equality"
    ).gate
    refit = None
    if refit_marginal_mode is not None:
        refit = refit_selective_ot(
            cost,
            source_gate,
            target_gate,
            backbone=backbone,
            marginal_mode=refit_marginal_mode,
            rejection_cost=rejection_cost,
            epsilon=epsilon,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            source_unmatched_cost=source_unmatched_cost,
            target_unmatched_cost=target_unmatched_cost,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=threshold,
            max_iterations=max_iterations,
            solver_options=solver_options,
        )
    return PosthocSelectiveOTResult(
        ungated_result=fixed,
        source_gate=source_gate,
        target_gate=target_gate,
        source_gate_coefficient=source_coefficient,
        target_gate_coefficient=target_coefficient,
        refit_result=refit,
    )


def calibrate_selective_rejection_cost(
    observed_cost: ArrayLike,
    *,
    source_null_costs: list[ArrayLike] | tuple[ArrayLike, ...],
    target_null_costs: list[ArrayLike] | tuple[ArrayLike, ...],
    candidate_costs: ArrayLike,
    maximum_source_raw_acceptance: float,
    maximum_target_raw_acceptance: float,
    backbone: Backbone,
    source_rejection_budget: float = 0.10,
    target_rejection_budget: float = 0.10,
    epsilon: float | None = None,
    lambda_a: float | None = None,
    lambda_b: float | None = None,
    source_unmatched_cost: ArrayLike | None = None,
    target_unmatched_cost: ArrayLike | None = None,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    max_outer_iterations: int = 100,
    solver_options: dict[str, object] | None = None,
) -> SelectiveCalibrationResult:
    """Select the largest cost jointly feasible on side-specific geometric nulls.

    The caller constructs the null geometry.  ``source_null_costs`` must hold
    source identities fixed (for example by perturbing/replacing the target),
    while ``target_null_costs`` must hold target identities fixed.  The
    routine calibrates on raw pre-projection acceptance and never pools sides.
    """
    observed = _cost_matrix(observed_cost)
    if not source_null_costs or not target_null_costs:
        raise ValueError("At least one side-specific null cost is required for each side.")
    source_null = tuple(_cost_matrix(value, name="source_null_cost") for value in source_null_costs)
    target_null = tuple(_cost_matrix(value, name="target_null_cost") for value in target_null_costs)
    if any(value.shape != observed.shape for value in source_null + target_null):
        raise ValueError("Every null cost matrix must match `observed_cost`.")
    candidates = np.asarray(candidate_costs, dtype=np.float64)
    if (
        candidates.ndim != 1
        or candidates.size == 0
        or not np.all(np.isfinite(candidates))
        or np.any(candidates <= 0.0)
    ):
        raise ValueError("`candidate_costs` must be a non-empty positive finite vector.")
    candidates = np.unique(candidates)
    source_limit = _positive_float(
        maximum_source_raw_acceptance,
        name="maximum_source_raw_acceptance",
        allow_zero=True,
    )
    target_limit = _positive_float(
        maximum_target_raw_acceptance,
        name="maximum_target_raw_acceptance",
        allow_zero=True,
    )
    if source_limit > 1.0 or target_limit > 1.0:
        raise ValueError("Maximum raw acceptance rates must lie in [0, 1].")
    common = {
        "backbone": backbone,
        "source_rejection_budget": source_rejection_budget,
        "target_rejection_budget": target_rejection_budget,
        "variant": "exact",
        "gate_budget_mode": "inequality",
        "epsilon": epsilon,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "source_unmatched_cost": source_unmatched_cost,
        "target_unmatched_cost": target_unmatched_cost,
        "source_weights": source_weights,
        "target_weights": target_weights,
        "threshold": threshold,
        "max_iterations": max_iterations,
        "max_outer_iterations": max_outer_iterations,
        "solver_options": solver_options,
    }
    source_curve = []
    target_curve = []
    for candidate in candidates:
        source_results = [
            selective_ot(value, rejection_cost=float(candidate), **common)
            for value in source_null
        ]
        target_results = [
            selective_ot(value, rejection_cost=float(candidate), **common)
            for value in target_null
        ]
        if any(result.status != "converged" for result in source_results + target_results):
            raise SelectiveOTError(
                "A null fit did not converge; calibration cannot use iteration-capped or cyclic fits."
            )
        source_curve.append(float(np.mean([result.source_raw_acceptance for result in source_results])))
        target_curve.append(float(np.mean([result.target_raw_acceptance for result in target_results])))
    source_curve_array = np.asarray(source_curve)
    target_curve_array = np.asarray(target_curve)
    feasible = (source_curve_array <= source_limit) & (target_curve_array <= target_limit)
    if not np.any(feasible):
        raise SelectiveOTError(
            "Geometric-null calibration failed: no candidate cost jointly controls both raw acceptance rates."
        )
    selected = float(candidates[np.flatnonzero(feasible)[-1]])
    observed_result = selective_ot(observed, rejection_cost=selected, **common)
    if observed_result.status != "converged":
        raise SelectiveOTError("The observed fit did not converge at the calibrated rejection cost.")
    return SelectiveCalibrationResult(
        backbone=backbone,
        rejection_cost=selected,
        candidate_costs=candidates,
        mean_source_raw_acceptance=source_curve_array,
        mean_target_raw_acceptance=target_curve_array,
        source_monotone=bool(np.all(np.diff(source_curve_array) >= -1e-12)),
        target_monotone=bool(np.all(np.diff(target_curve_array) >= -1e-12)),
        feasible=feasible,
        source_null_replicates=len(source_null),
        target_null_replicates=len(target_null),
        observed_result=observed_result,
    )
