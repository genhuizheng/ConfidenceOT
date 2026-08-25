"""Coupling-regularized partial OT and exact confidence-filtered ePOT.

The fixed-gate transport block solves

    min <C~, P> + eps KL(P || a b^T) + <k_s, u> + <k_t, v>

subject to ``P 1 + u = a`` and ``P.T 1 + v = b``.  Only the coupling is
regularized.  This is deliberately not the entropy-on-augmented-support
construction and not the variant that also regularizes the slack variables.

The confidence-filtered method alternates the unique fixed-gate transport
block with exact, budgeted Gauss--Seidel gate minimizers.  It implements the
objective in the frozen CF-POT theory note; it is not post-hoc subsampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from traditional_ot.bidirectional import (
    GateUpdateDiagnostics,
    _binary_gate,
    _coverage_floor,
    _rejection_budget,
    constrained_gate_update,
    filtered_cost,
)
from traditional_ot.icpot import _pointwise_cost
from traditional_ot.unbalanced import (
    _conditional_from_log_coupling,
    _cost_matrix,
    _generalized_kl,
    _logsumexp,
    _marginal,
    _positive_finite,
    _positive_integer,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
GateBudgetMode = Literal["inequality", "equality"]
RefitMarginalMode = Literal["submeasure", "renormalized"]


class PartialInnerSolverError(RuntimeError):
    """Raised when the fixed-gate capped-scaling solve does not converge."""


@dataclass(frozen=True)
class EntropicPartialOTResult:
    """Unique solution of coupling-only entropic partial OT."""

    coupling: FloatArray
    log_coupling: FloatArray
    transition_probability: FloatArray
    reverse_transition_probability: FloatArray
    source_conditional_loss: FloatArray
    target_conditional_loss: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    source_unmatched_mass: FloatArray
    target_unmatched_mass: FloatArray
    source_unmatched_fraction: FloatArray
    target_unmatched_fraction: FloatArray
    source_marginal: FloatArray
    target_marginal: FloatArray
    source_unmatched_cost: FloatArray
    target_unmatched_cost: FloatArray
    optimization_cost: FloatArray
    scoring_cost: FloatArray
    admissible_mask: BoolArray
    inadmissible_mass: float
    transported_mass: float
    generalized_kl: float
    objective: float
    reduced_objective: float
    epsilon: float
    log_source_scaling: FloatArray
    log_target_scaling: FloatArray
    converged: bool
    n_iterations: int
    fixed_point_error: float
    max_source_capacity_violation: float
    max_target_capacity_violation: float


@dataclass(frozen=True)
class ConfidenceFilteredEntropicPartialOTResult:
    """Exact bidirectional confidence-filtered ePOT result."""

    coupling: FloatArray
    log_coupling: FloatArray
    transition_probability: FloatArray
    reverse_transition_probability: FloatArray
    source_conditional_loss: FloatArray
    target_conditional_loss: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    source_unmatched_mass: FloatArray
    target_unmatched_mass: FloatArray
    source_unmatched_fraction: FloatArray
    target_unmatched_fraction: FloatArray
    source_marginal: FloatArray
    target_marginal: FloatArray
    source_unmatched_cost: FloatArray
    target_unmatched_cost: FloatArray
    admissible_mask: BoolArray
    inadmissible_mass: float
    transported_mass: float
    source_gate: BoolArray
    target_gate: BoolArray
    source_raw_gate: BoolArray
    target_raw_gate: BoolArray
    source_gate_coefficient: FloatArray
    target_gate_coefficient: FloatArray
    source_effective_reduced_min: FloatArray
    target_effective_reduced_min: FloatArray
    source_signal_upper_bound: FloatArray
    target_signal_upper_bound: FloatArray
    source_log_signal_upper_bound: FloatArray
    target_log_signal_upper_bound: FloatArray
    objective: float
    reduced_objective: float
    objective_history: tuple[float, ...]
    objective_stage_history: tuple[str, ...]
    source_gate_history: tuple[BoolArray, ...]
    target_gate_history: tuple[BoolArray, ...]
    rejection_cost: float
    epsilon: float
    source_rejection_budget: float
    target_rejection_budget: float
    gate_budget_mode: GateBudgetMode
    initialization: str
    source_min_accepted: int
    target_min_accepted: int
    source_budget_binding: bool
    target_budget_binding: bool
    source_boundary_count: int
    target_boundary_count: int
    log_source_scaling: FloatArray
    log_target_scaling: FloatArray
    inner_converged: bool
    outer_converged: bool
    cycle_detected: bool
    cycle_length: int
    n_outer_iterations: int
    n_transport_solves: int
    total_inner_iterations: int
    fixed_point_error: float

    @property
    def source_raw_acceptance(self) -> float:
        return float(np.mean(self.source_raw_gate))

    @property
    def target_raw_acceptance(self) -> float:
        return float(np.mean(self.target_raw_gate))

    @property
    def source_projected_acceptance(self) -> float:
        return float(np.mean(self.source_gate))

    @property
    def target_projected_acceptance(self) -> float:
        return float(np.mean(self.target_gate))


@dataclass(frozen=True)
class EntropicPartialRefitResult:
    """Gate-consistent ePOT refit embedded in the original index space."""

    result: EntropicPartialOTResult
    coupling: FloatArray
    source_gate: BoolArray
    target_gate: BoolArray
    marginal_mode: RefitMarginalMode


@dataclass(frozen=True)
class TwoStageEntropicPartialOTResult:
    """Native eCF-POT ranking followed by one exact-coverage fixed-gate solve."""

    native_result: ConfidenceFilteredEntropicPartialOTResult
    fixed_gate_result: EntropicPartialOTResult
    source_gate: BoolArray
    target_gate: BoolArray


def _reverse_from_log_coupling(
    log_coupling: FloatArray, scoring_cost: FloatArray
) -> tuple[FloatArray, FloatArray]:
    transition, loss = _conditional_from_log_coupling(
        log_coupling.T, scoring_cost.T
    )
    return transition.T, loss


def _validate_warm_start(
    warm_start: tuple[ArrayLike, ArrayLike] | None,
    *,
    n_source: int,
    n_target: int,
) -> tuple[FloatArray, FloatArray]:
    if warm_start is None:
        return np.zeros(n_source, dtype=np.float64), np.zeros(n_target, dtype=np.float64)
    if not isinstance(warm_start, tuple) or len(warm_start) != 2:
        raise ValueError("`warm_start` must be a pair of log-scaling arrays or None.")
    source = np.asarray(warm_start[0], dtype=np.float64)
    target = np.asarray(warm_start[1], dtype=np.float64)
    if source.shape != (n_source,) or target.shape != (n_target,):
        raise ValueError("`warm_start` log scalings have incompatible shapes.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("`warm_start` log scalings must be finite.")
    # Dual inequality multipliers imply alpha,beta <= 1.  Project a merely
    # approximate warm start back to the valid domain rather than propagating
    # small positive numerical errors.
    return np.minimum(source, 0.0).copy(), np.minimum(target, 0.0).copy()


def _measure(
    weights: ArrayLike | None,
    *,
    n: int,
    name: str,
    normalize: bool,
) -> FloatArray:
    """Validate a positive measure, optionally preserving its total mass."""
    if normalize:
        return _marginal(weights, n=n, name=name)
    if weights is None:
        raise ValueError(f"`{name}` is required when `normalize_weights=False`.")
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.shape != (n,):
        raise ValueError(f"`{name}` must have shape ({n},), found {values.shape}.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"`{name}` must contain strictly positive finite values.")
    return values.copy()


def _solve_entropic_partial(
    optimization_cost: FloatArray,
    *,
    scoring_cost: FloatArray,
    source_unmatched_cost: ArrayLike,
    target_unmatched_cost: ArrayLike,
    epsilon: float,
    source_weights: ArrayLike | None,
    target_weights: ArrayLike | None,
    normalize_weights: bool,
    threshold: float,
    max_iterations: int,
    warm_start: tuple[ArrayLike, ArrayLike] | None,
) -> EntropicPartialOTResult:
    n_source, n_target = optimization_cost.shape
    source = _measure(
        source_weights,
        n=n_source,
        name="source_weights",
        normalize=normalize_weights,
    )
    target = _measure(
        target_weights,
        n=n_target,
        name="target_weights",
        normalize=normalize_weights,
    )
    source_cost = _pointwise_cost(
        source_unmatched_cost, n=n_source, name="source_unmatched_cost"
    )
    target_cost = _pointwise_cost(
        target_unmatched_cost, n=n_target, name="target_unmatched_cost"
    )
    reduced_cost = optimization_cost - source_cost[:, None] - target_cost[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        log_kernel = (
            np.log(source)[:, None]
            + np.log(target)[None, :]
            - reduced_cost / epsilon
        )
    if not np.all(np.isfinite(log_kernel)):
        raise FloatingPointError(
            "The ePOT log-kernel is not finite. Increase `epsilon` or rescale "
            "the cost and unmatched-cost arrays."
        )
    log_source, log_target = _validate_warm_start(
        warm_start, n_source=n_source, n_target=n_target
    )
    converged = False
    fixed_error = np.inf
    completed = 0
    for iteration in range(1, max_iterations + 1):
        new_source = np.minimum(
            0.0,
            np.log(source) - _logsumexp(log_kernel + log_target[None, :], axis=1),
        )
        new_target = np.minimum(
            0.0,
            np.log(target)
            - _logsumexp(log_kernel + new_source[:, None], axis=0),
        )
        fixed_error = float(
            max(
                np.max(np.abs(new_source - log_source)),
                np.max(np.abs(new_target - log_target)),
            )
        )
        log_source, log_target = new_source, new_target
        completed = iteration
        if fixed_error <= threshold:
            converged = True
            break

    log_coupling = log_source[:, None] + log_kernel + log_target[None, :]
    with np.errstate(under="ignore"):
        coupling = np.exp(log_coupling)
    source_mass = coupling.sum(axis=1)
    target_mass = coupling.sum(axis=0)
    source_unmatched = np.maximum(source - source_mass, 0.0)
    target_unmatched = np.maximum(target - target_mass, 0.0)
    source_fraction = source_unmatched / source
    target_fraction = target_unmatched / target
    transition, source_loss = _conditional_from_log_coupling(
        log_coupling, scoring_cost
    )
    reverse, target_loss = _reverse_from_log_coupling(log_coupling, scoring_cost)
    reference = source[:, None] * target[None, :]
    generalized_kl = _generalized_kl(coupling, reference)
    objective = float(
        np.sum(coupling * optimization_cost)
        + epsilon * generalized_kl
        + np.dot(source_cost, source_unmatched)
        + np.dot(target_cost, target_unmatched)
    )
    reduced_objective = float(
        np.sum(coupling * reduced_cost) + epsilon * generalized_kl
    )
    admissible = optimization_cost <= source_cost[:, None] + target_cost[None, :]
    return EntropicPartialOTResult(
        coupling=coupling,
        log_coupling=log_coupling,
        transition_probability=transition,
        reverse_transition_probability=reverse,
        source_conditional_loss=source_loss,
        target_conditional_loss=target_loss,
        source_mass=source_mass,
        target_mass=target_mass,
        source_unmatched_mass=source_unmatched,
        target_unmatched_mass=target_unmatched,
        source_unmatched_fraction=source_fraction,
        target_unmatched_fraction=target_fraction,
        source_marginal=source,
        target_marginal=target,
        source_unmatched_cost=source_cost,
        target_unmatched_cost=target_cost,
        optimization_cost=optimization_cost,
        scoring_cost=scoring_cost,
        admissible_mask=admissible,
        inadmissible_mass=float(coupling[~admissible].sum()),
        transported_mass=float(coupling.sum()),
        generalized_kl=generalized_kl,
        objective=objective,
        reduced_objective=reduced_objective,
        epsilon=epsilon,
        log_source_scaling=log_source,
        log_target_scaling=log_target,
        converged=converged,
        n_iterations=completed,
        fixed_point_error=fixed_error,
        max_source_capacity_violation=float(np.max(np.maximum(source_mass - source, 0.0))),
        max_target_capacity_violation=float(np.max(np.maximum(target_mass - target, 0.0))),
    )


def entropic_partial_ot(
    cost_matrix: ArrayLike,
    *,
    source_unmatched_cost: ArrayLike,
    target_unmatched_cost: ArrayLike,
    epsilon: float,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    normalize_weights: bool = True,
    threshold: float = 1e-10,
    max_iterations: int = 10_000,
    warm_start: tuple[ArrayLike, ArrayLike] | None = None,
) -> EntropicPartialOTResult:
    """Solve coupling-only entropic partial OT.

    ``normalize_weights=False`` preserves the supplied total masses and is
    intended for submeasure refits.  Both weight arrays are then required.
    """
    cost = _cost_matrix(cost_matrix)
    eps = _positive_finite(epsilon, name="epsilon")
    tolerance = _positive_finite(threshold, name="threshold")
    iterations = _positive_integer(max_iterations, name="max_iterations")
    if not isinstance(normalize_weights, (bool, np.bool_)):
        raise ValueError("`normalize_weights` must be boolean.")
    return _solve_entropic_partial(
        cost,
        scoring_cost=cost,
        source_unmatched_cost=source_unmatched_cost,
        target_unmatched_cost=target_unmatched_cost,
        epsilon=eps,
        source_weights=source_weights,
        target_weights=target_weights,
        normalize_weights=normalize_weights,
        threshold=tolerance,
        max_iterations=iterations,
        warm_start=warm_start,
    )


def solve_fixed_confidence_filtered_partial_ot(
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
    *,
    rejection_cost: float,
    source_unmatched_cost: ArrayLike,
    target_unmatched_cost: ArrayLike,
    epsilon: float,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-10,
    max_iterations: int = 10_000,
    warm_start: tuple[ArrayLike, ArrayLike] | None = None,
    normalize_weights: bool = True,
) -> EntropicPartialOTResult:
    """Solve the unique eCF-POT transport block for fixed non-empty gates."""
    cost = _cost_matrix(cost_matrix)
    source = _binary_gate(source_gate, n=cost.shape[0], name="source_gate")
    target = _binary_gate(target_gate, n=cost.shape[1], name="target_gate")
    c = _positive_finite(rejection_cost, name="rejection_cost")
    eps = _positive_finite(epsilon, name="epsilon")
    tolerance = _positive_finite(threshold, name="threshold")
    iterations = _positive_integer(max_iterations, name="max_iterations")
    optimization_cost = filtered_cost(
        cost, source, target, rejection_cost=c
    )
    return _solve_entropic_partial(
        optimization_cost,
        scoring_cost=cost,
        source_unmatched_cost=source_unmatched_cost,
        target_unmatched_cost=target_unmatched_cost,
        epsilon=eps,
        source_weights=source_weights,
        target_weights=target_weights,
        normalize_weights=normalize_weights,
        threshold=tolerance,
        max_iterations=iterations,
        warm_start=warm_start,
    )


def refit_entropic_partial_ot(
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
    *,
    source_unmatched_cost: ArrayLike,
    target_unmatched_cost: ArrayLike,
    epsilon: float,
    marginal_mode: RefitMarginalMode,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-10,
    max_iterations: int = 10_000,
) -> EntropicPartialRefitResult:
    """Refit ePOT on a fixed gate using submeasure or renormalized masses."""
    cost = _cost_matrix(cost_matrix)
    source = _binary_gate(source_gate, n=cost.shape[0], name="source_gate")
    target = _binary_gate(target_gate, n=cost.shape[1], name="target_gate")
    if marginal_mode not in ("submeasure", "renormalized"):
        raise ValueError("`marginal_mode` must be 'submeasure' or 'renormalized'.")
    full_source = _marginal(
        source_weights, n=cost.shape[0], name="source_weights"
    )
    full_target = _marginal(
        target_weights, n=cost.shape[1], name="target_weights"
    )
    source_cost = _pointwise_cost(
        source_unmatched_cost,
        n=cost.shape[0],
        name="source_unmatched_cost",
    )
    target_cost = _pointwise_cost(
        target_unmatched_cost,
        n=cost.shape[1],
        name="target_unmatched_cost",
    )
    fit = entropic_partial_ot(
        cost[np.ix_(source, target)],
        source_unmatched_cost=source_cost[source],
        target_unmatched_cost=target_cost[target],
        epsilon=epsilon,
        source_weights=full_source[source],
        target_weights=full_target[target],
        normalize_weights=marginal_mode == "renormalized",
        threshold=threshold,
        max_iterations=max_iterations,
    )
    coupling = np.zeros_like(cost)
    coupling[np.ix_(source, target)] = fit.coupling
    return EntropicPartialRefitResult(
        result=fit,
        coupling=coupling,
        source_gate=source,
        target_gate=target,
        marginal_mode=marginal_mode,
    )


def partial_gate_coefficients(
    coupling: ArrayLike,
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
    *,
    rejection_cost: float,
) -> tuple[FloatArray, FloatArray]:
    """Return exact source and target gate-block coefficients."""
    cost = _cost_matrix(cost_matrix)
    plan = np.asarray(coupling, dtype=np.float64)
    if plan.shape != cost.shape or not np.all(np.isfinite(plan)) or np.any(plan < 0.0):
        raise ValueError("`coupling` must be a finite non-negative matrix matching the cost.")
    source = _binary_gate(
        source_gate, n=cost.shape[0], name="source_gate", allow_empty=True
    )
    target = _binary_gate(
        target_gate, n=cost.shape[1], name="target_gate", allow_empty=True
    )
    c = _positive_finite(rejection_cost, name="rejection_cost")
    source_coefficient = np.sum(
        plan * target[None, :] * (cost - c), axis=1
    )
    target_coefficient = np.sum(
        plan * source[:, None] * (cost - c), axis=0
    )
    return source_coefficient, target_coefficient


def _fixed_coupling_objective(
    result: EntropicPartialOTResult,
    cost: FloatArray,
    source_gate: BoolArray,
    target_gate: BoolArray,
    rejection_cost: float,
) -> float:
    optimization_cost = np.where(
        source_gate[:, None] & target_gate[None, :], cost, rejection_cost
    )
    return float(
        np.sum(result.coupling * optimization_cost)
        + result.epsilon * result.generalized_kl
        + np.dot(result.source_unmatched_cost, result.source_unmatched_mass)
        + np.dot(result.target_unmatched_cost, result.target_unmatched_mass)
    )


def _signal_bounds(
    result: EntropicPartialOTResult,
    cost: FloatArray,
    source_gate: BoolArray,
    target_gate: BoolArray,
    rejection_cost: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    reduced = (
        result.optimization_cost
        - result.source_unmatched_cost[:, None]
        - result.target_unmatched_cost[None, :]
    )
    source_min = np.min(reduced, axis=1)
    target_min = np.min(reduced, axis=0)
    source_m = np.max(
        target_gate[None, :] * np.abs(cost - rejection_cost), axis=1
    )
    target_m = np.max(
        source_gate[:, None] * np.abs(cost - rejection_cost), axis=0
    )
    source_log = np.full(cost.shape[0], -np.inf, dtype=np.float64)
    target_log = np.full(cost.shape[1], -np.inf, dtype=np.float64)
    positive_source = source_m > 0.0
    positive_target = target_m > 0.0
    source_log[positive_source] = (
        np.log(source_m[positive_source])
        + np.log(result.source_marginal[positive_source])
        + np.log(np.sum(result.target_marginal))
        - source_min[positive_source] / result.epsilon
    )
    target_log[positive_target] = (
        np.log(target_m[positive_target])
        + np.log(result.target_marginal[positive_target])
        + np.log(np.sum(result.source_marginal))
        - target_min[positive_target] / result.epsilon
    )
    with np.errstate(over="ignore", under="ignore"):
        source_bound = np.exp(source_log)
        target_bound = np.exp(target_log)
    return source_min, target_min, source_bound, target_bound, source_log, target_log


def exact_cardinality_gate_update(
    coefficients: ArrayLike,
    current_gate: ArrayLike,
    *,
    n_accepted: int,
    tau_loss: float = 0.0,
    partner_mass: ArrayLike | None = None,
    mass_floor: float = 0.0,
) -> GateUpdateDiagnostics:
    """Minimize a gate block subject to exactly ``n_accepted`` accepted endpoints."""
    score = np.asarray(coefficients, dtype=np.float64)
    if score.ndim != 1 or score.size == 0 or not np.all(np.isfinite(score)):
        raise ValueError("`coefficients` must be a non-empty finite 1D array.")
    old = _binary_gate(
        current_gate, n=score.size, name="current_gate", allow_empty=True
    )
    accepted = _positive_integer(n_accepted, name="n_accepted")
    if accepted > score.size:
        raise ValueError("`n_accepted` cannot exceed the number of coefficients.")
    tolerance = _positive_finite(tau_loss, name="tau_loss", allow_zero=True)
    floor = _positive_finite(mass_floor, name="mass_floor", allow_zero=True)
    if partner_mass is None:
        scale = np.ones(score.size, dtype=np.float64)
    else:
        scale = np.asarray(partner_mass, dtype=np.float64)
        if (
            scale.shape != score.shape
            or not np.all(np.isfinite(scale))
            or np.any(scale < 0.0)
        ):
            raise ValueError(
                "`partner_mass` must be a finite non-negative array matching "
                "`coefficients`."
            )
    boundary = tolerance * scale
    on_floor = scale < floor
    tie = (np.abs(score) <= boundary) | on_floor
    ranking_score = score.copy()
    ranking_score[tie] = 0.0
    order = sorted(
        range(score.size),
        key=lambda index: (ranking_score[index], -int(old[index]), index),
    )
    gate = np.zeros(score.size, dtype=bool)
    gate[np.asarray(order[:accepted], dtype=np.int64)] = True
    changed_by_index = sum(
        1
        for index in order[:accepted]
        if tie[index] and not old[index]
    )
    return GateUpdateDiagnostics(
        gate=gate,
        tie_count=int(np.count_nonzero(tie)),
        tie_fill=changed_by_index > 0,
        constraint_active=True,
        approximate=tolerance > 0.0 or floor > 0.0,
        min_accepted=accepted,
        accepted_before_projection=int(np.count_nonzero(ranking_score < 0.0)),
        tie_fill_count=int(changed_by_index),
        forced_acceptance_count=max(
            0, accepted - int(np.count_nonzero(ranking_score <= 0.0))
        ),
    )


def _partner_masses(
    coupling: FloatArray,
    source_gate: BoolArray,
    target_gate: BoolArray,
) -> tuple[FloatArray, FloatArray]:
    source = np.sum(coupling * target_gate[None, :], axis=1)
    target = np.sum(coupling * source_gate[:, None], axis=0)
    return source, target


def confidence_filtered_entropic_partial_ot(
    cost_matrix: ArrayLike,
    *,
    rejection_cost: float,
    source_unmatched_cost: ArrayLike,
    target_unmatched_cost: ArrayLike,
    epsilon: float,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    initial_source_gate: ArrayLike | None = None,
    initial_target_gate: ArrayLike | None = None,
    source_rejection_budget: float = 0.10,
    target_rejection_budget: float = 0.10,
    gate_budget_mode: GateBudgetMode = "inequality",
    update_source: bool = True,
    update_target: bool = True,
    tau_s: float = 0.0,
    gate_mass_floor: float = 0.0,
    threshold: float = 1e-10,
    max_iterations: int = 10_000,
    max_outer_iterations: int = 100,
    warm_start: bool = True,
) -> ConfidenceFilteredEntropicPartialOTResult:
    """Run inequality-budget eCF-POT-I-E or equality-budget eCF-POT-EQ.

    The source gate is updated first and the target gate sees that new source
    gate, as required by the Gauss--Seidel block-minimization specification.
    ``tau_s=0`` gives exact gate updates; positive values are intentionally
    approximate and should be reported as such.
    """
    cost = _cost_matrix(cost_matrix)
    c = _positive_finite(rejection_cost, name="rejection_cost")
    eps = _positive_finite(epsilon, name="epsilon")
    tolerance = _positive_finite(threshold, name="threshold")
    iterations = _positive_integer(max_iterations, name="max_iterations")
    outer_limit = _positive_integer(
        max_outer_iterations, name="max_outer_iterations"
    )
    gate_tolerance = _positive_finite(tau_s, name="tau_s", allow_zero=True)
    mass_floor = _positive_finite(
        gate_mass_floor, name="gate_mass_floor", allow_zero=True
    )
    if gate_budget_mode not in ("inequality", "equality"):
        raise ValueError("`gate_budget_mode` must be 'inequality' or 'equality'.")
    source_budget = _rejection_budget(
        source_rejection_budget, name="source_rejection_budget"
    )
    target_budget = _rejection_budget(
        target_rejection_budget, name="target_rejection_budget"
    )
    source_min_accepted = _coverage_floor(cost.shape[0], source_budget)
    target_min_accepted = _coverage_floor(cost.shape[1], target_budget)
    if not isinstance(update_source, (bool, np.bool_)) or not isinstance(
        update_target, (bool, np.bool_)
    ):
        raise ValueError("`update_source` and `update_target` must be boolean.")
    if not update_source and not update_target:
        raise ValueError("At least one gate must be updated.")
    if not isinstance(warm_start, (bool, np.bool_)):
        raise ValueError("`warm_start` must be boolean.")
    initialization = "provided"
    if gate_budget_mode == "equality" and (
        initial_source_gate is None or initial_target_gate is None
    ):
        if initial_source_gate is not None or initial_target_gate is not None:
            raise ValueError(
                "Equality mode requires both initial gates or neither initial gate."
            )
        ungated = entropic_partial_ot(
            cost,
            source_unmatched_cost=source_unmatched_cost,
            target_unmatched_cost=target_unmatched_cost,
            epsilon=eps,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=tolerance,
            max_iterations=iterations,
        )
        if not ungated.converged:
            raise PartialInnerSolverError(
                "Ungated ePOT failed during equality-gate initialization."
            )
        all_source = np.ones(cost.shape[0], dtype=bool)
        all_target = np.ones(cost.shape[1], dtype=bool)
        source_coefficient, target_coefficient = partial_gate_coefficients(
            ungated.coupling,
            cost,
            all_source,
            all_target,
            rejection_cost=c,
        )
        source_partner, target_partner = _partner_masses(
            ungated.coupling, all_source, all_target
        )
        source_gate = exact_cardinality_gate_update(
            source_coefficient,
            all_source,
            n_accepted=source_min_accepted,
            tau_loss=gate_tolerance,
            partner_mass=source_partner,
            mass_floor=mass_floor,
        ).gate
        target_gate = exact_cardinality_gate_update(
            target_coefficient,
            all_target,
            n_accepted=target_min_accepted,
            tau_loss=gate_tolerance,
            partner_mass=target_partner,
            mass_floor=mass_floor,
        ).gate
        initialization = "ungated_epot_projection"
    else:
        source_gate = _binary_gate(
            initial_source_gate, n=cost.shape[0], name="initial_source_gate"
        )
        target_gate = _binary_gate(
            initial_target_gate, n=cost.shape[1], name="initial_target_gate"
        )
        initialization = "all_accepted" if (
            initial_source_gate is None and initial_target_gate is None
        ) else "provided"
    source_count = int(source_gate.sum())
    target_count = int(target_gate.sum())
    if gate_budget_mode == "equality":
        if source_count != source_min_accepted:
            raise ValueError("`initial_source_gate` violates the equality budget.")
        if target_count != target_min_accepted:
            raise ValueError("`initial_target_gate` violates the equality budget.")
    else:
        if source_count < source_min_accepted:
            raise ValueError("`initial_source_gate` violates `source_rejection_budget`.")
        if target_count < target_min_accepted:
            raise ValueError("`initial_target_gate` violates `target_rejection_budget`.")

    objectives: list[float] = []
    stages: list[str] = []
    source_history: list[BoolArray] = [source_gate.copy()]
    target_history: list[BoolArray] = [target_gate.copy()]
    log_warm: tuple[FloatArray, FloatArray] | None = None
    total_inner = 0
    transport_solves = 0
    all_inner_converged = True
    outer_converged = False
    cycle_detected = False
    cycle_length = 0
    completed_outer = 0
    result: EntropicPartialOTResult | None = None
    solved_source: BoolArray | None = None
    solved_target: BoolArray | None = None
    seen: dict[tuple[bytes, bytes], int] = {
        (source_gate.tobytes(), target_gate.tobytes()): 0
    }

    for outer in range(outer_limit):
        completed_outer = outer + 1
        solved_source = source_gate.copy()
        solved_target = target_gate.copy()
        result = solve_fixed_confidence_filtered_partial_ot(
            cost,
            source_gate,
            target_gate,
            rejection_cost=c,
            source_unmatched_cost=source_unmatched_cost,
            target_unmatched_cost=target_unmatched_cost,
            epsilon=eps,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=tolerance,
            max_iterations=iterations,
            warm_start=log_warm if warm_start else None,
        )
        transport_solves += 1
        total_inner += result.n_iterations
        all_inner_converged &= result.converged
        if not result.converged:
            raise PartialInnerSolverError(
                "ePOT capped scaling failed during outer iteration "
                f"{outer}: residual={result.fixed_point_error:.6g}."
            )
        objectives.append(result.objective)
        stages.append("transport")
        previous_source = source_gate.copy()
        previous_target = target_gate.copy()

        if update_source:
            source_coefficient, _ = partial_gate_coefficients(
                result.coupling,
                cost,
                source_gate,
                target_gate,
                rejection_cost=c,
            )
            source_partner, _ = _partner_masses(
                result.coupling, source_gate, target_gate
            )
            update_coefficient = source_coefficient.copy()
            update_coefficient[source_partner < mass_floor] = 0.0
            if gate_budget_mode == "equality":
                source_update = exact_cardinality_gate_update(
                    update_coefficient,
                    source_gate,
                    n_accepted=source_min_accepted,
                    tau_loss=gate_tolerance,
                    partner_mass=source_partner,
                    mass_floor=mass_floor,
                )
            else:
                source_update = constrained_gate_update(
                    update_coefficient,
                    source_gate,
                    min_accepted=source_min_accepted,
                    tau_s=gate_tolerance,
                    tolerance_scale=source_partner,
                )
            source_gate = source_update.gate
            objectives.append(
                _fixed_coupling_objective(result, cost, source_gate, target_gate, c)
            )
            stages.append("source_gate")

        if update_target:
            _, target_coefficient = partial_gate_coefficients(
                result.coupling,
                cost,
                source_gate,
                target_gate,
                rejection_cost=c,
            )
            _, target_partner = _partner_masses(
                result.coupling, source_gate, target_gate
            )
            update_coefficient = target_coefficient.copy()
            update_coefficient[target_partner < mass_floor] = 0.0
            if gate_budget_mode == "equality":
                target_update = exact_cardinality_gate_update(
                    update_coefficient,
                    target_gate,
                    n_accepted=target_min_accepted,
                    tau_loss=gate_tolerance,
                    partner_mass=target_partner,
                    mass_floor=mass_floor,
                )
            else:
                target_update = constrained_gate_update(
                    update_coefficient,
                    target_gate,
                    min_accepted=target_min_accepted,
                    tau_s=gate_tolerance,
                    tolerance_scale=target_partner,
                )
            target_gate = target_update.gate
            objectives.append(
                _fixed_coupling_objective(result, cost, source_gate, target_gate, c)
            )
            stages.append("target_gate")

        source_history.append(source_gate.copy())
        target_history.append(target_gate.copy())
        if np.array_equal(source_gate, previous_source) and np.array_equal(
            target_gate, previous_target
        ):
            outer_converged = True
            break
        key = (source_gate.tobytes(), target_gate.tobytes())
        if key in seen:
            cycle_detected = True
            length = outer + 1 - seen[key]
            cycle_length = length if cycle_length == 0 else min(cycle_length, length)
        else:
            seen[key] = outer + 1
        if warm_start:
            log_warm = (
                result.log_source_scaling.copy(),
                result.log_target_scaling.copy(),
            )

    assert result is not None and solved_source is not None and solved_target is not None
    if not np.array_equal(source_gate, solved_source) or not np.array_equal(
        target_gate, solved_target
    ):
        result = solve_fixed_confidence_filtered_partial_ot(
            cost,
            source_gate,
            target_gate,
            rejection_cost=c,
            source_unmatched_cost=source_unmatched_cost,
            target_unmatched_cost=target_unmatched_cost,
            epsilon=eps,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=tolerance,
            max_iterations=iterations,
            warm_start=(result.log_source_scaling, result.log_target_scaling)
            if warm_start
            else None,
        )
        transport_solves += 1
        total_inner += result.n_iterations
        all_inner_converged &= result.converged
        if not result.converged:
            raise PartialInnerSolverError(
                "ePOT capped scaling failed during terminal consistency solve: "
                f"residual={result.fixed_point_error:.6g}."
            )
        objectives.append(result.objective)
        stages.append("terminal_transport")

    source_coefficient, target_coefficient = partial_gate_coefficients(
        result.coupling,
        cost,
        source_gate,
        target_gate,
        rejection_cost=c,
    )
    source_partner, target_partner = _partner_masses(
        result.coupling, source_gate, target_gate
    )
    source_boundary = gate_tolerance * source_partner
    target_boundary = gate_tolerance * target_partner
    source_raw = source_coefficient < -source_boundary
    target_raw = target_coefficient < -target_boundary
    source_update_coefficient = source_coefficient.copy()
    target_update_coefficient = target_coefficient.copy()
    source_update_coefficient[source_partner < mass_floor] = 0.0
    target_update_coefficient[target_partner < mass_floor] = 0.0
    if gate_budget_mode == "equality":
        source_terminal = exact_cardinality_gate_update(
            source_update_coefficient,
            source_gate,
            n_accepted=source_min_accepted,
            tau_loss=gate_tolerance,
            partner_mass=source_partner,
            mass_floor=mass_floor,
        )
        target_terminal = exact_cardinality_gate_update(
            target_update_coefficient,
            target_gate,
            n_accepted=target_min_accepted,
            tau_loss=gate_tolerance,
            partner_mass=target_partner,
            mass_floor=mass_floor,
        )
    else:
        source_terminal = constrained_gate_update(
            source_update_coefficient,
            source_gate,
            min_accepted=source_min_accepted,
            tau_s=gate_tolerance,
            tolerance_scale=source_partner,
        )
        target_terminal = constrained_gate_update(
            target_update_coefficient,
            target_gate,
            min_accepted=target_min_accepted,
            tau_s=gate_tolerance,
            tolerance_scale=target_partner,
        )
    (
        source_min,
        target_min,
        source_bound,
        target_bound,
        source_log_bound,
        target_log_bound,
    ) = _signal_bounds(result, cost, source_gate, target_gate, c)

    return ConfidenceFilteredEntropicPartialOTResult(
        coupling=result.coupling,
        log_coupling=result.log_coupling,
        transition_probability=result.transition_probability,
        reverse_transition_probability=result.reverse_transition_probability,
        source_conditional_loss=result.source_conditional_loss,
        target_conditional_loss=result.target_conditional_loss,
        source_mass=result.source_mass,
        target_mass=result.target_mass,
        source_unmatched_mass=result.source_unmatched_mass,
        target_unmatched_mass=result.target_unmatched_mass,
        source_unmatched_fraction=result.source_unmatched_fraction,
        target_unmatched_fraction=result.target_unmatched_fraction,
        source_marginal=result.source_marginal,
        target_marginal=result.target_marginal,
        source_unmatched_cost=result.source_unmatched_cost,
        target_unmatched_cost=result.target_unmatched_cost,
        admissible_mask=result.admissible_mask,
        inadmissible_mass=result.inadmissible_mass,
        transported_mass=result.transported_mass,
        source_gate=source_gate,
        target_gate=target_gate,
        source_raw_gate=source_raw,
        target_raw_gate=target_raw,
        source_gate_coefficient=source_coefficient,
        target_gate_coefficient=target_coefficient,
        source_effective_reduced_min=source_min,
        target_effective_reduced_min=target_min,
        source_signal_upper_bound=source_bound,
        target_signal_upper_bound=target_bound,
        source_log_signal_upper_bound=source_log_bound,
        target_log_signal_upper_bound=target_log_bound,
        objective=result.objective,
        reduced_objective=result.reduced_objective,
        objective_history=tuple(objectives),
        objective_stage_history=tuple(stages),
        source_gate_history=tuple(source_history),
        target_gate_history=tuple(target_history),
        rejection_cost=c,
        epsilon=eps,
        source_rejection_budget=source_budget,
        target_rejection_budget=target_budget,
        gate_budget_mode=gate_budget_mode,
        initialization=initialization,
        source_min_accepted=source_min_accepted,
        target_min_accepted=target_min_accepted,
        source_budget_binding=int(np.count_nonzero(source_gate)) == source_min_accepted,
        target_budget_binding=int(np.count_nonzero(target_gate)) == target_min_accepted,
        source_boundary_count=int(
            np.count_nonzero(
                (np.abs(source_coefficient) <= source_boundary)
                | (source_partner < mass_floor)
            )
        ),
        target_boundary_count=int(
            np.count_nonzero(
                (np.abs(target_coefficient) <= target_boundary)
                | (target_partner < mass_floor)
            )
        ),
        log_source_scaling=result.log_source_scaling,
        log_target_scaling=result.log_target_scaling,
        inner_converged=all_inner_converged,
        outer_converged=outer_converged,
        cycle_detected=cycle_detected,
        cycle_length=cycle_length,
        n_outer_iterations=completed_outer,
        n_transport_solves=transport_solves,
        total_inner_iterations=total_inner,
        fixed_point_error=result.fixed_point_error,
    )


def two_stage_confidence_filtered_entropic_partial_ot(
    cost_matrix: ArrayLike,
    *,
    rejection_cost: float,
    source_unmatched_cost: ArrayLike,
    target_unmatched_cost: ArrayLike,
    epsilon: float,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    source_rejection_budget: float = 0.10,
    target_rejection_budget: float = 0.10,
    tau_s: float = 0.0,
    gate_mass_floor: float = 0.0,
    threshold: float = 1e-10,
    max_iterations: int = 10_000,
    max_outer_iterations: int = 100,
    warm_start: bool = True,
) -> TwoStageEntropicPartialOTResult:
    """Fit eCF-POT-I-E, project its terminal ranking, and re-solve once."""
    native = confidence_filtered_entropic_partial_ot(
        cost_matrix,
        rejection_cost=rejection_cost,
        source_unmatched_cost=source_unmatched_cost,
        target_unmatched_cost=target_unmatched_cost,
        epsilon=epsilon,
        source_weights=source_weights,
        target_weights=target_weights,
        source_rejection_budget=source_rejection_budget,
        target_rejection_budget=target_rejection_budget,
        gate_budget_mode="inequality",
        tau_s=tau_s,
        gate_mass_floor=gate_mass_floor,
        threshold=threshold,
        max_iterations=max_iterations,
        max_outer_iterations=max_outer_iterations,
        warm_start=warm_start,
    )
    source_partner, target_partner = _partner_masses(
        native.coupling, native.source_gate, native.target_gate
    )
    source_gate = exact_cardinality_gate_update(
        native.source_gate_coefficient,
        native.source_gate,
        n_accepted=native.source_min_accepted,
        tau_loss=tau_s,
        partner_mass=source_partner,
        mass_floor=gate_mass_floor,
    ).gate
    target_gate = exact_cardinality_gate_update(
        native.target_gate_coefficient,
        native.target_gate,
        n_accepted=native.target_min_accepted,
        tau_loss=tau_s,
        partner_mass=target_partner,
        mass_floor=gate_mass_floor,
    ).gate
    fixed = solve_fixed_confidence_filtered_partial_ot(
        cost_matrix,
        source_gate,
        target_gate,
        rejection_cost=rejection_cost,
        source_unmatched_cost=source_unmatched_cost,
        target_unmatched_cost=target_unmatched_cost,
        epsilon=epsilon,
        source_weights=source_weights,
        target_weights=target_weights,
        threshold=threshold,
        max_iterations=max_iterations,
    )
    return TwoStageEntropicPartialOTResult(
        native_result=native,
        fixed_gate_result=fixed,
        source_gate=source_gate,
        target_gate=target_gate,
    )
