"""Bidirectional confidence-filtered entropic KL-unbalanced OT.

This module implements the final two-gate specification in the accompanying
manuscript.  The exact variant uses sequential (Gauss--Seidel) constrained
block minimizers.  The reversible variant uses counterfactual ungated scores
at the current dual potentials and is intentionally reported without a
monotone-objective guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from traditional_ot.unbalanced import (
    UOTResult,
    _cost_matrix,
    _logsumexp,
    _marginal,
    _objective,
    _positive_finite,
    _positive_integer,
    _solve_uot,
    unbalanced_ot,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Variant = Literal["exact", "reversible"]
Adjustment = Literal["by", "bh", "none"]
SolverStatus = Literal["converged", "iteration_capped"]


class InnerSolverError(RuntimeError):
    """Raised only when a fixed-gate Sinkhorn terminal state is non-finite."""


class CalibrationError(RuntimeError):
    """Compatibility exception for unrecoverable calibration failures."""


@dataclass(frozen=True)
class GateUpdateDiagnostics:
    """Diagnostics from one constrained binary gate minimization."""

    gate: BoolArray
    tie_count: int
    tie_fill: bool
    constraint_active: bool
    approximate: bool
    min_accepted: int
    accepted_before_projection: int
    tie_fill_count: int
    forced_acceptance_count: int


@dataclass(frozen=True)
class BidirectionalUOTResult:
    """Result of exact or reversible bidirectional confidence filtering."""

    coupling: FloatArray
    transition_probability: FloatArray
    reverse_transition_probability: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    source_gate: BoolArray
    target_gate: BoolArray
    source_raw_gate: BoolArray
    target_raw_gate: BoolArray
    source_conditional_loss: FloatArray
    target_conditional_loss: FloatArray
    source_counterfactual_loss: FloatArray
    target_counterfactual_loss: FloatArray
    source_gate_score: FloatArray
    target_gate_score: FloatArray
    source_gate_coefficient: FloatArray
    target_gate_coefficient: FloatArray
    log_source_scaling: FloatArray
    log_target_scaling: FloatArray
    objective: float
    objective_history: tuple[float, ...]
    objective_stage_history: tuple[str, ...]
    source_gate_history: tuple[BoolArray, ...]
    target_gate_history: tuple[BoolArray, ...]
    variant: Variant
    update_order: str
    rejection_cost: float
    tau_s: float
    source_tie_fill_count: int
    target_tie_fill_count: int
    source_constraint_count: int
    target_constraint_count: int
    source_forced_acceptance_history: tuple[int, ...]
    target_forced_acceptance_history: tuple[int, ...]
    source_tie_fill_history: tuple[int, ...]
    target_tie_fill_history: tuple[int, ...]
    source_budget_binding: bool
    target_budget_binding: bool
    source_min_accepted: int
    target_min_accepted: int
    source_rejection_budget: float
    target_rejection_budget: float
    source_boundary_count: int
    target_boundary_count: int
    inner_converged: bool
    outer_converged: bool
    cycle_detected: bool
    cycle_length: int
    status: SolverStatus
    n_outer_iterations: int
    n_transport_solves: int
    total_inner_iterations: int

    @property
    def source_raw_acceptance(self) -> float:
        return float(np.mean(self.source_raw_gate))

    @property
    def source_projected_acceptance(self) -> float:
        return float(np.mean(self.source_gate))

    @property
    def target_raw_acceptance(self) -> float:
        return float(np.mean(self.target_raw_gate))

    @property
    def target_projected_acceptance(self) -> float:
        return float(np.mean(self.target_gate))


@dataclass(frozen=True)
class BidirectionalCalibrationResult:
    """Shared rejection cost calibrated against two null acceptance curves."""

    rejection_cost: float
    variant: Variant
    quantile_method: str
    selection_status: str
    rate_constraints_satisfied: bool
    numerically_certified: bool
    calibration_valid: bool
    warning_messages: tuple[str, ...]
    initial_estimate: float
    source_initial_estimate: float
    target_initial_estimate: float
    observed_source_raw_acceptance: float
    observed_source_projected_acceptance: float
    observed_target_raw_acceptance: float
    observed_target_projected_acceptance: float
    null_source_raw_acceptance: float
    null_source_projected_acceptance: float
    null_target_raw_acceptance: float
    null_target_projected_acceptance: float
    curve_costs: FloatArray
    source_raw_acceptance_curve: FloatArray
    source_projected_acceptance_curve: FloatArray
    target_raw_acceptance_curve: FloatArray
    target_projected_acceptance_curve: FloatArray
    source_monotone: bool
    target_monotone: bool
    refinement_method: str
    observed_result: BidirectionalUOTResult
    null_results: tuple[BidirectionalUOTResult, ...]

    # Compatibility aliases for pre-budget callers.  In the corrected method,
    # the unqualified calibration rate always means the raw sign decision.
    @property
    def observed_source_acceptance(self) -> float:
        return self.observed_source_raw_acceptance

    @property
    def observed_target_acceptance(self) -> float:
        return self.observed_target_raw_acceptance

    @property
    def null_source_acceptance(self) -> float:
        return self.null_source_raw_acceptance

    @property
    def null_target_acceptance(self) -> float:
        return self.null_target_raw_acceptance

    @property
    def source_acceptance_curve(self) -> FloatArray:
        return self.source_raw_acceptance_curve

    @property
    def target_acceptance_curve(self) -> FloatArray:
        return self.target_raw_acceptance_curve


@dataclass(frozen=True)
class PostSelectionUOTResult:
    """Ordinary UOT refit on the source/target endpoints selected by O1."""

    uot_result: UOTResult
    source_indices: NDArray[np.int64]
    target_indices: NDArray[np.int64]
    source_gate: BoolArray
    target_gate: BoolArray
    original_shape: tuple[int, int]


@dataclass(frozen=True)
class PopulationTestResult:
    """Population-level Monte Carlo tests from identity-preserving scores."""

    groups: NDArray[np.str_]
    observed_statistic: FloatArray
    null_statistics: FloatArray
    p_value: FloatArray
    q_value: FloatArray
    adjustment: Adjustment
    n_null_replicates: int


def _binary_gate(
    gate: ArrayLike | None, *, n: int, name: str, allow_empty: bool = False
) -> BoolArray:
    if gate is None:
        return np.ones(n, dtype=bool)
    values = np.asarray(gate)
    if values.shape != (n,) or values.ndim != 1:
        raise ValueError(f"`{name}` must have shape ({n},), found {values.shape}.")
    if not np.all(np.isin(values, (0, 1, False, True))):
        raise ValueError(f"`{name}` must contain only boolean or 0/1 values.")
    result = values.astype(bool, copy=True)
    if not allow_empty and not np.any(result):
        raise ValueError(f"`{name}` must accept at least one index.")
    return result


def _rejection_budget(value: object, *, name: str) -> float:
    budget = _positive_finite(value, name=name, allow_zero=True)
    if budget >= 1.0:
        raise ValueError(f"`{name}` must be in [0, 1).")
    return budget


def _coverage_floor(n: int, rejection_budget: float) -> int:
    """Return ``ceil((1-rho)N)`` with the theoretical cardinality semantics."""
    return max(1, min(n, int(ceil((1.0 - rejection_budget) * n))))


def _validate_inner_terminal(result: UOTResult, *, context: str) -> None:
    if not result.converged:
        finite_terminal = bool(
            np.isfinite(result.objective)
            and np.isfinite(result.fixed_point_error)
            and np.all(np.isfinite(result.coupling))
            and np.all(np.isfinite(result.log_source_scaling))
            and np.all(np.isfinite(result.log_target_scaling))
        )
        if not finite_terminal:
            raise InnerSolverError(
                f"Generalized Sinkhorn produced a non-finite state during {context}."
            )
        # A finite capped state is intentionally retained.  The enclosing
        # result/calibration records the warning once, avoiding one warning
        # per inner solve during a calibration grid scan.


def filtered_cost(
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
    *,
    rejection_cost: float,
) -> FloatArray:
    """Return ``delta_i eta_j C_ij + (1-delta_i eta_j)c``."""
    cost = _cost_matrix(cost_matrix)
    source = _binary_gate(source_gate, n=cost.shape[0], name="source_gate")
    target = _binary_gate(target_gate, n=cost.shape[1], name="target_gate")
    c = _positive_finite(rejection_cost, name="rejection_cost")
    trusted = source[:, None] & target[None, :]
    return np.where(trusted, cost, c)


def solve_fixed_bidirectional_uot(
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
    *,
    rejection_cost: float,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-4,
    max_iterations: int = 10_000,
    warm_start: tuple[ArrayLike, ArrayLike] | None = None,
    normalize_weights: bool = True,
) -> UOTResult:
    """Solve KL-UOT for fixed non-empty source and target gates."""
    cost = _cost_matrix(cost_matrix)
    source = _binary_gate(source_gate, n=cost.shape[0], name="source_gate")
    target = _binary_gate(target_gate, n=cost.shape[1], name="target_gate")
    c = _positive_finite(rejection_cost, name="rejection_cost")
    epsilon = _positive_finite(epsilon, name="epsilon")
    lambda_a = _positive_finite(lambda_a, name="lambda_a", allow_zero=True)
    lambda_b = _positive_finite(lambda_b, name="lambda_b", allow_zero=True)
    threshold = _positive_finite(threshold, name="threshold")
    max_iterations = _positive_integer(max_iterations, name="max_iterations")
    optimization_cost = np.where(source[:, None] & target[None, :], cost, c)
    return _solve_uot(
        optimization_cost,
        scoring_cost=cost,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        source_weights=source_weights,
        target_weights=target_weights,
        threshold=threshold,
        max_iterations=max_iterations,
        warm_start=warm_start,
        normalize_weights=normalize_weights,
    )


def _partner_restricted_losses(
    coupling: FloatArray,
    cost: FloatArray,
    source_gate: BoolArray,
    target_gate: BoolArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    weighted = coupling * cost
    source_partner_mass = coupling @ target_gate.astype(np.float64)
    target_partner_mass = coupling.T @ source_gate.astype(np.float64)
    source_numerator = weighted @ target_gate.astype(np.float64)
    target_numerator = weighted.T @ source_gate.astype(np.float64)
    source_loss = np.full(cost.shape[0], np.nan, dtype=np.float64)
    target_loss = np.full(cost.shape[1], np.nan, dtype=np.float64)
    np.divide(
        source_numerator,
        source_partner_mass,
        out=source_loss,
        where=source_partner_mass > 0.0,
    )
    np.divide(
        target_numerator,
        target_partner_mass,
        out=target_loss,
        where=target_partner_mass > 0.0,
    )
    return source_loss, target_loss, source_partner_mass, target_partner_mass


def _counterfactual_losses(
    result: UOTResult, cost: FloatArray
) -> tuple[FloatArray, FloatArray]:
    source_log_weights = (
        np.log(result.target_marginal)[None, :]
        + result.log_target_scaling[None, :]
        - cost / result.epsilon
    )
    source_normalizer = _logsumexp(source_log_weights, axis=1)
    source_probability = np.exp(source_log_weights - source_normalizer[:, None])
    source_loss = np.sum(source_probability * cost, axis=1)

    target_log_weights = (
        np.log(result.source_marginal)[:, None]
        + result.log_source_scaling[:, None]
        - cost / result.epsilon
    )
    target_normalizer = _logsumexp(target_log_weights, axis=0)
    target_probability = np.exp(target_log_weights - target_normalizer[None, :])
    target_loss = np.sum(target_probability * cost, axis=0)
    return source_loss, target_loss


def _reverse_transition(coupling: FloatArray) -> FloatArray:
    mass = coupling.sum(axis=0)
    result = np.zeros_like(coupling)
    np.divide(coupling, mass[None, :], out=result, where=mass[None, :] > 0.0)
    return result


def constrained_gate_update(
    coefficients: ArrayLike,
    current_gate: ArrayLike,
    *,
    min_accepted: int = 1,
    tau_s: float = 0.0,
    tolerance_scale: ArrayLike | None = None,
) -> GateUpdateDiagnostics:
    """Budgeted tie-aware minimizer of ``sum_k gate_k * coefficients_k``.

    ``tau_s=0`` is the exact update.  A positive tolerance intentionally yields
    an approximate coordinate update and is exposed in the diagnostics.  When
    ``tolerance_scale`` is supplied, index ``k`` is treated as a tie whenever
    ``abs(coefficients[k]) <= tau_s * tolerance_scale[k]``.  This permits a
    tolerance expressed in conditional-loss units while the constrained
    projection continues to rank the original objective coefficients.  The
    returned gate always contains at least ``min_accepted`` indices.
    """
    score = np.asarray(coefficients, dtype=np.float64)
    if score.ndim != 1 or score.size == 0 or not np.all(np.isfinite(score)):
        raise ValueError("`coefficients` must be a non-empty finite 1D array.")
    old = _binary_gate(
        current_gate, n=score.size, name="current_gate", allow_empty=True
    )
    minimum = _positive_integer(min_accepted, name="min_accepted")
    if minimum > score.size:
        raise ValueError("`min_accepted` cannot exceed the number of coefficients.")
    tolerance = _positive_finite(tau_s, name="tau_s", allow_zero=True)
    if tolerance_scale is None:
        scale = np.ones(score.size, dtype=np.float64)
    else:
        scale = np.asarray(tolerance_scale, dtype=np.float64)
        if (
            scale.ndim != 1
            or scale.shape != score.shape
            or not np.all(np.isfinite(scale))
            or np.any(scale < 0.0)
        ):
            raise ValueError(
                "`tolerance_scale` must be a finite non-negative 1D array "
                "matching `coefficients`."
            )
    boundary = tolerance * scale
    negative = score < -boundary
    positive = score > boundary
    tie = ~(negative | positive)
    gate = np.zeros(score.size, dtype=bool)
    gate[negative] = True
    gate[tie] = old[tie]
    accepted_before_projection = int(np.count_nonzero(gate))
    tie_fill_count = 0
    forced_acceptance_count = 0
    if accepted_before_projection < minimum:
        candidates = np.flatnonzero(~gate)
        order = sorted(
            candidates.tolist(),
            key=lambda index: (score[index], -int(old[index]), index),
        )
        chosen = np.asarray(
            order[: minimum - accepted_before_projection], dtype=np.int64
        )
        if chosen.size:
            tie_fill_count = int(np.count_nonzero(tie[chosen]))
            forced_acceptance_count = int(np.count_nonzero(positive[chosen]))
            gate[chosen] = True
    return GateUpdateDiagnostics(
        gate=gate,
        tie_count=int(np.count_nonzero(tie)),
        tie_fill=tie_fill_count > 0,
        constraint_active=forced_acceptance_count > 0,
        approximate=tolerance > 0.0,
        min_accepted=minimum,
        accepted_before_projection=accepted_before_projection,
        tie_fill_count=tie_fill_count,
        forced_acceptance_count=forced_acceptance_count,
    )


def _block_objective(
    result: UOTResult,
    cost: FloatArray,
    source_gate: BoolArray,
    target_gate: BoolArray,
    rejection_cost: float,
) -> float:
    optimization_cost = np.where(
        source_gate[:, None] & target_gate[None, :], cost, rejection_cost
    )
    return _objective(
        result.coupling,
        optimization_cost,
        result.source_marginal,
        result.target_marginal,
        result.epsilon,
        result.lambda_a,
        result.lambda_b,
    )


def confidence_filtered_bidirectional_uot(
    cost_matrix: ArrayLike,
    *,
    rejection_cost: float,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    variant: Variant = "reversible",
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    initial_source_gate: ArrayLike | None = None,
    initial_target_gate: ArrayLike | None = None,
    source_rejection_budget: float = 0.10,
    target_rejection_budget: float = 0.10,
    update_source: bool = True,
    update_target: bool = True,
    tau_s: float = 0.0,
    threshold: float = 1e-4,
    max_iterations: int = 10_000,
    max_outer_iterations: int = 100,
    warm_start: bool = True,
) -> BidirectionalUOTResult:
    """Run budgeted M3, M4-E, or M4-R confidence-filtered KL-UOT.

    Set ``update_target=False`` for the M3 source-only ablation.  At least one
    side must be updated.  The target update always sees the newly updated
    source gate, as required by the exact Gauss--Seidel scheme.  ``tau_s`` is
    measured in conditional-loss units: internally each endpoint uses the
    coefficient tolerance ``partner_mass * tau_s``.  Thus a fixed stability
    band does not silently grow with the number of cells.
    """
    cost = _cost_matrix(cost_matrix)
    c = _positive_finite(rejection_cost, name="rejection_cost")
    epsilon = _positive_finite(epsilon, name="epsilon")
    lambda_a = _positive_finite(lambda_a, name="lambda_a", allow_zero=True)
    lambda_b = _positive_finite(lambda_b, name="lambda_b", allow_zero=True)
    tau_s = _positive_finite(tau_s, name="tau_s", allow_zero=True)
    threshold = _positive_finite(threshold, name="threshold")
    max_iterations = _positive_integer(max_iterations, name="max_iterations")
    max_outer_iterations = _positive_integer(
        max_outer_iterations, name="max_outer_iterations"
    )
    source_budget = _rejection_budget(
        source_rejection_budget, name="source_rejection_budget"
    )
    target_budget = _rejection_budget(
        target_rejection_budget, name="target_rejection_budget"
    )
    source_min_accepted = _coverage_floor(cost.shape[0], source_budget)
    target_min_accepted = _coverage_floor(cost.shape[1], target_budget)
    if variant not in ("exact", "reversible"):
        raise ValueError("`variant` must be 'exact' or 'reversible'.")
    if not isinstance(update_source, (bool, np.bool_)) or not isinstance(
        update_target, (bool, np.bool_)
    ):
        raise ValueError("`update_source` and `update_target` must be boolean.")
    if not update_source and not update_target:
        raise ValueError("At least one gate must be updated.")
    if not isinstance(warm_start, (bool, np.bool_)):
        raise ValueError("`warm_start` must be boolean.")

    source_gate = _binary_gate(
        initial_source_gate, n=cost.shape[0], name="initial_source_gate"
    )
    target_gate = _binary_gate(
        initial_target_gate, n=cost.shape[1], name="initial_target_gate"
    )
    if int(np.count_nonzero(source_gate)) < source_min_accepted:
        raise ValueError(
            "`initial_source_gate` violates `source_rejection_budget`."
        )
    if int(np.count_nonzero(target_gate)) < target_min_accepted:
        raise ValueError(
            "`initial_target_gate` violates `target_rejection_budget`."
        )
    source_history: list[BoolArray] = [source_gate.copy()]
    target_history: list[BoolArray] = [target_gate.copy()]
    objectives: list[float] = []
    stages: list[str] = []
    log_warm: tuple[FloatArray, FloatArray] | None = None
    total_inner = 0
    transport_solves = 0
    all_inner_converged = True
    outer_converged = False
    cycle_detected = False
    cycle_length = 0
    source_tie_fills = target_tie_fills = 0
    source_constraints = target_constraints = 0
    source_forced_history: list[int] = []
    target_forced_history: list[int] = []
    source_tie_history: list[int] = []
    target_tie_history: list[int] = []
    result: UOTResult | None = None
    solved_source: BoolArray | None = None
    solved_target: BoolArray | None = None
    completed_iterations = 0
    seen_gate_pairs: dict[tuple[bytes, bytes], int] = {
        (source_gate.tobytes(), target_gate.tobytes()): 0
    }

    for outer in range(max_outer_iterations):
        completed_iterations = outer + 1
        solved_source = source_gate.copy()
        solved_target = target_gate.copy()
        result = solve_fixed_bidirectional_uot(
            cost,
            source_gate,
            target_gate,
            rejection_cost=c,
            epsilon=epsilon,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=threshold,
            max_iterations=max_iterations,
            warm_start=log_warm if warm_start else None,
        )
        transport_solves += 1
        total_inner += result.n_iterations
        all_inner_converged &= result.converged
        _validate_inner_terminal(result, context=f"outer iteration {outer}")
        objectives.append(result.objective)
        stages.append("transport")

        source_loss, _, source_partner_mass, _ = _partner_restricted_losses(
            result.coupling, cost, source_gate, target_gate
        )
        source_cf, target_cf = _counterfactual_losses(result, cost)
        previous_source = source_gate.copy()
        previous_target = target_gate.copy()

        if update_source:
            if variant == "exact":
                source_coeff = np.sum(
                    result.coupling
                    * target_gate[None, :]
                    * (cost - c),
                    axis=1,
                )
            else:
                source_coeff = source_partner_mass * (source_cf - c)
            source_update = constrained_gate_update(
                source_coeff,
                source_gate,
                min_accepted=source_min_accepted,
                tau_s=tau_s,
                tolerance_scale=source_partner_mass,
            )
            source_gate = source_update.gate
            source_tie_fills += int(source_update.tie_fill)
            source_constraints += int(source_update.constraint_active)
            source_forced_history.append(source_update.forced_acceptance_count)
            source_tie_history.append(source_update.tie_fill_count)
            objectives.append(
                _block_objective(result, cost, source_gate, target_gate, c)
            )
            stages.append("source_gate")
        else:
            source_forced_history.append(0)
            source_tie_history.append(0)

        # Recompute the target partner mass/loss against the NEW source gate.
        _, target_loss, _, target_partner_mass = _partner_restricted_losses(
            result.coupling, cost, source_gate, target_gate
        )
        if update_target:
            if variant == "exact":
                target_coeff = np.sum(
                    result.coupling
                    * source_gate[:, None]
                    * (cost - c),
                    axis=0,
                )
            else:
                target_coeff = target_partner_mass * (target_cf - c)
            target_update = constrained_gate_update(
                target_coeff,
                target_gate,
                min_accepted=target_min_accepted,
                tau_s=tau_s,
                tolerance_scale=target_partner_mass,
            )
            target_gate = target_update.gate
            target_tie_fills += int(target_update.tie_fill)
            target_constraints += int(target_update.constraint_active)
            target_forced_history.append(target_update.forced_acceptance_count)
            target_tie_history.append(target_update.tie_fill_count)
            objectives.append(
                _block_objective(result, cost, source_gate, target_gate, c)
            )
            stages.append("target_gate")
        else:
            target_forced_history.append(0)
            target_tie_history.append(0)

        source_history.append(source_gate.copy())
        target_history.append(target_gate.copy())
        if np.array_equal(source_gate, previous_source) and np.array_equal(
            target_gate, previous_target
        ):
            outer_converged = True
            break
        gate_key = (source_gate.tobytes(), target_gate.tobytes())
        if gate_key in seen_gate_pairs:
            cycle_detected = True
            detected_length = outer + 1 - seen_gate_pairs[gate_key]
            cycle_length = (
                detected_length
                if cycle_length == 0
                else min(cycle_length, detected_length)
            )
        else:
            seen_gate_pairs[gate_key] = outer + 1
        if warm_start:
            log_warm = (
                result.log_source_scaling.copy(),
                result.log_target_scaling.copy(),
            )

    assert result is not None and solved_source is not None and solved_target is not None
    if not np.array_equal(solved_source, source_gate) or not np.array_equal(
        solved_target, target_gate
    ):
        result = solve_fixed_bidirectional_uot(
            cost,
            source_gate,
            target_gate,
            rejection_cost=c,
            epsilon=epsilon,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=threshold,
            max_iterations=max_iterations,
            warm_start=(
                result.log_source_scaling,
                result.log_target_scaling,
            )
            if warm_start
            else None,
        )
        transport_solves += 1
        total_inner += result.n_iterations
        all_inner_converged &= result.converged
        _validate_inner_terminal(result, context="terminal consistency re-solve")
        objectives.append(result.objective)
        stages.append("terminal_transport")

    source_loss, target_loss, source_partner_mass, target_partner_mass = (
        _partner_restricted_losses(result.coupling, cost, source_gate, target_gate)
    )
    source_cf, target_cf = _counterfactual_losses(result, cost)
    source_coeff = np.sum(
        result.coupling * target_gate[None, :] * (cost - c), axis=1
    )
    target_coeff = np.sum(
        result.coupling * source_gate[:, None] * (cost - c), axis=0
    )
    source_score = source_loss if variant == "exact" else source_cf
    target_score = target_loss if variant == "exact" else target_cf
    if variant == "reversible":
        reported_source_coeff = source_partner_mass * (source_cf - c)
        reported_target_coeff = target_partner_mass * (target_cf - c)
    else:
        reported_source_coeff = source_coeff
        reported_target_coeff = target_coeff

    source_raw_gate = reported_source_coeff < 0.0
    target_raw_gate = reported_target_coeff < 0.0
    terminal_source_update = constrained_gate_update(
        reported_source_coeff,
        source_gate,
        min_accepted=source_min_accepted,
        tau_s=tau_s,
        tolerance_scale=source_partner_mass,
    )
    terminal_target_update = constrained_gate_update(
        reported_target_coeff,
        target_gate,
        min_accepted=target_min_accepted,
        tau_s=tau_s,
        tolerance_scale=target_partner_mass,
    )
    status: SolverStatus = (
        "converged"
        if outer_converged and all_inner_converged
        else "iteration_capped"
    )

    return BidirectionalUOTResult(
        coupling=result.coupling,
        transition_probability=result.transition_probability,
        reverse_transition_probability=_reverse_transition(result.coupling),
        source_mass=result.source_mass,
        target_mass=result.target_mass,
        source_gate=source_gate,
        target_gate=target_gate,
        source_raw_gate=source_raw_gate,
        target_raw_gate=target_raw_gate,
        source_conditional_loss=source_loss,
        target_conditional_loss=target_loss,
        source_counterfactual_loss=source_cf,
        target_counterfactual_loss=target_cf,
        source_gate_score=source_score,
        target_gate_score=target_score,
        source_gate_coefficient=reported_source_coeff,
        target_gate_coefficient=reported_target_coeff,
        log_source_scaling=result.log_source_scaling,
        log_target_scaling=result.log_target_scaling,
        objective=result.objective,
        objective_history=tuple(objectives),
        objective_stage_history=tuple(stages),
        source_gate_history=tuple(source_history),
        target_gate_history=tuple(target_history),
        variant=variant,
        update_order="source_then_target",
        rejection_cost=c,
        tau_s=tau_s,
        source_tie_fill_count=source_tie_fills,
        target_tie_fill_count=target_tie_fills,
        source_constraint_count=source_constraints,
        target_constraint_count=target_constraints,
        source_forced_acceptance_history=tuple(source_forced_history),
        target_forced_acceptance_history=tuple(target_forced_history),
        source_tie_fill_history=tuple(source_tie_history),
        target_tie_fill_history=tuple(target_tie_history),
        source_budget_binding=terminal_source_update.constraint_active,
        target_budget_binding=terminal_target_update.constraint_active,
        source_min_accepted=source_min_accepted,
        target_min_accepted=target_min_accepted,
        source_rejection_budget=source_budget,
        target_rejection_budget=target_budget,
        source_boundary_count=int(
            np.count_nonzero(
                np.abs(reported_source_coeff) <= tau_s * source_partner_mass
            )
        ),
        target_boundary_count=int(
            np.count_nonzero(
                np.abs(reported_target_coeff) <= tau_s * target_partner_mass
            )
        ),
        inner_converged=all_inner_converged,
        outer_converged=outer_converged,
        cycle_detected=cycle_detected,
        cycle_length=cycle_length,
        status=status,
        n_outer_iterations=completed_iterations,
        n_transport_solves=transport_solves,
        total_inner_iterations=total_inner,
    )


def _validate_rate(value: object, *, minimum: float, name: str) -> float:
    rate = _positive_finite(value, name=name, allow_zero=True)
    if rate > 1.0:
        raise ValueError(f"`{name}` must not exceed 1.")
    if rate + 1e-15 < minimum:
        raise ValueError(
            f"`{name}` must be at least {minimum:.8g}."
        )
    return rate


def calibrate_bidirectional_rejection_cost(
    observed_cost: ArrayLike,
    null_costs: Sequence[ArrayLike],
    *,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    source_acceptance_target: float = 0.10,
    target_acceptance_target: float = 0.10,
    variant: Variant = "reversible",
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    source_rejection_budget: float = 0.10,
    target_rejection_budget: float = 0.10,
    tau_s: float = 0.0,
    threshold: float = 1e-4,
    max_iterations: int = 10_000,
    max_outer_iterations: int = 100,
    grid_size: int = 15,
    refinement_relative_tolerance: float = 1e-4,
) -> BidirectionalCalibrationResult:
    """Select a shared ``c`` from both mean raw null-rate curves.

    Calibration always thresholds terminal pre-projection sign decisions.
    Projected rates are returned only as diagnostics because the rejection
    budgets bound them from below.  Capped or cycling terminal fits are
    retained with warnings.  If the grid has no jointly feasible value, the
    least-violating grid value is returned with warning provenance.
    """
    observed = _cost_matrix(observed_cost, name="observed_cost")
    if not isinstance(null_costs, Sequence) or len(null_costs) == 0:
        raise ValueError("`null_costs` must contain at least one null cost matrix.")
    null = tuple(_cost_matrix(item, name="null_cost") for item in null_costs)
    if any(item.shape != observed.shape for item in null):
        raise ValueError("Every null cost matrix must match `observed_cost` shape.")
    source_target = _validate_rate(
        source_acceptance_target,
        minimum=0.0,
        name="source_acceptance_target",
    )
    target_target = _validate_rate(
        target_acceptance_target,
        minimum=0.0,
        name="target_acceptance_target",
    )
    grid_size = _positive_integer(grid_size, name="grid_size")
    if grid_size < 3:
        raise ValueError("`grid_size` must be at least 3.")
    relative_tolerance = _positive_finite(
        refinement_relative_tolerance, name="refinement_relative_tolerance"
    )
    source_budget = _rejection_budget(
        source_rejection_budget, name="source_rejection_budget"
    )
    target_budget = _rejection_budget(
        target_rejection_budget, name="target_rejection_budget"
    )

    solver_arguments = dict(
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        source_weights=source_weights,
        target_weights=target_weights,
        threshold=threshold,
        max_iterations=max_iterations,
    )
    calibration_warnings: set[str] = set()
    source_quantiles: list[float] = []
    target_quantiles: list[float] = []
    all_scores: list[FloatArray] = []
    for null_cost in null:
        base = _solve_uot(
            null_cost,
            scoring_cost=null_cost,
            warm_start=None,
            **solver_arguments,
        )
        _validate_inner_terminal(base, context="ungated null initialization")
        if not base.converged:
            calibration_warnings.add(
                "At least one ungated null initialization reached the inner iteration "
                "cap; its finite terminal scores were retained."
            )
        source_score, target_score = _counterfactual_losses(base, null_cost)
        source_quantiles.append(float(np.quantile(
            source_score, source_target, method="linear"
        )))
        target_quantiles.append(float(np.quantile(
            target_score, target_target, method="linear"
        )))
        all_scores.extend((source_score, target_score))
    source_initial = float(np.median(source_quantiles))
    target_initial = float(np.median(target_quantiles))
    initial = min(source_initial, target_initial)
    positive_scores = np.concatenate(all_scores)
    positive_scores = positive_scores[positive_scores > 0.0]
    lower = (
        max(float(np.min(positive_scores)) * 0.1, np.finfo(np.float64).tiny)
        if positive_scores.size
        else max(initial * 0.1, 1e-12)
    )
    upper = max(float(np.max(np.concatenate(all_scores))) * 1.1, initial * 2.0)
    curve_costs = np.geomspace(lower, upper, grid_size)

    cache: dict[float, tuple[float, float, float, float]] = {}
    def fit_nulls(c: float) -> tuple[BidirectionalUOTResult, ...]:
        fitted_results: list[BidirectionalUOTResult] = []
        for null_cost in null:
            fitted = confidence_filtered_bidirectional_uot(
                null_cost,
                rejection_cost=c,
                variant=variant,
                source_rejection_budget=source_budget,
                target_rejection_budget=target_budget,
                tau_s=tau_s,
                max_outer_iterations=max_outer_iterations,
                **solver_arguments,
            )
            if not fitted.outer_converged:
                calibration_warnings.add(
                    "At least one null CF-UOT fit was iteration-capped; its terminal "
                    "coefficients were retained in the calibration curve."
                )
            if not fitted.inner_converged:
                calibration_warnings.add(
                    "At least one null CF-UOT fit reached an inner iteration cap; its "
                    "finite terminal coupling and coefficients were retained."
                )
            if fitted.cycle_detected:
                calibration_warnings.add(
                    "At least one null M4 fit cycled; its terminal coefficients were "
                    "retained in the calibration curve."
                )
            fitted_results.append(fitted)
        return tuple(fitted_results)

    def evaluate(c: float) -> tuple[float, float, float, float]:
        key = float(c)
        if key not in cache:
            fitted_results = fit_nulls(key)
            cache[key] = (
                float(np.mean([fit.source_raw_acceptance for fit in fitted_results])),
                float(
                    np.mean([fit.source_projected_acceptance for fit in fitted_results])
                ),
                float(np.mean([fit.target_raw_acceptance for fit in fitted_results])),
                float(
                    np.mean([fit.target_projected_acceptance for fit in fitted_results])
                ),
            )
        return cache[key]

    curve = np.asarray([evaluate(c) for c in curve_costs], dtype=np.float64)
    source_raw_curve = curve[:, 0]
    source_projected_curve = curve[:, 1]
    target_raw_curve = curve[:, 2]
    target_projected_curve = curve[:, 3]
    source_monotone = bool(np.all(np.diff(source_raw_curve) >= -1e-12))
    target_monotone = bool(np.all(np.diff(target_raw_curve) >= -1e-12))
    if not source_monotone or not target_monotone:
        calibration_warnings.add(
            "At least one empirical raw-acceptance curve was nonmonotone; the grid "
            "selection is retained, but continuous c refinement is skipped."
        )
    feasible = (source_raw_curve <= source_target) & (
        target_raw_curve <= target_target
    )
    rate_constraints_satisfied = bool(np.any(feasible))
    if rate_constraints_satisfied:
        feasible_indices = np.flatnonzero(feasible)
        best_index = int(feasible_indices[-1])
        selection_status = "largest_jointly_feasible"
    else:
        joint_violation = np.maximum(
            np.maximum(source_raw_curve - source_target, 0.0),
            np.maximum(target_raw_curve - target_target, 0.0),
        )
        minimum_violation = float(np.min(joint_violation))
        best_index = int(np.flatnonzero(
            np.isclose(joint_violation, minimum_violation, rtol=0.0, atol=1e-15)
        )[-1])
        selection_status = "minimum_joint_violation_fallback"
        calibration_warnings.add(
            "No jointly feasible rejection cost was found; the returned grid point "
            "minimizes the maximum source/target raw-rate violation, with ties resolved "
            "toward the largest rejection cost."
        )
    c_star = float(curve_costs[best_index])
    refinement = selection_status
    if (
        rate_constraints_satisfied
        and source_monotone
        and target_monotone
        and best_index + 1 < grid_size
    ):
        low = c_star
        high = float(curve_costs[best_index + 1])
        while (high - low) / max(low, np.finfo(np.float64).tiny) > relative_tolerance:
            middle = float(np.sqrt(low * high))
            source_raw, _, target_raw, _ = evaluate(middle)
            if source_raw <= source_target and target_raw <= target_target:
                low = middle
            else:
                high = middle
        c_star = low
        refinement = "joint_bisection"

    (
        null_source_raw,
        null_source_projected,
        null_target_raw,
        null_target_projected,
    ) = evaluate(c_star)
    null_results = fit_nulls(c_star)
    observed_result = confidence_filtered_bidirectional_uot(
        observed,
        rejection_cost=c_star,
        variant=variant,
        source_rejection_budget=source_budget,
        target_rejection_budget=target_budget,
        tau_s=tau_s,
        max_outer_iterations=max_outer_iterations,
        **solver_arguments,
    )
    if not observed_result.outer_converged:
        calibration_warnings.add(
            "The observed CF-UOT fit was iteration-capped; its terminal result was "
            "retained with warning status."
        )
    if not observed_result.inner_converged:
        calibration_warnings.add(
            "The observed CF-UOT fit reached an inner iteration cap; its finite "
            "terminal coupling and coefficients were retained with warning status."
        )
    if observed_result.cycle_detected:
        calibration_warnings.add(
            "The observed M4 fit cycled; its terminal result was retained with warning "
            "status."
        )
    warning_messages = tuple(sorted(calibration_warnings))
    for message in warning_messages:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    numerically_certified = not warning_messages
    evaluated_costs = np.asarray(sorted(cache), dtype=np.float64)
    evaluated_curve = np.asarray([cache[c] for c in evaluated_costs])
    return BidirectionalCalibrationResult(
        rejection_cost=c_star,
        variant=variant,
        quantile_method="linear",
        selection_status=selection_status,
        rate_constraints_satisfied=rate_constraints_satisfied,
        numerically_certified=numerically_certified,
        calibration_valid=bool(
            rate_constraints_satisfied and numerically_certified
        ),
        warning_messages=warning_messages,
        initial_estimate=initial,
        source_initial_estimate=source_initial,
        target_initial_estimate=target_initial,
        observed_source_raw_acceptance=observed_result.source_raw_acceptance,
        observed_source_projected_acceptance=(
            observed_result.source_projected_acceptance
        ),
        observed_target_raw_acceptance=observed_result.target_raw_acceptance,
        observed_target_projected_acceptance=(
            observed_result.target_projected_acceptance
        ),
        null_source_raw_acceptance=null_source_raw,
        null_source_projected_acceptance=null_source_projected,
        null_target_raw_acceptance=null_target_raw,
        null_target_projected_acceptance=null_target_projected,
        curve_costs=evaluated_costs,
        source_raw_acceptance_curve=evaluated_curve[:, 0],
        source_projected_acceptance_curve=evaluated_curve[:, 1],
        target_raw_acceptance_curve=evaluated_curve[:, 2],
        target_projected_acceptance_curve=evaluated_curve[:, 3],
        source_monotone=source_monotone,
        target_monotone=target_monotone,
        refinement_method=refinement,
        observed_result=observed_result,
        null_results=null_results,
    )


def refit_post_selection_uot(
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
    *,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-4,
    max_iterations: int = 10_000,
) -> PostSelectionUOTResult:
    """Construct O2 by refitting ordinary UOT on the accepted endpoints.

    ``cost_matrix`` is the original, already-scaled cost matrix.  The function
    takes no labels or planted truth and only subsets rows and columns according
    to the supplied O1 gates.  Subset weights are renormalized by the ordinary
    UOT solver.
    """
    cost = _cost_matrix(cost_matrix)
    source = _binary_gate(source_gate, n=cost.shape[0], name="source_gate")
    target = _binary_gate(target_gate, n=cost.shape[1], name="target_gate")
    source_indices = np.flatnonzero(source).astype(np.int64, copy=False)
    target_indices = np.flatnonzero(target).astype(np.int64, copy=False)
    source_full = (
        None
        if source_weights is None
        else _marginal(source_weights, n=cost.shape[0], name="source_weights")
    )
    target_full = (
        None
        if target_weights is None
        else _marginal(target_weights, n=cost.shape[1], name="target_weights")
    )
    result = unbalanced_ot(
        cost[np.ix_(source_indices, target_indices)],
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        source_weights=None if source_full is None else source_full[source_indices],
        target_weights=None if target_full is None else target_full[target_indices],
        threshold=threshold,
        max_iterations=max_iterations,
    )
    _validate_inner_terminal(result, context="post-selection O2 refit")
    return PostSelectionUOTResult(
        uot_result=result,
        source_indices=source_indices,
        target_indices=target_indices,
        source_gate=source.copy(),
        target_gate=target.copy(),
        original_shape=cost.shape,
    )


def _adjust_pvalues(p_value: FloatArray, method: Adjustment) -> FloatArray:
    if method == "none":
        return p_value.copy()
    count = p_value.size
    order = np.argsort(p_value)
    ranked = p_value[order]
    factor = float(np.sum(1.0 / np.arange(1, count + 1))) if method == "by" else 1.0
    adjusted_ranked = ranked * count * factor / np.arange(1, count + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def population_monte_carlo_test(
    observed_scores: ArrayLike,
    null_scores: ArrayLike,
    groups: ArrayLike,
    *,
    adjustment: Adjustment = "by",
) -> PopulationTestResult:
    """Test fixed populations with lower-tail median-score Monte Carlo tests.

    Rows of ``null_scores`` are identity-preserving null replicates of the same
    cells represented by ``observed_scores``.  The caller is responsible for
    the conditional-exchangeability design.  BY is the conservative default
    because group statistics share couplings and null supports.
    """
    observed = np.asarray(observed_scores, dtype=np.float64)
    null = np.asarray(null_scores, dtype=np.float64)
    labels = np.asarray(groups)
    if observed.ndim != 1 or observed.size == 0 or not np.all(np.isfinite(observed)):
        raise ValueError("`observed_scores` must be a non-empty finite 1D array.")
    if null.ndim != 2 or null.shape[1] != observed.size or null.shape[0] == 0:
        raise ValueError(
            "`null_scores` must have shape (n_null_replicates, n_observed_scores)."
        )
    if not np.all(np.isfinite(null)):
        raise ValueError("`null_scores` must contain only finite values.")
    if labels.ndim != 1 or labels.shape != observed.shape:
        raise ValueError("`groups` must have one label per observed score.")
    if adjustment not in ("by", "bh", "none"):
        raise ValueError("`adjustment` must be 'by', 'bh', or 'none'.")
    unique = np.asarray(sorted(np.unique(labels).astype(str)))
    observed_statistic = np.empty(unique.size, dtype=np.float64)
    null_statistics = np.empty((null.shape[0], unique.size), dtype=np.float64)
    string_labels = labels.astype(str)
    for index, group in enumerate(unique):
        mask = string_labels == group
        observed_statistic[index] = float(np.median(observed[mask]))
        null_statistics[:, index] = np.median(null[:, mask], axis=1)
    p_value = (
        1.0
        + np.sum(null_statistics <= observed_statistic[None, :], axis=0)
    ) / (null.shape[0] + 1.0)
    q_value = _adjust_pvalues(p_value, adjustment)
    return PopulationTestResult(
        groups=unique,
        observed_statistic=observed_statistic,
        null_statistics=null_statistics,
        p_value=p_value,
        q_value=q_value,
        adjustment=adjustment,
        n_null_replicates=int(null.shape[0]),
    )
