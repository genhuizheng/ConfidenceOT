"""Strict bidirectional confidence-filtered balanced entropic OT.

This module extends the original source-only CF-BOT with independent source
and target gates.  For fixed gates it solves ordinary balanced entropic OT
with

    C~_ij = delta_i eta_j C_ij + (1 - delta_i eta_j) c.

Both marginals remain exact: rejection means abstaining from cell-specific
correspondence geometry, not deleting mass.  The exact variant uses sequential
budgeted block minimizers; the reversible variant uses counterfactual ungated
scores and is intentionally not given a monotone-objective guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from traditional_ot.balanced import (
    BalancedOTResult,
    _cost_matrix,
    _generalized_kl,
    _logsumexp,
    _positive_finite,
    _positive_integer,
    _solve_balanced,
)
from confidenceot._cpu_uot import (
    GateUpdateDiagnostics,
    _binary_gate,
    _coverage_floor,
    _rejection_budget,
    constrained_gate_update,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
BalancedVariant = Literal["exact", "reversible"]


class BalancedInnerSolverError(RuntimeError):
    """Raised when a fixed-gate balanced Sinkhorn solve does not converge."""


@dataclass(frozen=True)
class BidirectionalBalancedOTResult:
    """Result of budgeted source-and-target confidence-filtered balanced OT."""

    coupling: FloatArray
    transition_probability: FloatArray
    reverse_transition_probability: FloatArray
    source_marginal: FloatArray
    target_marginal: FloatArray
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
    variant: BalancedVariant
    rejection_cost: float
    epsilon: float
    tau_s: float
    source_rejection_budget: float
    target_rejection_budget: float
    source_min_accepted: int
    target_min_accepted: int
    source_budget_binding: bool
    target_budget_binding: bool
    source_boundary_count: int
    target_boundary_count: int
    source_constraint_count: int
    target_constraint_count: int
    inner_converged: bool
    outer_converged: bool
    cycle_detected: bool
    cycle_length: int
    n_outer_iterations: int
    n_transport_solves: int
    total_inner_iterations: int

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


def bidirectional_balanced_filtered_cost(
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
    return np.where(source[:, None] & target[None, :], cost, c)


def solve_fixed_bidirectional_balanced_ot(
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
    *,
    rejection_cost: float,
    epsilon: float,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    warm_start: tuple[ArrayLike, ArrayLike] | None = None,
    normalize_weights: bool = True,
) -> BalancedOTResult:
    """Solve balanced entropic OT for fixed source and target gates."""
    cost = _cost_matrix(cost_matrix)
    filtered = bidirectional_balanced_filtered_cost(
        cost_matrix,
        source_gate,
        target_gate,
        rejection_cost=rejection_cost,
    )
    epsilon = _positive_finite(epsilon, name="epsilon")
    threshold = _positive_finite(threshold, name="threshold")
    max_iterations = _positive_integer(max_iterations, name="max_iterations")
    return _solve_balanced(
        filtered,
        scoring_cost=cost,
        epsilon=epsilon,
        source_weights=source_weights,
        target_weights=target_weights,
        threshold=threshold,
        max_iterations=max_iterations,
        warm_start=warm_start,
        normalize_weights=normalize_weights,
    )


def _reverse_transition(coupling: FloatArray) -> FloatArray:
    mass = coupling.sum(axis=0)
    result = np.zeros_like(coupling)
    np.divide(coupling, mass[None, :], out=result, where=mass[None, :] > 0.0)
    return result


def _partner_losses(
    coupling: FloatArray,
    cost: FloatArray,
    source_gate: BoolArray,
    target_gate: BoolArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    weighted = coupling * cost
    source_partner_mass = coupling @ target_gate.astype(np.float64)
    target_partner_mass = coupling.T @ source_gate.astype(np.float64)
    source_loss = np.divide(
        weighted @ target_gate.astype(np.float64), source_partner_mass
    )
    target_loss = np.divide(
        weighted.T @ source_gate.astype(np.float64), target_partner_mass
    )
    return source_loss, target_loss, source_partner_mass, target_partner_mass


def _counterfactual_losses(
    result: BalancedOTResult, cost: FloatArray
) -> tuple[FloatArray, FloatArray]:
    source_log_weights = (
        np.log(result.target_marginal)[None, :]
        + result.log_target_scaling[None, :]
        - cost / result.epsilon
    )
    source_probability = np.exp(
        source_log_weights
        - _logsumexp(source_log_weights, axis=1)[:, None]
    )
    target_log_weights = (
        np.log(result.source_marginal)[:, None]
        + result.log_source_scaling[:, None]
        - cost / result.epsilon
    )
    target_probability = np.exp(
        target_log_weights
        - _logsumexp(target_log_weights, axis=0)[None, :]
    )
    return (
        np.sum(source_probability * cost, axis=1),
        np.sum(target_probability * cost, axis=0),
    )


def _objective(
    result: BalancedOTResult,
    cost: FloatArray,
    source_gate: BoolArray,
    target_gate: BoolArray,
    rejection_cost: float,
) -> float:
    filtered = np.where(
        source_gate[:, None] & target_gate[None, :], cost, rejection_cost
    )
    reference = result.source_marginal[:, None] * result.target_marginal[None, :]
    return float(
        np.sum(result.coupling * filtered)
        + result.epsilon * _generalized_kl(result.coupling, reference)
    )


def confidence_filtered_bidirectional_balanced_ot(
    cost_matrix: ArrayLike,
    *,
    rejection_cost: float,
    epsilon: float,
    variant: BalancedVariant = "exact",
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    initial_source_gate: ArrayLike | None = None,
    initial_target_gate: ArrayLike | None = None,
    source_rejection_budget: float = 0.10,
    target_rejection_budget: float = 0.10,
    update_source: bool = True,
    update_target: bool = True,
    tau_s: float = 0.0,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    max_outer_iterations: int = 100,
    warm_start: bool = True,
) -> BidirectionalBalancedOTResult:
    """Run exact or reversible bidirectional confidence-filtered balanced OT.

    ``tau_s`` is measured in conditional-loss units.  Each endpoint therefore
    uses ``partner_mass * tau_s`` as its coefficient tolerance, avoiding a
    hidden dependence on the number of uniformly weighted cells.  The exact
    ``tau_s=0`` algorithm is unchanged.
    """
    cost = _cost_matrix(cost_matrix)
    c = _positive_finite(rejection_cost, name="rejection_cost")
    epsilon = _positive_finite(epsilon, name="epsilon")
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
    source_min = _coverage_floor(cost.shape[0], source_budget)
    target_min = _coverage_floor(cost.shape[1], target_budget)
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
    if np.count_nonzero(source_gate) < source_min:
        raise ValueError("`initial_source_gate` violates `source_rejection_budget`.")
    if np.count_nonzero(target_gate) < target_min:
        raise ValueError("`initial_target_gate` violates `target_rejection_budget`.")

    source_history: list[BoolArray] = [source_gate.copy()]
    target_history: list[BoolArray] = [target_gate.copy()]
    objective_history: list[float] = []
    stage_history: list[str] = []
    seen = {(source_gate.tobytes(), target_gate.tobytes()): 0}
    log_warm: tuple[FloatArray, FloatArray] | None = None
    result: BalancedOTResult | None = None
    solved_source = source_gate.copy()
    solved_target = target_gate.copy()
    all_inner = True
    outer_converged = False
    cycle_detected = False
    cycle_length = 0
    source_constraints = 0
    target_constraints = 0
    transport_solves = 0
    total_inner = 0
    completed = 0

    for outer in range(max_outer_iterations):
        completed = outer + 1
        solved_source = source_gate.copy()
        solved_target = target_gate.copy()
        result = solve_fixed_bidirectional_balanced_ot(
            cost,
            source_gate,
            target_gate,
            rejection_cost=c,
            epsilon=epsilon,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=threshold,
            max_iterations=max_iterations,
            warm_start=log_warm if warm_start else None,
        )
        transport_solves += 1
        total_inner += result.n_iterations
        all_inner &= result.converged
        if not result.converged:
            raise BalancedInnerSolverError(
                f"Balanced Sinkhorn failed at outer iteration {outer}: "
                f"marginal error={result.marginal_error:.6g}."
            )
        objective_history.append(result.objective)
        stage_history.append("transport")
        source_loss, _, source_partner_mass, _ = _partner_losses(
            result.coupling, cost, source_gate, target_gate
        )
        source_cf, target_cf = _counterfactual_losses(result, cost)
        previous_source = source_gate.copy()
        previous_target = target_gate.copy()

        if update_source:
            source_coefficient = (
                np.sum(
                    result.coupling
                    * target_gate[None, :]
                    * (cost - c),
                    axis=1,
                )
                if variant == "exact"
                else source_partner_mass * (source_cf - c)
            )
            update: GateUpdateDiagnostics = constrained_gate_update(
                source_coefficient,
                source_gate,
                min_accepted=source_min,
                tau_s=tau_s,
                tolerance_scale=source_partner_mass,
            )
            source_gate = update.gate
            source_constraints += int(update.constraint_active)
            objective_history.append(
                _objective(result, cost, source_gate, target_gate, c)
            )
            stage_history.append("source_gate")

        _, target_loss, _, target_partner_mass = _partner_losses(
            result.coupling, cost, source_gate, target_gate
        )
        if update_target:
            target_coefficient = (
                np.sum(
                    result.coupling
                    * source_gate[:, None]
                    * (cost - c),
                    axis=0,
                )
                if variant == "exact"
                else target_partner_mass * (target_cf - c)
            )
            update = constrained_gate_update(
                target_coefficient,
                target_gate,
                min_accepted=target_min,
                tau_s=tau_s,
                tolerance_scale=target_partner_mass,
            )
            target_gate = update.gate
            target_constraints += int(update.constraint_active)
            objective_history.append(
                _objective(result, cost, source_gate, target_gate, c)
            )
            stage_history.append("target_gate")

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

    assert result is not None
    if not np.array_equal(solved_source, source_gate) or not np.array_equal(
        solved_target, target_gate
    ):
        result = solve_fixed_bidirectional_balanced_ot(
            cost,
            source_gate,
            target_gate,
            rejection_cost=c,
            epsilon=epsilon,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=threshold,
            max_iterations=max_iterations,
            warm_start=(result.log_source_scaling, result.log_target_scaling)
            if warm_start
            else None,
        )
        transport_solves += 1
        total_inner += result.n_iterations
        all_inner &= result.converged
        if not result.converged:
            raise BalancedInnerSolverError(
                "Balanced Sinkhorn failed during terminal consistency re-solve."
            )
        objective_history.append(result.objective)
        stage_history.append("terminal_transport")

    source_loss, target_loss, source_partner_mass, target_partner_mass = _partner_losses(
        result.coupling, cost, source_gate, target_gate
    )
    source_cf, target_cf = _counterfactual_losses(result, cost)
    exact_source_coefficient = np.sum(
        result.coupling * target_gate[None, :] * (cost - c), axis=1
    )
    exact_target_coefficient = np.sum(
        result.coupling * source_gate[:, None] * (cost - c), axis=0
    )
    if variant == "exact":
        source_coefficient = exact_source_coefficient
        target_coefficient = exact_target_coefficient
        source_score = source_loss
        target_score = target_loss
    else:
        source_coefficient = source_partner_mass * (source_cf - c)
        target_coefficient = target_partner_mass * (target_cf - c)
        source_score = source_cf
        target_score = target_cf

    source_raw_gate = source_coefficient < 0.0
    target_raw_gate = target_coefficient < 0.0
    terminal_source = constrained_gate_update(
        source_coefficient,
        source_gate,
        min_accepted=source_min,
        tau_s=tau_s,
        tolerance_scale=source_partner_mass,
    )
    terminal_target = constrained_gate_update(
        target_coefficient,
        target_gate,
        min_accepted=target_min,
        tau_s=tau_s,
        tolerance_scale=target_partner_mass,
    )
    return BidirectionalBalancedOTResult(
        coupling=result.coupling,
        transition_probability=result.transition_probability,
        reverse_transition_probability=_reverse_transition(result.coupling),
        source_marginal=result.source_marginal,
        target_marginal=result.target_marginal,
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
        source_gate_coefficient=source_coefficient,
        target_gate_coefficient=target_coefficient,
        log_source_scaling=result.log_source_scaling,
        log_target_scaling=result.log_target_scaling,
        objective=result.objective,
        objective_history=tuple(objective_history),
        objective_stage_history=tuple(stage_history),
        source_gate_history=tuple(source_history),
        target_gate_history=tuple(target_history),
        variant=variant,
        rejection_cost=c,
        epsilon=epsilon,
        tau_s=tau_s,
        source_rejection_budget=source_budget,
        target_rejection_budget=target_budget,
        source_min_accepted=source_min,
        target_min_accepted=target_min,
        source_budget_binding=terminal_source.constraint_active,
        target_budget_binding=terminal_target.constraint_active,
        source_boundary_count=int(
            np.count_nonzero(
                np.abs(source_coefficient) <= tau_s * source_partner_mass
            )
        ),
        target_boundary_count=int(
            np.count_nonzero(
                np.abs(target_coefficient) <= tau_s * target_partner_mass
            )
        ),
        source_constraint_count=source_constraints,
        target_constraint_count=target_constraints,
        inner_converged=all_inner,
        outer_converged=outer_converged,
        cycle_detected=cycle_detected,
        cycle_length=cycle_length,
        n_outer_iterations=completed,
        n_transport_solves=transport_solves,
        total_inner_iterations=total_inner,
    )
