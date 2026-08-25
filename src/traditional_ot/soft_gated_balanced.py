"""Soft-gated bidirectional confidence filtering over balanced entropic OT.

The gates reweight and renormalize the empirical marginals.  Unlike the
unbalanced implementation, the envelope gradient is computed from centered
balanced dual potentials.  Finite Sinkhorn iterates are rounded onto the
transport polytope before they are used as primal upper bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from time import perf_counter
from typing import Literal, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from traditional_ot.balanced import (
    BalancedOTResult,
    _cost_matrix,
    _generalized_kl,
    _marginal,
    _positive_finite,
    _positive_integer,
    _solve_balanced,
)
from traditional_ot.soft_gated import project_soft_coverage
from traditional_ot.support_restricted import solve_fixed_support_ot


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
BalancedSoftStatus = Literal[
    "numerically-soft-stationary",
    "line-search-failure",
    "iteration-capped",
    "inner-iteration-capped",
    "inner-solver-failure",
]


@dataclass(frozen=True)
class BalancedSoftInnerState:
    source_gate: FloatArray
    target_gate: FloatArray
    source_reference: FloatArray
    target_reference: FloatArray
    result: BalancedOTResult
    rounded_coupling: FloatArray
    source_potential: FloatArray
    target_potential: FloatArray
    upper_bound: float
    lower_bound: float
    gap: float
    marginal_residual: float
    rounded_marginal_residual: float
    gauge: str = "native Sinkhorn scaling; gradients are centered and gauge-invariant"


@dataclass(frozen=True)
class SoftGatedBalancedOTResult:
    status: BalancedSoftStatus
    soft_coupling: FloatArray
    rounded_soft_coupling: FloatArray
    coupling: FloatArray
    transition_probability: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    source_soft_gate: FloatArray
    target_soft_gate: FloatArray
    source_gate: BoolArray
    target_gate: BoolArray
    source_raw_gate: BoolArray
    target_raw_gate: BoolArray
    source_readout_overridden: bool
    target_readout_overridden: bool
    source_fractional_suppression: float
    target_fractional_suppression: float
    source_projected_gradient_norm: float
    target_projected_gradient_norm: float
    source_potential: FloatArray
    target_potential: FloatArray
    source_escape_score: FloatArray
    target_escape_score: FloatArray
    objective: float
    lower_bound: float
    primal_dual_gap: float
    inner_converged: bool
    inner_marginal_residual: float
    rounded_marginal_residual: float
    objective_history: tuple[float, ...]
    n_outer_iterations: int
    n_transport_solves: int
    total_inner_iterations: int
    line_search_backtracks: int
    source_rejection_budget: float
    target_rejection_budget: float
    gate_floor: float
    readout_threshold: float


@dataclass(frozen=True)
class SoftGatedBalancedRun:
    initialization: str
    initial_source_gate: FloatArray
    initial_target_gate: FloatArray
    result: SoftGatedBalancedOTResult
    fit_seconds: float


@dataclass(frozen=True)
class SoftGatedBalancedMultiStartResult:
    runs: tuple[SoftGatedBalancedRun, ...]
    source_escape_score: FloatArray
    target_escape_score: FloatArray
    diagnostic_seconds: float


def _soft_gate(
    value: ArrayLike | None,
    *,
    n: int,
    gate_floor: float,
    budget: int,
    name: str,
) -> FloatArray:
    gate = np.ones(n, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64).copy()
    if gate.shape != (n,) or not np.all(np.isfinite(gate)):
        raise ValueError(f"`{name}` must be a finite vector of shape ({n},).")
    if np.any(gate < gate_floor) or np.any(gate > 1.0):
        raise ValueError(f"`{name}` must lie in [gate_floor, 1].")
    if float(np.sum(1.0 - gate)) > budget + 1e-10:
        raise ValueError(f"`{name}` violates its soft rejection budget.")
    return gate


def _gated_reference(gate: FloatArray, base: FloatArray) -> FloatArray:
    weighted = gate * base
    total = float(weighted.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError("A gated balanced reference has non-positive mass.")
    return weighted / total


def round_transport_plan(
    coupling: ArrayLike,
    source_marginal: ArrayLike,
    target_marginal: ArrayLike,
) -> FloatArray:
    """Round a non-negative plan onto ``Pi(source_marginal, target_marginal)``.

    This is the diagonal-truncation plus rank-one correction of Altschuler,
    Weed and Rigollet (2017).  A tiny final correction is applied only for
    floating-point drift.
    """
    plan = np.asarray(coupling, dtype=np.float64).copy()
    a = np.asarray(source_marginal, dtype=np.float64)
    b = np.asarray(target_marginal, dtype=np.float64)
    if plan.shape != (a.size, b.size) or np.any(plan < 0.0):
        raise ValueError("The coupling shape/mass is incompatible with the marginals.")
    row = plan.sum(axis=1)
    row_scale = np.ones_like(a)
    np.divide(a, row, out=row_scale, where=row > 0.0)
    plan *= np.minimum(row_scale, 1.0)[:, None]
    column = plan.sum(axis=0)
    column_scale = np.ones_like(b)
    np.divide(b, column, out=column_scale, where=column > 0.0)
    plan *= np.minimum(column_scale, 1.0)[None, :]
    row_error = np.maximum(a - plan.sum(axis=1), 0.0)
    column_error = np.maximum(b - plan.sum(axis=0), 0.0)
    missing = float(row_error.sum())
    if missing > 0.0:
        column_missing = float(column_error.sum())
        if column_missing <= 0.0:
            raise FloatingPointError("Transport rounding produced inconsistent deficits.")
        plan += np.outer(row_error, column_error) / column_missing
    # The construction is exact algebraically; these two rank-one corrections
    # remove only machine-level drift and preserve non-negativity.
    row_delta = a - plan.sum(axis=1)
    column_delta = b - plan.sum(axis=0)
    if np.max(np.abs(row_delta)) > 5e-13 or np.max(np.abs(column_delta)) > 5e-13:
        raise FloatingPointError("Rounded plan failed the marginal-feasibility audit.")
    return plan


def _safe_transition(coupling: FloatArray) -> FloatArray:
    mass = coupling.sum(axis=1)
    result = np.zeros_like(coupling)
    np.divide(coupling, mass[:, None], out=result, where=mass[:, None] > 0.0)
    return result


def _readout(gate: FloatArray, *, threshold: float, max_rejections: int) -> tuple[BoolArray, BoolArray, bool]:
    raw = gate >= threshold
    projected = raw.copy()
    if int((~projected).sum()) > max_rejections:
        order = np.asarray(sorted(range(gate.size), key=lambda index: (-gate[index], index)))
        projected[:] = False
        projected[order[: gate.size - max_rejections]] = True
        return raw, projected, True
    if not np.any(projected):
        projected[int(np.argmax(gate))] = True
        return raw, projected, True
    return raw, projected, False


def _solve_inner(
    cost: FloatArray,
    source_gate: FloatArray,
    target_gate: FloatArray,
    *,
    base_source: FloatArray,
    base_target: FloatArray,
    epsilon: float,
    c_s: float,
    c_t: float,
    threshold: float,
    max_iterations: int,
    warm_start: tuple[ArrayLike, ArrayLike] | None = None,
) -> BalancedSoftInnerState:
    source_reference = _gated_reference(source_gate, base_source)
    target_reference = _gated_reference(target_gate, base_target)
    result = _solve_balanced(
        cost,
        scoring_cost=cost,
        epsilon=epsilon,
        source_weights=source_reference,
        target_weights=target_reference,
        threshold=threshold,
        max_iterations=max_iterations,
        warm_start=warm_start,
        normalize_weights=False,
    )
    rounded = round_transport_plan(result.coupling, source_reference, target_reference)
    reference = source_reference[:, None] * target_reference[None, :]
    penalty = c_s * float(np.sum(1.0 - source_gate)) + c_t * float(np.sum(1.0 - target_gate))
    upper = float(np.sum(rounded * cost) + epsilon * _generalized_kl(rounded, reference) + penalty)
    source_potential = epsilon * result.log_source_scaling
    target_potential = epsilon * result.log_target_scaling
    lower = float(
        np.dot(source_potential, source_reference)
        + np.dot(target_potential, target_reference)
        - epsilon * (float(result.coupling.sum()) - 1.0)
        + penalty
    )
    marginal_residual = float(max(
        np.max(np.abs(result.coupling.sum(axis=1) - source_reference)),
        np.max(np.abs(result.coupling.sum(axis=0) - target_reference)),
    ))
    rounded_residual = float(max(
        np.max(np.abs(rounded.sum(axis=1) - source_reference)),
        np.max(np.abs(rounded.sum(axis=0) - target_reference)),
    ))
    raw_gap = upper - lower
    audit_tolerance = 1e-10 * max(1.0, abs(upper), abs(lower))
    if raw_gap < -audit_tolerance:
        raise FloatingPointError(
            "Balanced primal/dual audit failed: the rounded primal upper "
            "bound lies below the dual lower bound."
        )
    gap = max(0.0, raw_gap)
    return BalancedSoftInnerState(
        source_gate=source_gate.copy(),
        target_gate=target_gate.copy(),
        source_reference=source_reference,
        target_reference=target_reference,
        result=result,
        rounded_coupling=rounded,
        source_potential=source_potential,
        target_potential=target_potential,
        upper_bound=upper,
        lower_bound=lower,
        gap=gap,
        marginal_residual=marginal_residual,
        rounded_marginal_residual=rounded_residual,
    )


def balanced_soft_envelope_gradient(
    state: BalancedSoftInnerState,
    *,
    base_source: ArrayLike,
    base_target: ArrayLike,
    c_s: float,
    c_t: float,
) -> tuple[FloatArray, FloatArray]:
    """Return the manuscript's gauge-invariant balanced envelope gradient."""
    a = np.asarray(base_source, dtype=np.float64)
    b = np.asarray(base_target, dtype=np.float64)
    source_denominator = float(np.dot(state.source_gate, a))
    target_denominator = float(np.dot(state.target_gate, b))
    centered_source = state.source_potential - float(
        np.dot(state.source_potential, state.source_reference)
    )
    centered_target = state.target_potential - float(
        np.dot(state.target_potential, state.target_reference)
    )
    return (
        a * centered_source / source_denominator - c_s,
        b * centered_target / target_denominator - c_t,
    )


def soft_gated_balanced_ot(
    cost_matrix: ArrayLike,
    *,
    epsilon: float,
    c_s: float,
    c_t: float,
    source_rejection_budget: float = 0.10,
    target_rejection_budget: float = 0.10,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    initial_source_gate: ArrayLike | None = None,
    initial_target_gate: ArrayLike | None = None,
    gate_floor: float = 1e-3,
    readout_threshold: float = 0.5,
    initial_step: float = 1.0,
    reference_step: float = 1.0,
    backtracking_factor: float = 0.5,
    armijo_constant: float = 1e-4,
    minimum_step: float = 1e-8,
    gate_tolerance: float = 1e-4,
    block_tolerance: float = 1e-4,
    gap_tolerance: float = 1e-7,
    threshold: float = 1e-9,
    max_iterations: int = 20_000,
    max_outer_iterations: int = 100,
    warn_on_terminal: bool = True,
) -> SoftGatedBalancedOTResult:
    """Fit the direct-box soft-gated balanced formulation.

    Capped and line-search-failed terminal states are retained.  They emit a
    warning by default and are never labelled certified stationary.
    """
    cost = _cost_matrix(cost_matrix)
    eps = _positive_finite(epsilon, name="epsilon")
    price_s = _positive_finite(c_s, name="c_s")
    price_t = _positive_finite(c_t, name="c_t")
    floor_value = _positive_finite(gate_floor, name="gate_floor")
    if floor_value >= 1.0:
        raise ValueError("`gate_floor` must lie in (0, 1).")
    readout = _positive_finite(readout_threshold, name="readout_threshold")
    if not floor_value < readout < 1.0:
        raise ValueError("`readout_threshold` must lie strictly between gate_floor and 1.")
    step0 = _positive_finite(initial_step, name="initial_step")
    step_ref = _positive_finite(reference_step, name="reference_step")
    shrink = _positive_finite(backtracking_factor, name="backtracking_factor")
    if shrink >= 1.0:
        raise ValueError("`backtracking_factor` must lie in (0, 1).")
    armijo = _positive_finite(armijo_constant, name="armijo_constant")
    if armijo >= 1.0:
        raise ValueError("`armijo_constant` must lie in (0, 1).")
    min_step = _positive_finite(minimum_step, name="minimum_step")
    gate_tol = _positive_finite(gate_tolerance, name="gate_tolerance")
    block_tol = _positive_finite(block_tolerance, name="block_tolerance")
    gap_tol = _positive_finite(gap_tolerance, name="gap_tolerance")
    inner_tol = _positive_finite(threshold, name="threshold")
    inner_cap = _positive_integer(max_iterations, name="max_iterations")
    outer_cap = _positive_integer(max_outer_iterations, name="max_outer_iterations")
    for value, name in ((source_rejection_budget, "source_rejection_budget"), (target_rejection_budget, "target_rejection_budget")):
        if not np.isfinite(value) or value < 0.0 or value >= 1.0:
            raise ValueError(f"`{name}` must lie in [0, 1).")
    max_source = int(floor(source_rejection_budget * cost.shape[0] + 1e-12))
    max_target = int(floor(target_rejection_budget * cost.shape[1] + 1e-12))
    base_source = _marginal(source_weights, n=cost.shape[0], name="source_weights")
    base_target = _marginal(target_weights, n=cost.shape[1], name="target_weights")
    source = _soft_gate(initial_source_gate, n=cost.shape[0], gate_floor=floor_value, budget=max_source, name="initial_source_gate")
    target = _soft_gate(initial_target_gate, n=cost.shape[1], gate_floor=floor_value, budget=max_target, name="initial_target_gate")
    state = _solve_inner(
        cost, source, target, base_source=base_source, base_target=base_target,
        epsilon=eps, c_s=price_s, c_t=price_t,
        threshold=inner_tol, max_iterations=inner_cap,
    )
    n_solves = 1
    total_inner = state.result.n_iterations
    backtracks = 0
    objective_history = [state.upper_bound]
    status: BalancedSoftStatus = "iteration-capped"
    completed = 0
    source_pg = target_pg = float("inf")
    if not np.isfinite(state.gap):
        status = "inner-solver-failure"
    elif not state.result.converged or state.marginal_residual > inner_tol:
        status = "inner-iteration-capped"

    if status == "iteration-capped":
        for outer in range(outer_cap):
            completed = outer + 1
            failed = False
            for side in ("source", "target"):
                gradient_s, gradient_t = balanced_soft_envelope_gradient(
                    state, base_source=base_source, base_target=base_target,
                    c_s=price_s, c_t=price_t,
                )
                current = state.source_gate if side == "source" else state.target_gate
                gradient = gradient_s if side == "source" else gradient_t
                budget = max_source if side == "source" else max_target
                reference = project_soft_coverage(
                    current - step_ref * gradient,
                    gate_floor=floor_value,
                    max_fractional_suppression=budget,
                )
                mapping_norm = float(np.linalg.norm(reference - current) / step_ref)
                if side == "source":
                    source_pg = mapping_norm
                else:
                    target_pg = mapping_norm
                if mapping_norm <= block_tol:
                    continue
                step = step0
                accepted = False
                while step >= min_step:
                    candidate_gate = project_soft_coverage(
                        current - step * gradient,
                        gate_floor=floor_value,
                        max_fractional_suppression=budget,
                    )
                    displacement = candidate_gate - current
                    if float(np.linalg.norm(displacement)) <= 1e-14:
                        accepted = True
                        break
                    candidate = _solve_inner(
                        cost,
                        candidate_gate if side == "source" else state.source_gate,
                        candidate_gate if side == "target" else state.target_gate,
                        base_source=base_source,
                        base_target=base_target,
                        epsilon=eps,
                        c_s=price_s,
                        c_t=price_t,
                        threshold=inner_tol,
                        max_iterations=inner_cap,
                        warm_start=(state.result.log_source_scaling, state.result.log_target_scaling),
                    )
                    n_solves += 1
                    total_inner += candidate.result.n_iterations
                    if not candidate.result.converged or candidate.marginal_residual > inner_tol:
                        status = "inner-iteration-capped"
                        failed = True
                        break
                    required = armijo * float(np.dot(displacement, displacement)) / step
                    if candidate.gap <= gap_tol and state.gap <= gap_tol and candidate.upper_bound <= state.lower_bound - required:
                        state = candidate
                        objective_history.append(state.upper_bound)
                        accepted = True
                        break
                    step *= shrink
                    backtracks += 1
                if failed:
                    break
                if not accepted:
                    status = "line-search-failure"
                    failed = True
                    break
            if failed:
                break
            gradient_s, gradient_t = balanced_soft_envelope_gradient(
                state, base_source=base_source, base_target=base_target,
                c_s=price_s, c_t=price_t,
            )
            source_pg = float(np.linalg.norm(project_soft_coverage(
                state.source_gate - step_ref * gradient_s,
                gate_floor=floor_value, max_fractional_suppression=max_source,
            ) - state.source_gate) / step_ref)
            target_pg = float(np.linalg.norm(project_soft_coverage(
                state.target_gate - step_ref * gradient_t,
                gate_floor=floor_value, max_fractional_suppression=max_target,
            ) - state.target_gate) / step_ref)
            if source_pg <= gate_tol and target_pg <= gate_tol and state.marginal_residual <= inner_tol and state.gap <= gap_tol:
                status = "numerically-soft-stationary"
                break

    raw_source, hard_source, source_override = _readout(state.source_gate, threshold=readout, max_rejections=max_source)
    raw_target, hard_target, target_override = _readout(state.target_gate, threshold=readout, max_rejections=max_target)
    if status == "numerically-soft-stationary":
        hard = solve_fixed_support_ot(
            cost, hard_source, hard_target,
            backbone="balanced", epsilon=eps,
            kappa_s=price_s * cost.shape[0], kappa_t=price_t * cost.shape[1],
            threshold=inner_tol, max_iterations=inner_cap,
        )
        coupling = hard.coupling
        transition = hard.transition_probability
        source_mass = hard.source_mass
        target_mass = hard.target_mass
        n_solves += 1
        total_inner += hard.n_iterations
    else:
        coupling = state.rounded_coupling
        transition = _safe_transition(coupling)
        source_mass = coupling.sum(axis=1)
        target_mass = coupling.sum(axis=0)
        if warn_on_terminal:
            warnings.warn(
                f"Soft-gated balanced OT retained terminal result with status={status}.",
                RuntimeWarning,
                stacklevel=2,
            )
    source_centered = state.source_potential - float(np.dot(state.source_potential, state.source_reference))
    target_centered = state.target_potential - float(np.dot(state.target_potential, state.target_reference))
    return SoftGatedBalancedOTResult(
        status=status,
        soft_coupling=state.result.coupling,
        rounded_soft_coupling=state.rounded_coupling,
        coupling=coupling,
        transition_probability=transition,
        source_mass=source_mass,
        target_mass=target_mass,
        source_soft_gate=state.source_gate,
        target_soft_gate=state.target_gate,
        source_gate=hard_source,
        target_gate=hard_target,
        source_raw_gate=raw_source,
        target_raw_gate=raw_target,
        source_readout_overridden=source_override,
        target_readout_overridden=target_override,
        source_fractional_suppression=float(np.sum(1.0 - state.source_gate)),
        target_fractional_suppression=float(np.sum(1.0 - state.target_gate)),
        source_projected_gradient_norm=source_pg,
        target_projected_gradient_norm=target_pg,
        source_potential=state.source_potential,
        target_potential=state.target_potential,
        source_escape_score=base_source * source_centered,
        target_escape_score=base_target * target_centered,
        objective=state.upper_bound,
        lower_bound=state.lower_bound,
        primal_dual_gap=state.gap,
        inner_converged=state.result.converged,
        inner_marginal_residual=state.marginal_residual,
        rounded_marginal_residual=state.rounded_marginal_residual,
        objective_history=tuple(objective_history),
        n_outer_iterations=completed,
        n_transport_solves=n_solves,
        total_inner_iterations=total_inner,
        line_search_backtracks=backtracks,
        source_rejection_budget=float(source_rejection_budget),
        target_rejection_budget=float(target_rejection_budget),
        gate_floor=floor_value,
        readout_threshold=readout,
    )


def multi_start_soft_gated_balanced_ot(
    cost_matrix: ArrayLike,
    *,
    epsilon: float,
    c_s: float,
    c_t: float,
    source_rejection_budget: float = 0.10,
    target_rejection_budget: float = 0.10,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    perturbation_amplitude: float = 0.5,
    perturbation_count: int | None = None,
    random_seeds: Sequence[int] = (0,),
    **fit_options: object,
) -> SoftGatedBalancedMultiStartResult:
    """Run all-one, deterministic, escape-score and fixed random starts."""
    if "initial_source_gate" in fit_options or "initial_target_gate" in fit_options:
        raise ValueError("Initial gates are constructed by the multistart wrapper.")
    cost = _cost_matrix(cost_matrix)
    eps = _positive_finite(epsilon, name="epsilon")
    price_s = _positive_finite(c_s, name="c_s")
    price_t = _positive_finite(c_t, name="c_t")
    amplitude = _positive_finite(perturbation_amplitude, name="perturbation_amplitude")
    gate_floor = float(fit_options.get("gate_floor", 1e-3))
    if amplitude >= 1.0 - gate_floor:
        raise ValueError("`perturbation_amplitude` must be smaller than 1 - gate_floor.")
    max_source = int(floor(source_rejection_budget * cost.shape[0] + 1e-12))
    max_target = int(floor(target_rejection_budget * cost.shape[1] + 1e-12))
    source_count = max_source if perturbation_count is None else min(int(perturbation_count), max_source)
    target_count = max_target if perturbation_count is None else min(int(perturbation_count), max_target)
    base_source = _marginal(source_weights, n=cost.shape[0], name="source_weights")
    base_target = _marginal(target_weights, n=cost.shape[1], name="target_weights")
    all_source = np.ones(cost.shape[0])
    all_target = np.ones(cost.shape[1])
    diagnostic_start = perf_counter()
    ungated = _solve_inner(
        cost, all_source, all_target,
        base_source=base_source, base_target=base_target,
        epsilon=eps, c_s=price_s, c_t=price_t,
        threshold=float(fit_options.get("threshold", 1e-9)),
        max_iterations=int(fit_options.get("max_iterations", 20_000)),
    )
    source_escape = base_source * (
        ungated.source_potential - float(np.dot(ungated.source_potential, base_source))
    )
    target_escape = base_target * (
        ungated.target_potential - float(np.dot(ungated.target_potential, base_target))
    )
    diagnostic_seconds = perf_counter() - diagnostic_start

    def ranked_gate(score: FloatArray, count: int) -> FloatArray:
        gate = np.ones(score.size)
        if count:
            order = np.asarray(sorted(range(score.size), key=lambda index: (-score[index], index)))
            gate[order[:count]] = 1.0 - amplitude
        return gate

    initializations: list[tuple[str, FloatArray, FloatArray]] = [("all-one", all_source.copy(), all_target.copy())]
    if max_source or max_target:
        initializations.append(("deterministic", ranked_gate(base_source, source_count), ranked_gate(base_target, target_count)))
        initializations.append(("escape-score", ranked_gate(source_escape, source_count), ranked_gate(target_escape, target_count)))
        for seed in random_seeds:
            rng = np.random.default_rng(int(seed))
            initializations.append((
                f"random-{int(seed)}",
                project_soft_coverage(rng.uniform(1.0 - amplitude, 1.0, cost.shape[0]), gate_floor=gate_floor, max_fractional_suppression=max_source),
                project_soft_coverage(rng.uniform(1.0 - amplitude, 1.0, cost.shape[1]), gate_floor=gate_floor, max_fractional_suppression=max_target),
            ))
    runs = []
    for label, initial_source, initial_target in initializations:
        start = perf_counter()
        result = soft_gated_balanced_ot(
            cost,
            epsilon=eps, c_s=price_s, c_t=price_t,
            source_rejection_budget=source_rejection_budget,
            target_rejection_budget=target_rejection_budget,
            source_weights=base_source, target_weights=base_target,
            initial_source_gate=initial_source, initial_target_gate=initial_target,
            **fit_options,
        )
        runs.append(SoftGatedBalancedRun(
            initialization=label,
            initial_source_gate=initial_source.copy(),
            initial_target_gate=initial_target.copy(),
            result=result,
            fit_seconds=perf_counter() - start,
        ))
    return SoftGatedBalancedMultiStartResult(
        runs=tuple(runs),
        source_escape_score=source_escape,
        target_escape_score=target_escape,
        diagnostic_seconds=diagnostic_seconds,
    )
