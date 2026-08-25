"""Soft-gated bidirectional confidence-filtered KL-unbalanced OT.

This module implements the direct-box continuous gate formulation.  Positive
soft gates reweight and renormalize the UOT reference marginals; a safeguarded
projected-gradient outer loop minimizes the reduced objective, followed by a
prespecified binary readout and hard-support UOT refit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from time import perf_counter
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from traditional_ot.support_restricted import solve_fixed_support_ot
from traditional_ot.unbalanced import (
    UOTResult,
    _cost_matrix,
    _marginal,
    _positive_finite,
    _positive_integer,
    _solve_uot,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
SoftStatus = Literal[
    "numerically-soft-stationary",
    "line-search-failure",
    "iteration-capped",
    "inner-iteration-capped",
    "inner-solver-failure",
]


@dataclass(frozen=True)
class SoftGateInnerState:
    source_gate: FloatArray
    target_gate: FloatArray
    source_reference: FloatArray
    target_reference: FloatArray
    result: UOTResult
    upper_bound: float
    lower_bound: float
    gap: float


@dataclass(frozen=True)
class SoftGatedUOTResult:
    status: SoftStatus
    soft_coupling: FloatArray
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
    objective: float
    lower_bound: float
    primal_dual_gap: float
    inner_converged: bool
    inner_fixed_point_error: float
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
class SoftGatedRun:
    """One explicitly labelled member of a soft-gate multistart fit."""

    initialization: str
    initial_source_gate: FloatArray
    initial_target_gate: FloatArray
    result: SoftGatedUOTResult
    fit_seconds: float


@dataclass(frozen=True)
class SoftGatedMultiStartResult:
    """All runs from the manuscript-prescribed initialization diagnostic."""

    runs: tuple[SoftGatedRun, ...]
    source_escape_score: FloatArray
    target_escape_score: FloatArray
    diagnostic_seconds: float


def project_soft_coverage(
    values: ArrayLike,
    *,
    gate_floor: float,
    max_fractional_suppression: int,
    tolerance: float = 1e-12,
) -> FloatArray:
    """Euclidean projection onto ``[gate_floor,1]^N`` with a coverage floor."""

    zeta = np.asarray(values, dtype=np.float64)
    if zeta.ndim != 1 or zeta.size == 0 or not np.all(np.isfinite(zeta)):
        raise ValueError("`values` must be a non-empty finite one-dimensional array.")
    floor_value = _positive_finite(gate_floor, name="gate_floor")
    if floor_value >= 1.0:
        raise ValueError("`gate_floor` must lie in (0, 1).")
    if isinstance(max_fractional_suppression, (bool, np.bool_)) or not isinstance(
        max_fractional_suppression, (int, np.integer)
    ):
        raise ValueError("`max_fractional_suppression` must be an integer.")
    budget = int(max_fractional_suppression)
    if budget < 0 or budget >= zeta.size:
        raise ValueError("`max_fractional_suppression` must lie in {0, ..., N-1}.")
    clipped = np.clip(zeta, floor_value, 1.0)
    required = float(zeta.size - budget)
    if float(clipped.sum()) >= required - tolerance:
        return clipped
    low = 0.0
    high = max(0.0, 1.0 - float(np.min(zeta)))
    for _ in range(100):
        midpoint = 0.5 * (low + high)
        candidate = np.clip(zeta + midpoint, floor_value, 1.0)
        if float(candidate.sum()) < required:
            low = midpoint
        else:
            high = midpoint
        if high - low <= tolerance:
            break
    return np.clip(zeta + high, floor_value, 1.0)


def _soft_gate(
    value: ArrayLike | None,
    *,
    n: int,
    gate_floor: float,
    budget: int,
    name: str,
) -> FloatArray:
    if value is None:
        result = np.ones(n, dtype=np.float64)
    else:
        result = np.asarray(value, dtype=np.float64).copy()
        if result.shape != (n,) or not np.all(np.isfinite(result)):
            raise ValueError(f"`{name}` must be a finite vector of shape ({n},).")
    if np.any(result < gate_floor) or np.any(result > 1.0):
        raise ValueError(f"`{name}` must lie in [gate_floor, 1].")
    if float(np.sum(1.0 - result)) > budget + 1e-10:
        raise ValueError(f"`{name}` violates its soft rejection budget.")
    return result


def _dual_value(result: UOTResult, cost: FloatArray) -> float:
    a = result.source_marginal
    b = result.target_marginal
    f = result.epsilon * result.log_source_scaling
    g = result.epsilon * result.log_target_scaling
    with np.errstate(over="raise", invalid="raise"):
        source = -result.lambda_a * np.sum(a * np.expm1(-f / result.lambda_a))
        target = -result.lambda_b * np.sum(b * np.expm1(-g / result.lambda_b))
        interaction = -result.epsilon * np.sum(
            a[:, None]
            * b[None, :]
            * np.expm1((f[:, None] + g[None, :] - cost) / result.epsilon)
        )
    return float(source + target + interaction)


def _solve_inner(
    cost: FloatArray,
    source_gate: FloatArray,
    target_gate: FloatArray,
    *,
    base_source: FloatArray,
    base_target: FloatArray,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    c_s: float,
    c_t: float,
    threshold: float,
    max_iterations: int,
    warm_start: tuple[FloatArray, FloatArray] | None = None,
) -> SoftGateInnerState:
    source_reference = source_gate * base_source
    source_reference /= source_reference.sum()
    target_reference = target_gate * base_target
    target_reference /= target_reference.sum()
    result = _solve_uot(
        cost,
        scoring_cost=cost,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        source_weights=source_reference,
        target_weights=target_reference,
        threshold=threshold,
        max_iterations=max_iterations,
        warm_start=warm_start,
    )
    penalty = c_s * float(np.sum(1.0 - source_gate)) + c_t * float(
        np.sum(1.0 - target_gate)
    )
    upper = float(result.objective + penalty)
    lower = float(_dual_value(result, cost) + penalty)
    return SoftGateInnerState(
        source_gate=source_gate.copy(),
        target_gate=target_gate.copy(),
        source_reference=source_reference,
        target_reference=target_reference,
        result=result,
        upper_bound=upper,
        lower_bound=lower,
        gap=float(upper - lower),
    )


def _gradient(
    state: SoftGateInnerState,
    *,
    base_source: FloatArray,
    base_target: FloatArray,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    c_s: float,
    c_t: float,
) -> tuple[FloatArray, FloatArray]:
    mass = float(state.result.coupling.sum())
    source_denominator = float(np.dot(state.source_gate, base_source))
    target_denominator = float(np.dot(state.target_gate, base_target))
    source = (epsilon + lambda_a) * (
        mass * base_source / source_denominator
        - state.result.source_mass / state.source_gate
    ) - c_s
    target = (epsilon + lambda_b) * (
        mass * base_target / target_denominator
        - state.result.target_mass / state.target_gate
    ) - c_t
    return source, target


def _readout(gate: FloatArray, *, threshold: float, max_rejections: int) -> tuple[BoolArray, BoolArray, bool]:
    raw = gate >= threshold
    projected = raw.copy()
    maximum = int(max_rejections)
    if int((~projected).sum()) > maximum:
        order = np.asarray(sorted(range(gate.size), key=lambda i: (-gate[i], i)))
        projected[:] = False
        projected[order[: gate.size - maximum]] = True
        return raw, projected, True
    if not np.any(projected):
        projected[int(np.argmax(gate))] = True
        return raw, projected, True
    return raw, projected, False


def soft_gated_uot(
    cost_matrix: ArrayLike,
    *,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
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
    gap_tolerance: float = 1e-4,
    threshold: float = 1e-4,
    max_iterations: int = 20_000,
    max_outer_iterations: int = 100,
) -> SoftGatedUOTResult:
    """Fit the safeguarded direct-box soft-gated UOT formulation."""

    cost = _cost_matrix(cost_matrix)
    eps = _positive_finite(epsilon, name="epsilon")
    la = _positive_finite(lambda_a, name="lambda_a")
    lb = _positive_finite(lambda_b, name="lambda_b")
    price_s = _positive_finite(c_s, name="c_s")
    price_t = _positive_finite(c_t, name="c_t")
    floor_value = _positive_finite(gate_floor, name="gate_floor")
    if floor_value >= 1.0:
        raise ValueError("`gate_floor` must lie in (0, 1).")
    readout = _positive_finite(readout_threshold, name="readout_threshold")
    if readout >= 1.0:
        raise ValueError("`readout_threshold` must lie in (0, 1).")
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
    for value, name in (
        (source_rejection_budget, "source_rejection_budget"),
        (target_rejection_budget, "target_rejection_budget"),
    ):
        if not np.isfinite(value) or value < 0.0 or value >= 1.0:
            raise ValueError(f"`{name}` must lie in [0, 1).")
    max_source = int(floor(source_rejection_budget * cost.shape[0] + 1e-12))
    max_target = int(floor(target_rejection_budget * cost.shape[1] + 1e-12))
    base_source = _marginal(source_weights, n=cost.shape[0], name="source_weights")
    base_target = _marginal(target_weights, n=cost.shape[1], name="target_weights")
    source = _soft_gate(
        initial_source_gate,
        n=cost.shape[0],
        gate_floor=floor_value,
        budget=max_source,
        name="initial_source_gate",
    )
    target = _soft_gate(
        initial_target_gate,
        n=cost.shape[1],
        gate_floor=floor_value,
        budget=max_target,
        name="initial_target_gate",
    )
    n_solves = 0
    total_inner = 0
    backtracks = 0
    objective_history: list[float] = []
    status: SoftStatus = "iteration-capped"
    state = _solve_inner(
        cost, source, target,
        base_source=base_source, base_target=base_target,
        epsilon=eps, lambda_a=la, lambda_b=lb,
        c_s=price_s, c_t=price_t,
        threshold=inner_tol, max_iterations=inner_cap,
    )
    n_solves += 1
    total_inner += state.result.n_iterations
    objective_history.append(state.upper_bound)
    if not np.isfinite(state.gap):
        status = "inner-solver-failure"
    elif not state.result.converged:
        status = "inner-iteration-capped"
    completed = 0
    source_pg = target_pg = float("inf")
    if status == "iteration-capped":
        for outer in range(outer_cap):
            completed = outer + 1
            failed = False
            for side in ("source", "target"):
                gradient_s, gradient_t = _gradient(
                    state,
                    base_source=base_source, base_target=base_target,
                    epsilon=eps, lambda_a=la, lambda_b=lb,
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
                    candidate_source = candidate_gate if side == "source" else state.source_gate
                    candidate_target = candidate_gate if side == "target" else state.target_gate
                    candidate = _solve_inner(
                        cost, candidate_source, candidate_target,
                        base_source=base_source, base_target=base_target,
                        epsilon=eps, lambda_a=la, lambda_b=lb,
                        c_s=price_s, c_t=price_t,
                        threshold=inner_tol, max_iterations=inner_cap,
                        warm_start=(
                            state.result.log_source_scaling,
                            state.result.log_target_scaling,
                        ),
                    )
                    n_solves += 1
                    total_inner += candidate.result.n_iterations
                    if not np.isfinite(candidate.gap):
                        status = "inner-solver-failure"
                        failed = True
                        break
                    if not candidate.result.converged:
                        status = "inner-iteration-capped"
                        failed = True
                        break
                    required = armijo * float(np.dot(displacement, displacement)) / step
                    if (
                        candidate.gap <= gap_tol
                        and state.gap <= gap_tol
                        and candidate.upper_bound <= state.lower_bound - required
                    ):
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
            gradient_s, gradient_t = _gradient(
                state,
                base_source=base_source, base_target=base_target,
                epsilon=eps, lambda_a=la, lambda_b=lb,
                c_s=price_s, c_t=price_t,
            )
            source_pg = float(np.linalg.norm(
                project_soft_coverage(
                    state.source_gate - step_ref * gradient_s,
                    gate_floor=floor_value,
                    max_fractional_suppression=max_source,
                ) - state.source_gate
            ) / step_ref)
            target_pg = float(np.linalg.norm(
                project_soft_coverage(
                    state.target_gate - step_ref * gradient_t,
                    gate_floor=floor_value,
                    max_fractional_suppression=max_target,
                ) - state.target_gate
            ) / step_ref)
            if (
                source_pg <= gate_tol
                and target_pg <= gate_tol
                and state.result.fixed_point_error <= inner_tol
                and state.gap <= gap_tol
            ):
                status = "numerically-soft-stationary"
                break

    raw_source, hard_source, source_override = _readout(
        state.source_gate, threshold=readout, max_rejections=max_source
    )
    raw_target, hard_target, target_override = _readout(
        state.target_gate, threshold=readout, max_rejections=max_target
    )
    if status == "numerically-soft-stationary":
        hard = solve_fixed_support_ot(
            cost, hard_source, hard_target,
            backbone="unbalanced", epsilon=eps,
            lambda_a=la, lambda_b=lb,
            kappa_s=price_s * cost.shape[0],
            kappa_t=price_t * cost.shape[1],
            threshold=inner_tol, max_iterations=inner_cap,
        )
        coupling = hard.coupling
        transition = hard.transition_probability
        source_mass = hard.source_mass
        target_mass = hard.target_mass
        n_solves += 1
        total_inner += hard.n_iterations
    else:
        # Failed/capped fits retain their terminal soft state and terminal
        # hard readout as a warning-level diagnostic.  No hard-support refit
        # is claimed: coupling remains the terminal soft coupling.
        coupling = state.result.coupling
        transition = state.result.transition_probability
        source_mass = state.result.source_mass
        target_mass = state.result.target_mass
    return SoftGatedUOTResult(
        status=status,
        soft_coupling=state.result.coupling,
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
        objective=state.upper_bound,
        lower_bound=state.lower_bound,
        primal_dual_gap=state.gap,
        inner_converged=state.result.converged,
        inner_fixed_point_error=state.result.fixed_point_error,
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


def multi_start_soft_gated_uot(
    cost_matrix: ArrayLike,
    *,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
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
) -> SoftGatedMultiStartResult:
    """Run all-one and nontrivial initializations without collapsing outcomes.

    The constructors follow the manuscript protocol: an all-one run, a
    deterministic weight-ranked perturbation, an ungated escape-score-ranked
    perturbation, and optional fixed-seed random perturbations.  Every run and
    its initial gates are returned separately because the outer problem is
    nonconvex.
    """

    if "initial_source_gate" in fit_options or "initial_target_gate" in fit_options:
        raise ValueError("Initial gates are constructed by the multistart wrapper.")
    cost = _cost_matrix(cost_matrix)
    eps = _positive_finite(epsilon, name="epsilon")
    la = _positive_finite(lambda_a, name="lambda_a")
    lb = _positive_finite(lambda_b, name="lambda_b")
    price_s = _positive_finite(c_s, name="c_s")
    price_t = _positive_finite(c_t, name="c_t")
    amplitude = _positive_finite(perturbation_amplitude, name="perturbation_amplitude")
    gate_floor = float(fit_options.get("gate_floor", 1e-3))
    if amplitude >= 1.0 - gate_floor:
        raise ValueError("`perturbation_amplitude` must be smaller than 1 - gate_floor.")
    for value, name in (
        (source_rejection_budget, "source_rejection_budget"),
        (target_rejection_budget, "target_rejection_budget"),
    ):
        if not np.isfinite(value) or value < 0.0 or value >= 1.0:
            raise ValueError(f"`{name}` must lie in [0, 1).")
    max_source = int(floor(source_rejection_budget * cost.shape[0] + 1e-12))
    max_target = int(floor(target_rejection_budget * cost.shape[1] + 1e-12))
    if perturbation_count is None:
        source_count = max_source
        target_count = max_target
    else:
        count = _positive_integer(perturbation_count, name="perturbation_count")
        source_count = min(count, max_source)
        target_count = min(count, max_target)
    base_source = _marginal(source_weights, n=cost.shape[0], name="source_weights")
    base_target = _marginal(target_weights, n=cost.shape[1], name="target_weights")

    all_source = np.ones(cost.shape[0], dtype=np.float64)
    all_target = np.ones(cost.shape[1], dtype=np.float64)
    threshold = float(fit_options.get("threshold", 1e-4))
    max_iterations = int(fit_options.get("max_iterations", 20_000))
    diagnostic_start = perf_counter()
    ungated = _solve_inner(
        cost, all_source, all_target,
        base_source=base_source, base_target=base_target,
        epsilon=eps, lambda_a=la, lambda_b=lb,
        c_s=price_s, c_t=price_t,
        threshold=threshold, max_iterations=max_iterations,
    )
    mass = float(ungated.result.coupling.sum())
    source_escape = (eps + la) * (mass * base_source - ungated.result.source_mass)
    target_escape = (eps + lb) * (mass * base_target - ungated.result.target_mass)
    diagnostic_seconds = perf_counter() - diagnostic_start

    initializations: list[tuple[str, FloatArray, FloatArray]] = [
        ("all-one", all_source.copy(), all_target.copy())
    ]

    def ranked_gate(weights: FloatArray, count: int) -> FloatArray:
        gate = np.ones(weights.size, dtype=np.float64)
        if count:
            order = np.asarray(sorted(range(weights.size), key=lambda i: (-weights[i], i)))
            gate[order[:count]] = 1.0 - amplitude
        return gate

    if max_source or max_target:
        initializations.append((
            "deterministic",
            ranked_gate(base_source, source_count),
            ranked_gate(base_target, target_count),
        ))
        initializations.append((
            "escape-score",
            ranked_gate(source_escape, source_count),
            ranked_gate(target_escape, target_count),
        ))
        for seed_value in random_seeds:
            if isinstance(seed_value, (bool, np.bool_)) or not isinstance(
                seed_value, (int, np.integer)
            ):
                raise ValueError("Every random seed must be an integer.")
            rng = np.random.default_rng(int(seed_value))
            source_random = project_soft_coverage(
                rng.uniform(1.0 - amplitude, 1.0, size=cost.shape[0]),
                gate_floor=gate_floor, max_fractional_suppression=max_source,
            )
            target_random = project_soft_coverage(
                rng.uniform(1.0 - amplitude, 1.0, size=cost.shape[1]),
                gate_floor=gate_floor, max_fractional_suppression=max_target,
            )
            initializations.append((
                f"random-{int(seed_value)}", source_random, target_random
            ))

    runs = []
    for label, initial_source, initial_target in initializations:
        fit_start = perf_counter()
        fit = soft_gated_uot(
            cost,
            epsilon=eps, lambda_a=la, lambda_b=lb,
            c_s=price_s, c_t=price_t,
            source_rejection_budget=source_rejection_budget,
            target_rejection_budget=target_rejection_budget,
            source_weights=base_source, target_weights=base_target,
            initial_source_gate=initial_source,
            initial_target_gate=initial_target,
            **fit_options,
        )
        fit_seconds = perf_counter() - fit_start
        runs.append(SoftGatedRun(
            initialization=label,
            initial_source_gate=initial_source.copy(),
            initial_target_gate=initial_target.copy(),
            result=fit,
            fit_seconds=fit_seconds,
        ))
    return SoftGatedMultiStartResult(
        runs=tuple(runs),
        source_escape_score=source_escape,
        target_escape_score=target_escape,
        diagnostic_seconds=diagnostic_seconds,
    )
