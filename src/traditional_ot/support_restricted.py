"""Support-restricted selective balanced and unbalanced optimal transport.

This module implements the frozen M4-P formulation.  Rejected endpoints are
removed from the transport support and the retained empirical marginals are
renormalized.  It is intentionally separate from :mod:`traditional_ot.selective`,
which implements the older cost-substitution objective.

Only uniform empirical cell weights are supported.  This is the setting in
which the exact prefix cardinality update in the accompanying theory applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from traditional_ot.balanced import BalancedOTResult, balanced_ot
from traditional_ot.unbalanced import UOTResult, _cost_matrix, _logsumexp, unbalanced_ot


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Backbone = Literal["balanced", "unbalanced"]
TerminalStatus = Literal[
    "conditionally-certified",
    "gate-stable-xi-large",
    "primal-dual-audit-failure",
    "nonfinite-audit-failure",
    "cycled",
    "iteration-capped",
    "inner-solver-failure",
]


@dataclass(frozen=True)
class PrefixSelection:
    """Exact cardinality sweep for one frozen-potential gate update."""

    gate: BoolArray
    order: NDArray[np.int64]
    accepted: int
    objective: float
    objectives: FloatArray
    admissible_cardinalities: NDArray[np.int64]
    budget_active: bool


@dataclass(frozen=True)
class FixedSupportResult:
    """One ordinary OT solve on a non-empty retained submatrix."""

    backbone: Backbone
    coupling: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    transition_probability: FloatArray
    reverse_transition_probability: FloatArray
    source_gate: BoolArray
    target_gate: BoolArray
    source_potential: FloatArray
    target_potential: FloatArray
    transport_objective: float
    full_objective: float
    solver_residual: float
    converged: bool
    n_iterations: int
    native_result: BalancedOTResult | UOTResult


@dataclass(frozen=True)
class SupportRestrictedOTResult:
    """Terminal result of the two-sided support-restricted M4-P iteration."""

    backbone: Backbone
    status: TerminalStatus
    coupling: FloatArray
    transition_probability: FloatArray
    reverse_transition_probability: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    source_gate: BoolArray
    target_gate: BoolArray
    source_price: FloatArray
    target_price: FloatArray
    source_potential: FloatArray
    target_potential: FloatArray
    transport_objective: float
    objective: float
    source_dual_bound: float
    target_dual_bound: float
    source_xi_raw: float
    target_xi_raw: float
    source_xi_report: float
    target_xi_report: float
    xi_certificate: float
    solver_residual: float
    source_rejection_budget: float
    target_rejection_budget: float
    max_source_rejections: int
    max_target_rejections: int
    source_budget_active: bool
    target_budget_active: bool
    gate_history: tuple[tuple[BoolArray, BoolArray], ...]
    objective_history: tuple[float, ...]
    n_outer_iterations: int
    n_transport_solves: int
    cycle_length: int
    source_cycle_acceptance_min: float
    source_cycle_acceptance_max: float
    target_cycle_acceptance_min: float
    target_cycle_acceptance_max: float
    terminal_gate_stable: bool
    inner_converged: bool


def _finite_nonnegative(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`{name}` must be a finite non-negative float.") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"`{name}` must be a finite non-negative float.")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result == 0.0:
        raise ValueError(f"`{name}` must be strictly positive.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"`{name}` must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return result


def _budget(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result >= 1.0:
        raise ValueError(f"`{name}` must lie in [0, 1).")
    return result


def _gate(value: ArrayLike | None, *, n: int, name: str, minimum: int) -> BoolArray:
    if value is None:
        result = np.ones(n, dtype=bool)
    else:
        raw = np.asarray(value)
        if raw.shape != (n,) or not np.all(np.isin(raw, (0, 1, False, True))):
            raise ValueError(f"`{name}` must have shape ({n},) and contain booleans.")
        result = raw.astype(bool, copy=True)
    if int(result.sum()) < minimum:
        raise ValueError(f"`{name}` violates its rejection budget.")
    return result


def prefix_select(
    prices: ArrayLike,
    previous_gate: ArrayLike,
    *,
    kappa: float,
    max_rejections: int,
) -> PrefixSelection:
    """Return the exact prefix/cardinality minimizer under a rejection budget.

    The minimized objective is

    ``mean(prices[accepted]) + (kappa / N) * (N - n_accepted)``.

    Prices may be signed.  All admissible cardinalities are swept because the
    cardinality objective need not be unimodal.  Equal prices prefer endpoints
    retained by the previous gate and then the smaller original index.  Exact
    objective ties use NumPy's first-minimum rule, i.e. the smaller cardinality.
    """

    rho = np.asarray(prices, dtype=np.float64)
    prev = np.asarray(previous_gate)
    if rho.ndim != 1 or rho.size == 0 or not np.all(np.isfinite(rho)):
        raise ValueError("`prices` must be a non-empty finite one-dimensional array.")
    if prev.shape != rho.shape or not np.all(np.isin(prev, (0, 1, False, True))):
        raise ValueError("`previous_gate` must be boolean and match `prices`.")
    if isinstance(max_rejections, (bool, np.bool_)) or not isinstance(
        max_rejections, (int, np.integer)
    ):
        raise ValueError("`max_rejections` must be an integer.")
    maximum = int(max_rejections)
    if maximum < 0 or maximum >= rho.size:
        raise ValueError("`max_rejections` must lie in {0, ..., N-1}.")
    kap = _finite_nonnegative(kappa, name="kappa")
    old = prev.astype(bool, copy=False)
    order = np.asarray(
        sorted(range(rho.size), key=lambda index: (rho[index], -int(old[index]), index)),
        dtype=np.int64,
    )
    sorted_prices = rho[order]
    cumulative = np.cumsum(sorted_prices)
    cardinalities = np.arange(max(1, rho.size - maximum), rho.size + 1, dtype=np.int64)
    values = cumulative[cardinalities - 1] / cardinalities
    values = values + (kap / rho.size) * (rho.size - cardinalities)
    position = int(np.argmin(values))
    accepted = int(cardinalities[position])
    gate = np.zeros(rho.size, dtype=bool)
    gate[order[:accepted]] = True
    return PrefixSelection(
        gate=gate,
        order=order,
        accepted=accepted,
        objective=float(values[position]),
        objectives=values,
        admissible_cardinalities=cardinalities,
        budget_active=(rho.size - accepted == maximum),
    )


def _safe_transitions(coupling: FloatArray) -> tuple[FloatArray, FloatArray]:
    source_mass = coupling.sum(axis=1)
    target_mass = coupling.sum(axis=0)
    forward = np.zeros_like(coupling)
    reverse = np.zeros_like(coupling)
    np.divide(coupling, source_mass[:, None], out=forward, where=source_mass[:, None] > 0.0)
    np.divide(coupling, target_mass[None, :], out=reverse, where=target_mass[None, :] > 0.0)
    return forward, reverse


def _penalty(
    source_gate: BoolArray,
    target_gate: BoolArray,
    kappa_s: float,
    kappa_t: float,
) -> float:
    return float(
        kappa_s * (1.0 - source_gate.mean())
        + kappa_t * (1.0 - target_gate.mean())
    )


def solve_fixed_support_ot(
    cost_matrix: ArrayLike,
    source_gate: ArrayLike,
    target_gate: ArrayLike,
    *,
    backbone: Backbone,
    epsilon: float,
    kappa_s: float,
    kappa_t: float,
    lambda_a: float | None = None,
    lambda_b: float | None = None,
    threshold: float = 1e-10,
    max_iterations: int = 20_000,
) -> FixedSupportResult:
    """Solve ordinary balanced OT or KL-UOT on the retained submatrix."""

    cost = _cost_matrix(cost_matrix)
    eps = _finite_positive(epsilon, name="epsilon")
    kap_s = _finite_nonnegative(kappa_s, name="kappa_s")
    kap_t = _finite_nonnegative(kappa_t, name="kappa_t")
    tol = _finite_positive(threshold, name="threshold")
    iterations = _positive_integer(max_iterations, name="max_iterations")
    source = _gate(source_gate, n=cost.shape[0], name="source_gate", minimum=1)
    target = _gate(target_gate, n=cost.shape[1], name="target_gate", minimum=1)
    source_index = np.flatnonzero(source)
    target_index = np.flatnonzero(target)
    restricted = cost[np.ix_(source_index, target_index)]

    if backbone == "balanced":
        if lambda_a is not None or lambda_b is not None:
            raise ValueError("`lambda_a` and `lambda_b` must be omitted for balanced OT.")
        native: BalancedOTResult | UOTResult = balanced_ot(
            restricted,
            epsilon=eps,
            threshold=tol,
            max_iterations=iterations,
        )
        residual = float(native.marginal_error)
    elif backbone == "unbalanced":
        if lambda_a is None or lambda_b is None:
            raise ValueError("Positive `lambda_a` and `lambda_b` are required for UOT.")
        la = _finite_positive(lambda_a, name="lambda_a")
        lb = _finite_positive(lambda_b, name="lambda_b")
        native = unbalanced_ot(
            restricted,
            epsilon=eps,
            lambda_a=la,
            lambda_b=lb,
            threshold=tol,
            max_iterations=iterations,
        )
        residual = float(native.fixed_point_error)
    else:
        raise ValueError("`backbone` must be 'balanced' or 'unbalanced'.")

    full = np.zeros(cost.shape, dtype=np.float64)
    full[np.ix_(source_index, target_index)] = native.coupling
    forward, reverse = _safe_transitions(full)
    f = np.full(cost.shape[0], np.nan, dtype=np.float64)
    g = np.full(cost.shape[1], np.nan, dtype=np.float64)
    f[source] = eps * native.log_source_scaling
    g[target] = eps * native.log_target_scaling
    penalty = _penalty(source, target, kap_s, kap_t)
    return FixedSupportResult(
        backbone=backbone,
        coupling=full,
        source_mass=full.sum(axis=1),
        target_mass=full.sum(axis=0),
        transition_probability=forward,
        reverse_transition_probability=reverse,
        source_gate=source,
        target_gate=target,
        source_potential=f,
        target_potential=g,
        transport_objective=float(native.objective),
        full_objective=float(native.objective + penalty),
        solver_residual=residual,
        converged=bool(native.converged),
        n_iterations=int(native.n_iterations),
        native_result=native,
    )


def _source_log_partition(cost: FloatArray, target: BoolArray, g: FloatArray, epsilon: float) -> FloatArray:
    retained = np.flatnonzero(target)
    terms = -np.log(retained.size) + (g[retained][None, :] - cost[:, retained]) / epsilon
    return _logsumexp(terms, axis=1)


def _target_log_partition(cost: FloatArray, source: BoolArray, f: FloatArray, epsilon: float) -> FloatArray:
    retained = np.flatnonzero(source)
    terms = -np.log(retained.size) + (f[retained][:, None] - cost[retained, :]) / epsilon
    return _logsumexp(terms, axis=0)


def _price(log_partition: FloatArray, *, epsilon: float, penalty: float | None) -> FloatArray:
    if penalty is None:
        return -epsilon * log_partition
    exponent = epsilon / (penalty + epsilon)
    return (penalty + epsilon) * (-np.expm1(exponent * log_partition))


def _handoff_source_potential(
    log_partition: FloatArray,
    *,
    epsilon: float,
    lambda_a: float | None,
) -> FloatArray:
    if lambda_a is None:
        return -epsilon * log_partition
    gamma = lambda_a * epsilon / (lambda_a + epsilon)
    return -gamma * log_partition


def _frozen_updates(
    cost: FloatArray,
    fixed: FixedSupportResult,
    *,
    kappa_s: float,
    kappa_t: float,
    max_source_rejections: int,
    max_target_rejections: int,
    lambda_a: float | None,
    lambda_b: float | None,
) -> tuple[PrefixSelection, PrefixSelection, FloatArray, FloatArray, FloatArray]:
    eps = float(fixed.native_result.epsilon)
    log_a = _source_log_partition(cost, fixed.target_gate, fixed.target_potential, eps)
    source_price = _price(log_a, epsilon=eps, penalty=lambda_a)
    source_step = prefix_select(
        source_price,
        fixed.source_gate,
        kappa=kappa_s,
        max_rejections=max_source_rejections,
    )
    handoff_f = _handoff_source_potential(log_a, epsilon=eps, lambda_a=lambda_a)
    log_b = _target_log_partition(cost, source_step.gate, handoff_f, eps)
    target_price = _price(log_b, epsilon=eps, penalty=lambda_b)
    target_step = prefix_select(
        target_price,
        fixed.target_gate,
        kappa=kappa_t,
        max_rejections=max_target_rejections,
    )
    return source_step, target_step, source_price, target_price, handoff_f


def _dual_bounds(
    fixed: FixedSupportResult,
    source_price: FloatArray,
    target_price: FloatArray,
    *,
    kappa_s: float,
    kappa_t: float,
    lambda_a: float | None,
    lambda_b: float | None,
) -> tuple[float, float]:
    source = fixed.source_gate
    target = fixed.target_gate
    penalty = _penalty(source, target, kappa_s, kappa_t)
    if fixed.backbone == "balanced":
        source_constant = float(np.mean(fixed.target_potential[target]))
        target_constant = float(np.mean(fixed.source_potential[source]))
    else:
        assert lambda_a is not None and lambda_b is not None
        source_constant = float(
            -lambda_b * np.mean(np.expm1(-fixed.target_potential[target] / lambda_b))
        )
        target_constant = float(
            -lambda_a * np.mean(np.expm1(-fixed.source_potential[source] / lambda_a))
        )
    source_bound = float(np.mean(source_price[source]) + source_constant + penalty)
    target_bound = float(np.mean(target_price[target]) + target_constant + penalty)
    return source_bound, target_bound


def _fixed_support_is_finite(result: FixedSupportResult) -> bool:
    """Return whether a fixed-support solve is numerically usable.

    Potentials outside the retained support are intentionally NaN, so only the
    active entries are audited.
    """

    return bool(
        np.isfinite(result.transport_objective)
        and np.isfinite(result.full_objective)
        and np.isfinite(result.solver_residual)
        and np.all(np.isfinite(result.coupling))
        and np.all(np.isfinite(result.source_potential[result.source_gate]))
        and np.all(np.isfinite(result.target_potential[result.target_gate]))
    )


def support_restricted_ot(
    cost_matrix: ArrayLike,
    *,
    backbone: Backbone,
    epsilon: float,
    kappa_s: float,
    kappa_t: float,
    lambda_a: float | None = None,
    lambda_b: float | None = None,
    source_rejection_budget: float = 0.10,
    target_rejection_budget: float = 0.10,
    initial_source_gate: ArrayLike | None = None,
    initial_target_gate: ArrayLike | None = None,
    threshold: float = 1e-10,
    max_iterations: int = 20_000,
    max_outer_iterations: int = 100,
    xi_tolerance: float = 1e-6,
    weak_duality_tolerance: float = 1e-8,
    cycle_objective_tolerance: float = 1e-9,
) -> SupportRestrictedOTResult:
    """Run the frozen support-restricted M4-P method.

    The two rejection budgets are fractions in ``[0,1)`` and are converted to
    integer limits with ``floor(rho * N)``.  Invalid inputs raise.  Inner-solver
    failure, cycling, and iteration capping are returned as explicit statuses.
    """

    cost = _cost_matrix(cost_matrix)
    if backbone not in ("balanced", "unbalanced"):
        raise ValueError("`backbone` must be 'balanced' or 'unbalanced'.")
    eps = _finite_positive(epsilon, name="epsilon")
    kap_s = _finite_nonnegative(kappa_s, name="kappa_s")
    kap_t = _finite_nonnegative(kappa_t, name="kappa_t")
    source_budget = _budget(source_rejection_budget, name="source_rejection_budget")
    target_budget = _budget(target_rejection_budget, name="target_rejection_budget")
    outer_cap = _positive_integer(max_outer_iterations, name="max_outer_iterations")
    xi_tol = _finite_nonnegative(xi_tolerance, name="xi_tolerance")
    weak_tol = _finite_nonnegative(weak_duality_tolerance, name="weak_duality_tolerance")
    cycle_objective_tol = _finite_nonnegative(
        cycle_objective_tolerance, name="cycle_objective_tolerance"
    )
    max_source_rejections = int(floor(source_budget * cost.shape[0] + 1e-12))
    max_target_rejections = int(floor(target_budget * cost.shape[1] + 1e-12))
    min_source = cost.shape[0] - max_source_rejections
    min_target = cost.shape[1] - max_target_rejections
    source = _gate(
        initial_source_gate,
        n=cost.shape[0],
        name="initial_source_gate",
        minimum=min_source,
    )
    target = _gate(
        initial_target_gate,
        n=cost.shape[1],
        name="initial_target_gate",
        minimum=min_target,
    )
    if backbone == "balanced":
        if lambda_a is not None or lambda_b is not None:
            raise ValueError("Lambdas must be omitted for the balanced backbone.")
        la = lb = None
    else:
        if lambda_a is None or lambda_b is None:
            raise ValueError("Positive lambdas are required for the unbalanced backbone.")
        la = _finite_positive(lambda_a, name="lambda_a")
        lb = _finite_positive(lambda_b, name="lambda_b")

    history: list[tuple[BoolArray, BoolArray]] = [(source.copy(), target.copy())]
    state_first_seen: dict[tuple[bytes, bytes], int] = {
        (source.tobytes(), target.tobytes()): 0
    }
    solved: list[FixedSupportResult] = []
    objective_history: list[float] = []
    n_solves = 0
    cycle_length = 0
    source_cycle_acceptance_min = float("nan")
    source_cycle_acceptance_max = float("nan")
    target_cycle_acceptance_min = float("nan")
    target_cycle_acceptance_max = float("nan")
    stable = False
    terminal_status: TerminalStatus = "iteration-capped"
    source_price = np.full(cost.shape[0], np.nan)
    target_price = np.full(cost.shape[1], np.nan)
    final: FixedSupportResult | None = None

    for _outer in range(outer_cap):
        fixed = solve_fixed_support_ot(
            cost,
            source,
            target,
            backbone=backbone,
            epsilon=eps,
            kappa_s=kap_s,
            kappa_t=kap_t,
            lambda_a=la,
            lambda_b=lb,
            threshold=threshold,
            max_iterations=max_iterations,
        )
        n_solves += 1
        solved.append(fixed)
        objective_history.append(fixed.full_objective)
        if not fixed.converged or not _fixed_support_is_finite(fixed):
            final = fixed
            terminal_status = "inner-solver-failure"
            break
        source_step, target_step, source_price, target_price, _ = _frozen_updates(
            cost,
            fixed,
            kappa_s=kap_s,
            kappa_t=kap_t,
            max_source_rejections=max_source_rejections,
            max_target_rejections=max_target_rejections,
            lambda_a=la,
            lambda_b=lb,
        )
        new_source = source_step.gate
        new_target = target_step.gate
        if np.array_equal(new_source, source) and np.array_equal(new_target, target):
            stable = True
            final = fixed
            break
        key = (new_source.tobytes(), new_target.tobytes())
        history.append((new_source.copy(), new_target.copy()))
        if key in state_first_seen:
            cycle_start = state_first_seen[key]
            cycle_length = len(history) - 1 - cycle_start
            orbit_states = history[cycle_start:-1]
            orbit_fits: list[FixedSupportResult] = []
            for orbit_source, orbit_target in orbit_states:
                orbit_fit = solve_fixed_support_ot(
                    cost,
                    orbit_source,
                    orbit_target,
                    backbone=backbone,
                    epsilon=eps,
                    kappa_s=kap_s,
                    kappa_t=kap_t,
                    lambda_a=la,
                    lambda_b=lb,
                    threshold=threshold,
                    max_iterations=max_iterations,
                )
                n_solves += 1
                orbit_fits.append(orbit_fit)
            valid_orbit = [
                item for item in orbit_fits
                if item.converged and _fixed_support_is_finite(item)
            ]
            if len(valid_orbit) != len(orbit_fits):
                final = orbit_fits[0]
                terminal_status = "inner-solver-failure"
            else:
                source_acceptance = np.asarray(
                    [item.source_gate.mean() for item in orbit_fits], dtype=np.float64
                )
                target_acceptance = np.asarray(
                    [item.target_gate.mean() for item in orbit_fits], dtype=np.float64
                )
                source_cycle_acceptance_min = float(source_acceptance.min())
                source_cycle_acceptance_max = float(source_acceptance.max())
                target_cycle_acceptance_min = float(target_acceptance.min())
                target_cycle_acceptance_max = float(target_acceptance.max())
                best_objective = min(item.full_objective for item in orbit_fits)
                selected = next(
                    item for item in orbit_fits
                    if item.full_objective <= best_objective + cycle_objective_tol
                )
                # The selected representative receives its own terminal solve;
                # orbit solves above are used only for numerical validation and
                # deterministic complete-objective comparison.
                final = solve_fixed_support_ot(
                    cost,
                    selected.source_gate,
                    selected.target_gate,
                    backbone=backbone,
                    epsilon=eps,
                    kappa_s=kap_s,
                    kappa_t=kap_t,
                    lambda_a=la,
                    lambda_b=lb,
                    threshold=threshold,
                    max_iterations=max_iterations,
                )
                n_solves += 1
                terminal_status = (
                    "cycled"
                    if final.converged and _fixed_support_is_finite(final)
                    else "inner-solver-failure"
                )
            source = final.source_gate.copy()
            target = final.target_gate.copy()
            if terminal_status == "cycled":
                source_step, target_step, source_price, target_price, _ = _frozen_updates(
                    cost,
                    final,
                    kappa_s=kap_s,
                    kappa_t=kap_t,
                    max_source_rejections=max_source_rejections,
                    max_target_rejections=max_target_rejections,
                    lambda_a=la,
                    lambda_b=lb,
                )
            break
        state_first_seen[key] = len(history) - 1
        source, target = new_source, new_target
    else:
        final = solve_fixed_support_ot(
            cost,
            source,
            target,
            backbone=backbone,
            epsilon=eps,
            kappa_s=kap_s,
            kappa_t=kap_t,
            lambda_a=la,
            lambda_b=lb,
            threshold=threshold,
            max_iterations=max_iterations,
        )
        n_solves += 1
        objective_history.append(final.full_objective)
        source_step, target_step, source_price, target_price, _ = _frozen_updates(
            cost,
            final,
            kappa_s=kap_s,
            kappa_t=kap_t,
            max_source_rejections=max_source_rejections,
            max_target_rejections=max_target_rejections,
            lambda_a=la,
            lambda_b=lb,
        )

    assert final is not None
    # The two coordinatewise bounds certify only a gate-stable incumbent.  A
    # cycle representative, an iteration-capped state, or an inner failure is
    # deliberately not assigned a misleading primal--dual gap.
    xi_certificate = float("nan")
    if stable and final.converged and _fixed_support_is_finite(final):
        audit_source_bound, audit_target_bound = _dual_bounds(
            final,
            source_price,
            target_price,
            kappa_s=kap_s,
            kappa_t=kap_t,
            lambda_a=la,
            lambda_b=lb,
        )
        audit_xi_x = float(final.full_objective - audit_source_bound)
        audit_xi_y = float(final.full_objective - audit_target_bound)
        finite_audit = bool(
            np.all(np.isfinite(source_price))
            and np.all(np.isfinite(target_price))
            and np.isfinite(audit_source_bound)
            and np.isfinite(audit_target_bound)
            and np.isfinite(audit_xi_x)
            and np.isfinite(audit_xi_y)
        )
    else:
        audit_source_bound = audit_target_bound = float("nan")
        audit_xi_x = audit_xi_y = float("nan")
        finite_audit = False
    source_bound = target_bound = float("nan")
    xi_x = xi_y = float("nan")
    if stable:
        if not finite_audit:
            terminal_status = "nonfinite-audit-failure"
        elif min(audit_xi_x, audit_xi_y) < -weak_tol:
            terminal_status = "primal-dual-audit-failure"
        else:
            source_bound, target_bound = audit_source_bound, audit_target_bound
            xi_x, xi_y = audit_xi_x, audit_xi_y
            xi_certificate = max(0.0, xi_x, xi_y)
            terminal_status = (
                "conditionally-certified"
                if xi_certificate <= xi_tol
                else "gate-stable-xi-large"
            )

    return SupportRestrictedOTResult(
        backbone=backbone,
        status=terminal_status,
        coupling=final.coupling,
        transition_probability=final.transition_probability,
        reverse_transition_probability=final.reverse_transition_probability,
        source_mass=final.source_mass,
        target_mass=final.target_mass,
        source_gate=final.source_gate,
        target_gate=final.target_gate,
        source_price=source_price,
        target_price=target_price,
        source_potential=final.source_potential,
        target_potential=final.target_potential,
        transport_objective=final.transport_objective,
        objective=final.full_objective,
        source_dual_bound=source_bound,
        target_dual_bound=target_bound,
        source_xi_raw=xi_x,
        target_xi_raw=xi_y,
        source_xi_report=(max(0.0, xi_x) if np.isfinite(xi_x) else float("nan")),
        target_xi_report=(max(0.0, xi_y) if np.isfinite(xi_y) else float("nan")),
        xi_certificate=xi_certificate,
        solver_residual=final.solver_residual,
        source_rejection_budget=source_budget,
        target_rejection_budget=target_budget,
        max_source_rejections=max_source_rejections,
        max_target_rejections=max_target_rejections,
        source_budget_active=int((~final.source_gate).sum()) == max_source_rejections,
        target_budget_active=int((~final.target_gate).sum()) == max_target_rejections,
        gate_history=tuple((left.copy(), right.copy()) for left, right in history),
        objective_history=tuple(objective_history),
        n_outer_iterations=max(0, len(objective_history)),
        n_transport_solves=n_solves,
        cycle_length=cycle_length,
        source_cycle_acceptance_min=source_cycle_acceptance_min,
        source_cycle_acceptance_max=source_cycle_acceptance_max,
        target_cycle_acceptance_min=target_cycle_acceptance_min,
        target_cycle_acceptance_max=target_cycle_acceptance_max,
        terminal_gate_stable=stable,
        inner_converged=final.converged,
    )
