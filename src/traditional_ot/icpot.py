"""Intent-controlled partial optimal transport (IC-POT).

This is an independent implementation of Eq. (2) and Proposition 1 in
Tripathi et al., *Take It or Leave It: Intent-Controlled Partial Optimal
Transport* (arXiv:2605.20030v1).  It solves the unregularized linear program;
it deliberately does not apply entropic Sinkhorn to the augmented dummy-point
problem, which the paper proves is a different objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from traditional_ot.unbalanced import _cost_matrix


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ICPOTResult:
    """Exact LP solution of intent-controlled partial OT."""

    coupling: FloatArray
    transition_probability: FloatArray
    reverse_transition_probability: FloatArray
    source_unmatched_mass: FloatArray
    target_unmatched_mass: FloatArray
    source_unmatched_fraction: FloatArray
    target_unmatched_fraction: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    source_marginal: FloatArray
    target_marginal: FloatArray
    source_unmatched_cost: FloatArray
    target_unmatched_cost: FloatArray
    admissible_mask: BoolArray
    transported_mass: float
    objective: float
    reduced_objective: float
    success: bool
    status: int
    message: str
    n_iterations: int


def _weights(
    values: ArrayLike | None, *, n: int, name: str, normalize: bool = True
) -> FloatArray:
    if values is None:
        if not normalize:
            raise ValueError(f"`{name}` is required when `normalize_weights=False`.")
        return np.full(n, 1.0 / n, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.shape != (n,):
        raise ValueError(f"`{name}` must have shape ({n},), found {array.shape}.")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"`{name}` must contain finite non-negative values.")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError(f"`{name}` must have positive total mass.")
    return array / total if normalize else array.copy()


def _pointwise_cost(values: ArrayLike, *, n: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(n, float(array), dtype=np.float64)
    if array.ndim != 1 or array.shape != (n,):
        raise ValueError(f"`{name}` must be scalar or have shape ({n},).")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"`{name}` must contain finite non-negative values.")
    return array


def _conditional(coupling: FloatArray, *, axis: int) -> FloatArray:
    if axis == 1:
        mass = coupling.sum(axis=1)
        result = np.zeros_like(coupling)
        np.divide(coupling, mass[:, None], out=result, where=mass[:, None] > 0.0)
        return result
    mass = coupling.sum(axis=0)
    result = np.zeros_like(coupling)
    np.divide(coupling, mass[None, :], out=result, where=mass[None, :] > 0.0)
    return result


def intent_controlled_partial_ot(
    cost_matrix: ArrayLike,
    *,
    source_unmatched_cost: ArrayLike,
    target_unmatched_cost: ArrayLike,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    admissibility_tolerance: float = 0.0,
    solver_options: dict[str, object] | None = None,
    normalize_weights: bool = True,
) -> ICPOTResult:
    """Solve the exact IC-POT slack problem by its sparse reduced LP.

    The optimized problem is

    ``min <C,P> + <c_s,u> + <c_t,v>``

    subject to ``P1 + u = mu``, ``P.T1 + v = nu`` and non-negativity.
    The equivalent reduced LP keeps only edges satisfying
    ``C_ij - c_s[i] - c_t[j] < -admissibility_tolerance``.  Equality edges
    can be set to zero in an optimum (paper Proposition 5), so excluding them
    chooses the sparse representative without changing the optimum value.
    """
    cost = _cost_matrix(cost_matrix)
    n_source, n_target = cost.shape
    if not isinstance(normalize_weights, (bool, np.bool_)):
        raise ValueError("`normalize_weights` must be boolean.")
    source = _weights(
        source_weights, n=n_source, name="source_weights", normalize=normalize_weights
    )
    target = _weights(
        target_weights, n=n_target, name="target_weights", normalize=normalize_weights
    )
    source_cost = _pointwise_cost(
        source_unmatched_cost, n=n_source, name="source_unmatched_cost"
    )
    target_cost = _pointwise_cost(
        target_unmatched_cost, n=n_target, name="target_unmatched_cost"
    )
    try:
        tolerance = float(admissibility_tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("`admissibility_tolerance` must be non-negative and finite.") from error
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("`admissibility_tolerance` must be non-negative and finite.")
    if solver_options is not None and not isinstance(solver_options, dict):
        raise ValueError("`solver_options` must be a dictionary or None.")

    reduced_cost = cost - source_cost[:, None] - target_cost[None, :]
    admissible = reduced_cost < -tolerance
    source_index, target_index = np.nonzero(admissible)
    n_edges = source_index.size
    coupling = np.zeros_like(cost)
    if n_edges:
        column_index = np.arange(n_edges, dtype=np.int64)
        constraint_rows = np.concatenate(
            (source_index, n_source + target_index)
        )
        constraint_columns = np.concatenate((column_index, column_index))
        constraint_values = np.ones(2 * n_edges, dtype=np.float64)
        constraints = coo_matrix(
            (constraint_values, (constraint_rows, constraint_columns)),
            shape=(n_source + n_target, n_edges),
        ).tocsr()
        solution = linprog(
            reduced_cost[admissible],
            A_ub=constraints,
            b_ub=np.concatenate((source, target)),
            bounds=(0.0, None),
            method="highs",
            options=solver_options,
        )
        if not solution.success:
            raise RuntimeError(
                f"IC-POT linear program failed (status {solution.status}): "
                f"{solution.message}"
            )
        coupling[source_index, target_index] = solution.x
        status = int(solution.status)
        message = str(solution.message)
        n_iterations = int(solution.nit)
    else:
        status = 0
        message = "Optimal all-unmatched solution; no strictly admissible edge."
        n_iterations = 0

    source_mass = coupling.sum(axis=1)
    target_mass = coupling.sum(axis=0)
    source_unmatched = np.maximum(source - source_mass, 0.0)
    target_unmatched = np.maximum(target - target_mass, 0.0)
    source_fraction = np.zeros_like(source)
    target_fraction = np.zeros_like(target)
    np.divide(source_unmatched, source, out=source_fraction, where=source > 0.0)
    np.divide(target_unmatched, target, out=target_fraction, where=target > 0.0)
    original_objective = float(
        np.sum(coupling * cost)
        + np.dot(source_cost, source_unmatched)
        + np.dot(target_cost, target_unmatched)
    )
    reduced_objective = float(np.sum(coupling * reduced_cost))
    return ICPOTResult(
        coupling=coupling,
        transition_probability=_conditional(coupling, axis=1),
        reverse_transition_probability=_conditional(coupling, axis=0),
        source_unmatched_mass=source_unmatched,
        target_unmatched_mass=target_unmatched,
        source_unmatched_fraction=source_fraction,
        target_unmatched_fraction=target_fraction,
        source_mass=source_mass,
        target_mass=target_mass,
        source_marginal=source,
        target_marginal=target,
        source_unmatched_cost=source_cost,
        target_unmatched_cost=target_cost,
        admissible_mask=admissible,
        transported_mass=float(coupling.sum()),
        objective=original_objective,
        reduced_objective=reduced_objective,
        success=True,
        status=status,
        message=message,
        n_iterations=n_iterations,
    )
