"""Exact fixed-mass partial optimal transport for uniform empirical measures.

The solver implements the classical partial-W objective

    min <C, P>
    s.t. P 1 <= a, P.T 1 <= b, sum(P) = m,

for equally sized, uniformly weighted empirical measures.  When ``m * n`` is
an integer, this is a minimum-cost cardinality matching problem.  The
implementation reduces it to one linear-sum assignment without entropic
regularization or dummy-mass leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from traditional_ot.unbalanced import _cost_matrix


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class PartialOTResult:
    """Exact fixed-mass partial-W solution for two uniform empirical measures."""

    coupling: FloatArray
    transition_probability: FloatArray
    reverse_transition_probability: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    source_unmatched_mass: FloatArray
    target_unmatched_mass: FloatArray
    source_unmatched_fraction: FloatArray
    target_unmatched_fraction: FloatArray
    source_gate: BoolArray
    target_gate: BoolArray
    transported_mass: float
    transported_count: int
    requested_transported_mass: float
    objective: float
    success: bool
    status: str


def _conditional(coupling: FloatArray, *, axis: int) -> FloatArray:
    mass = coupling.sum(axis=axis)
    result = np.zeros_like(coupling)
    if axis == 1:
        np.divide(coupling, mass[:, None], out=result, where=mass[:, None] > 0.0)
    else:
        np.divide(coupling, mass[None, :], out=result, where=mass[None, :] > 0.0)
    return result


def partial_wasserstein_uniform(
    cost_matrix: ArrayLike,
    *,
    transported_mass: float,
    cardinality_tolerance: float = 1e-9,
) -> PartialOTResult:
    """Solve exact fixed-mass partial OT for equal-size uniform measures.

    ``transported_mass`` is expressed on the unit-mass scale.  It must select
    an integer number of samples, i.e. ``transported_mass * n`` must be an
    integer up to ``cardinality_tolerance``.  This holds for the benchmark's
    N=(100, 500, 1000) and transported mass 0.85.
    """
    cost = _cost_matrix(cost_matrix)
    n_source, n_target = cost.shape
    if n_source != n_target:
        raise ValueError(
            "`partial_wasserstein_uniform` requires equally sized supports."
        )
    try:
        mass = float(transported_mass)
        tolerance = float(cardinality_tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("Transported mass and tolerance must be finite scalars.") from error
    if not np.isfinite(mass) or not 0.0 < mass <= 1.0:
        raise ValueError("`transported_mass` must lie in (0, 1].")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("`cardinality_tolerance` must be finite and non-negative.")

    n = n_source
    exact_count = mass * n
    transported_count = int(round(exact_count))
    if abs(exact_count - transported_count) > tolerance:
        raise ValueError(
            "`transported_mass * n` must be an integer for exact cardinality "
            f"matching; found {exact_count:.12g}."
        )
    if transported_count < 1 or transported_count > n:
        raise ValueError("The requested transported cardinality is outside [1, n].")

    # Add n-k dummy sources and n-k dummy targets.  Dummy-real and real-dummy
    # edges have zero cost; dummy-dummy edges are forbidden by a cost larger
    # than every feasible complete assignment.  A perfect augmented matching
    # then contains exactly k real-real edges.
    rejected_count = n - transported_count
    if rejected_count == 0:
        source_index, target_index = linear_sum_assignment(cost)
    else:
        augmented_size = n + rejected_count
        maximum = float(np.max(cost))
        forbidden = (n + 1.0) * maximum + 1.0
        augmented = np.full(
            (augmented_size, augmented_size), forbidden, dtype=np.float64
        )
        augmented[:n, :n] = cost
        augmented[:n, n:] = 0.0
        augmented[n:, :n] = 0.0
        rows, columns = linear_sum_assignment(augmented)
        real = (rows < n) & (columns < n)
        source_index = rows[real]
        target_index = columns[real]
        if source_index.size != transported_count:
            raise RuntimeError(
                "Augmented cardinality assignment returned an invalid number "
                f"of transported pairs ({source_index.size}, expected {transported_count})."
            )

    unit_mass = 1.0 / n
    coupling = np.zeros_like(cost)
    coupling[source_index, target_index] = unit_mass
    source_mass = coupling.sum(axis=1)
    target_mass = coupling.sum(axis=0)
    source_unmatched = np.maximum(unit_mass - source_mass, 0.0)
    target_unmatched = np.maximum(unit_mass - target_mass, 0.0)
    source_fraction = source_unmatched / unit_mass
    target_fraction = target_unmatched / unit_mass
    source_gate = source_mass > 0.5 * unit_mass
    target_gate = target_mass > 0.5 * unit_mass
    achieved_mass = float(coupling.sum())
    return PartialOTResult(
        coupling=coupling,
        transition_probability=_conditional(coupling, axis=1),
        reverse_transition_probability=_conditional(coupling, axis=0),
        source_mass=source_mass,
        target_mass=target_mass,
        source_unmatched_mass=source_unmatched,
        target_unmatched_mass=target_unmatched,
        source_unmatched_fraction=source_fraction,
        target_unmatched_fraction=target_fraction,
        source_gate=source_gate,
        target_gate=target_gate,
        transported_mass=achieved_mass,
        transported_count=transported_count,
        requested_transported_mass=mass,
        objective=float(np.sum(coupling * cost)),
        success=True,
        status="optimal",
    )
