"""Two-sided birth/death reservoir extension of KL-unbalanced OT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from traditional_ot.unbalanced import UOTResult, unbalanced_ot


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BirthDeathUOTResult:
    """UOT solution augmented by one birth row and one death column."""

    coupling: FloatArray
    real_coupling: FloatArray
    real_transition_probability: FloatArray
    source_death_probability: FloatArray
    target_birth_probability: FloatArray
    source_real_mass: FloatArray
    target_real_mass: FloatArray
    death_mass_fraction: float
    birth_mass_fraction: float
    birth_to_death_mass: float
    extended_cost_matrix: FloatArray
    extended_source_marginal: FloatArray
    extended_target_marginal: FloatArray
    reservoir_weight: float
    birth_cost: float
    death_cost: float
    solver_result: UOTResult


def _positive_finite(value: object, *, name: str, allow_zero: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`{name}` must be a finite float.") from error
    invalid = converted < 0.0 if allow_zero else converted <= 0.0
    if not np.isfinite(converted) or invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"`{name}` must be {qualifier} and finite.")
    return converted


def _cost(cost_matrix: ArrayLike) -> FloatArray:
    try:
        cost = np.asarray(cost_matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("`cost_matrix` must be a numeric 2D matrix.") from error
    if cost.ndim != 2 or min(cost.shape) == 0:
        raise ValueError("`cost_matrix` must be a non-empty 2D matrix.")
    if not np.all(np.isfinite(cost)) or np.any(cost < 0.0):
        raise ValueError("`cost_matrix` must be finite and non-negative.")
    return np.ascontiguousarray(cost)


def _weights(weights: ArrayLike | None, *, n: int, name: str) -> FloatArray:
    if weights is None:
        return np.full(n, 1.0 / n)
    try:
        values = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`{name}` must be a numeric 1D array.") from error
    if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"`{name}` must have shape ({n},) and be strictly positive and finite.")
    return values / values.sum()


def birth_death_uot(
    cost_matrix: ArrayLike,
    *,
    birth_cost: float,
    death_cost: float,
    reservoir_weight: float = 0.25,
    reservoir_interaction_cost: float = 0.0,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
) -> BirthDeathUOTResult:
    """Solve KL-UOT after adding a birth source and a death target.

    The extended cost is::

        [ real-to-real C      real-to-death c_d ]
        [ birth-to-real c_b   reservoir-to-reservoir 0 ]

    ``reservoir_weight`` is relative to unit total real mass before the
    extended marginals are normalized.  Rejection probabilities are conditional
    fractions within each real source row or real target column.
    """
    cost = _cost(cost_matrix)
    birth_cost = _positive_finite(birth_cost, name="birth_cost")
    death_cost = _positive_finite(death_cost, name="death_cost")
    reservoir_weight = _positive_finite(reservoir_weight, name="reservoir_weight")
    reservoir_interaction_cost = _positive_finite(
        reservoir_interaction_cost, name="reservoir_interaction_cost", allow_zero=True
    )
    n_source, n_target = cost.shape
    a = _weights(source_weights, n=n_source, name="source_weights")
    b = _weights(target_weights, n=n_target, name="target_weights")
    extended = np.empty((n_source + 1, n_target + 1), dtype=np.float64)
    extended[:n_source, :n_target] = cost
    extended[:n_source, n_target] = death_cost
    extended[n_source, :n_target] = birth_cost
    extended[n_source, n_target] = reservoir_interaction_cost
    extended_a = np.concatenate([a, [reservoir_weight]])
    extended_b = np.concatenate([b, [reservoir_weight]])
    result = unbalanced_ot(
        extended,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        source_weights=extended_a,
        target_weights=extended_b,
        threshold=threshold,
        max_iterations=max_iterations,
    )
    coupling = result.coupling
    real = coupling[:n_source, :n_target]
    death = coupling[:n_source, n_target]
    birth = coupling[n_source, :n_target]
    source_total = real.sum(axis=1) + death
    target_total = real.sum(axis=0) + birth
    source_death_probability = np.divide(
        death, source_total, out=np.zeros_like(death), where=source_total > 0.0
    )
    target_birth_probability = np.divide(
        birth, target_total, out=np.zeros_like(birth), where=target_total > 0.0
    )
    real_row_mass = real.sum(axis=1, keepdims=True)
    real_transition = np.divide(
        real, real_row_mass, out=np.zeros_like(real), where=real_row_mass > 0.0
    )
    total_mass = float(coupling.sum())
    return BirthDeathUOTResult(
        coupling=coupling,
        real_coupling=real,
        real_transition_probability=real_transition,
        source_death_probability=source_death_probability,
        target_birth_probability=target_birth_probability,
        source_real_mass=source_total,
        target_real_mass=target_total,
        death_mass_fraction=float(death.sum() / total_mass),
        birth_mass_fraction=float(birth.sum() / total_mass),
        birth_to_death_mass=float(coupling[n_source, n_target]),
        extended_cost_matrix=extended,
        extended_source_marginal=result.source_marginal,
        extended_target_marginal=result.target_marginal,
        reservoir_weight=reservoir_weight,
        birth_cost=birth_cost,
        death_cost=death_cost,
        solver_result=result,
    )
