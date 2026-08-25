"""Minimal identifiable-to-impossible Gaussian benchmark for CF-UOT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class S6Simulation:
    source: FloatArray
    target: FloatArray
    source_population: NDArray[np.str_]
    target_population: NDArray[np.str_]
    true_rejection: NDArray[np.bool_]
    rho: float
    separation: float
    noise: float


@dataclass(frozen=True)
class ScenarioSimulation:
    source: FloatArray
    target: FloatArray
    source_population: NDArray[np.str_]
    target_population: NDArray[np.str_]
    true_rejection: NDArray[np.bool_]
    true_target_rejection: NDArray[np.bool_]
    population_coupling: dict[str, dict[str, float]]
    scenario: str

    @property
    def true_source_rejection(self) -> NDArray[np.bool_]:
        """Alias retained alongside the legacy ``true_rejection`` field."""
        return self.true_rejection


SCENARIOS = (
    "S0_no_death_far_move",
    "S1_extinction",
    "S2_novel_target",
    "S3_outliers",
    "S4_bifurcation",
    "S5_abundance",
)


def simulate_scenario(
    scenario: str,
    *,
    seed: int,
    n_per_population: int = 20,
    separation: float = 3.0,
    noise: float = 0.25,
    dimension: int = 10,
) -> ScenarioSimulation:
    """Generate one orthogonal L0 benchmark scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"`scenario` must be one of {SCENARIOS}.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("`seed` must be an integer.")
    if not isinstance(n_per_population, int) or n_per_population < 10:
        raise ValueError("`n_per_population` must be an integer >= 10.")
    if not isinstance(dimension, int) or dimension < 2:
        raise ValueError("`dimension` must be an integer >= 2.")
    if not np.isfinite(separation) or separation <= 0 or not np.isfinite(noise) or noise <= 0:
        raise ValueError("`separation` and `noise` must be positive and finite.")
    rng = np.random.default_rng(seed)
    base_names = list("ABCDEF")
    means_2d = np.array([
        [0.0, 0.0], [separation, 0.0], [0.0, separation],
        [separation, separation], [-separation, separation], [0.0, -separation],
    ])
    means = np.zeros((6, dimension))
    means[:, :2] = means_2d
    source_names = base_names.copy()
    target_names = base_names.copy()
    source_counts = {name: n_per_population for name in source_names}
    target_counts = {name: n_per_population for name in target_names}
    target_means = {name: means[index].copy() for index, name in enumerate(base_names)}
    truth = {name: {name: 1.0} for name in base_names}
    rejected_names: set[str] = set()
    rejected_target_names: set[str] = set()

    if scenario == "S0_no_death_far_move":
        target_means["B"][0] -= 0.4 * separation
    elif scenario == "S1_extinction":
        target_names.remove("A")
        target_counts.pop("A")
        target_means.pop("A")
        truth["A"] = {}
        rejected_names.add("A")
    elif scenario == "S2_novel_target":
        target_names.append("G")
        target_counts["G"] = n_per_population
        novel = np.zeros(dimension)
        novel[:2] = (separation, -separation)
        target_means["G"] = novel
        rejected_target_names.add("G")
    elif scenario == "S3_outliers":
        source_names.append("O")
        source_counts["O"] = n_per_population
        truth["O"] = {}
        rejected_names.add("O")
    elif scenario == "S4_bifurcation":
        target_names.remove("B")
        target_counts.pop("B")
        target_means.pop("B")
        target_names.extend(("B1", "B2"))
        target_counts["B1"] = n_per_population // 2
        target_counts["B2"] = n_per_population - n_per_population // 2
        branch_1, branch_2 = means[1].copy(), means[1].copy()
        branch_1[:2] += (0.65 * separation, 0.45 * separation)
        branch_2[:2] += (0.65 * separation, -0.45 * separation)
        target_means["B1"], target_means["B2"] = branch_1, branch_2
        truth["B"] = {"B1": 0.5, "B2": 0.5}
    elif scenario == "S5_abundance":
        target_counts["A"] = 5 * n_per_population
        target_counts["B"] = max(2, int(round(0.2 * n_per_population)))

    source_blocks, source_labels = [], []
    for name in source_names:
        count = source_counts[name]
        if name == "O":
            block = rng.uniform(-2.5 * separation, 2.5 * separation, size=(count, dimension))
        else:
            block = rng.normal(means[base_names.index(name)], noise, size=(count, dimension))
        source_blocks.append(block)
        source_labels.extend([name] * count)
    target_blocks, target_labels = [], []
    for name in target_names:
        count = target_counts[name]
        block = rng.normal(target_means[name], noise, size=(count, dimension))
        target_blocks.append(block)
        target_labels.extend([name] * count)
    source_population = np.asarray(source_labels)
    target_population = np.asarray(target_labels)
    return ScenarioSimulation(
        source=np.vstack(source_blocks),
        target=np.vstack(target_blocks),
        source_population=source_population,
        target_population=target_population,
        true_rejection=np.isin(source_population, list(rejected_names)),
        true_target_rejection=np.isin(
            target_population, list(rejected_target_names)
        ),
        population_coupling=truth,
        scenario=scenario,
    )


def simulate_s6(
    *,
    rho: float,
    seed: int,
    n_per_population: int = 40,
    separation: float = 3.0,
    noise: float = 0.25,
    dimension: int = 10,
) -> S6Simulation:
    """Simulate extinct A together with a far-moving surviving B.

    Source A is extinct. Source B starts one separation unit to its right and
    moves left by ``rho * separation``. Populations C--F persist with small,
    fixed displacements. Pre and post cells are independent Gaussian draws;
    no observed cell has a parent identifier.
    """
    if not np.isfinite(rho) or rho < 0.0:
        raise ValueError("`rho` must be a non-negative finite float.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("`seed` must be an integer.")
    if not isinstance(n_per_population, int) or n_per_population < 5:
        raise ValueError("`n_per_population` must be an integer >= 5.")
    if not np.isfinite(separation) or separation <= 0.0:
        raise ValueError("`separation` must be positive and finite.")
    if not np.isfinite(noise) or noise <= 0.0:
        raise ValueError("`noise` must be positive and finite.")
    if not isinstance(dimension, int) or dimension < 2:
        raise ValueError("`dimension` must be an integer >= 2.")

    rng = np.random.default_rng(seed)
    names = np.array(list("ABCDEF"))
    means_2d = np.array([
        [0.0, 0.0],
        [separation, 0.0],
        [0.0, separation],
        [separation, separation],
        [-separation, separation],
        [0.0, -separation],
    ])
    source_means = np.zeros((len(names), dimension))
    source_means[:, :2] = means_2d
    target_means = source_means.copy()
    target_means[1, 0] -= rho * separation
    # Small non-confounding motion prevents the stable populations from being
    # exact duplicates across independently drawn snapshots.
    target_means[2, 1] += 0.10 * separation
    target_means[3, 0] += 0.08 * separation
    target_means[4, 0] -= 0.06 * separation
    target_means[5, 1] -= 0.08 * separation

    source_blocks = [
        rng.normal(mean, noise, size=(n_per_population, dimension))
        for mean in source_means
    ]
    # A is absent at t1. Target samples are independently drawn from the five
    # persistent terminal distributions.
    target_blocks = [
        rng.normal(target_means[index], noise, size=(n_per_population, dimension))
        for index in range(1, len(names))
    ]
    source = np.vstack(source_blocks)
    target = np.vstack(target_blocks)
    source_population = np.repeat(names, n_per_population)
    target_population = np.repeat(names[1:], n_per_population)
    return S6Simulation(
        source=source,
        target=target,
        source_population=source_population,
        target_population=target_population,
        true_rejection=source_population == "A",
        rho=float(rho),
        separation=float(separation),
        noise=float(noise),
    )
