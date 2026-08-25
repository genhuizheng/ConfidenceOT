"""Synthetic snapshot pairs for high-confidence transport filtering.

The benchmark separates biological differentiation from endpoint-specific
contamination.  There are no observed cell-to-cell parent identifiers: truth
is defined at the population level, as in real destructive scRNA snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


FILTER_SCENARIOS = (
    "Q0_clean_differentiation",
    "Q1_shared_contamination_negative_control",
    "Q2_extinction_contamination",
    "Q3_emergence_contamination",
    "Q4_differentiation_contamination",
    "Q5_turnover_differentiation_contamination",
)


@dataclass(frozen=True)
class FilterSimulation:
    source: FloatArray
    target: FloatArray
    source_population: NDArray[np.str_]
    target_population: NDArray[np.str_]
    true_source_rejection: BoolArray
    true_target_rejection: BoolArray
    source_contamination_kind: NDArray[np.str_]
    target_contamination_kind: NDArray[np.str_]
    population_coupling: dict[str, dict[str, float]]
    scenario: str


def _clean_block(rng, mean, count, noise):
    return rng.normal(mean, noise, size=(count, mean.size))


def _contaminants(
    rng,
    *,
    count: int,
    clean_means: FloatArray,
    separation: float,
    noise: float,
    side: str,
) -> tuple[FloatArray, NDArray[np.str_]]:
    """Generate a graded mixture of technical artifacts.

    The high-noise component can overlap genuine populations and is therefore
    deliberately harder than the snapshot-specific off-manifold component.
    Source and target doublets use shifted population pairs so that artifacts
    do not acquire an artificial one-to-one continuation.
    """
    if count == 0:
        return np.empty((0, clean_means.shape[1])), np.empty(0, dtype="U16")
    blocks, kinds = [], []
    split = np.array_split(np.arange(count), 3)
    # High-variance cells around a real state: the intentionally ambiguous case.
    n = split[0].size
    indices = rng.integers(0, clean_means.shape[0], size=n)
    blocks.append(rng.normal(clean_means[indices], 3.0 * noise, size=(n, clean_means.shape[1])))
    kinds.extend(["high_noise"] * n)
    # Doublet-like cells between state pairs; use different pair offsets by side.
    n = split[1].size
    offset = 1 if side in {"source", "shared"} else 2
    first = rng.integers(0, clean_means.shape[0], size=n)
    second = (first + offset) % clean_means.shape[0]
    doublet_mean = 0.5 * (clean_means[first] + clean_means[second])
    blocks.append(rng.normal(doublet_mean, 1.4 * noise, size=(n, clean_means.shape[1])))
    kinds.extend(["doublet"] * n)
    # Snapshot-specific off-manifold technical direction.
    n = split[2].size
    technical = np.zeros(clean_means.shape[1])
    technical[2] = (1.25 if side in {"source", "shared"} else -1.25) * separation
    if clean_means.shape[1] > 3:
        technical[3] = (0.75 if side in {"source", "shared"} else -0.75) * separation
    center = clean_means.mean(axis=0) + technical
    blocks.append(rng.normal(center, 1.2 * noise, size=(n, clean_means.shape[1])))
    kinds.extend(["off_manifold"] * n)
    return np.vstack(blocks), np.asarray(kinds)


def _unidentifiable_contaminants(rng, *, count, clean_means, noise):
    """Draw labeled contaminants from exactly the clean mixture distribution.

    Their labels are exchangeable with genuine cells conditional on expression,
    so no expression-only filter should beat chance systematically.
    """
    indices = rng.integers(0, clean_means.shape[0], size=count)
    values = rng.normal(clean_means[indices], noise, size=(count, clean_means.shape[1]))
    return values, np.full(count, "unidentifiable", dtype="U16")


def simulate_filter_scenario(
    scenario: str,
    *,
    seed: int,
    n_per_population: int = 12,
    dimension: int = 10,
    separation: float = 3.0,
    noise: float = 0.35,
    contamination_fraction: float = 0.15,
) -> FilterSimulation:
    """Generate an independent pre/post pair with filtering ground truth."""
    if scenario not in FILTER_SCENARIOS:
        raise ValueError(f"`scenario` must be one of {FILTER_SCENARIOS}.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("`seed` must be an integer.")
    if not isinstance(n_per_population, int) or n_per_population < 6:
        raise ValueError("`n_per_population` must be an integer >= 6.")
    if not isinstance(dimension, int) or dimension < 4:
        raise ValueError("`dimension` must be an integer >= 4.")
    if not np.isfinite(noise) or noise <= 0 or not np.isfinite(separation) or separation <= 0:
        raise ValueError("`noise` and `separation` must be positive and finite.")
    if not np.isfinite(contamination_fraction) or not 0 <= contamination_fraction < 0.5:
        raise ValueError("`contamination_fraction` must be in [0, 0.5).")

    rng = np.random.default_rng(seed)
    names = list("ABCDE")
    means_2d = np.array([
        [0.0, 0.0], [separation, 0.0], [0.0, separation],
        [separation, separation], [-separation, separation],
    ])
    means = np.zeros((len(names), dimension))
    means[:, :2] = means_2d
    source_names = names.copy()
    target_names = names.copy()
    target_means = {name: means[index].copy() for index, name in enumerate(names)}
    truth = {name: {name: 1.0} for name in names}

    differentiates = scenario in {
        "Q0_clean_differentiation",
        "Q4_differentiation_contamination",
        "Q5_turnover_differentiation_contamination",
    }
    extinction = scenario in {
        "Q2_extinction_contamination",
        "Q5_turnover_differentiation_contamination",
    }
    emergence = scenario in {
        "Q3_emergence_contamination",
        "Q5_turnover_differentiation_contamination",
    }
    contaminated = scenario != "Q0_clean_differentiation"
    shared_contamination = scenario in {
        "Q1_shared_contamination_negative_control",
        "Q2_extinction_contamination",
        "Q3_emergence_contamination",
    }
    effective_noise = noise * (1.45 if "differentiation_contamination" in scenario else 1.0)

    if extinction:
        target_names.remove("A")
        target_means.pop("A")
        truth["A"] = {}
    if differentiates:
        target_names.remove("B")
        target_means.pop("B")
        target_names.extend(("B1", "B2"))
        branch_1, branch_2 = means[1].copy(), means[1].copy()
        branch_1[:2] += (0.65 * separation, 0.45 * separation)
        branch_2[:2] += (0.65 * separation, -0.45 * separation)
        target_means["B1"], target_means["B2"] = branch_1, branch_2
        truth["B"] = {"B1": 0.5, "B2": 0.5}
    if emergence:
        target_names.append("G")
        novel = np.zeros(dimension)
        novel[:2] = (separation, -separation)
        target_means["G"] = novel

    # Small genuine shifts prevent clean pre/post distributions from being copies.
    for index, name in enumerate(list(target_means)):
        if name not in {"B1", "B2", "G"}:
            target_means[name] = target_means[name].copy()
            target_means[name][index % 2] += (0.08 + 0.02 * index) * separation

    source_blocks, source_labels = [], []
    for index, name in enumerate(source_names):
        source_blocks.append(_clean_block(rng, means[index], n_per_population, effective_noise))
        source_labels.extend([name] * n_per_population)
    target_blocks, target_labels = [], []
    for name in target_names:
        count = n_per_population
        if name in {"B1", "B2"}:
            count = n_per_population // 2 if name == "B1" else n_per_population - n_per_population // 2
        target_blocks.append(_clean_block(rng, target_means[name], count, effective_noise))
        target_labels.extend([name] * count)

    source = np.vstack(source_blocks)
    target = np.vstack(target_blocks)
    source_population = np.asarray(source_labels)
    target_population = np.asarray(target_labels)
    source_kind = np.full(source.shape[0], "clean", dtype="U16")
    target_kind = np.full(target.shape[0], "clean", dtype="U16")
    source_bad = np.zeros(source.shape[0], bool)
    target_bad = np.isin(target_population, ["G"]) if emergence else np.zeros(target.shape[0], bool)
    if extinction:
        source_bad |= source_population == "A"

    if contaminated:
        source_count = max(3, int(round(source.shape[0] * contamination_fraction)))
        target_count = max(3, int(round(target.shape[0] * contamination_fraction)))
        target_reference = np.vstack(list(target_means.values()))
        if scenario == "Q1_shared_contamination_negative_control":
            source_contam, source_contam_kind = _unidentifiable_contaminants(
                rng, count=source_count, clean_means=means, noise=effective_noise
            )
            target_contam, target_contam_kind = _unidentifiable_contaminants(
                rng, count=target_count, clean_means=target_reference, noise=effective_noise
            )
        else:
            source_contam, source_contam_kind = _contaminants(
                rng, count=source_count, clean_means=means, separation=separation,
                noise=effective_noise, side="shared" if shared_contamination else "source",
            )
            target_contam, target_contam_kind = _contaminants(
                rng, count=target_count, clean_means=target_reference, separation=separation,
                noise=effective_noise, side="shared" if shared_contamination else "target",
            )
        source = np.vstack((source, source_contam))
        target = np.vstack((target, target_contam))
        source_population = np.concatenate((source_population, np.full(source_count, "QC")))
        target_population = np.concatenate((target_population, np.full(target_count, "QC")))
        source_kind = np.concatenate((source_kind, source_contam_kind))
        target_kind = np.concatenate((target_kind, target_contam_kind))
        source_bad = np.concatenate((source_bad, np.ones(source_count, bool)))
        target_bad = np.concatenate((target_bad, np.ones(target_count, bool)))

    return FilterSimulation(
        source=source,
        target=target,
        source_population=source_population,
        target_population=target_population,
        true_source_rejection=source_bad,
        true_target_rejection=target_bad,
        source_contamination_kind=source_kind,
        target_contamination_kind=target_kind,
        population_coupling=truth,
        scenario=scenario,
    )
