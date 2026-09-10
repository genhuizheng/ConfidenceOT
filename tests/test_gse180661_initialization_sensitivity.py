from __future__ import annotations

import numpy as np
import pandas as pd

from cancer_metastasis.gse180661.initialization_deg import (
    complete_configuration,
    configuration_name,
)
from cancer_metastasis.gse180661.initialization_sensitivity import (
    feasible_random_gate,
    jaccard,
    projected_zero_gate,
)


def test_feasible_random_gate_respects_rejection_budget():
    gate = feasible_random_gate(101, 0.85, 0.0, np.random.default_rng(1))
    assert gate.dtype == bool
    assert gate.sum() == 16


def test_random_half_gate_is_above_floor():
    gate = feasible_random_gate(100, 0.85, 0.5, np.random.default_rng(2))
    assert gate.sum() == 50


def test_projected_zero_is_the_budget_floor():
    gate = projected_zero_gate(100, 0.85)
    assert gate.sum() == 15
    assert gate[:15].all()
    assert not gate[15:].any()


def test_jaccard_uses_retained_sets():
    left = np.array([True, True, False, False])
    right = np.array([True, False, True, False])
    assert np.isclose(jaccard(left, right), 1 / 3)


def test_initialization_configuration_names_are_unique():
    assert configuration_name("all_one", 0) == "all_one"
    assert configuration_name("all_zero_projected_to_budget_floor", 0) == "projected_zero"
    assert configuration_name("random_50_percent", 3) == "random_03"


def test_complete_configuration_drops_pairs_with_small_states():
    metadata = pd.DataFrame({
        "sample_id": ["a_case", "a_ref", "b_case", "b_ref"],
        "pair_id": ["a", "a", "b", "b"],
        "strategy": ["all_one"] * 4,
        "replicate": [0] * 4,
        "comparison_status": ["case", "reference", "case", "reference"],
        "cell_n": [25, 30, 10, 40],
    }).set_index("sample_id")
    counts = pd.DataFrame({"G1": [1, 2, 3, 4]}, index=metadata.index)
    selected_counts, selected_metadata = complete_configuration(
        counts, metadata, "all_one", 0, 20
    )
    assert list(selected_counts.index) == ["a_case", "a_ref"]
    assert selected_metadata["pair_id"].unique().tolist() == ["a"]
