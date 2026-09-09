from __future__ import annotations

import numpy as np

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
