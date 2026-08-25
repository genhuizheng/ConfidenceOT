import numpy as np
import pytest

from traditional_ot import (
    UnsupportedSelectiveVariantError,
    balanced_ot,
    budgeted_gate_update,
    calibrate_selective_rejection_cost,
    entropic_partial_ot,
    intent_controlled_partial_ot,
    posthoc_selective_ot,
    refit_selective_ot,
    selective_ot,
    solve_fixed_gate_selective_ot,
    two_stage_selective_ot,
    unbalanced_ot,
)


COST = np.array(
    [
        [0.05, 1.4, 2.0, 2.2],
        [1.2, 0.08, 1.5, 2.0],
        [1.7, 1.3, 0.10, 1.1],
        [2.3, 2.0, 1.2, 3.0],
    ],
    dtype=float,
)


def _kwargs(backbone):
    if backbone == "balanced":
        return {"epsilon": 0.35}
    if backbone == "unbalanced":
        return {"epsilon": 0.35, "lambda_a": 1.2, "lambda_b": 1.1}
    if backbone == "partial":
        return {"source_unmatched_cost": 0.8, "target_unmatched_cost": 0.8}
    return {
        "epsilon": 0.35,
        "source_unmatched_cost": 0.8,
        "target_unmatched_cost": 0.8,
    }


@pytest.mark.parametrize(
    "backbone", ["balanced", "unbalanced", "partial", "entropic_partial"]
)
def test_all_accepted_fixed_gate_matches_ungated_backbone(backbone):
    gate = np.ones(4, dtype=bool)
    result = solve_fixed_gate_selective_ot(
        COST,
        gate,
        gate,
        backbone=backbone,
        rejection_cost=0.9,
        **_kwargs(backbone),
    )
    if backbone == "balanced":
        baseline = balanced_ot(COST, epsilon=0.35)
    elif backbone == "unbalanced":
        baseline = unbalanced_ot(COST, epsilon=0.35, lambda_a=1.2, lambda_b=1.1)
    elif backbone == "partial":
        baseline = intent_controlled_partial_ot(
            COST, source_unmatched_cost=0.8, target_unmatched_cost=0.8
        )
    else:
        baseline = entropic_partial_ot(
            COST,
            source_unmatched_cost=0.8,
            target_unmatched_cost=0.8,
            epsilon=0.35,
            threshold=1e-9,
        )
    np.testing.assert_allclose(result.coupling, baseline.coupling, atol=2e-8, rtol=2e-8)


def test_budgeted_gate_update_is_exact_and_tie_preserving():
    coefficients = np.array([-2.0, 0.0, 0.0, 3.0])
    previous = np.array([0, 1, 0, 1], dtype=bool)
    inequality = budgeted_gate_update(
        coefficients, previous, n_accepted=2, mode="inequality"
    )
    np.testing.assert_array_equal(inequality.gate, [1, 1, 0, 0])
    equality = budgeted_gate_update(
        coefficients, previous, n_accepted=3, mode="equality"
    )
    np.testing.assert_array_equal(equality.gate, [1, 1, 1, 0])


@pytest.mark.parametrize(
    "backbone", ["balanced", "unbalanced", "partial", "entropic_partial"]
)
def test_exact_shared_outer_loop_is_monotone_and_terminally_consistent(backbone):
    result = selective_ot(
        COST,
        backbone=backbone,
        rejection_cost=0.75,
        source_rejection_budget=0.25,
        target_rejection_budget=0.25,
        variant="exact",
        **_kwargs(backbone),
    )
    assert result.status == "converged"
    assert result.outer_converged
    assert result.source_terminal_gate_consistent
    assert result.target_terminal_gate_consistent
    assert result.source_gate.sum() >= 3
    assert result.target_gate.sum() >= 3
    difference = np.diff(np.asarray(result.objective_history))
    assert np.all(difference <= 2e-8), difference
    fixed = solve_fixed_gate_selective_ot(
        COST,
        result.source_gate,
        result.target_gate,
        backbone=backbone,
        rejection_cost=0.75,
        **_kwargs(backbone),
    )
    np.testing.assert_allclose(result.coupling, fixed.coupling, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize(
    "backbone", ["balanced", "unbalanced", "partial", "entropic_partial"]
)
def test_equality_mode_uses_exact_counts(backbone):
    result = selective_ot(
        COST,
        backbone=backbone,
        rejection_cost=0.75,
        source_rejection_budget=0.25,
        target_rejection_budget=0.50,
        gate_budget_mode="equality",
        variant="exact",
        **_kwargs(backbone),
    )
    assert result.initialization == "ungated_projection"
    assert result.source_gate.sum() == 3
    assert result.target_gate.sum() == 2
    assert result.source_budget_binding
    assert result.target_budget_binding


def test_reversible_scope_is_explicit():
    with pytest.raises(UnsupportedSelectiveVariantError, match="entropic backbones"):
        selective_ot(
            COST,
            backbone="partial",
            rejection_cost=0.75,
            variant="reversible",
            source_unmatched_cost=0.8,
            target_unmatched_cost=0.8,
        )
    result = selective_ot(
        COST,
        backbone="entropic_partial",
        rejection_cost=0.75,
        variant="reversible",
        epsilon=0.35,
        source_unmatched_cost=0.8,
        target_unmatched_cost=0.8,
        gate_tolerance=0.05,
    )
    assert result.variant == "reversible"
    assert result.status in ("converged", "cycle_detected", "iteration_capped")


def test_two_stage_projects_native_ranking_then_resolves():
    result = two_stage_selective_ot(
        COST,
        backbone="unbalanced",
        rejection_cost=0.75,
        epsilon=0.35,
        lambda_a=1.2,
        lambda_b=1.1,
        source_rejection_budget=0.25,
        target_rejection_budget=0.50,
        variant="exact",
    )
    assert result.source_gate.sum() == 3
    assert result.target_gate.sum() == 2
    np.testing.assert_array_equal(result.fixed_gate_result.source_gate, result.source_gate)
    np.testing.assert_array_equal(result.fixed_gate_result.target_gate, result.target_gate)


def test_strict_backbone_parameters_and_exact_tolerance():
    with pytest.raises(ValueError, match="epsilon.*required"):
        selective_ot(COST, backbone="balanced", rejection_cost=0.8)
    with pytest.raises(ValueError, match="only valid for unbalanced"):
        selective_ot(
            COST,
            backbone="balanced",
            rejection_cost=0.8,
            epsilon=0.3,
            lambda_a=1.0,
        )
    with pytest.raises(ValueError, match="exact variant requires"):
        selective_ot(
            COST,
            backbone="balanced",
            rejection_cost=0.8,
            epsilon=0.3,
            gate_tolerance=0.01,
        )


@pytest.mark.parametrize("backbone", ["unbalanced", "partial", "entropic_partial"])
def test_submeasure_refit_preserves_retained_measure(backbone):
    source_gate = np.array([1, 1, 0, 0], dtype=bool)
    target_gate = np.array([1, 1, 1, 0], dtype=bool)
    result = refit_selective_ot(
        COST,
        source_gate,
        target_gate,
        backbone=backbone,
        marginal_mode="submeasure",
        rejection_cost=0.75,
        **_kwargs(backbone),
    )
    assert result.fixed_gate_result.source_marginal.sum() == pytest.approx(0.5)
    assert result.fixed_gate_result.target_marginal.sum() == pytest.approx(0.75)
    assert result.coupling.shape == COST.shape
    assert np.all(result.coupling[~source_gate] == 0.0)
    assert np.all(result.coupling[:, ~target_gate] == 0.0)


def test_balanced_submeasure_refit_reports_infeasibility_instead_of_renormalizing():
    with pytest.raises(ValueError, match="submeasure is infeasible"):
        refit_selective_ot(
            COST,
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            backbone="balanced",
            marginal_mode="submeasure",
            rejection_cost=0.75,
            epsilon=0.35,
        )
    feasible = refit_selective_ot(
        COST,
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        backbone="balanced",
        marginal_mode="submeasure",
        rejection_cost=0.75,
        epsilon=0.35,
    )
    assert feasible.fixed_gate_result.source_marginal.sum() == pytest.approx(0.5)
    assert feasible.fixed_gate_result.target_marginal.sum() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "backbone", ["balanced", "unbalanced", "partial", "entropic_partial"]
)
def test_posthoc_baseline_has_matched_coverage_and_own_refit(backbone):
    result = posthoc_selective_ot(
        COST,
        backbone=backbone,
        rejection_cost=0.75,
        source_rejection_budget=0.25,
        target_rejection_budget=0.50,
        refit_marginal_mode="renormalized",
        **_kwargs(backbone),
    )
    assert result.source_gate.sum() == 3
    assert result.target_gate.sum() == 2
    assert result.refit_result is not None
    np.testing.assert_array_equal(result.refit_result.source_gate, result.source_gate)
    np.testing.assert_array_equal(result.refit_result.target_gate, result.target_gate)


def test_two_sided_calibration_uses_raw_rates_and_largest_jointly_feasible_cost():
    null = np.full_like(COST, 1.5)
    calibration = calibrate_selective_rejection_cost(
        COST,
        source_null_costs=[null, null + 0.05],
        target_null_costs=[null + 0.1, null + 0.15],
        candidate_costs=[0.2, 0.6, 2.0],
        maximum_source_raw_acceptance=0.25,
        maximum_target_raw_acceptance=0.25,
        backbone="balanced",
        epsilon=0.35,
        source_rejection_budget=0.25,
        target_rejection_budget=0.25,
    )
    assert calibration.rejection_cost == pytest.approx(0.6)
    np.testing.assert_array_equal(calibration.feasible, [True, True, False])
    assert calibration.source_null_replicates == 2
    assert calibration.target_null_replicates == 2
