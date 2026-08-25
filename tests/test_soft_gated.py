import numpy as np

from traditional_ot.soft_gated import (
    _gradient,
    _solve_inner,
    multi_start_soft_gated_uot,
    project_soft_coverage,
    soft_gated_uot,
)
from traditional_ot.unbalanced import unbalanced_ot


def _cost(seed=9, n=7, m=6, d=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = rng.normal(size=(m, d))
    cost = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    return cost / np.median(cost[cost > 0])


def test_projection_enforces_box_and_coverage():
    rng = np.random.default_rng(3)
    for n in (4, 9, 20):
        for budget in range(n):
            projected = project_soft_coverage(
                rng.normal(size=n), gate_floor=0.05,
                max_fractional_suppression=budget,
            )
            assert np.all(projected >= 0.05)
            assert np.all(projected <= 1.0)
            assert projected.sum() >= n - budget - 1e-9


def test_all_one_inner_solve_recovers_vanilla_uot_and_duality():
    cost = _cost()
    source = np.ones(cost.shape[0])
    target = np.ones(cost.shape[1])
    a = np.full(cost.shape[0], 1.0 / cost.shape[0])
    b = np.full(cost.shape[1], 1.0 / cost.shape[1])
    state = _solve_inner(
        cost, source, target,
        base_source=a, base_target=b,
        epsilon=0.3, lambda_a=1.0, lambda_b=1.2,
        c_s=0.1 / cost.shape[0], c_t=0.1 / cost.shape[1],
        threshold=1e-12, max_iterations=20_000,
    )
    vanilla = unbalanced_ot(
        cost, epsilon=0.3, lambda_a=1.0, lambda_b=1.2,
        threshold=1e-12, max_iterations=20_000,
    )
    np.testing.assert_allclose(state.result.coupling, vanilla.coupling, atol=1e-11)
    assert abs(state.gap) < 1e-10


def test_envelope_gradient_matches_resolved_finite_difference():
    cost = _cost(n=5, m=4)
    rng = np.random.default_rng(11)
    source = rng.uniform(0.65, 1.0, size=cost.shape[0])
    target = rng.uniform(0.65, 1.0, size=cost.shape[1])
    a = np.full(cost.shape[0], 1.0 / cost.shape[0])
    b = np.full(cost.shape[1], 1.0 / cost.shape[1])
    kwargs = dict(
        base_source=a, base_target=b,
        epsilon=0.4, lambda_a=1.1, lambda_b=0.9,
        c_s=0.03, c_t=0.04,
        threshold=1e-12, max_iterations=30_000,
    )
    state = _solve_inner(cost, source, target, **kwargs)
    grad_source, grad_target = _gradient(
        state, base_source=a, base_target=b,
        epsilon=0.4, lambda_a=1.1, lambda_b=0.9,
        c_s=0.03, c_t=0.04,
    )
    step = 1e-5
    for index in range(source.size):
        plus = source.copy(); plus[index] += step
        minus = source.copy(); minus[index] -= step
        numeric = (
            _solve_inner(cost, plus, target, **kwargs).upper_bound
            - _solve_inner(cost, minus, target, **kwargs).upper_bound
        ) / (2 * step)
        assert np.isclose(grad_source[index], numeric, atol=2e-6)
    for index in range(target.size):
        plus = target.copy(); plus[index] += step
        minus = target.copy(); minus[index] -= step
        numeric = (
            _solve_inner(cost, source, plus, **kwargs).upper_bound
            - _solve_inner(cost, source, minus, **kwargs).upper_bound
        ) / (2 * step)
        assert np.isclose(grad_target[index], numeric, atol=2e-6)


def test_soft_fit_is_budget_safe_and_terminally_consistent():
    cost = _cost(seed=17, n=10, m=9)
    fit = soft_gated_uot(
        cost,
        epsilon=0.3,
        lambda_a=1.0,
        lambda_b=1.0,
        c_s=0.2 / cost.shape[0],
        c_t=0.2 / cost.shape[1],
        source_rejection_budget=0.2,
        target_rejection_budget=0.2,
        max_outer_iterations=50,
        gate_tolerance=1e-4,
    )
    assert fit.status in {
        "numerically-soft-stationary", "line-search-failure", "iteration-capped",
        "inner-iteration-capped"
    }
    assert fit.source_fractional_suppression <= 2 + 1e-8
    assert fit.target_fractional_suppression <= 1 + 1e-8
    assert (~fit.source_gate).sum() <= 2
    assert (~fit.target_gate).sum() <= 1
    if fit.status == "numerically-soft-stationary":
        assert np.all(fit.coupling[~fit.source_gate] == 0.0)
        assert np.all(fit.coupling[:, ~fit.target_gate] == 0.0)


def test_multistart_constructs_feasible_labelled_nontrivial_runs():
    cost = _cost(seed=21, n=8, m=7)
    fit = multi_start_soft_gated_uot(
        cost,
        epsilon=0.3,
        lambda_a=1.0,
        lambda_b=1.0,
        c_s=0.15 / cost.shape[0],
        c_t=0.15 / cost.shape[1],
        source_rejection_budget=0.25,
        target_rejection_budget=0.29,
        perturbation_amplitude=0.4,
        random_seeds=(13,),
        threshold=1e-11,
        max_iterations=20_000,
        max_outer_iterations=10,
        gate_tolerance=1e-4,
    )
    assert [run.initialization for run in fit.runs] == [
        "all-one", "deterministic", "escape-score", "random-13"
    ]
    assert abs(fit.source_escape_score.sum()) < 1e-12
    assert abs(fit.target_escape_score.sum()) < 1e-12
    assert np.all(fit.runs[0].initial_source_gate == 1.0)
    assert any(np.any(run.initial_source_gate < 1.0) for run in fit.runs[1:])
    for run in fit.runs:
        assert np.sum(1.0 - run.initial_source_gate) <= 2 + 1e-10
        assert np.sum(1.0 - run.initial_target_gate) <= 2 + 1e-10
