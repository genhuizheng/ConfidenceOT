import itertools

import numpy as np

from traditional_ot.balanced import balanced_ot
from traditional_ot.support_restricted import (
    prefix_select,
    solve_fixed_support_ot,
    support_restricted_ot,
)
from traditional_ot.unbalanced import unbalanced_ot


def _cost(seed=4, n=10, m=9, d=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = rng.normal(size=(m, d))
    C = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    return C / np.median(C)


def test_prefix_select_matches_brute_force_with_signed_prices():
    rng = np.random.default_rng(42)
    for _ in range(250):
        n = int(rng.integers(3, 9))
        prices = np.round(rng.normal(size=n), 1)
        previous = rng.integers(0, 2, size=n)
        maximum = int(rng.integers(0, n))
        kappa = float(rng.uniform(0.0, 2.0))
        selected = prefix_select(
            prices, previous, kappa=kappa, max_rejections=maximum
        )
        brute = np.inf
        for size in range(max(1, n - maximum), n + 1):
            for subset in itertools.combinations(range(n), size):
                value = prices[list(subset)].mean() + (kappa / n) * (n - size)
                brute = min(brute, value)
        assert np.isclose(selected.objective, brute, atol=1e-12)
        assert n - selected.gate.sum() <= maximum


def test_all_accepted_fixed_support_matches_vanilla_balanced_and_uot():
    C = _cost()
    source = np.ones(C.shape[0], dtype=bool)
    target = np.ones(C.shape[1], dtype=bool)
    fixed_b = solve_fixed_support_ot(
        C, source, target, backbone="balanced", epsilon=0.3,
        kappa_s=0.4, kappa_t=0.6,
    )
    vanilla_b = balanced_ot(C, epsilon=0.3, threshold=1e-10, max_iterations=20_000)
    np.testing.assert_allclose(fixed_b.coupling, vanilla_b.coupling, rtol=1e-9, atol=1e-11)
    fixed_u = solve_fixed_support_ot(
        C, source, target, backbone="unbalanced", epsilon=0.3,
        lambda_a=1.0, lambda_b=1.4, kappa_s=0.4, kappa_t=0.6,
    )
    vanilla_u = unbalanced_ot(
        C, epsilon=0.3, lambda_a=1.0, lambda_b=1.4,
        threshold=1e-10, max_iterations=20_000,
    )
    np.testing.assert_allclose(fixed_u.coupling, vanilla_u.coupling, rtol=1e-9, atol=1e-11)


def test_fixed_support_rejects_endpoints_and_renormalizes_balanced_marginals():
    C = _cost()
    source = np.ones(C.shape[0], dtype=bool)
    target = np.ones(C.shape[1], dtype=bool)
    source[[1, 8]] = False
    target[[0, 6]] = False
    fixed = solve_fixed_support_ot(
        C, source, target, backbone="balanced", epsilon=0.25,
        kappa_s=0.5, kappa_t=0.5,
    )
    assert np.all(fixed.coupling[~source] == 0.0)
    assert np.all(fixed.coupling[:, ~target] == 0.0)
    np.testing.assert_allclose(fixed.source_mass[source], 1.0 / source.sum(), atol=1e-9)
    np.testing.assert_allclose(fixed.target_mass[target], 1.0 / target.sum(), atol=1e-9)


def test_terminal_prices_are_tight_and_budgets_hold_for_both_backbones():
    C = _cost(seed=8, n=12, m=11)
    settings = [
        dict(backbone="balanced"),
        dict(backbone="unbalanced", lambda_a=1.0, lambda_b=1.0),
    ]
    for extra in settings:
        fit = support_restricted_ot(
            C,
            epsilon=0.35,
            kappa_s=0.8,
            kappa_t=0.8,
            source_rejection_budget=0.25,
            target_rejection_budget=0.25,
            max_outer_iterations=30,
            xi_tolerance=1e-5,
            **extra,
        )
        assert fit.inner_converged
        assert (~fit.source_gate).sum() <= fit.max_source_rejections
        assert (~fit.target_gate).sum() <= fit.max_target_rejections
        assert fit.status in {
            "conditionally-certified", "gate-stable-xi-large", "cycled"
        }
        if fit.terminal_gate_stable:
            assert np.isfinite(fit.source_xi_raw)
            assert np.isfinite(fit.target_xi_raw)
            assert fit.source_xi_raw >= -1e-7
            assert fit.target_xi_raw >= -1e-7
            assert fit.xi_certificate == max(
                0.0, fit.source_xi_raw, fit.target_xi_raw
            )
        else:
            assert np.isnan(fit.source_dual_bound)
            assert np.isnan(fit.target_dual_bound)
            assert np.isnan(fit.source_xi_raw)
            assert np.isnan(fit.target_xi_raw)
            assert np.isnan(fit.source_xi_report)
            assert np.isnan(fit.target_xi_report)
            assert np.isnan(fit.xi_certificate)


def test_cycle_representative_is_uncertified_and_re_solved():
    C = _cost(seed=1, n=5, m=5, d=2)
    fit = support_restricted_ot(
        C,
        backbone="unbalanced",
        epsilon=0.5,
        kappa_s=0.2,
        kappa_t=0.2,
        lambda_a=1.0,
        lambda_b=1.0,
        source_rejection_budget=0.6,
        target_rejection_budget=0.6,
        max_outer_iterations=30,
    )
    assert fit.status == "cycled"
    assert fit.cycle_length == 2
    assert not fit.terminal_gate_stable
    assert fit.n_transport_solves >= fit.n_outer_iterations + fit.cycle_length + 1
    assert fit.source_cycle_acceptance_min <= fit.source_cycle_acceptance_max
    assert fit.target_cycle_acceptance_min <= fit.target_cycle_acceptance_max
    for value in (
        fit.source_dual_bound,
        fit.target_dual_bound,
        fit.source_xi_raw,
        fit.target_xi_raw,
        fit.source_xi_report,
        fit.target_xi_report,
        fit.xi_certificate,
    ):
        assert np.isnan(value)


def test_strict_backbone_parameters_raise():
    C = _cost(n=5, m=4)
    try:
        support_restricted_ot(
            C, backbone="unbalanced", epsilon=0.3, kappa_s=1.0, kappa_t=1.0
        )
    except ValueError as error:
        assert "lambdas" in str(error).lower()
    else:
        raise AssertionError("missing UOT lambdas did not raise")
