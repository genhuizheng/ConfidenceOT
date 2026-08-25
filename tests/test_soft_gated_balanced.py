import unittest

import numpy as np

from traditional_ot import (
    balanced_ot,
    balanced_soft_envelope_gradient,
    round_transport_plan,
    soft_gated_balanced_ot,
)
from traditional_ot.soft_gated_balanced import _solve_inner


class TestSoftGatedBalancedOT(unittest.TestCase):
    def test_rounding_returns_exact_marginals(self):
        plan = np.array([[0.20, 0.25], [0.15, 0.35]])
        a = np.array([0.4, 0.6])
        b = np.array([0.3, 0.7])
        rounded = round_transport_plan(plan, a, b)
        self.assertTrue(np.all(rounded >= 0.0))
        np.testing.assert_allclose(rounded.sum(axis=1), a, atol=1e-12)
        np.testing.assert_allclose(rounded.sum(axis=0), b, atol=1e-12)

    def test_zero_budgets_recover_balanced_ot(self):
        cost = np.array([[0.1, 1.2, 2.0], [1.1, 0.2, 0.7]])
        baseline = balanced_ot(cost, epsilon=0.4, threshold=1e-10)
        fit = soft_gated_balanced_ot(
            cost,
            epsilon=0.4,
            c_s=0.01,
            c_t=0.01,
            source_rejection_budget=0.0,
            target_rejection_budget=0.0,
            threshold=1e-10,
            gap_tolerance=1e-7,
            warn_on_terminal=False,
        )
        np.testing.assert_allclose(fit.coupling, baseline.coupling, atol=1e-8)
        self.assertTrue(np.all(fit.source_gate))
        self.assertTrue(np.all(fit.target_gate))

    def test_centered_gradient_is_gauge_invariant_and_matches_finite_difference(self):
        cost = np.array(
            [[0.1, 0.7, 1.3], [0.8, 0.2, 0.5], [1.1, 0.4, 0.3]]
        )
        a = np.array([0.2, 0.3, 0.5])
        b = np.array([0.4, 0.35, 0.25])
        source = np.array([0.85, 0.95, 1.0])
        target = np.array([1.0, 0.9, 0.92])
        state = _solve_inner(
            cost,
            source,
            target,
            base_source=a,
            base_target=b,
            epsilon=0.35,
            c_s=0.01,
            c_t=0.015,
            threshold=1e-12,
            max_iterations=50_000,
        )
        gradient_s, gradient_t = balanced_soft_envelope_gradient(
            state, base_source=a, base_target=b, c_s=0.01, c_t=0.015
        )
        shifted = state.__class__(
            **{
                **state.__dict__,
                "source_potential": state.source_potential + 7.0,
                "target_potential": state.target_potential - 7.0,
            }
        )
        shifted_s, shifted_t = balanced_soft_envelope_gradient(
            shifted, base_source=a, base_target=b, c_s=0.01, c_t=0.015
        )
        np.testing.assert_allclose(gradient_s, shifted_s, atol=1e-12)
        np.testing.assert_allclose(gradient_t, shifted_t, atol=1e-12)

        step = 1e-5
        direction = np.array([1.0, -0.5, 0.25])

        def value(gate):
            return _solve_inner(
                cost,
                gate,
                target,
                base_source=a,
                base_target=b,
                epsilon=0.35,
                c_s=0.01,
                c_t=0.015,
                threshold=1e-12,
                max_iterations=50_000,
            ).upper_bound

        finite_difference = (value(source + step * direction) - value(source - step * direction)) / (2 * step)
        self.assertAlmostEqual(finite_difference, float(np.dot(gradient_s, direction)), places=5)

    def test_escape_scores_are_centered(self):
        cost = np.array([[0.1, 1.0], [0.8, 0.2], [1.4, 0.7]])
        fit = soft_gated_balanced_ot(
            cost,
            epsilon=0.3,
            c_s=0.1,
            c_t=0.1,
            source_rejection_budget=0.0,
            target_rejection_budget=0.0,
            threshold=1e-10,
            warn_on_terminal=False,
        )
        self.assertAlmostEqual(float(fit.source_escape_score.sum()), 0.0, places=12)
        self.assertAlmostEqual(float(fit.target_escape_score.sum()), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
