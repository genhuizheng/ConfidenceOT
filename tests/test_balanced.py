import unittest

import numpy as np

from traditional_ot import (
    balanced_ot,
    confidence_filtered_balanced_ot,
    solve_fixed_gate_balanced_ot,
)
from traditional_ot.unbalanced import squared_euclidean_cost


class TestBalancedOT(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        source = rng.normal(size=(24, 4))
        target = rng.normal(size=(29, 4))
        cost = squared_euclidean_cost(source, target)
        self.cost = cost / cost.mean()
        self.parameters = dict(epsilon=0.2, threshold=1e-10, max_iterations=20_000)

    def test_balanced_marginals(self):
        result = balanced_ot(self.cost, **self.parameters)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.source_mass, result.source_marginal, atol=1e-10)
        np.testing.assert_allclose(result.target_mass, result.target_marginal, atol=1e-12)
        np.testing.assert_allclose(result.transition_probability.sum(axis=1), 1.0, atol=1e-12)

    def test_all_accepted_recovers_vanilla(self):
        vanilla = balanced_ot(self.cost, **self.parameters)
        fixed = solve_fixed_gate_balanced_ot(
            self.cost, np.ones(len(self.cost), dtype=bool), rejection_cost=1.7, **self.parameters
        )
        np.testing.assert_allclose(fixed.coupling, vanilla.coupling, rtol=1e-10, atol=1e-12)

    def test_rejected_rows_share_profile_and_keep_mass(self):
        gate = np.ones(len(self.cost), dtype=bool)
        gate[[2, 8, 17]] = False
        result = solve_fixed_gate_balanced_ot(
            self.cost, gate, rejection_cost=1.3, **self.parameters
        )
        np.testing.assert_allclose(result.source_mass, result.source_marginal, atol=1e-10)
        expected = result.target_marginal * np.exp(result.log_target_scaling)
        expected /= expected.sum()
        np.testing.assert_allclose(
            result.transition_probability[~gate],
            np.broadcast_to(expected, (np.count_nonzero(~gate), len(expected))),
            rtol=1e-9,
            atol=1e-11,
        )

    def test_rejection_cost_does_not_change_fixed_gate_coupling(self):
        gate = np.ones(len(self.cost), dtype=bool)
        gate[::5] = False
        low = solve_fixed_gate_balanced_ot(self.cost, gate, rejection_cost=0.4, **self.parameters)
        high = solve_fixed_gate_balanced_ot(self.cost, gate, rejection_cost=4.0, **self.parameters)
        np.testing.assert_allclose(low.coupling, high.coupling, rtol=1e-9, atol=1e-11)

    def test_exact_objective_is_monotone(self):
        vanilla = balanced_ot(self.cost, **self.parameters)
        result = confidence_filtered_balanced_ot(
            self.cost,
            rejection_cost=float(np.quantile(vanilla.conditional_loss, 0.65)),
            variant="exact",
            max_outer_iterations=30,
            **self.parameters,
        )
        self.assertTrue(result.inner_converged)
        self.assertTrue(result.outer_converged)
        self.assertTrue(np.all(np.diff(result.objective_history) <= 1e-9))
        np.testing.assert_array_equal(result.gate, result.gate_score < result.rejection_cost)

    def test_invalid_variant_raises(self):
        with self.assertRaisesRegex(ValueError, "variant"):
            confidence_filtered_balanced_ot(
                self.cost, rejection_cost=1.0, epsilon=0.2, variant="wrong"
            )


if __name__ == "__main__":
    unittest.main()
