import unittest

import numpy as np

from traditional_ot import intent_controlled_partial_ot


class TestIntentControlledPartialOT(unittest.TestCase):
    def test_one_by_one_accept_or_reject(self):
        accepted = intent_controlled_partial_ot(
            [[1.0]], source_unmatched_cost=0.7, target_unmatched_cost=0.6
        )
        self.assertAlmostEqual(accepted.coupling[0, 0], 1.0)
        self.assertAlmostEqual(accepted.objective, 1.0)
        rejected = intent_controlled_partial_ot(
            [[1.0]], source_unmatched_cost=0.2, target_unmatched_cost=0.3
        )
        self.assertAlmostEqual(rejected.coupling[0, 0], 0.0)
        self.assertAlmostEqual(rejected.objective, 0.5)

    def test_slack_constraints_and_objective_identity(self):
        cost = np.array([[0.2, 2.0, 3.0], [2.0, 0.1, 4.0]])
        result = intent_controlled_partial_ot(
            cost,
            source_unmatched_cost=[0.7, 0.8],
            target_unmatched_cost=[0.6, 0.6, 0.05],
            source_weights=[2, 1],
            target_weights=[1, 1, 1],
        )
        np.testing.assert_allclose(
            result.source_mass + result.source_unmatched_mass,
            result.source_marginal,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.target_mass + result.target_unmatched_mass,
            result.target_marginal,
            atol=1e-12,
        )
        constant = np.dot(result.source_unmatched_cost, result.source_marginal)
        constant += np.dot(result.target_unmatched_cost, result.target_marginal)
        self.assertAlmostEqual(result.objective, constant + result.reduced_objective)

    def test_admissible_support_is_exact(self):
        cost = np.array([[0.1, 2.0], [3.0, 0.2]])
        source_cost = np.array([0.4, 0.4])
        target_cost = np.array([0.4, 0.4])
        result = intent_controlled_partial_ot(
            cost,
            source_unmatched_cost=source_cost,
            target_unmatched_cost=target_cost,
        )
        expected = cost < source_cost[:, None] + target_cost[None, :]
        np.testing.assert_array_equal(result.admissible_mask, expected)
        self.assertTrue(np.all(result.coupling[~expected] == 0.0))
        self.assertAlmostEqual(result.transported_mass, 1.0)

    def test_uniform_high_cost_recovers_full_balanced_assignment(self):
        cost = np.array([[0.1, 4.0], [4.0, 0.2]])
        result = intent_controlled_partial_ot(
            cost, source_unmatched_cost=10.0, target_unmatched_cost=10.0
        )
        np.testing.assert_allclose(result.source_mass, [0.5, 0.5])
        np.testing.assert_allclose(result.target_mass, [0.5, 0.5])
        np.testing.assert_allclose(result.transition_probability, np.eye(2))

    def test_invalid_inputs_raise(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            intent_controlled_partial_ot(
                [[1.0]], source_unmatched_cost=-1.0, target_unmatched_cost=1.0
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            intent_controlled_partial_ot(
                [[1.0, 2.0]],
                source_unmatched_cost=[1.0, 2.0],
                target_unmatched_cost=1.0,
            )


if __name__ == "__main__":
    unittest.main()
