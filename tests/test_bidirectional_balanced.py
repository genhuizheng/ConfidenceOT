import unittest

import numpy as np

from traditional_ot import (
    balanced_ot,
    confidence_filtered_bidirectional_balanced_ot,
    solve_fixed_bidirectional_balanced_ot,
)


class TestBidirectionalBalancedOT(unittest.TestCase):
    def test_all_accepted_fixed_gates_recover_balanced_ot(self):
        cost = np.array([[0.1, 1.2, 2.0], [1.1, 0.2, 0.7]])
        baseline = balanced_ot(cost, epsilon=0.4)
        fixed = solve_fixed_bidirectional_balanced_ot(
            cost,
            [True, True],
            [True, True, True],
            rejection_cost=0.8,
            epsilon=0.4,
        )
        np.testing.assert_allclose(fixed.coupling, baseline.coupling, atol=1e-10)

    def test_zero_budgets_recover_balanced_ot(self):
        cost = np.array([[0.1, 1.2, 2.0], [1.1, 0.2, 0.7]])
        baseline = balanced_ot(cost, epsilon=0.4)
        fit = confidence_filtered_bidirectional_balanced_ot(
            cost,
            rejection_cost=0.5,
            epsilon=0.4,
            source_rejection_budget=0.0,
            target_rejection_budget=0.0,
        )
        np.testing.assert_allclose(fit.coupling, baseline.coupling, atol=1e-10)
        self.assertTrue(np.all(fit.source_gate))
        self.assertTrue(np.all(fit.target_gate))

    def test_target_gate_rejects_target_only_outlier(self):
        cost = np.array([[0.0, 4.0, 10.0], [4.0, 0.0, 10.0]])
        fit = confidence_filtered_bidirectional_balanced_ot(
            cost,
            rejection_cost=2.0,
            epsilon=0.2,
            source_rejection_budget=0.0,
            target_rejection_budget=0.34,
            update_source=False,
        )
        np.testing.assert_array_equal(fit.source_gate, [True, True])
        np.testing.assert_array_equal(fit.target_gate, [True, True, False])
        np.testing.assert_allclose(fit.coupling.sum(axis=0), np.full(3, 1 / 3), atol=1e-8)

    def test_reversible_loss_tolerance_is_invariant_to_cell_duplication(self):
        cost = np.array([[0.1, 0.8], [0.8, 0.1], [2.0, 2.0]])
        parameters = dict(
            rejection_cost=0.7,
            epsilon=0.3,
            variant="reversible",
            source_rejection_budget=0.34,
            target_rejection_budget=0.0,
            tau_s=0.05,
        )
        original = confidence_filtered_bidirectional_balanced_ot(
            cost, **parameters
        )
        duplicated_cost = np.repeat(np.repeat(cost, 20, axis=0), 20, axis=1)
        duplicated = confidence_filtered_bidirectional_balanced_ot(
            duplicated_cost, **parameters
        )

        np.testing.assert_array_equal(original.source_gate, [True, True, False])
        duplicated_by_population = duplicated.source_gate.reshape(3, 20)
        self.assertTrue(np.all(duplicated_by_population[0]))
        self.assertTrue(np.all(duplicated_by_population[1]))
        self.assertFalse(np.any(duplicated_by_population[2]))
        self.assertTrue(original.outer_converged)
        self.assertTrue(duplicated.outer_converged)

    def test_exact_objective_history_is_monotone(self):
        cost = np.array(
            [[0.1, 0.5, 3.0], [0.4, 0.2, 2.5], [3.0, 2.7, 0.3]]
        )
        fit = confidence_filtered_bidirectional_balanced_ot(
            cost,
            rejection_cost=0.8,
            epsilon=0.3,
            source_rejection_budget=0.34,
            target_rejection_budget=0.34,
            variant="exact",
        )
        self.assertTrue(fit.outer_converged)
        self.assertTrue(np.all(np.diff(fit.objective_history) <= 1e-9))

    def test_transpose_symmetry(self):
        cost = np.array([[0.1, 0.6, 2.2], [1.8, 0.2, 0.4]])
        fit = confidence_filtered_bidirectional_balanced_ot(
            cost,
            rejection_cost=0.75,
            epsilon=0.35,
            source_rejection_budget=0.5,
            target_rejection_budget=0.34,
        )
        transposed = confidence_filtered_bidirectional_balanced_ot(
            cost.T,
            rejection_cost=0.75,
            epsilon=0.35,
            source_rejection_budget=0.34,
            target_rejection_budget=0.5,
        )
        np.testing.assert_allclose(fit.coupling, transposed.coupling.T, atol=1e-8)
        np.testing.assert_array_equal(fit.source_gate, transposed.target_gate)
        np.testing.assert_array_equal(fit.target_gate, transposed.source_gate)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "At least one gate"):
            confidence_filtered_bidirectional_balanced_ot(
                np.ones((2, 2)), rejection_cost=1, epsilon=1,
                update_source=False, update_target=False,
            )
        with self.assertRaisesRegex(ValueError, "target_rejection_budget"):
            confidence_filtered_bidirectional_balanced_ot(
                np.ones((2, 2)), rejection_cost=1, epsilon=1,
                target_rejection_budget=1.0,
            )


if __name__ == "__main__":
    unittest.main()
