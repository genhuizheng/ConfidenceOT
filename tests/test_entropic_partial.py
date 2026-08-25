import unittest

import numpy as np

from traditional_ot import (
    confidence_filtered_entropic_partial_ot,
    entropic_partial_ot,
    exact_cardinality_gate_update,
    partial_gate_coefficients,
    refit_entropic_partial_ot,
    solve_fixed_confidence_filtered_partial_ot,
    two_stage_confidence_filtered_entropic_partial_ot,
)


class TestEntropicPartialOT(unittest.TestCase):
    def test_one_by_one_matches_closed_form_capacity_solution(self):
        # D = C-k_s-k_t = 0.5 and eps=0.5, hence the unconstrained
        # generalized-KL minimizer is exp(-D/eps)=exp(-1) < 1.
        result = entropic_partial_ot(
            [[1.0]],
            source_unmatched_cost=0.25,
            target_unmatched_cost=0.25,
            epsilon=0.5,
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.coupling[0, 0], np.exp(-1.0), places=12)
        self.assertLessEqual(result.max_source_capacity_violation, 1e-12)
        self.assertLessEqual(result.max_target_capacity_violation, 1e-12)

    def test_slacks_and_reduced_objective_identity(self):
        cost = np.array([[0.2, 2.0, 3.0], [2.0, 0.1, 4.0]])
        result = entropic_partial_ot(
            cost,
            source_unmatched_cost=[0.7, 0.8],
            target_unmatched_cost=[0.6, 0.6, 0.05],
            epsilon=0.3,
            source_weights=[2.0, 1.0],
            target_weights=[1.0, 1.0, 1.0],
        )
        np.testing.assert_allclose(
            result.source_mass + result.source_unmatched_mass,
            result.source_marginal,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            result.target_mass + result.target_unmatched_mass,
            result.target_marginal,
            atol=1e-11,
        )
        constant = np.dot(result.source_unmatched_cost, result.source_marginal)
        constant += np.dot(result.target_unmatched_cost, result.target_marginal)
        self.assertAlmostEqual(result.objective, constant + result.reduced_objective)

    def test_all_accepted_fixed_gate_reduces_to_epot(self):
        cost = np.array([[0.1, 1.3], [1.4, 0.2], [0.8, 0.7]])
        kwargs = dict(
            source_unmatched_cost=[0.6, 0.7, 0.8],
            target_unmatched_cost=[0.5, 0.9],
            epsilon=0.25,
        )
        baseline = entropic_partial_ot(cost, **kwargs)
        fixed = solve_fixed_confidence_filtered_partial_ot(
            cost,
            np.ones(3, bool),
            np.ones(2, bool),
            rejection_cost=0.4,
            **kwargs,
        )
        np.testing.assert_allclose(fixed.coupling, baseline.coupling, atol=1e-12)
        self.assertAlmostEqual(fixed.objective, baseline.objective, places=12)

    def test_exact_gate_updates_are_monotone_and_reject_outlier_pair(self):
        cost = np.array(
            [[0.1, 2.5, 3.0], [2.5, 0.1, 3.0], [3.0, 3.0, 3.0]]
        )
        result = confidence_filtered_entropic_partial_ot(
            cost,
            rejection_cost=0.5,
            source_unmatched_cost=2.0,
            target_unmatched_cost=2.0,
            epsilon=0.5,
            source_rejection_budget=0.34,
            target_rejection_budget=0.34,
        )
        np.testing.assert_array_equal(result.source_gate, [True, True, False])
        np.testing.assert_array_equal(result.target_gate, [True, True, False])
        self.assertTrue(result.inner_converged)
        self.assertTrue(result.outer_converged)
        differences = np.diff(np.asarray(result.objective_history))
        self.assertTrue(np.all(differences <= 1e-10), differences)

    def test_zero_transport_row_has_zero_exact_gate_coefficient(self):
        cost = np.array([[1.0, 2.0], [0.1, 0.2]])
        coupling = np.array([[0.0, 0.0], [0.4, 0.6]])
        source, target = partial_gate_coefficients(
            coupling,
            cost,
            [1, 1],
            [1, 1],
            rejection_cost=0.5,
        )
        self.assertEqual(source[0], 0.0)
        self.assertTrue(np.all(np.isfinite(target)))

    def test_reported_exponential_signal_bound_holds(self):
        rng = np.random.default_rng(17)
        cost = rng.uniform(0.05, 2.0, size=(5, 4))
        result = confidence_filtered_entropic_partial_ot(
            cost,
            rejection_cost=0.7,
            source_unmatched_cost=np.linspace(0.5, 1.0, 5),
            target_unmatched_cost=np.linspace(0.6, 0.9, 4),
            epsilon=0.4,
            source_rejection_budget=0.2,
            target_rejection_budget=0.25,
        )
        self.assertTrue(
            np.all(
                np.abs(result.source_gate_coefficient)
                <= result.source_signal_upper_bound + 1e-12
            )
        )
        self.assertTrue(
            np.all(
                np.abs(result.target_gate_coefficient)
                <= result.target_signal_upper_bound + 1e-12
            )
        )

    def test_zero_rejection_budget_is_exact_epot(self):
        cost = np.array([[0.2, 1.0], [1.1, 0.3]])
        kwargs = dict(
            source_unmatched_cost=0.8,
            target_unmatched_cost=0.8,
            epsilon=0.3,
        )
        baseline = entropic_partial_ot(cost, **kwargs)
        filtered = confidence_filtered_entropic_partial_ot(
            cost,
            rejection_cost=0.5,
            source_rejection_budget=0.0,
            target_rejection_budget=0.0,
            **kwargs,
        )
        np.testing.assert_allclose(filtered.coupling, baseline.coupling, atol=1e-12)
        self.assertTrue(np.all(filtered.source_gate))
        self.assertTrue(np.all(filtered.target_gate))

    def test_exact_cardinality_update_uses_deterministic_tie_breaking(self):
        update = exact_cardinality_gate_update(
            [0.0, -1.0, 0.0, 2.0],
            [False, True, True, False],
            n_accepted=2,
        )
        np.testing.assert_array_equal(update.gate, [False, True, True, False])

    def test_equality_budget_has_exact_gate_counts_and_monotone_history(self):
        cost = np.array(
            [[0.1, 2.0, 2.5], [2.0, 0.1, 2.5], [2.5, 2.5, 2.5]]
        )
        result = confidence_filtered_entropic_partial_ot(
            cost,
            rejection_cost=0.6,
            source_unmatched_cost=1.5,
            target_unmatched_cost=1.5,
            epsilon=0.4,
            source_rejection_budget=0.34,
            target_rejection_budget=0.34,
            gate_budget_mode="equality",
        )
        self.assertEqual(result.gate_budget_mode, "equality")
        self.assertEqual(result.initialization, "ungated_epot_projection")
        self.assertEqual(int(result.source_gate.sum()), result.source_min_accepted)
        self.assertEqual(int(result.target_gate.sum()), result.target_min_accepted)
        self.assertTrue(result.outer_converged)
        differences = np.diff(np.asarray(result.objective_history))
        self.assertTrue(np.all(differences <= 1e-10), differences)

    def test_refit_distinguishes_submeasure_and_renormalized_marginals(self):
        cost = np.array([[0.1, 1.0], [1.0, 0.2], [0.8, 0.7]])
        kwargs = dict(
            source_unmatched_cost=[0.8, 0.8, 0.8],
            target_unmatched_cost=[0.8, 0.8],
            epsilon=0.3,
            source_weights=[0.2, 0.3, 0.5],
            target_weights=[0.4, 0.6],
        )
        submeasure = refit_entropic_partial_ot(
            cost,
            [True, False, True],
            [True, False],
            marginal_mode="submeasure",
            **kwargs,
        )
        renormalized = refit_entropic_partial_ot(
            cost,
            [True, False, True],
            [True, False],
            marginal_mode="renormalized",
            **kwargs,
        )
        self.assertAlmostEqual(submeasure.result.source_marginal.sum(), 0.7)
        self.assertAlmostEqual(submeasure.result.target_marginal.sum(), 0.4)
        self.assertAlmostEqual(renormalized.result.source_marginal.sum(), 1.0)
        self.assertAlmostEqual(renormalized.result.target_marginal.sum(), 1.0)
        self.assertEqual(submeasure.coupling.shape, cost.shape)
        self.assertTrue(np.all(submeasure.coupling[1] == 0.0))
        self.assertTrue(np.all(submeasure.coupling[:, 1] == 0.0))

    def test_two_stage_uses_exact_coverage_and_fixed_gate_resolve(self):
        cost = np.array(
            [[0.1, 2.0, 2.5], [2.0, 0.1, 2.5], [2.5, 2.5, 2.5]]
        )
        result = two_stage_confidence_filtered_entropic_partial_ot(
            cost,
            rejection_cost=0.6,
            source_unmatched_cost=1.5,
            target_unmatched_cost=1.5,
            epsilon=0.4,
            source_rejection_budget=0.34,
            target_rejection_budget=0.34,
        )
        self.assertEqual(
            int(result.source_gate.sum()), result.native_result.source_min_accepted
        )
        self.assertEqual(
            int(result.target_gate.sum()), result.native_result.target_min_accepted
        )
        self.assertTrue(result.fixed_gate_result.converged)

    def test_invalid_inputs_raise(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            entropic_partial_ot(
                [[1.0]],
                source_unmatched_cost=1.0,
                target_unmatched_cost=1.0,
                epsilon=0.0,
            )
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            entropic_partial_ot(
                [[1.0], [2.0]],
                source_unmatched_cost=1.0,
                target_unmatched_cost=1.0,
                epsilon=0.5,
                source_weights=[1.0, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
