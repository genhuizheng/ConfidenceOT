import unittest
import warnings
from itertools import combinations

import numpy as np

from traditional_ot import (
    calibrate_bidirectional_rejection_cost,
    CalibrationError,
    confidence_filtered_bidirectional_uot,
    constrained_gate_update,
    population_monte_carlo_test,
    refit_post_selection_uot,
    solve_fixed_bidirectional_uot,
    unbalanced_ot,
)


class TestBidirectionalConfidenceFilteredUOT(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(33)
        source = rng.normal(size=(18, 4))
        target = rng.normal(size=(21, 4))
        self.cost = np.sum(
            (source[:, None, :] - target[None, :, :]) ** 2, axis=2
        )
        self.parameters = dict(
            epsilon=0.6,
            lambda_a=3.0,
            lambda_b=4.0,
            threshold=1e-11,
            max_iterations=20_000,
        )

    def test_all_accepted_fixed_gates_recover_vanilla(self):
        baseline = unbalanced_ot(self.cost, **self.parameters)
        fixed = solve_fixed_bidirectional_uot(
            self.cost,
            np.ones(self.cost.shape[0], dtype=bool),
            np.ones(self.cost.shape[1], dtype=bool),
            rejection_cost=2.0,
            **self.parameters,
        )
        np.testing.assert_allclose(
            fixed.coupling, baseline.coupling, rtol=2e-11, atol=1e-13
        )

    def test_rejected_rows_and_columns_have_common_conditionals(self):
        source_gate = np.ones(self.cost.shape[0], dtype=bool)
        target_gate = np.ones(self.cost.shape[1], dtype=bool)
        source_gate[[1, 7]] = False
        target_gate[[2, 9, 14]] = False
        result = solve_fixed_bidirectional_uot(
            self.cost,
            source_gate,
            target_gate,
            rejection_cost=2.5,
            **self.parameters,
        )
        v = np.exp(result.log_target_scaling)
        expected_row = result.target_marginal * v
        expected_row /= expected_row.sum()
        np.testing.assert_allclose(
            result.transition_probability[~source_gate],
            np.broadcast_to(expected_row, (np.count_nonzero(~source_gate), expected_row.size)),
            rtol=2e-11,
            atol=1e-13,
        )
        reverse = result.coupling / result.target_mass[None, :]
        u = np.exp(result.log_source_scaling)
        expected_column = result.source_marginal * u
        expected_column /= expected_column.sum()
        np.testing.assert_allclose(
            reverse[:, ~target_gate],
            np.broadcast_to(
                expected_column[:, None],
                (expected_column.size, np.count_nonzero(~target_gate)),
            ),
            rtol=2e-11,
            atol=1e-13,
        )

    def test_constrained_gate_update_is_coefficient_based_and_tie_aware(self):
        collapsed = constrained_gate_update(
            [0.20, 0.05, 0.12], [True, False, False]
        )
        np.testing.assert_array_equal(collapsed.gate, [False, True, False])
        self.assertTrue(collapsed.constraint_active)

        tied = constrained_gate_update(
            [-0.3, 0.0, 0.2, 0.0], [False, True, True, False]
        )
        np.testing.assert_array_equal(tied.gate, [True, True, False, False])
        self.assertFalse(tied.constraint_active)
        self.assertFalse(tied.tie_fill)

        tie_fill = constrained_gate_update([0.0, 0.4], [False, True])
        np.testing.assert_array_equal(tie_fill.gate, [True, False])
        self.assertTrue(tie_fill.tie_fill)

    def test_loss_unit_tolerance_is_invariant_to_cell_mass_scaling(self):
        excess_loss = np.array([-0.020, -0.003, 0.004, 0.030])
        current = np.array([False, True, False, True])
        first_mass = np.array([0.20, 0.10, 0.30, 0.40])
        second_mass = first_mass / 50.0

        first = constrained_gate_update(
            first_mass * excess_loss,
            current,
            min_accepted=2,
            tau_s=0.005,
            tolerance_scale=first_mass,
        )
        second = constrained_gate_update(
            second_mass * excess_loss,
            current,
            min_accepted=2,
            tau_s=0.005,
            tolerance_scale=second_mass,
        )

        np.testing.assert_array_equal(first.gate, [True, True, False, False])
        np.testing.assert_array_equal(second.gate, first.gate)
        self.assertEqual(first.tie_count, 2)
        self.assertEqual(second.tie_count, 2)

    def test_tolerance_scale_validation(self):
        with self.assertRaisesRegex(ValueError, "tolerance_scale"):
            constrained_gate_update(
                [0.1, -0.1],
                [True, True],
                tau_s=0.01,
                tolerance_scale=[1.0],
            )
        with self.assertRaisesRegex(ValueError, "tolerance_scale"):
            constrained_gate_update(
                [0.1, -0.1],
                [True, True],
                tau_s=0.01,
                tolerance_scale=[1.0, -1.0],
            )

    def test_t7_budgeted_gate_matches_exhaustive_optimum(self):
        rng = np.random.default_rng(0)
        zero_tie_cases = 0
        for _ in range(300):
            n = int(rng.integers(4, 9))
            score = np.round(rng.normal(size=n), 1)
            minimum = int(rng.integers(1, n + 1))
            current = rng.integers(0, 2, n).astype(bool)
            update = constrained_gate_update(
                score, current, min_accepted=minimum
            )
            best = np.inf
            for size in range(minimum, n + 1):
                for indices in combinations(range(n), size):
                    best = min(best, float(score[list(indices)].sum()))
            self.assertGreaterEqual(update.gate.sum(), minimum)
            self.assertAlmostEqual(float(score[update.gate].sum()), best)
            zero = (score == 0.0) & current
            if np.any(zero):
                zero_tie_cases += 1
                self.assertTrue(np.all(update.gate[zero]))
        self.assertGreater(zero_tie_cases, 0)

    def test_t9_t10_budget_extremes_and_unequal_floors(self):
        cost = np.full((5, 7), 10.0)
        all_accepted = confidence_filtered_bidirectional_uot(
            cost,
            rejection_cost=1.0,
            epsilon=0.5,
            lambda_a=3.0,
            lambda_b=3.0,
            variant="exact",
            source_rejection_budget=0.0,
            target_rejection_budget=0.0,
        )
        self.assertTrue(np.all(all_accepted.source_gate))
        self.assertTrue(np.all(all_accepted.target_gate))

        unequal = confidence_filtered_bidirectional_uot(
            cost,
            rejection_cost=1.0,
            epsilon=0.5,
            lambda_a=3.0,
            lambda_b=3.0,
            variant="exact",
            source_rejection_budget=0.20,
            target_rejection_budget=0.40,
        )
        self.assertEqual(unequal.source_gate.sum(), 4)
        self.assertEqual(unequal.target_gate.sum(), 5)
        self.assertEqual(unequal.source_raw_gate.sum(), 0)
        self.assertEqual(unequal.target_raw_gate.sum(), 0)
        self.assertTrue(unequal.source_budget_binding)
        self.assertTrue(unequal.target_budget_binding)

    def test_t11_budget_active_exact_blocks_descend(self):
        result = confidence_filtered_bidirectional_uot(
            np.full((8, 9), 10.0),
            rejection_cost=1.0,
            epsilon=0.5,
            lambda_a=3.0,
            lambda_b=4.0,
            variant="exact",
            source_rejection_budget=0.25,
            target_rejection_budget=0.34,
            threshold=1e-12,
        )
        self.assertTrue(result.outer_converged)
        self.assertTrue(result.source_budget_binding)
        self.assertTrue(result.target_budget_binding)
        self.assertTrue(np.all(np.diff(result.objective_history) <= 2e-10))
        self.assertTrue(any(x > 0 for x in result.source_forced_acceptance_history))
        self.assertTrue(any(x > 0 for x in result.target_forced_acceptance_history))

    def test_t12_terminal_re_solve_matches_returned_gates(self):
        cost = np.array(
            [[0.1, 3.0, 3.0], [3.0, 0.1, 3.0], [8.0, 8.0, 8.0]],
            dtype=float,
        )
        capped = confidence_filtered_bidirectional_uot(
            cost,
            rejection_cost=1.0,
            epsilon=0.5,
            lambda_a=3.0,
            lambda_b=3.0,
            variant="exact",
            source_rejection_budget=0.34,
            target_rejection_budget=0.34,
            threshold=1e-12,
            max_outer_iterations=1,
        )
        self.assertEqual(capped.status, "iteration_capped")
        self.assertEqual(capped.objective_stage_history[-1], "terminal_transport")
        fixed = solve_fixed_bidirectional_uot(
            cost,
            capped.source_gate,
            capped.target_gate,
            rejection_cost=1.0,
            epsilon=0.5,
            lambda_a=3.0,
            lambda_b=3.0,
            threshold=1e-12,
        )
        np.testing.assert_allclose(capped.coupling, fixed.coupling, rtol=1e-11)

    def test_t14_source_target_symmetry_with_unequal_budgets(self):
        # The gate projection itself is side-symmetric even though the fixed
        # source-then-target Gauss--Seidel trajectory is order-dependent.
        rng = np.random.default_rng(144)
        for n, budget in ((18, 0.20), (21, 0.35)):
            score = rng.normal(size=n)
            current = rng.integers(0, 2, n).astype(bool)
            minimum = int(np.ceil((1.0 - budget) * n))
            source_update = constrained_gate_update(
                score, current, min_accepted=minimum
            )
            target_update = constrained_gate_update(
                score.copy(), current.copy(), min_accepted=minimum
            )
            np.testing.assert_array_equal(source_update.gate, target_update.gate)
            self.assertGreaterEqual(source_update.gate.sum(), minimum)

        result = confidence_filtered_bidirectional_uot(
            self.cost,
            rejection_cost=2.0,
            epsilon=0.6,
            lambda_a=3.0,
            lambda_b=3.0,
            variant="reversible",
            source_rejection_budget=0.20,
            target_rejection_budget=0.35,
            threshold=1e-11,
        )
        self.assertEqual(result.source_min_accepted, 15)
        self.assertEqual(result.target_min_accepted, 14)
        self.assertEqual(result.variant, "reversible")
        self.assertEqual(result.update_order, "source_then_target")

    def test_exact_three_block_objective_is_monotone(self):
        cost = np.array(
            [[0.1, 3.0, 3.0], [3.0, 0.1, 3.0], [8.0, 8.0, 8.0]],
            dtype=float,
        )
        result = confidence_filtered_bidirectional_uot(
            cost,
            rejection_cost=1.0,
            epsilon=0.5,
            lambda_a=3.0,
            lambda_b=3.0,
            variant="exact",
            source_rejection_budget=0.34,
            target_rejection_budget=0.34,
            tau_s=0.0,
            threshold=1e-12,
            max_outer_iterations=20,
        )
        self.assertTrue(result.outer_converged)
        self.assertTrue(result.inner_converged)
        self.assertTrue(np.all(np.diff(result.objective_history) <= 2e-10))
        np.testing.assert_array_equal(result.source_gate, [True, True, False])
        np.testing.assert_array_equal(result.target_gate, [True, True, False])
        self.assertGreaterEqual(result.source_gate.sum(), 1)
        self.assertGreaterEqual(result.target_gate.sum(), 1)

    def test_permutation_invariance_on_both_sides(self):
        rng = np.random.default_rng(91)
        source_permutation = rng.permutation(self.cost.shape[0])
        target_permutation = rng.permutation(self.cost.shape[1])
        source_gate = np.ones(self.cost.shape[0], dtype=bool)
        target_gate = np.ones(self.cost.shape[1], dtype=bool)
        original = solve_fixed_bidirectional_uot(
            self.cost,
            source_gate,
            target_gate,
            rejection_cost=2.0,
            **self.parameters,
        )
        permuted = solve_fixed_bidirectional_uot(
            self.cost[source_permutation][:, target_permutation],
            source_gate[source_permutation],
            target_gate[target_permutation],
            rejection_cost=2.0,
            **self.parameters,
        )
        inverse_source = np.argsort(source_permutation)
        inverse_target = np.argsort(target_permutation)
        np.testing.assert_allclose(
            original.coupling,
            permuted.coupling[inverse_source][:, inverse_target],
            rtol=2e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            original.conditional_loss,
            permuted.conditional_loss[inverse_source],
            rtol=2e-10,
            atol=1e-11,
        )

    def test_joint_calibration_and_infeasible_rates(self):
        observed = np.full((6, 7), 4.0)
        for index in range(6):
            observed[index, index] = 0.05
        null_1 = np.arange(42, dtype=float).reshape(6, 7) / 20.0 + 1.0
        null_2 = np.flip(null_1, axis=1) + 0.15
        calibration = calibrate_bidirectional_rejection_cost(
            observed,
            [null_1, null_2],
            epsilon=0.5,
            lambda_a=3.0,
            lambda_b=3.0,
            source_acceptance_target=0.34,
            target_acceptance_target=0.30,
            grid_size=7,
            max_outer_iterations=15,
        )
        self.assertGreater(calibration.rejection_cost, 0.0)
        self.assertLessEqual(calibration.null_source_acceptance, 0.34 + 1e-12)
        self.assertLessEqual(calibration.null_target_acceptance, 0.30 + 1e-12)
        self.assertGreaterEqual(calibration.null_source_projected_acceptance, 0.9)
        self.assertGreaterEqual(calibration.null_target_projected_acceptance, 0.9)
        self.assertEqual(calibration.observed_result.variant, "reversible")
        self.assertEqual(calibration.variant, "reversible")
        self.assertEqual(calibration.quantile_method, "linear")

    def test_t13_calibration_retains_iteration_capped_fit_with_warning(self):
        observed = np.full((4, 4), 5.0)
        np.fill_diagonal(observed, 0.1)
        null = np.full((4, 4), 10.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calibration = calibrate_bidirectional_rejection_cost(
                observed,
                [null],
                epsilon=0.5,
                lambda_a=3.0,
                lambda_b=3.0,
                source_acceptance_target=0.25,
                target_acceptance_target=0.25,
                source_rejection_budget=0.50,
                target_rejection_budget=0.50,
                max_outer_iterations=1,
                grid_size=3,
            )
        self.assertFalse(calibration.numerically_certified)
        self.assertFalse(calibration.calibration_valid)
        self.assertTrue(any("iteration-capped" in str(item.message) for item in caught))
        self.assertIsNotNone(calibration.observed_result)

    def test_t15_t19_post_selection_refit_and_index_mapping(self):
        cost = np.arange(30, dtype=float).reshape(5, 6) / 10.0
        source_gate = np.array([True, False, True, False, True])
        target_gate = np.array([False, True, True, False, False, True])
        o2 = refit_post_selection_uot(
            cost,
            source_gate,
            target_gate,
            epsilon=0.5,
            lambda_a=2.0,
            lambda_b=3.0,
            threshold=1e-11,
        )
        np.testing.assert_array_equal(o2.source_indices, [0, 2, 4])
        np.testing.assert_array_equal(o2.target_indices, [1, 2, 5])
        np.testing.assert_array_equal(
            o2.uot_result.cost_matrix, cost[np.ix_([0, 2, 4], [1, 2, 5])]
        )
        self.assertEqual(o2.uot_result.coupling.shape, (3, 3))
        self.assertEqual(o2.original_shape, (5, 6))

    def test_invalid_budget_and_initial_gate_raise(self):
        with self.assertRaisesRegex(ValueError, "source_rejection_budget"):
            confidence_filtered_bidirectional_uot(
                np.ones((3, 4)), rejection_cost=1, epsilon=1,
                lambda_a=1, lambda_b=1, source_rejection_budget=1.0,
            )
        with self.assertRaisesRegex(ValueError, "violates"):
            confidence_filtered_bidirectional_uot(
                np.ones((3, 4)), rejection_cost=1, epsilon=1,
                lambda_a=1, lambda_b=1, source_rejection_budget=0.1,
                initial_source_gate=[True, True, False],
            )

    def test_population_test_uses_identity_preserving_replicates_and_by(self):
        observed = np.array([0.1, 0.2, 2.0, 2.1])
        null = np.array(
            [
                [0.8, 0.9, 1.9, 2.0],
                [0.7, 1.0, 2.2, 2.3],
                [0.9, 0.8, 1.8, 2.4],
                [1.1, 1.2, 2.1, 2.2],
            ]
        )
        result = population_monte_carlo_test(
            observed, null, ["A", "A", "B", "B"], adjustment="by"
        )
        np.testing.assert_array_equal(result.groups, ["A", "B"])
        self.assertAlmostEqual(result.p_value[0], 0.2)
        self.assertTrue(np.all(result.q_value >= result.p_value))
        with self.assertRaisesRegex(ValueError, "shape"):
            population_monte_carlo_test(observed, null[:, :3], ["A"] * 4)


if __name__ == "__main__":
    unittest.main()
