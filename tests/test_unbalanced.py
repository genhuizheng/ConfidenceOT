import unittest

import numpy as np

from traditional_ot import (
    calibrate_rejection_cost,
    confidence_filtered_uot,
    solve_fixed_gate_uot,
    unbalanced_ot,
)
from traditional_ot.unbalanced import (
    feature_permutation_null,
    random_rotation_null,
    squared_euclidean_cost,
)


class TestUnbalancedOT(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(60, 5))
        target = rng.normal(size=(70, 5))
        self.cost = squared_euclidean_cost(source, target)
        self.parameters = dict(
            epsilon=0.5,
            lambda_a=1.0,
            lambda_b=5.0,
            threshold=1e-12,
            max_iterations=20_000,
        )

    def test_one_by_one_analytic_solution(self):
        cost = 2.3
        epsilon = 0.4
        lambda_a = 1.2
        lambda_b = 2.1
        result = unbalanced_ot(
            [[cost]],
            epsilon=epsilon,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            threshold=1e-13,
        )
        expected = np.exp(-cost / (epsilon + lambda_a + lambda_b))
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.coupling[0, 0], expected, places=11)
        self.assertAlmostEqual(result.transition_probability[0, 0], 1.0)

    def test_t1_all_accepted_recovers_vanilla(self):
        vanilla = unbalanced_ot(self.cost, **self.parameters)
        fixed = solve_fixed_gate_uot(
            self.cost,
            np.ones(60, dtype=bool),
            rejection_cost=2.0,
            **self.parameters,
        )
        self.assertTrue(vanilla.converged and fixed.converged)
        np.testing.assert_allclose(fixed.coupling, vanilla.coupling, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(fixed.conditional_loss, vanilla.conditional_loss)

    def test_t2_rejected_row_closed_form(self):
        gate = np.ones(60, dtype=bool)
        gate[[1, 7, 15, 31]] = False
        rejection_cost = 2.4
        result = solve_fixed_gate_uot(
            self.cost,
            gate,
            rejection_cost=rejection_cost,
            **self.parameters,
        )
        self.assertTrue(result.converged)
        v = np.exp(result.log_target_scaling)
        target_profile = result.target_marginal * v
        target_profile /= target_profile.sum()
        expected_mass = (
            result.source_marginal
            * np.sum(result.target_marginal * v) ** (1.0 - result.alpha)
            * np.exp(-rejection_cost / (result.lambda_a + result.epsilon))
        )
        np.testing.assert_allclose(
            result.transition_probability[~gate],
            np.broadcast_to(target_profile, (np.count_nonzero(~gate), len(target_profile))),
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            result.source_mass[~gate], expected_mass[~gate], rtol=1e-11, atol=1e-14
        )

    def test_t3_target_penalty_zero_is_posthoc_on_accepted_rows(self):
        gate = np.ones(60, dtype=bool)
        gate[::4] = False
        parameters = dict(self.parameters)
        parameters["lambda_b"] = 0.0
        vanilla = unbalanced_ot(self.cost, **parameters)
        fixed = solve_fixed_gate_uot(
            self.cost,
            gate,
            rejection_cost=2.0,
            **parameters,
        )
        np.testing.assert_allclose(
            fixed.coupling[gate], vanilla.coupling[gate], rtol=1e-12, atol=1e-14
        )

    def test_t4_target_permutation_invariance(self):
        rng = np.random.default_rng(8)
        permutation = rng.permutation(self.cost.shape[1])
        original = unbalanced_ot(self.cost, **self.parameters)
        permuted = unbalanced_ot(self.cost[:, permutation], **self.parameters)
        np.testing.assert_allclose(
            original.conditional_loss, permuted.conditional_loss, rtol=1e-11, atol=1e-12
        )
        inverse = np.argsort(permutation)
        np.testing.assert_allclose(
            original.coupling, permuted.coupling[:, inverse], rtol=1e-11, atol=1e-13
        )

    def test_exact_outer_objective_is_monotone(self):
        result = confidence_filtered_uot(
            self.cost,
            rejection_cost=float(np.quantile(self.cost.min(axis=1), 0.45)),
            variant="exact",
            max_outer_iterations=20,
            **self.parameters,
        )
        self.assertTrue(result.inner_converged)
        self.assertTrue(result.outer_converged)
        self.assertTrue(np.all(np.diff(result.objective_history) <= 1e-10))
        np.testing.assert_array_equal(
            result.gate,
            result.gate_score < result.rejection_cost,
        )

    def test_invalid_inputs_raise(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            unbalanced_ot([[-1.0]], epsilon=1, lambda_a=1, lambda_b=1)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            unbalanced_ot([[1.0]], epsilon=0, lambda_a=1, lambda_b=1)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            unbalanced_ot(
                np.ones((2, 2)), epsilon=1, lambda_a=1, lambda_b=1,
                source_weights=[1, 0],
            )
        with self.assertRaisesRegex(ValueError, "variant"):
            confidence_filtered_uot(
                np.ones((2, 2)), rejection_cost=1, epsilon=1,
                lambda_a=1, lambda_b=1, variant="wrong",
            )

    def test_geometric_null_helpers(self):
        rng = np.random.default_rng(12)
        source = rng.normal(size=(20, 4))
        target = rng.normal(size=(25, 4))
        rotation_cost = random_rotation_null(
            source, target, rng=np.random.default_rng(13)
        )
        self.assertEqual(rotation_cost.shape, (20, 25))

        # Recover the feature-permuted source marginals indirectly by repeating
        # the helper's construction with the same seed.
        permutation_cost = feature_permutation_null(
            source, target, rng=np.random.default_rng(14)
        )
        self.assertEqual(permutation_cost.shape, (20, 25))
        self.assertTrue(np.all(rotation_cost >= 0.0))
        self.assertTrue(np.all(permutation_cost >= 0.0))

    def test_null_calibration_hits_requested_acceptance(self):
        observed = np.full((8, 8), 4.0)
        np.fill_diagonal(observed, 0.1)
        null_1 = np.array([
            [1.0 + i * 0.2 + j * 0.1 for j in range(8)] for i in range(8)
        ])
        null_2 = np.array([
            [1.2 + i * 0.15 + (7 - j) * 0.12 for j in range(8)]
            for i in range(8)
        ])
        result = calibrate_rejection_cost(
            observed,
            [null_1, null_2],
            epsilon=0.5,
            lambda_a=1.0,
            lambda_b=5.0,
            acceptance_target=0.25,
            acceptance_tolerance=0.05,
            max_outer_iterations=20,
            grid_size=9,
        )
        self.assertGreater(result.rejection_cost, 0.0)
        self.assertAlmostEqual(result.null_acceptance, 0.25)
        self.assertEqual(result.observed_acceptance, 1.0)
        self.assertTrue(result.monotonic_on_grid)


if __name__ == "__main__":
    unittest.main()
