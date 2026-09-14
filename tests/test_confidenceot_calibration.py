import unittest
import warnings

import numpy as np

from confidenceot import (
    ConfidenceOT,
    calibrate_confidence_cost,
    rotation_null_costs,
    within_side_null_costs,
)


class ConfidenceOTCalibrationTest(unittest.TestCase):
    def test_rotation_nulls_and_two_stage_calibration(self) -> None:
        rng = np.random.default_rng(11)
        source = rng.normal(size=(12, 4))
        target = rng.normal(size=(12, 4))
        observed = np.sum((source[:, None] - target[None, :]) ** 2, axis=2)
        scale = float(np.median(observed[observed > 0]))
        source_nulls, target_nulls = rotation_null_costs(
            source, target, observed_scale=scale, seed=19, n_replicates=2
        )
        self.assertEqual(len(source_nulls), 2)
        self.assertEqual(source_nulls[0].shape, (12, 12))
        common = dict(
            backbone="uot", null_semantics="cross_side_rotation",
            source_raw_acceptance_target=1.0,
            target_raw_acceptance_target=1.0, grid_size=3,
            tolerance=1e-3, max_iterations=1_000,
            max_outer_iterations=30, device="cpu",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = calibrate_confidence_cost(
                [source_nulls[0], target_nulls[0]],
                [source_nulls[1], target_nulls[1]], workers=1, **common,
            )
            parallel = calibrate_confidence_cost(
                [source_nulls[0], target_nulls[0]],
                [source_nulls[1], target_nulls[1]], workers=2, **common,
            )
        self.assertGreater(result.rejection_cost, 0.0)
        self.assertEqual(result.backbone, "uot")
        self.assertEqual(len(result.validation), 2)
        self.assertAlmostEqual(
            result.validation_source_raw_acceptance,
            np.mean([record.source_raw_acceptance for record in result.validation]),
        )
        self.assertAlmostEqual(
            result.validation_target_raw_acceptance,
            np.mean([record.target_raw_acceptance for record in result.validation]),
        )
        self.assertTrue(result.validation_aggregate_valid)
        self.assertTrue(np.all(np.isfinite(result.curve_costs)))
        self.assertEqual(parallel.rejection_cost, result.rejection_cost)
        np.testing.assert_array_equal(parallel.curve_costs, result.curve_costs)
        np.testing.assert_array_equal(
            parallel.source_raw_acceptance_curve, result.source_raw_acceptance_curve
        )
        np.testing.assert_array_equal(
            parallel.target_raw_acceptance_curve, result.target_raw_acceptance_curve
        )
        self.assertEqual(parallel.validation, result.validation)

    def test_within_side_null_accepts_a_homogeneous_population(self) -> None:
        """A homogeneous cloud must not be rejected wholesale.

        Source and target are drawn from one distribution, so no cell is
        incompatible and a specific method should accept nearly all of them.
        A rotation null rotates the target about its own centroid, preserving
        every point's radial position, so it resembles the observed data and
        drives the rejection cost below the median cost: almost everything is
        rejected. A within-side split null contains no incompatible cells by
        construction, so it yields a cost that accepts the population.
        """
        rng = np.random.default_rng(5)
        source = rng.normal(size=(60, 5))
        target = rng.normal(size=(60, 5))
        observed = np.sum((source[:, None] - target[None, :]) ** 2, axis=2)
        scale = float(np.median(observed[observed > 0]))
        cost = observed / scale
        shared = dict(
            backbone="uot", grid_size=5, tolerance=1e-3,
            max_iterations=2_000, max_outer_iterations=30, device="cpu",
            source_rejection_budget=0.85, target_rejection_budget=0.85,
            emit_warnings=False,
        )
        source_nulls, target_nulls = rotation_null_costs(
            source, target, observed_scale=scale, seed=7, n_replicates=2
        )
        within_nulls = within_side_null_costs(
            source, observed_scale=scale, seed=7, n_replicates=2
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rotation = calibrate_confidence_cost(
                [source_nulls[0], target_nulls[0]],
                [source_nulls[1], target_nulls[1]],
                null_semantics="cross_side_rotation", **shared,
            )
            within = calibrate_confidence_cost(
                [within_nulls[0]], [within_nulls[1]],
                null_semantics="within_side_split", **shared,
            )

            def raw_acceptance(rejection_cost: float) -> float:
                model = ConfidenceOT(
                    backbone="uot", variant="exact",
                    rejection_cost=rejection_cost, device="cpu",
                    source_rejection_budget=0.85, target_rejection_budget=0.85,
                    tolerance=1e-3, warn_on_terminal=False,
                )
                return model.fit(cost).source_raw_acceptance

            rotation_acceptance = raw_acceptance(rotation.rejection_cost)
            within_acceptance = raw_acceptance(within.rejection_cost)

        self.assertEqual(rotation.null_semantics, "cross_side_rotation")
        self.assertEqual(within.null_semantics, "within_side_split")
        self.assertIn("0.9", within.acceptance_requirement)
        # The defect and its fix, on data whose correct answer is "reject none".
        self.assertLess(rotation_acceptance, 0.5)
        self.assertGreater(within_acceptance, 0.8)
        self.assertGreater(within.rejection_cost, rotation.rejection_cost)


if __name__ == "__main__":
    unittest.main()
