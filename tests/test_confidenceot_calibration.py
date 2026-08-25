import unittest
import warnings

import numpy as np

from confidenceot import calibrate_confidence_cost, rotation_null_costs


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
            backbone="uot", source_raw_acceptance_target=1.0,
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


if __name__ == "__main__":
    unittest.main()
