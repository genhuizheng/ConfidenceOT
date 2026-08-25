import unittest

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
        result = calibrate_confidence_cost(
            [source_nulls[0], target_nulls[0]],
            [source_nulls[1], target_nulls[1]],
            backbone="uot",
            source_raw_acceptance_target=1.0,
            target_raw_acceptance_target=1.0,
            grid_size=3,
            tolerance=1e-3,
            max_iterations=1_000,
            max_outer_iterations=30,
            device="cpu",
        )
        self.assertGreater(result.rejection_cost, 0.0)
        self.assertEqual(result.backbone, "uot")
        self.assertEqual(len(result.validation), 2)
        self.assertTrue(np.all(np.isfinite(result.curve_costs)))


if __name__ == "__main__":
    unittest.main()
