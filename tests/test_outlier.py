import unittest

import numpy as np

from traditional_ot import cross_snapshot_outlier_scores, outlier_trimmed_ot, outlier_trimmed_uot


class TestOutlierTrimmedOT(unittest.TestCase):
    def test_unsupported_groups_have_largest_scores(self):
        rng = np.random.default_rng(3)
        source = np.vstack([rng.normal(0, 0.1, (30, 3)), rng.normal(5, 0.1, (10, 3))])
        target = np.vstack([rng.normal(0, 0.1, (30, 3)), rng.normal(-5, 0.1, (10, 3))])
        source_score, target_score = cross_snapshot_outlier_scores(source, target, k_neighbors=5)
        self.assertGreater(source_score[30:].mean(), source_score[:30].mean())
        self.assertGreater(target_score[30:].mean(), target_score[:30].mean())

    def test_trimmed_rows_and_columns_are_zero(self):
        rng = np.random.default_rng(5)
        source = np.vstack([rng.normal(0, 0.1, (30, 2)), rng.normal(5, 0.1, (10, 2))])
        target = np.vstack([rng.normal(0, 0.1, (30, 2)), rng.normal(-5, 0.1, (10, 2))])
        result = outlier_trimmed_ot(
            source, target, source_outlier_fraction=0.25, target_outlier_fraction=0.25,
            k_neighbors=5, epsilon=0.1,
        )
        self.assertTrue(result.inlier_result.converged)
        np.testing.assert_allclose(result.transition_probability[result.source_outlier], 0.0)
        np.testing.assert_allclose(result.transition_probability[:, result.target_outlier], 0.0)
        np.testing.assert_allclose(result.transition_probability[result.source_inlier].sum(axis=1), 1.0)
        self.assertEqual(result.solver, "balanced")

    def test_unbalanced_trimmed_uses_same_masks(self):
        rng = np.random.default_rng(7)
        source = np.vstack([rng.normal(0, 0.1, (30, 2)), rng.normal(5, 0.1, (10, 2))])
        target = np.vstack([rng.normal(0, 0.1, (30, 2)), rng.normal(-5, 0.1, (10, 2))])
        balanced = outlier_trimmed_ot(
            source, target, source_outlier_fraction=.25, target_outlier_fraction=.25,
            k_neighbors=5, epsilon=.1,
        )
        unbalanced = outlier_trimmed_uot(
            source, target, source_outlier_fraction=.25, target_outlier_fraction=.25,
            k_neighbors=5, epsilon=.1, lambda_a=.5, lambda_b=5,
        )
        np.testing.assert_array_equal(unbalanced.source_outlier, balanced.source_outlier)
        np.testing.assert_array_equal(unbalanced.target_outlier, balanced.target_outlier)
        np.testing.assert_allclose(unbalanced.transition_probability[unbalanced.source_inlier].sum(axis=1), 1.0)
        self.assertEqual(unbalanced.solver, "unbalanced")

    def test_invalid_fraction_raises(self):
        with self.assertRaisesRegex(ValueError, "fraction"):
            outlier_trimmed_ot(
                np.ones((12, 2)), np.zeros((12, 2)), source_outlier_fraction=1.0,
                target_outlier_fraction=0.0,
            )


if __name__ == "__main__":
    unittest.main()
