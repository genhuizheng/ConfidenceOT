import itertools
import unittest

import numpy as np

from traditional_ot.partial import partial_wasserstein_uniform


class PartialWassersteinUniformTest(unittest.TestCase):
    def test_selects_low_cost_cardinality_matching(self) -> None:
        cost = np.array(
            [
                [0.0, 9.0, 9.0, 9.0],
                [9.0, 0.1, 9.0, 9.0],
                [9.0, 9.0, 0.2, 9.0],
                [9.0, 9.0, 9.0, 8.0],
            ]
        )
        result = partial_wasserstein_uniform(cost, transported_mass=0.75)
        np.testing.assert_array_equal(result.source_gate, [True, True, True, False])
        np.testing.assert_array_equal(result.target_gate, [True, True, True, False])
        self.assertEqual(result.transported_count, 3)
        self.assertAlmostEqual(result.transported_mass, 0.75)
        self.assertAlmostEqual(result.objective, (0.0 + 0.1 + 0.2) / 4.0)

    def test_matches_brute_force(self) -> None:
        cost = np.array(
            [
                [0.7, 0.2, 0.8, 0.9],
                [0.3, 0.6, 0.1, 0.8],
                [0.9, 0.4, 0.5, 0.2],
                [0.8, 0.7, 0.3, 0.6],
            ]
        )
        result = partial_wasserstein_uniform(cost, transported_mass=0.5)
        brute = min(
            sum(cost[i, j] for i, j in zip(rows, columns))
            for rows in itertools.combinations(range(4), 2)
            for columns_subset in itertools.combinations(range(4), 2)
            for columns in itertools.permutations(columns_subset)
        )
        self.assertAlmostEqual(result.objective, brute / 4.0)
        self.assertEqual(int(result.source_gate.sum()), 2)
        self.assertEqual(int(result.target_gate.sum()), 2)

    def test_rejects_noninteger_cardinality(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            partial_wasserstein_uniform(np.eye(3), transported_mass=0.5)

    def test_requires_equal_support_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "equally sized"):
            partial_wasserstein_uniform(
                np.zeros((3, 4)), transported_mass=2.0 / 3.0
            )


if __name__ == "__main__":
    unittest.main()
