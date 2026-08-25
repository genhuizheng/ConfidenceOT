import unittest

import numpy as np

from traditional_ot.simulation import simulate_s6


class TestS6Simulation(unittest.TestCase):
    def test_population_truth_and_independent_snapshots(self):
        data = simulate_s6(rho=0.6, seed=3, n_per_population=10, dimension=4)
        self.assertEqual(data.source.shape, (60, 4))
        self.assertEqual(data.target.shape, (50, 4))
        self.assertEqual(set(data.source_population), set("ABCDEF"))
        self.assertEqual(set(data.target_population), set("BCDEF"))
        np.testing.assert_array_equal(data.true_rejection, data.source_population == "A")
        self.assertFalse(np.any(data.target_population == "A"))

    def test_rho_controls_b_displacement(self):
        low = simulate_s6(rho=0.0, seed=9, n_per_population=100, noise=0.05)
        high = simulate_s6(rho=1.0, seed=9, n_per_population=100, noise=0.05)
        low_b = low.target[low.target_population == "B", 0].mean()
        high_b = high.target[high.target_population == "B", 0].mean()
        self.assertAlmostEqual(low_b - high_b, low.separation, places=10)


if __name__ == "__main__":
    unittest.main()

