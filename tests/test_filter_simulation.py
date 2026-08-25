import unittest

import numpy as np

from traditional_ot.filter_simulation import (
    FILTER_SCENARIOS,
    simulate_filter_scenario,
)


class TestFilterSimulation(unittest.TestCase):
    def test_all_scenarios_have_consistent_arrays(self):
        for scenario in FILTER_SCENARIOS:
            with self.subTest(scenario=scenario):
                result = simulate_filter_scenario(scenario, seed=4, n_per_population=8)
                self.assertEqual(result.source.shape[0], result.source_population.size)
                self.assertEqual(result.target.shape[0], result.target_population.size)
                self.assertEqual(result.true_source_rejection.shape, result.source_population.shape)
                self.assertEqual(result.true_target_rejection.shape, result.target_population.shape)
                self.assertEqual(result.source_contamination_kind.shape, result.source_population.shape)
                self.assertEqual(result.target_contamination_kind.shape, result.target_population.shape)
                self.assertTrue(np.all(np.isfinite(result.source)))
                self.assertTrue(np.all(np.isfinite(result.target)))

    def test_separate_and_combined_truth(self):
        clean = simulate_filter_scenario("Q0_clean_differentiation", seed=1, n_per_population=8)
        self.assertFalse(np.any(clean.true_source_rejection))
        self.assertFalse(np.any(clean.true_target_rejection))
        self.assertEqual(clean.population_coupling["B"], {"B1": 0.5, "B2": 0.5})

        shared = simulate_filter_scenario(
            "Q1_shared_contamination_negative_control", seed=1, n_per_population=8
        )
        self.assertTrue(np.any(shared.source_contamination_kind != "clean"))
        self.assertTrue(np.any(shared.target_contamination_kind != "clean"))
        self.assertTrue(np.all(
            shared.source_contamination_kind[shared.source_population == "QC"]
            == "unidentifiable"
        ))

        extinction = simulate_filter_scenario("Q2_extinction_contamination", seed=1, n_per_population=8)
        self.assertEqual(extinction.population_coupling["A"], {})
        self.assertTrue(np.all(extinction.true_source_rejection[extinction.source_population == "A"]))
        self.assertFalse(np.any(extinction.target_population == "G"))

        emergence = simulate_filter_scenario("Q3_emergence_contamination", seed=1, n_per_population=8)
        self.assertTrue(np.all(emergence.true_target_rejection[emergence.target_population == "G"]))
        self.assertEqual(emergence.population_coupling["B"], {"B": 1.0})

        combined = simulate_filter_scenario(
            "Q5_turnover_differentiation_contamination", seed=1, n_per_population=8
        )
        self.assertEqual(combined.population_coupling["A"], {})
        self.assertEqual(combined.population_coupling["B"], {"B1": 0.5, "B2": 0.5})
        self.assertTrue(np.any(combined.target_population == "G"))
        self.assertTrue(np.any(combined.source_contamination_kind != "clean"))
        self.assertTrue(np.any(combined.target_contamination_kind != "clean"))

    def test_validation(self):
        with self.assertRaises(ValueError):
            simulate_filter_scenario("bad", seed=0)
        with self.assertRaises(ValueError):
            simulate_filter_scenario(FILTER_SCENARIOS[0], seed=0, dimension=3)
        with self.assertRaises(ValueError):
            simulate_filter_scenario(FILTER_SCENARIOS[0], seed=0, contamination_fraction=.5)


if __name__ == "__main__":
    unittest.main()
