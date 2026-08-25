import unittest

import numpy as np

from traditional_ot.simulation import SCENARIOS, simulate_scenario


class TestScenarioSimulation(unittest.TestCase):
    def test_all_scenarios_have_consistent_truth(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                data = simulate_scenario(scenario, seed=2)
                self.assertEqual(len(data.source), len(data.source_population))
                self.assertEqual(len(data.target), len(data.target_population))
                self.assertEqual(len(data.target), len(data.true_target_rejection))
                for population in np.unique(data.source_population):
                    self.assertIn(population, data.population_coupling)
                for population, destinations in data.population_coupling.items():
                    if destinations:
                        self.assertAlmostEqual(sum(destinations.values()), 1.0)

    def test_scenario_specific_contracts(self):
        extinction = simulate_scenario("S1_extinction", seed=1)
        self.assertTrue(np.all(extinction.source_population[extinction.true_rejection] == "A"))
        novel = simulate_scenario("S2_novel_target", seed=1)
        self.assertFalse(novel.true_rejection.any())
        self.assertIn("G", novel.target_population)
        self.assertTrue(np.all(novel.target_population[novel.true_target_rejection] == "G"))
        split = simulate_scenario("S4_bifurcation", seed=1)
        self.assertEqual(split.population_coupling["B"], {"B1": 0.5, "B2": 0.5})
        abundance = simulate_scenario("S5_abundance", seed=1)
        counts = {name: int(np.sum(abundance.target_population == name)) for name in "AB"}
        self.assertEqual(counts["A"], 100)
        self.assertEqual(counts["B"], 4)


if __name__ == "__main__":
    unittest.main()
