import unittest

import numpy as np

import cellot
from cellot.baselines import balanced_ot
from cellot.cost_matrix_gate import fit_balanced_cost_matrix_gate
from cellot.soft_gate import fit_soft_gate_balanced


class TestCellOTPublicAPI(unittest.TestCase):
    def test_small_partial_ot_baseline(self) -> None:
        result = cellot.partial_wasserstein_uniform(
            np.array([[0.0, 2.0], [2.0, 3.0]]), transported_mass=0.5
        )
        self.assertEqual(result.transported_count, 1)
        self.assertAlmostEqual(result.transported_mass, 0.5)
        self.assertEqual(int(result.source_gate.sum()), 1)
        self.assertEqual(int(result.target_gate.sum()), 1)

    def test_method_families_are_separate_public_modules(self):
        self.assertIs(cellot.balanced_ot, balanced_ot)
        self.assertIs(cellot.fit_balanced_cost_matrix_gate, fit_balanced_cost_matrix_gate)
        self.assertIs(cellot.fit_soft_gate_balanced, fit_soft_gate_balanced)

    def test_cost_matrix_gate_is_not_baseline_api(self):
        from cellot import baselines

        self.assertFalse(hasattr(baselines, "fit_balanced_cost_matrix_gate"))
        self.assertFalse(hasattr(baselines, "fit_soft_gate_balanced"))

    def test_small_balanced_cost_matrix_fit(self):
        cost = np.array([[0.1, 1.0], [1.0, 0.1], [2.0, 2.0]])
        fit = fit_balanced_cost_matrix_gate(
            cost,
            rejection_cost=0.5,
            epsilon=0.2,
            source_rejection_budget=0.34,
            target_rejection_budget=0.0,
            update_target=False,
        )
        self.assertEqual(fit.coupling.shape, cost.shape)


if __name__ == "__main__":
    unittest.main()
