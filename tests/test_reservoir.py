import unittest

import numpy as np

from traditional_ot import birth_death_uot


class TestBirthDeathReservoir(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            epsilon=0.08,
            lambda_a=0.5,
            lambda_b=10.0,
            reservoir_weight=0.5,
            threshold=1e-10,
            max_iterations=20_000,
        )

    def test_shapes_probabilities_and_real_conditionals(self):
        cost = np.array([[0.0, 2.0, 3.0], [2.0, 0.0, 2.0]])
        result = birth_death_uot(cost, birth_cost=0.8, death_cost=0.8, **self.parameters)
        self.assertEqual(result.coupling.shape, (3, 4))
        self.assertEqual(result.real_coupling.shape, cost.shape)
        self.assertEqual(result.source_death_probability.shape, (2,))
        self.assertEqual(result.target_birth_probability.shape, (3,))
        self.assertTrue(np.all((0 <= result.source_death_probability) & (result.source_death_probability <= 1)))
        self.assertTrue(np.all((0 <= result.target_birth_probability) & (result.target_birth_probability <= 1)))
        np.testing.assert_allclose(result.real_transition_probability.sum(axis=1), 1.0, atol=1e-12)

    def test_unmatched_source_and_target_receive_highest_scores(self):
        cost = np.array([
            [0.0, 4.0, 4.0],
            [4.0, 0.0, 4.0],
            [4.0, 4.0, 4.0],
        ])
        result = birth_death_uot(cost, birth_cost=0.5, death_cost=0.5, **self.parameters)
        self.assertEqual(int(np.argmax(result.source_death_probability)), 2)
        self.assertEqual(int(np.argmax(result.target_birth_probability)), 2)

    def test_expensive_reservoir_suppresses_reservoir_use(self):
        cost = np.array([[0.0, 2.0], [2.0, 0.0]])
        cheap = birth_death_uot(cost, birth_cost=0.3, death_cost=0.3, **self.parameters)
        expensive = birth_death_uot(cost, birth_cost=5.0, death_cost=5.0, **self.parameters)
        self.assertLess(expensive.death_mass_fraction, cheap.death_mass_fraction)
        self.assertLess(expensive.birth_mass_fraction, cheap.birth_mass_fraction)

    def test_invalid_inputs_raise(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            birth_death_uot([[-1.0]], birth_cost=1.0, death_cost=1.0, **self.parameters)
        with self.assertRaisesRegex(ValueError, "reservoir_weight"):
            birth_death_uot([[1.0]], birth_cost=1.0, death_cost=1.0, reservoir_weight=0.0,
                            epsilon=0.1, lambda_a=1.0, lambda_b=1.0)


if __name__ == "__main__":
    unittest.main()
