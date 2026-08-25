import unittest

import numpy as np

from traditional_ot import traditional_method
from traditional_ot.sinkhorn import _solve_traditional_ot


class TestTraditionalOT(unittest.TestCase):
    def test_identical_columns_match_themselves(self):
        matrix = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 0.0]])
        result = _solve_traditional_ot(matrix, matrix, epsilon=0.05, threshold=1e-10)

        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.coupling.sum(axis=1), np.full(3, 1 / 3), atol=1e-8)
        np.testing.assert_allclose(result.coupling.sum(axis=0), np.full(3, 1 / 3), atol=1e-8)
        np.testing.assert_allclose(result.transition_probability.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.all(np.argmax(result.transition_probability, axis=1) == np.arange(3)))

    def test_default_orientation_is_columns(self):
        source_columns = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
        target_columns = np.array([[0.1, 2.1, 4.0], [1.1, 3.1, 5.0], [2.1, 4.1, 6.0]])

        by_columns = _solve_traditional_ot(source_columns, target_columns, epsilon=0.1)
        by_rows = _solve_traditional_ot(
            source_columns.T, target_columns.T, cells_axis=0, epsilon=0.1
        )

        np.testing.assert_allclose(by_columns.coupling, by_rows.coupling)
        np.testing.assert_allclose(
            by_columns.transition_probability, by_rows.transition_probability
        )

    def test_nonuniform_marginals(self):
        source = np.array([[0.0, 1.0], [0.0, 1.0]])
        target = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        a = np.array([1.0, 3.0])
        b = np.array([1.0, 1.0, 2.0])

        result = _solve_traditional_ot(
            source,
            target,
            source_weights=a,
            target_weights=b,
            epsilon=0.2,
            threshold=1e-10,
        )

        np.testing.assert_allclose(result.coupling.sum(axis=1), a / a.sum(), atol=1e-8)
        np.testing.assert_allclose(result.coupling.sum(axis=0), b / b.sum(), atol=1e-8)
        np.testing.assert_allclose(result.transition_probability.sum(axis=1), 1.0, atol=1e-12)

    def test_zero_weight_cells_are_supported(self):
        source = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 0.0]])
        target = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = _solve_traditional_ot(
            source,
            target,
            source_weights=np.array([1.0, 0.0, 1.0]),
            target_weights=np.array([1.0, 1.0]),
            epsilon=0.2,
            threshold=1e-10,
        )

        np.testing.assert_allclose(result.coupling[1], 0.0)
        np.testing.assert_allclose(result.transition_probability[1], 0.0)
        np.testing.assert_allclose(result.coupling.sum(axis=0), 0.5, atol=1e-8)

    def test_mean_cost_scaling(self):
        source = np.array([[0.0, 2.0]])
        target = np.array([[1.0, 3.0]])
        result = _solve_traditional_ot(source, target, epsilon=0.2)
        np.testing.assert_allclose(
            result.scaled_cost_matrix,
            result.cost_matrix / result.cost_matrix.mean(),
        )

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "feature dimension"):
            traditional_method(np.zeros((2, 3)), np.zeros((4, 2)))
        with self.assertRaisesRegex(ValueError, "epsilon"):
            traditional_method(np.zeros((2, 2)), np.ones((2, 2)), epsilon=0.0)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            traditional_method(np.zeros((2, 2)), np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "numeric 2D"):
            traditional_method([["not-a-number"]], [[1.0]])
        with self.assertRaisesRegex(ValueError, "scale_cost"):
            traditional_method(
                np.zeros((2, 2)), np.ones((2, 2)), scale_cost=None
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            traditional_method(
                np.zeros((2, 2)), np.ones((2, 2)), max_iterations=1.5
            )

    def test_public_function_returns_only_transition_matrix(self):
        source = np.array([[0.0, 1.0], [0.0, 1.0]])
        target = np.array([[0.1, 0.9], [0.1, 1.1]])
        transition = traditional_method(source, target)

        self.assertIsInstance(transition, np.ndarray)
        self.assertEqual(transition.shape, (2, 2))
        np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-12)

    def test_public_function_raises_when_solver_does_not_converge(self):
        source = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 0.0]])
        target = np.array([[0.2, 0.9, 2.1, 3.0], [0.1, 1.2, 0.1, 1.0]])
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            traditional_method(
                source,
                target,
                epsilon=0.1,
                threshold=1e-15,
                max_iterations=1,
            )


if __name__ == "__main__":
    unittest.main()
