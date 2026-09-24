"""Pin common.normalize_expression against the copy in 11_run_robust_target_deg.py.

The function was moved into ``common`` so that method B's per-patient
differential expression and the target-side analysis cannot drift apart.
``11_`` keeps its own copy, because its results are complete and reproducing
them must not depend on an edit to a shared module -- which leaves exactly the
duplication this project has already been bitten by once, when the depth screen
spent a week measuring a configuration the production runner assembled
separately by hand.

This test is what makes the duplication safe: if either copy changes, it fails.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "cancer_metastasis"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from common import normalize_expression  # noqa: E402


def _eleven():
    """Import the numbered script, whose name is not a valid module name."""
    path = REPO / "cancer_metastasis" / "11_run_robust_target_deg.py"
    spec = importlib.util.spec_from_file_location("robust_target_deg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NormalizeExpressionMatchesEleven(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        dense = rng.poisson(1.4, size=(30, 50)).astype(np.float64)
        # No zero-library row: the raw-counts branch rejects those on purpose,
        # and that rejection is covered by its own test below.
        dense[dense.sum(axis=1) == 0, 0] = 1.0
        self.matrix = sparse.csr_matrix(dense)
        self.eleven = _eleven()

    def test_every_branch_agrees(self) -> None:
        for kind in ("raw counts", "Raw Counts", "normalized",
                     "log-normalized", "log normalized"):
            with self.subTest(kind=kind):
                mine, my_label = normalize_expression(
                    self.matrix, kind, 10_000.0)
                theirs, their_label = self.eleven.normalize_expression(
                    sparse.csr_matrix(self.matrix), kind, 10_000.0)
                self.assertEqual(my_label, their_label)
                np.testing.assert_allclose(
                    mine.toarray(), theirs.toarray(), rtol=0, atol=0)

    def test_target_sum_is_honoured(self) -> None:
        matrix, label = normalize_expression(self.matrix, "raw counts", 5_000.0)
        self.assertIn("5000", label)
        # log1p is applied after scaling, so undo it before checking the total.
        restored = matrix.copy()
        restored.data = np.expm1(restored.data)
        np.testing.assert_allclose(
            np.asarray(restored.sum(axis=1)).ravel(),
            np.full(matrix.shape[0], 5_000.0), rtol=1e-9)

    def test_zero_library_is_refused_rather_than_divided_by(self) -> None:
        dense = np.zeros((3, 4))
        dense[0, 0] = 2.0
        with self.assertRaises(ValueError):
            normalize_expression(sparse.csr_matrix(dense), "raw counts", 10_000.0)

    def test_a_negative_stored_matrix_is_left_alone(self) -> None:
        # Pearson residuals are stored normalised and go negative; taking log1p
        # of them would produce NaN, so that branch must pass them through.
        dense = np.array([[-1.5, 0.2], [0.4, -0.3]])
        matrix, label = normalize_expression(
            sparse.csr_matrix(dense), "normalized", 10_000.0)
        self.assertEqual(label, "stored_normalized_expression_used_as_provided")
        np.testing.assert_allclose(matrix.toarray(), dense)


if __name__ == "__main__":
    unittest.main()
