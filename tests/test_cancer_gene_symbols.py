"""Pin common.gene_symbols against the copy in the four-state pseudobulk script,
and against gene_keys, which is the function it exists to not be.

The first proliferation score matched zero genes because it reached for
gene_keys, which prefers gene_id and so returns Ensembl identifiers on these
objects. Nothing failed: the score came back NaN for every patient and every
state, and the cause was three layers down. These tests make the distinction
explicit so the next caller picks by intent rather than by which name came to
mind.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "cancer_metastasis"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from common import gene_keys, gene_symbols  # noqa: E402


def _four_state():
    path = REPO / "cancer_metastasis" / "21_prepare_four_state_malignant_pseudobulk.py"
    spec = importlib.util.spec_from_file_location("four_state", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Var:
    """The smallest thing both functions accept: var_names plus a var frame."""

    def __init__(self, names, **columns):
        self.var_names = pd.Index([str(value) for value in names])
        self.var = pd.DataFrame(columns, index=self.var_names)


class GeneSymbolsMatchesTwentyOne(unittest.TestCase):
    def setUp(self) -> None:
        self.four_state = _four_state()

    def _both(self, data):
        return gene_symbols(data), self.four_state.gene_symbols(data)

    def test_plain_var_names(self) -> None:
        data = _Var(["TP53", "MKI67"])
        mine, theirs = self._both(data)
        np.testing.assert_array_equal(mine, theirs)
        np.testing.assert_array_equal(mine, ["TP53", "MKI67"])

    def test_gene_symbol_column_wins(self) -> None:
        data = _Var(["ENSG1", "ENSG2"], gene_symbol=["TP53", "MKI67"])
        mine, theirs = self._both(data)
        np.testing.assert_array_equal(mine, theirs)
        np.testing.assert_array_equal(mine, ["TP53", "MKI67"])

    def test_placeholder_symbols_fall_back_to_var_names(self) -> None:
        data = _Var(["ENSG1", "ENSG2", "ENSG3"],
                    gene_symbol=["TP53", "NA", ""])
        mine, theirs = self._both(data)
        np.testing.assert_array_equal(mine, theirs)
        np.testing.assert_array_equal(mine, ["TP53", "ENSG2", "ENSG3"])

    def test_whitespace_is_stripped(self) -> None:
        data = _Var(["a", "b"], gene_symbol=["  TP53 ", "MKI67"])
        mine, theirs = self._both(data)
        np.testing.assert_array_equal(mine, theirs)
        np.testing.assert_array_equal(mine, ["TP53", "MKI67"])


class SymbolsAreNotKeys(unittest.TestCase):
    """gene_keys stops at gene_id. That is the whole bug, in one line.

    common.gene_keys iterates ("gene_id", "gene_symbol") and breaks after
    gene_id, so on an object carrying both it never reads the symbol column.
    It is the right key for joining a representation to a manifest and the
    wrong one for naming genes, and reaching for it to score a marker set
    matches nothing while raising nothing.
    """

    def test_gene_id_wins_outright_even_when_symbols_are_present(self) -> None:
        data = _Var(["TP53", "MKI67"],
                    gene_id=["ENSG00000141510", "ENSG00000148773"],
                    gene_symbol=["TP53", "MKI67"])
        np.testing.assert_array_equal(
            gene_keys(data), ["ENSG00000141510", "ENSG00000148773"])
        np.testing.assert_array_equal(gene_symbols(data), ["TP53", "MKI67"])

    def test_a_marker_set_matches_nothing_against_keys(self) -> None:
        data = _Var(["TP53", "MKI67"],
                    gene_id=["ENSG00000141510", "ENSG00000148773"],
                    gene_symbol=["TP53", "MKI67"])
        markers = {"TP53", "MKI67"}
        self.assertEqual(len(markers & set(gene_keys(data))), 0)
        self.assertEqual(len(markers & set(gene_symbols(data))), 2)

    def test_without_gene_id_the_two_agree(self) -> None:
        data = _Var(["ENSG1", "ENSG2"], gene_symbol=["TP53", "MKI67"])
        np.testing.assert_array_equal(gene_keys(data), gene_symbols(data))


if __name__ == "__main__":
    unittest.main()
