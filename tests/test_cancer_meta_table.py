"""Pin common.meta_table against the copy in 12_meta_analyze_robust_target_deg.py,
and cover the design floor that makes a whole meta-analysis unreachable.

Same arrangement as ``test_cancer_normalize_expression.py``: the function moved
into ``common`` so the primary contrast's combination step and the target-side
one cannot drift, ``12_`` keeps its copy because its results are complete, and
this test is what makes the duplication safe.
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

from common import meta_table  # noqa: E402

TARGET_LABEL = "robust_rejected_minus_robust_retained_target_malignant_state"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "cancer_metastasis" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _effects(patient_n: int, gene_n: int, planted: set[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(gene_n):
        gene = f"G{index:03d}"
        centre = 1.5 if gene in planted else 0.0
        for patient in range(patient_n):
            rows.append({"gene": gene, "patient_id": f"P{patient}",
                         "log2_fold_change": float(rng.normal(centre, 0.4))})
    return pd.DataFrame(rows)


class MetaTableMatchesTwelve(unittest.TestCase):
    def test_every_column_agrees(self) -> None:
        effects = _effects(9, 40, {"G000", "G001"}, seed=12)
        mine = meta_table(effects, 6, TARGET_LABEL).reset_index(drop=True)
        theirs = _load("meta_twelve", "12_meta_analyze_robust_target_deg.py") \
            .meta_table(effects, 6).reset_index(drop=True)
        self.assertEqual(list(mine.columns), list(theirs.columns))
        for column in mine.columns:
            with self.subTest(column=column):
                if mine[column].dtype.kind in "fc":
                    np.testing.assert_allclose(
                        mine[column].to_numpy(), theirs[column].to_numpy(),
                        rtol=0, atol=0, equal_nan=True)
                else:
                    self.assertTrue(mine[column].equals(theirs[column]))

    def test_interpretation_is_a_parameter_not_a_constant(self) -> None:
        effects = _effects(7, 5, set(), seed=3)
        table = meta_table(effects, 6, "primary_rejected_minus_primary_retained")
        self.assertEqual(set(table["interpretation"]),
                         {"primary_rejected_minus_primary_retained"})
        self.assertEqual(set(table["inference_unit"]), {"patient"})


class MetaTableBehaviour(unittest.TestCase):
    def test_direction_consistency_counts_agreement_with_the_median(self) -> None:
        # Seven patients up, two down: 7/9 agree with the positive median.
        values = [2.0] * 7 + [-1.0] * 2
        effects = pd.DataFrame({"gene": ["G"] * 9,
                                "log2_fold_change": values})
        row = meta_table(effects, 6, "x").iloc[0]
        self.assertAlmostEqual(row["direction_consistency"], 7 / 9)
        self.assertEqual(row["patient_n"], 9)

    def test_a_gene_below_the_patient_floor_gets_no_p_value(self) -> None:
        effects = pd.DataFrame({"gene": ["G"] * 4,
                                "log2_fold_change": [1.0, 1.1, 0.9, 1.2]})
        row = meta_table(effects, 6, "x").iloc[0]
        self.assertTrue(np.isnan(row["patient_level_wilcoxon_p_value"]))
        self.assertEqual(row["patient_n"], 4)


class AttainableFdrFloor(unittest.TestCase):
    """The floor that turns an underpowered design into a clean-looking null.

    A two-sided exact signed-rank test over n non-zero effects cannot return a
    p below 2/2**n. Benjamini-Hochberg multiplies the smallest p by the gene
    count, so with too few patients no gene can be significant whatever the
    data does -- and the output looks like a negative result rather than an
    error.
    """

    def setUp(self) -> None:
        self.combine = _load("combine_thirtythree",
                             "33_combine_per_patient_primary_deg.py")

    def test_matches_the_closed_form(self) -> None:
        for patient_n, gene_n in ((9, 60), (20, 20_000), (29, 20_000)):
            with self.subTest(patient_n=patient_n, gene_n=gene_n):
                self.assertAlmostEqual(
                    self.combine.attainable_fdr(patient_n, gene_n),
                    gene_n * 2.0 ** (1 - patient_n))

    def test_twenty_patients_is_the_threshold_at_twenty_thousand_genes(self) -> None:
        self.assertGreater(self.combine.attainable_fdr(19, 20_000), 0.05)
        self.assertLess(self.combine.attainable_fdr(20, 20_000), 0.05)

    def test_the_floor_is_reachable_in_practice(self) -> None:
        # A perfectly consistent gene over nine patients should land exactly on
        # the closed form, which is what makes the closed form the right guard.
        effects = pd.DataFrame({"gene": ["G"] * 9,
                                "log2_fold_change": [1.0] * 9})
        row = meta_table(effects, 6, "x").iloc[0]
        self.assertAlmostEqual(row["patient_level_wilcoxon_p_value"],
                               2.0 ** (1 - 9), places=12)


if __name__ == "__main__":
    unittest.main()
