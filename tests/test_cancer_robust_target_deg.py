from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "cancer_metastasis" / file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


META = load("robust_target_meta", "12_meta_analyze_robust_target_deg.py")


class CancerRobustTargetDegTest(unittest.TestCase):
    def test_bh_adjust_is_monotone_in_rank(self):
        p = np.array([0.01, 0.04, 0.03, np.nan])
        adjusted = META.bh_adjust(p)
        self.assertTrue(np.isnan(adjusted[3]))
        self.assertTrue(np.all((adjusted[:3] >= p[:3]) & (adjusted[:3] <= 1)))
        order = np.argsort(p[:3])
        self.assertTrue(np.all(np.diff(adjusted[:3][order]) >= -1e-12))

    def test_jaccard_uses_direction_specific_gene_sets(self):
        table = pd.DataFrame({
            "group_id": ["a", "a", "b", "b"],
            "gene": ["G1", "G2", "G1", "G3"],
            "direction": ["rejected_enriched"] * 4,
        })
        result = META.jaccard_table(table, "rejected_enriched")
        value = result[
            result["left_group"].eq("a") & result["right_group"].eq("b")
        ].iloc[0]
        self.assertEqual(value["intersection_gene_n"], 1)
        self.assertEqual(value["union_gene_n"], 3)
        self.assertAlmostEqual(value["jaccard"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
