from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cancer_asymmetric_caps",
    ROOT / "cancer_metastasis" / "09_compare_asymmetric_caps.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CancerAsymmetricCapTest(unittest.TestCase):
    def test_winner_stability_uses_target_then_continuous_score(self):
        rows = []
        for label, source_cap, left_rate, right_rate in (
            ("baseline", 0.95, 0.2, 0.4),
            ("source090", 0.90, 0.2, 0.4),
        ):
            for source, rate, score in (
                ("left", left_rate, 0.3), ("right", right_rate, 0.5)
            ):
                rows.append({
                    "run_label": label,
                    "dataset_id": "dataset_a",
                    "patient_id": "patient_a",
                    "target_sample": "metastasis",
                    "analysis_scope": "malignant",
                    "source_sample": source,
                    "target_final_rejection_rate": rate,
                    "target_mean_rejection_score": score,
                    "source_final_rejection_rate": source_cap,
                    "transported_mass": 0.8,
                })
        result = MODULE.winner_stability(pd.DataFrame(rows), "baseline")
        changed = result[result["run_label"].eq("source090")].iloc[0]
        self.assertEqual(changed["source_sample"], "left")
        self.assertTrue(bool(changed["winner_matches_baseline"]))


if __name__ == "__main__":
    unittest.main()
