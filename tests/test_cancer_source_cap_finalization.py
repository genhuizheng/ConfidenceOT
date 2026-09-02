from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cancer_source_cap_finalization",
    ROOT / "cancer_metastasis" / "10_finalize_source_cap_sensitivity.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CancerSourceCapFinalizationTest(unittest.TestCase):
    def test_anatomical_resolution(self):
        self.assertEqual(MODULE.collapse_laterality("right_adnexa"), "adnexa")
        self.assertEqual(MODULE.collapse_laterality("left-adnexa"), "adnexa")
        self.assertEqual(
            MODULE.anatomical_compartment("left_fallopian_tube"),
            "tubo_ovarian_compartment",
        )
        self.assertEqual(
            MODULE.anatomical_compartment("right_ovary"),
            "tubo_ovarian_compartment",
        )
        self.assertEqual(MODULE.anatomical_compartment("bowel"), "bowel")

    def test_laterality_flip_is_not_compartment_flip(self):
        winners = pd.DataFrame([
            {
                "run_label": "baseline", "dataset_id": "d", "patient_id": "p",
                "target_sample": "omentum", "analysis_scope": "malignant",
                "source_sample": "right_adnexa", "pair_id": "right",
                "target_final_rejection_rate": 0.4,
                "target_mean_rejection_score": 0.4,
                "source_final_rejection_rate": 0.8, "transported_mass": 0.5,
                "second_source_sample": "left_adnexa",
                "second_target_rejection_rate": 0.5,
                "second_target_rejection_score": 0.5,
                "margin_to_second_target_rejection_rate": 0.1,
                "margin_to_second_target_rejection_score": 0.1,
                "winner_laterality_collapsed": "adnexa",
                "winner_anatomical_compartment": "tubo_ovarian_compartment",
            },
            {
                "run_label": "source090", "dataset_id": "d", "patient_id": "p",
                "target_sample": "omentum", "analysis_scope": "malignant",
                "source_sample": "left_adnexa", "pair_id": "left",
                "target_final_rejection_rate": 0.39,
                "target_mean_rejection_score": 0.39,
                "source_final_rejection_rate": 0.75, "transported_mass": 0.5,
                "second_source_sample": "right_adnexa",
                "second_target_rejection_rate": 0.41,
                "second_target_rejection_score": 0.41,
                "margin_to_second_target_rejection_rate": 0.02,
                "margin_to_second_target_rejection_score": 0.02,
                "winner_laterality_collapsed": "adnexa",
                "winner_anatomical_compartment": "tubo_ovarian_compartment",
            },
        ])
        result = MODULE.winner_stability(winners, "baseline")
        changed = result[result["run_label"].eq("source090")].iloc[0]
        self.assertFalse(bool(changed["exact_winner_matches_baseline"]))
        self.assertTrue(bool(changed["laterality_collapsed_matches_baseline"]))
        self.assertTrue(bool(changed["anatomical_compartment_matches_baseline"]))
        robustness = MODULE.origin_group_robustness(
            result, "baseline", "source090"
        ).iloc[0]
        self.assertEqual(
            robustness["interpretation"],
            "laterality_ambiguous_but_site_robust",
        )
        self.assertFalse(bool(robustness["recommended_range_exact_robust"]))
        self.assertTrue(bool(robustness["recommended_range_laterality_robust"]))


if __name__ == "__main__":
    unittest.main()
