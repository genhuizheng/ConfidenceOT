from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cancer_origin_finalization",
    ROOT / "cancer_metastasis" / "08_finalize_origin_analysis.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CancerOriginFinalizationTest(unittest.TestCase):
    def write_pair(self, root: Path, pair_id: str, source: str, rejection: float) -> None:
        run = root / pair_id / "scope_malignant" / "budget_0.95"
        run.mkdir(parents=True)
        common = {
            "pair_id": pair_id,
            "dataset_id": "dataset_a",
            "patient_id": "patient_a",
            "source_sample": source,
            "target_sample": "metastasis",
            "analysis_scope": "malignant",
            "rejection_budget_cap": 0.95,
            "rejection_cost": 0.5,
            "source_raw_rejection_rate": 0.2,
            "target_raw_rejection_rate": rejection,
            "source_final_rejection_rate": 0.2,
            "target_final_rejection_rate": rejection,
            "source_budget_override_rate": 0.0,
            "target_budget_override_rate": 0.0,
            "transported_mass": 0.8,
            "fit_seconds": 1.0,
            "calibration_seconds_shared": 2.0,
            "pipeline_seconds_shared": 4.0,
            "calibration_valid_for_m4r": False,
            "inner_converged": True,
            "outer_converged": True,
            "cycle_detected": False,
        }
        pd.DataFrame([
            {**common, "method": "M4-E"},
            {**common, "method": "M4-R", "outer_converged": False, "cycle_detected": True},
        ]).to_csv(run / "pair_metrics.csv", index=False)
        pd.DataFrame([
            {
                "method": method,
                "side": side,
                "annotation": "Tumor",
                "n": 10,
                "raw_rejection_rate": rejection,
                "final_rejection_rate": rejection,
                "budget_override_rate": 0.0,
                "mean_confidence_score": rejection,
            }
            for method in ("M4-E", "M4-R") for side in ("source", "target")
        ]).to_csv(run / "population_rejection.csv", index=False)
        pd.DataFrame([{
            "method": "M4-E", "source_annotation": "Tumor",
            "target_annotation": "Tumor", "transported_mass": 0.8,
            "source_conditional_probability": 1.0,
        }]).to_csv(run / "population_transitions.csv", index=False)
        pd.DataFrame([{"method": "M4-E", "side": "source"}]).to_csv(
            run / "cell_confidence.csv", index=False
        )
        (run / "run.json").write_text("{}", encoding="utf-8")
        (run / "SUCCESS").write_text("complete\n", encoding="utf-8")
        (run / "calibration.json").write_text(json.dumps({
            "rejection_cost": 0.5,
            "selection_status": "largest_jointly_feasible",
            "source_monotone": True,
            "target_monotone": True,
            "calibration_valid": False,
            "validation_aggregate_valid": True,
            "validation_source_raw_acceptance": 0.05,
            "validation_target_raw_acceptance": 0.05,
            "warning_messages": [
                "At least one held-out M4-R validation fit ended with a terminal warning."
            ],
            "validation": [{
                "inner_converged": True,
                "outer_converged": False,
                "cycle_detected": True,
            }],
        }), encoding="utf-8")

    def test_complete_m4e_primary_and_m4r_diagnostic_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            self.write_pair(results, "pair_left", "left", 0.1)
            self.write_pair(results, "pair_right", "right", 0.4)
            manifest = root / "manifest.csv"
            pd.DataFrame({"pair_id": ["pair_left", "pair_right"]}).to_csv(
                manifest, index=False
            )
            output = root / "analysis"
            summary = MODULE.finalize_origin_analysis(manifest, results, output)
            self.assertEqual(summary["complete_pair_n"], 2)
            self.assertEqual(summary["multi_primary_group_n"], 1)
            self.assertEqual(summary["m4e_valid_origin_group_n"], 1)
            self.assertEqual(summary["m4r_usable_candidate_n"], 0)
            winner = pd.read_csv(output / "m4e_origin_group_winners.csv").iloc[0]
            self.assertEqual(winner["source_sample"], "left")
            self.assertTrue(bool(winner["inference_valid"]))
            self.assertAlmostEqual(
                winner["margin_to_second_target_rejection_rate"], 0.3
            )
            self.assertAlmostEqual(
                winner["margin_to_second_target_rejection_score"], 0.3
            )

    def test_incomplete_pair_stops_finalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            pd.DataFrame({"pair_id": ["missing_pair"]}).to_csv(manifest, index=False)
            with self.assertRaises(RuntimeError):
                MODULE.finalize_origin_analysis(manifest, root / "results", root / "analysis")


if __name__ == "__main__":
    unittest.main()
