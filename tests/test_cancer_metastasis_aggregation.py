from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "cancer_metastasis" / "04_aggregate_results.py"
SPEC = importlib.util.spec_from_file_location("cancer_aggregate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CancerAggregationTest(unittest.TestCase):
    def test_methods_remain_separate_and_shared_time_is_not_doubled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "results" / "pair_a" / "scope_malignant" / "budget_0.95"
            run.mkdir(parents=True)

            common = {
                "pair_id": "pair_a",
                "dataset_id": "dataset_a",
                "patient_id": "patient_a",
                "analysis_scope": "malignant",
                "rejection_budget_cap": 0.95,
                "rejection_cost": 0.4,
                "source_raw_rejection_rate": 0.2,
                "target_raw_rejection_rate": 0.3,
                "source_final_rejection_rate": 0.2,
                "target_final_rejection_rate": 0.3,
                "source_budget_override_rate": 0.0,
                "target_budget_override_rate": 0.0,
                "transported_mass": 0.8,
                "calibration_seconds_shared": 12.0,
                "pipeline_seconds_shared": 20.0,
                "inner_converged": True,
                "calibration_valid_for_m4r": False,
            }
            pd.DataFrame([
                {**common, "method": "M4-E", "fit_seconds": 1.0,
                 "outer_converged": True, "cycle_detected": False},
                {**common, "method": "M4-R", "fit_seconds": 2.0,
                 "outer_converged": False, "cycle_detected": True},
            ]).to_csv(run / "pair_metrics.csv", index=False)
            pd.DataFrame([
                {"method": "M4-E", "side": "source", "annotation": "Tumor",
                 "n": 10, "raw_rejection_rate": 0.2,
                 "final_rejection_rate": 0.2, "budget_override_rate": 0.0,
                 "mean_confidence_score": 0.6},
                {"method": "M4-R", "side": "source", "annotation": "Tumor",
                 "n": 10, "raw_rejection_rate": 0.3,
                 "final_rejection_rate": 0.3, "budget_override_rate": 0.0,
                 "mean_confidence_score": 0.5},
            ]).to_csv(run / "population_rejection.csv", index=False)

            output = root / "aggregate"
            MODULE.aggregate_results(root / "results", output)

            patient = pd.read_csv(output / "patient_level_metrics.csv")
            self.assertEqual(set(patient["method"]), {"M4-E", "M4-R"})
            self.assertTrue((patient["source_rejection_budget_cap"] == 0.95).all())
            self.assertTrue((patient["target_rejection_budget_cap"] == 0.95).all())

            timing = pd.read_csv(output / "dataset_run_timing.csv")
            self.assertEqual(int(timing.loc[0, "run_n"]), 1)
            self.assertEqual(float(timing.loc[0, "total_calibration_seconds"]), 12.0)
            self.assertEqual(float(timing.loc[0, "total_pipeline_seconds"]), 20.0)

            diagnostics = pd.read_csv(output / "method_terminal_diagnostics.csv")
            exact = diagnostics.loc[diagnostics["method"].eq("M4-E")].iloc[0]
            reversible = diagnostics.loc[diagnostics["method"].eq("M4-R")].iloc[0]
            self.assertEqual(float(exact["cycle_detected_rate"]), 0.0)
            self.assertEqual(float(reversible["cycle_detected_rate"]), 1.0)
            self.assertEqual(float(reversible["outer_converged_rate"]), 0.0)

            population = pd.read_csv(output / "all_population_rejection.csv")
            self.assertEqual(set(population["method"]), {"M4-E", "M4-R"})


if __name__ == "__main__":
    unittest.main()
