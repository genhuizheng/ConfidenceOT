from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AUDIT = load_script(
    "cancer_calibration_audit",
    "cancer_metastasis/05_audit_calibration_certificates.py",
)
RANK = load_script(
    "cancer_origin_rank",
    "cancer_metastasis/06_rank_primary_origins.py",
)


class CancerOriginPostprocessingTest(unittest.TestCase):
    def test_calibration_certificates_and_origin_ranking_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            for pair_id, source, target_rejection, target_score in (
                ("pair_left", "left", 0.10, 0.20),
                ("pair_right", "right", 0.40, 0.60),
            ):
                run = results / pair_id / "scope_malignant" / "budget_0.95"
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
                    "source_final_rejection_rate": 0.2,
                    "target_final_rejection_rate": target_rejection,
                    "transported_mass": 0.8,
                    "outer_converged": True,
                    "cycle_detected": False,
                }
                pd.DataFrame([
                    {**common, "method": "M4-E"},
                    {**common, "method": "M4-R"},
                ]).to_csv(run / "pair_metrics.csv", index=False)
                pd.DataFrame([
                    {"method": method, "side": side, "annotation": "Tumor", "n": 10,
                     "mean_confidence_score": target_score if side == "target" else 0.3}
                    for method in ("M4-E", "M4-R") for side in ("source", "target")
                ]).to_csv(run / "population_rejection.csv", index=False)
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

            audit_out = root / "audit"
            certificates = AUDIT.audit_calibrations(results, audit_out)
            self.assertTrue(certificates["m4e_cost_selection_valid"].all())
            self.assertTrue(certificates["m4r_rate_validation_valid"].all())
            self.assertFalse(certificates["m4r_terminal_validation_valid"].any())
            self.assertFalse(certificates["m4r_deployment_valid"].any())

            rank_out = root / "rank"
            ranking = RANK.rank_primary_origins(results, rank_out)
            winner = ranking[
                ranking["method"].eq("M4-E") & ranking["primary_rank"].eq(1)
            ].iloc[0]
            self.assertEqual(winner["source_sample"], "left")
            self.assertEqual(winner["interpretation"],
                             "candidate malignant-cell origin compatibility")
            agreement = pd.read_csv(rank_out / "origin_method_agreement.csv")
            self.assertTrue(bool(agreement.loc[0, "m4e_m4r_top_source_agree"]))


if __name__ == "__main__":
    unittest.main()
