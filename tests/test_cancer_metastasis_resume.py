from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class CancerResumeTest(unittest.TestCase):
    def test_skip_completed_returns_before_loading_h5ad(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            output = root / "results"
            pair_id = "patient_a__primary__metastasis"
            pd.DataFrame([{
                "pair_id": pair_id,
                "dataset_id": "dataset_a",
                "patient_id": "patient_a",
                "source_sample": "primary",
                "target_sample": "metastasis",
                "source_h5ad": str(root / "intentionally_missing_source.h5ad"),
                "target_h5ad": str(root / "intentionally_missing_target.h5ad"),
                "eligible": True,
            }]).to_csv(manifest, index=False)
            completed = output / pair_id / "scope_all" / "budget_0.95"
            completed.mkdir(parents=True)
            (completed / "SUCCESS").write_text("complete\n", encoding="utf-8")

            script = Path(__file__).parents[1] / "cancer_metastasis" / "02_run_pair.py"
            environment = os.environ.copy()
            source_root = str(Path(__file__).parents[1] / "src")
            environment["PYTHONPATH"] = os.pathsep.join(
                value for value in (source_root, environment.get("PYTHONPATH", "")) if value
            )
            process = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(manifest),
                    str(output),
                    "--index", "0",
                    "--skip-completed",
                ],
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("SKIP completed index=0", process.stdout)

    def test_asymmetric_budget_uses_unambiguous_output_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            output = root / "results"
            pair_id = "patient_a__primary__metastasis"
            pd.DataFrame([{
                "pair_id": pair_id,
                "dataset_id": "dataset_a",
                "patient_id": "patient_a",
                "source_sample": "primary",
                "target_sample": "metastasis",
                "source_h5ad": str(root / "missing_source.h5ad"),
                "target_h5ad": str(root / "missing_target.h5ad"),
                "eligible": True,
            }]).to_csv(manifest, index=False)
            completed = (
                output / pair_id / "scope_all"
                / "budget_source_0.90_target_0.95"
            )
            completed.mkdir(parents=True)
            (completed / "SUCCESS").write_text("complete\n", encoding="utf-8")
            script = Path(__file__).parents[1] / "cancer_metastasis" / "02_run_pair.py"
            environment = os.environ.copy()
            source_root = str(Path(__file__).parents[1] / "src")
            environment["PYTHONPATH"] = os.pathsep.join(
                value for value in (source_root, environment.get("PYTHONPATH", "")) if value
            )
            process = subprocess.run([
                sys.executable, str(script), str(manifest), str(output),
                "--index", "0", "--source-rejection-budget", "0.90",
                "--target-rejection-budget", "0.95", "--skip-completed",
            ], capture_output=True, text=True, env=environment)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("budget_source_0.90_target_0.95", process.stdout)


if __name__ == "__main__":
    unittest.main()
