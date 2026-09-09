"""Aggregate independently completed local or SLURM ConfidenceOT cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    return parser.parse_args()


def concatenate(root: Path, filename: str, output_name: str) -> int:
    paths = sorted(path for path in root.rglob(filename) if path.parent != root)
    frames = [pd.read_csv(path) for path in paths if path.stat().st_size > 0]
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(root / output_name, index=False)
    return len(combined)


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    manifests = sorted(path for path in root.rglob("manifest.json") if path.parent != root)
    manifest_payloads = [json.loads(path.read_text(encoding="utf-8-sig")) for path in manifests]
    outputs = {
        "runs_combined.csv": concatenate(root, "runs.csv", "runs_combined.csv"),
        "calibration_diagnostics_combined.csv": concatenate(root, "calibration_diagnostics.csv", "calibration_diagnostics_combined.csv"),
        "population_rejection_rates_combined.csv": concatenate(root, "population_rejection_rates.csv", "population_rejection_rates_combined.csv"),
        "population_mass_diagnostics_combined.csv": concatenate(root, "population_mass_diagnostics.csv", "population_mass_diagnostics_combined.csv"),
        "population_transitions_combined.csv": concatenate(root, "population_transitions.csv", "population_transitions_combined.csv"),
    }
    summary = {
        "completed_cases": len(manifests),
        "total_case_cpu_seconds": float(sum(float(value.get("actual_cpu_seconds", 0.0)) for value in manifest_payloads)),
        "total_case_wall_seconds": float(sum(float(value.get("actual_wall_seconds", 0.0)) for value in manifest_payloads)),
        "rows": outputs,
    }
    (root / "aggregate_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
