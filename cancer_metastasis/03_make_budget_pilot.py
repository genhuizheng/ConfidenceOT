"""Choose one deterministic eligible pair per dataset and cross it with budget caps."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eligible_manifest", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.15, 0.30, 0.50, 0.80, 0.95])
    args = parser.parse_args()
    manifest = pd.read_csv(args.eligible_manifest).sort_values(
        ["dataset_id", "patient_id", "pair_id"], kind="stable"
    )
    selected = manifest.groupby("dataset_id", as_index=False, sort=True).head(1)
    rows = []
    for _, pair in selected.iterrows():
        for budget in args.budgets:
            rows.append({
                "pilot_index": len(rows), "eligible_index": int(pair["eligible_index"]),
                "pair_id": pair["pair_id"], "dataset_id": pair["dataset_id"],
                "patient_id": pair["patient_id"], "rejection_budget": budget,
            })
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Pilot runs: {len(rows)}; pairs: {len(selected)}; budgets: {len(args.budgets)}")


if __name__ == "__main__":
    main()
