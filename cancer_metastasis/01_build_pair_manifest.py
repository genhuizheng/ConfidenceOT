"""Build an audited primary-to-metastasis pair manifest from converted H5ADs."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from common import build_pair_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("converted_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-cells-per-side", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = build_pair_manifest(args.converted_root, args.minimum_cells_per_side)
    table.to_csv(args.output_dir / "pair_manifest_all.csv", index=False)
    eligible = table[table["eligible"]].copy()
    eligible.to_csv(args.output_dir / "pair_manifest_eligible.csv", index=False)
    skipped = table[~table["eligible"]].copy()
    skipped.to_csv(args.output_dir / "pair_manifest_skipped.csv", index=False)
    summary = {
        "converted_root": str(args.converted_root.resolve()),
        "discovered_pair_ids": len(table), "eligible_pairs": len(eligible),
        "skipped_pairs": len(skipped), "eligible_patients": int(eligible["patient_id"].nunique()),
        "eligible_datasets": int(eligible["dataset_id"].nunique()),
    }
    (args.output_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
