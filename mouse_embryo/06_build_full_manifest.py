"""Build all adjacent-stage MOSTA section pairs for E9.5--E11.5."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PATTERN = re.compile(r"^(E(?:9\.5|10\.5|11\.5))_(E\d+S\d+)\.MOSTA\.h5ad$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    by_stage: dict[str, list[tuple[str, Path]]] = {stage: [] for stage in ("E9.5", "E10.5", "E11.5")}
    for path in sorted(args.data_root.glob("*.MOSTA.h5ad")):
        match = PATTERN.match(path.name)
        if match:
            stage, section = match.groups()
            by_stage[stage].append((f"{stage}_{section}", path.resolve()))
    missing = [stage for stage, values in by_stage.items() if not values]
    if missing:
        raise FileNotFoundError(f"No MOSTA sections found for: {', '.join(missing)}")
    rows = []
    for source_stage, target_stage in (("E9.5", "E10.5"), ("E10.5", "E11.5")):
        for source_sample, source_path in by_stage[source_stage]:
            for target_sample, target_path in by_stage[target_stage]:
                rows.append({
                    "task_id": len(rows), "source_stage": source_stage,
                    "target_stage": target_stage, "source_sample": source_sample,
                    "target_sample": target_sample,
                    "pair_id": f"{source_sample}_to_{target_sample}",
                    "source_h5ad": str(source_path), "target_h5ad": str(target_path),
                })
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.manifest, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Manifest: {args.manifest.resolve()} ({len(rows)} tasks)")


if __name__ == "__main__":
    main()
