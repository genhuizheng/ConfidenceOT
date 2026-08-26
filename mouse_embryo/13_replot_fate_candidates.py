"""Rebuild the developmental-candidate figure from exported summary tables."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-labels", type=int, default=14)
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("mouse_embryo.08_visualize_biological_discovery")
    composition = pd.read_csv(args.input_dir / "annotation_abundance_by_stage.csv")
    rejection = pd.read_csv(args.input_dir / "population_rejection_reproducibility.csv")
    candidates = module.build_fate_candidates(composition, rejection)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_zh" if args.language == "zh" else ""
    candidates.to_csv(
        args.output_dir / f"developmental_fate_candidates_corrected{suffix}.csv",
        index=False,
    )
    module.plot_fate_candidates(
        candidates,
        args.output_dir / f"developmental_fate_candidates_corrected{suffix}",
        args.dpi,
        args.max_labels,
        language=args.language,
    )
    print(f"Corrected candidate figure: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
