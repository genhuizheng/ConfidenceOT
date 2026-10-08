"""List every MOSTA slice and every pair the all-stage run analyses.

Pairs are every source x target slice combination between consecutive stages
(E9.5 -> E10.5, ..., E15.5 -> E16.5) and every unordered pair of slices within
one stage, each once, the slice that sorts first as source.  Nothing is
selected: a slice file that is present is in the run.  The expected counts
from the 2026-10-08 listing (53 slices, 237 adjacent, 227 same-stage) are
printed beside the found ones as a check, never used to drop anything.

Pairs get one global index, adjacent first.  The index is the seed offset of
cancer_metastasis/02_run_pair.py's seed rule, so it must stay fixed once the
run has started; rebuilding the manifest from the same files reproduces it.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosta_common import STAGES, Layout  # noqa: E402

PATTERN = re.compile(r"^(?P<stage>E\d+(?:\.\d+)?)_(?P<section>E\d+S\d+)\.MOSTA\.h5ad$")
EXPECTED = {"slices": 53, "adjacent": 237, "same_stage": 227}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    layout = Layout(args.root)
    slices = []
    unrecognised = []
    for path in sorted(args.data_root.glob("*.MOSTA.h5ad")):
        match = PATTERN.match(path.name)
        if match is None or match.group("stage") not in STAGES:
            unrecognised.append(path.name)
            continue
        slices.append({
            "sample": f"{match.group('stage')}_{match.group('section')}",
            "stage": match.group("stage"),
            "section": match.group("section"),
            "path": str(path.resolve()),
            "file_bytes": path.stat().st_size,
        })
    if unrecognised:
        print(f"files not matching <stage>_<E#S#>.MOSTA.h5ad for a stage in {STAGES}: {unrecognised}")
    table = pd.DataFrame(slices)
    table["stage_order"] = table["stage"].map({stage: i for i, stage in enumerate(STAGES)})
    table = table.sort_values(["stage_order", "sample"]).drop(columns="stage_order").reset_index(drop=True)
    by_stage = {stage: table.loc[table["stage"] == stage, "sample"].tolist() for stage in STAGES}
    rows = []
    for source_stage, target_stage in zip(STAGES[:-1], STAGES[1:]):
        for source, target in itertools.product(by_stage[source_stage], by_stage[target_stage]):
            rows.append({"kind": "adjacent", "group": f"{source_stage}->{target_stage}",
                         "source_stage": source_stage, "target_stage": target_stage,
                         "source_sample": source, "target_sample": target})
    for stage in STAGES:
        for source, target in itertools.combinations(by_stage[stage], 2):
            rows.append({"kind": "same_stage", "group": stage,
                         "source_stage": stage, "target_stage": stage,
                         "source_sample": source, "target_sample": target})
    pairs = pd.DataFrame(rows)
    pairs.insert(0, "index", range(len(pairs)))
    pairs.insert(1, "pair_id", pairs["source_sample"] + "__" + pairs["target_sample"])
    layout.manifest.mkdir(parents=True, exist_ok=True)
    table.to_csv(layout.slices_csv, index=False)
    pairs.to_csv(layout.pairs_csv, index=False)
    found = {"slices": len(table), "adjacent": int((pairs["kind"] == "adjacent").sum()),
             "same_stage": int((pairs["kind"] == "same_stage").sum())}
    print(table.groupby("stage", sort=False).size().rename("slices").to_string())
    print(pairs.groupby(["kind", "group"], sort=False).size().rename("pairs").to_string())
    for key, value in found.items():
        note = "" if value == EXPECTED[key] else f"   <-- the 2026-10-08 listing had {EXPECTED[key]}"
        print(f"{key}: {value}{note}")
    print(f"wrote {layout.slices_csv} and {layout.pairs_csv}")


if __name__ == "__main__":
    main()
