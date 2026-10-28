"""Read-only: can a finished MOSTA run's OT be rerun from its saved representations alone?

Before the OT of a finished run is repeated on its representations (to save
the couplings, submit_mosta_all_stages.sh with MOSTA_SHARED_REPRESENTATIONS),
this checks, for every pair of manifest/pairs.csv, that everything the rerun
reads is there and agrees with what the finished run recorded:

* representations/<pair_id>/REPRESENTATION_SUCCESS, representation.npz with
  float32 ``source`` and ``target`` of 30 columns, finite, and
  representation.json with the expected label, its seed and bin counts equal
  to the arrays' rows;
* the slice tables give the same number of analysed bins (the rows 23_ joins
  on);
* the finished pair's run.json names the same label, bin counts and
  representation seed, and records the cost scale, rejection cost, epsilon,
  lambda and tolerance; calibration.json is readable.

It also lists the code commits the finished pairs recorded and, for each, the
files of the solver path (src/confidenceot except preprocessing.py, and the
MOSTA runner) that differ from the checked-out commit; a non-empty list means
the rerun does not run the same solver code.

Prints a summary and writes nothing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosta_common import Layout, read_pairs  # noqa: E402

SOLVER_PATH = ("src/confidenceot/api.py", "src/confidenceot/cuda.py", "src/confidenceot/blockwise.py",
               "src/confidenceot/calibration.py", "src/confidenceot/_cpu_uot.py", "src/confidenceot/_cpu_balanced.py",
               "src/confidenceot/result.py", "mouse_embryo/mosta_common.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--label", default="ranknm256_noscale_ds_cos")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    layout = Layout(args.root)
    pairs = read_pairs(layout)
    analysed = {}
    problems: list[str] = []
    commits: Counter = Counter()
    recorded = Counter()
    for _, row in pairs.iterrows():
        pair_id = row["pair_id"]
        directory = layout.representation(pair_id)
        if not (directory / "REPRESENTATION_SUCCESS").is_file():
            problems.append(f"{pair_id}: no REPRESENTATION_SUCCESS")
            continue
        try:
            record = json.loads((directory / "representation.json").read_text(encoding="utf-8"))
            with np.load(directory / "representation.npz") as stored:
                source, target = stored["source"], stored["target"]
            for side, array in (("source", source), ("target", target)):
                if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 30 or not np.isfinite(array).all():
                    problems.append(f"{pair_id}: {side} is {array.dtype} {array.shape}, finite {bool(np.isfinite(array).all())}")
            if record["preprocessing"]["label"] != args.label:
                problems.append(f"{pair_id}: representation label {record['preprocessing']['label']}")
            if (record["source_bins"], record["target_bins"]) != (len(source), len(target)):
                problems.append(f"{pair_id}: representation.json bins {record['source_bins']}x{record['target_bins']} "
                                f"vs arrays {len(source)}x{len(target)}")
            for side, sample, count in (("source", row["source_sample"], len(source)),
                                        ("target", row["target_sample"], len(target))):
                if sample not in analysed:
                    table = pd.read_csv(layout.slice_bins(sample), usecols=["equalised_row"])
                    analysed[sample] = int((table["equalised_row"] >= 0).sum())
                if analysed[sample] != count:
                    problems.append(f"{pair_id}: slices/{sample} has {analysed[sample]} analysed bins, {side} has {count}")
        except Exception as error:  # reported, never repaired
            problems.append(f"{pair_id}: representation unreadable: {error}")
            continue
        pair = layout.pair(pair_id)
        if not (pair / "SUCCESS").is_file():
            problems.append(f"{pair_id}: the finished run has no SUCCESS")
            continue
        try:
            run = json.loads((pair / "run.json").read_text(encoding="utf-8"))
            json.loads((pair / "calibration.json").read_text(encoding="utf-8"))
            expected = {"preprocessing_label": args.label, "source_bins": len(source), "target_bins": len(target),
                        "representation_seed": record["seed"]}
            for key, value in expected.items():
                if run.get(key) != value:
                    problems.append(f"{pair_id}: run.json {key}={run.get(key)!r}, representation says {value!r}")
            for key in ("cost_scale", "rejection_cost", "epsilon", "lambda", "tolerance"):
                if not isinstance(run.get(key), (int, float)):
                    problems.append(f"{pair_id}: run.json has no {key}")
                else:
                    recorded[key] += 1
            commits[run.get("code_commit", "missing")] += 1
            if run.get("coupling_format"):
                problems.append(f"{pair_id}: already has saved couplings ({run['coupling_format']})")
        except Exception as error:
            problems.append(f"{pair_id}: finished pair unreadable: {error}")
    print(f"root {layout.root}; pairs {len(pairs)}; problems {len(problems)}")
    for line in problems[:50]:
        print("  " + line)
    if len(problems) > 50:
        print(f"  ... {len(problems) - 50} more")
    print("run.json fields recorded:", dict(recorded))
    print("code commits of the finished pairs:", dict(commits))
    head = subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for commit in commits:
        changed = subprocess.run(["git", "-C", str(args.repo), "diff", "--name-only", commit, head, "--", *SOLVER_PATH],
                                 capture_output=True, text=True)
        files = changed.stdout.split() if changed.returncode == 0 else [f"git diff failed: {changed.stderr.strip()}"]
        print(f"solver-path files changed from {commit[:12]} to {head[:12]}: {files if files else 'none'}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
