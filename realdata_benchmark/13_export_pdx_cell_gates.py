"""Export every LTL331 benchmark fit's per-cell gates in one compressed table, for the local biological readout (runs on
TACC; reads each row's cell_confidence.csv).

    python realdata_benchmark/13_export_pdx_cell_gates.py --root ROOT

For every row of ROOT/worklist_ltl331.tsv with SUCCESS: the fit's cell_confidence.csv (05_ writes it under
ROOT/ltl331/<config>/pair<i>/fits/pair<i>/<config>/out/<pair>/scope_malignant/budget_*/), M4-E and M4-R rows, columns
config_id, pair_index, pair_id, method, side, observation_id, rejected, signed_rejection_margin. Writes
ROOT/PDX_CELL_GATES.tsv.gz and ROOT/PDX_CELL_GATES_STATUS.tsv (per row: found or why not). Refuses if the table exists.
The cell annotations are joined locally (they are in the converted h5ad obs already downloaded).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

COLS = ["method", "side", "observation_id", "rejected", "signed_rejection_margin"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--worklist", default="worklist_ltl331.tsv", help="another LTL331 worklist writes PDX_CELL_GATES_<stem>.tsv.gz")
    a = p.parse_args()
    tag = "" if a.worklist == "worklist_ltl331.tsv" else "_" + Path(a.worklist).stem
    target = a.root / f"PDX_CELL_GATES{tag}.tsv.gz"
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    w = pd.read_csv(a.root / a.worklist, sep="\t", dtype=str)
    parts, status = [], []
    for _, r in w.iterrows():
        out = Path(r.out)
        files = sorted(out.glob("fits/pair*/*/out/*/scope_malignant/budget_*/cell_confidence.csv"))
        st = {"config_id": r.config_id, "pair_index": r.pair_index, "success": (out / "SUCCESS").exists(), "files": len(files)}
        if len(files) == 1:
            c = pd.read_csv(files[0], usecols=COLS)
            c.insert(0, "pair_id", r.pair_id); c.insert(0, "pair_index", int(r.pair_index)); c.insert(0, "config_id", r.config_id)
            parts.append(c)
        status.append(st)
    pd.concat(parts, ignore_index=True).to_csv(target, sep="\t", index=False, compression="gzip")
    s = pd.DataFrame(status); s.to_csv(a.root / f"PDX_CELL_GATES_STATUS{tag}.tsv", sep="\t", index=False)
    print(s.groupby(["success", "files"]).size().to_string())
    print(f"{sum(len(x) for x in parts)} rows -> {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
