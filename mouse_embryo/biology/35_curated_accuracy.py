"""Level B: moscot's curated-transition accuracy (Fig. 4b) on our couplings, from the runs' own transition tables.

    python mouse_embryo/biology/35_curated_accuracy.py --out <analysis root> --run ds=<root> [--run logcpm=<root> ...]

moscot's score (mosta_st_map_accuracies.py, see literature/LITERATURE_NOTES.md
A4): the forward annotation transition matrix, each source annotation's row
normalised to 1, scored as the mean over source annotations of the row's mass
on curated edges (self edges included; Suppl. Tab. 5, 138 edges).  Computed
here from audit/annotation_transitions.csv.gz (forward direction, nominal
masses written by 23_) for every adjacent pair, for:

* bot            balanced OT, every source bin
* co             the M4-E coupling on its retained x retained support (rows of
                 annotations with no retained mass are left out and counted)
* bot_E_retained balanced OT restricted to the M4-E-retained source bins
                 (bot minus bot_co_rejected), the same bins as ``co``
* bot_E_rejected balanced OT on the M4-E-rejected source bins

co against bot_E_retained is the paired comparison on identical bins; bot
against bot_E_retained is the selection effect of the gate.  The E1S1 pairs
(the sections moscot used, inferred from bin counts) are flagged, with
moscot's reported spatiotemporal and temporal scores beside them.  The
settings differ (our UOT with calibrated rejection and expression-only cost vs
moscot's tuned low-rank fused OT, Cavity bins excluded here), so this is a
harmonised-annotation comparison, not a method benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bio_common import Outputs, curated_transitions  # noqa: E402

MOSCOT = {  # LITERATURE_NOTES.md A4 (accuracy notebook cells 23/24)
    "E9.5->E10.5": (0.3670, 0.3108), "E10.5->E11.5": (0.3922, 0.3186), "E11.5->E12.5": (0.3914, 0.3860),
    "E12.5->E13.5": (0.3911, 0.3453), "E13.5->E14.5": (0.4879, 0.4423), "E14.5->E15.5": (0.4910, 0.4797),
    "E15.5->E16.5": (0.4573, 0.3538),
}


def score(flow: pd.DataFrame, allowed: set[tuple[str, str]]) -> tuple[float, int, int]:
    """flow: annotation x partner nominal mass.  Mean over rows with mass of the allowed share."""
    totals = flow.sum(axis=1)
    rows = flow[totals > 0]
    if rows.empty:
        return np.nan, 0, len(flow)
    share = rows.div(rows.sum(axis=1), axis=0)
    mask = np.array([[(a, b) in allowed for b in share.columns] for a in share.index])
    return float((share.to_numpy() * mask).sum(axis=1).mean()), len(rows), int((totals <= 0).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="name=root")
    args = parser.parse_args()
    out = Outputs(args.out)
    curated = curated_transitions()
    if curated is None:
        raise SystemExit("no curated transition table")
    allowed = set(zip(curated["source_annotation"], curated["target_annotation"]))
    rows = []
    for item in args.run:
        name, root = item.split("=", 1)
        path = Path(root) / "audit" / "annotation_transitions.csv.gz"
        if not path.is_file():
            print(f"{name}: no {path}; left out", flush=True)
            continue
        t = pd.read_csv(path)
        t = t[(t["direction"] == "forward") & (t["kind"] == "adjacent")]
        for (pid, group), g in t.groupby(["pair_id", "group"]):
            flows = {m: g[g["method"] == m].pivot_table(index="annotation", columns="partner_annotation",
                                                        values="nominal_mass", aggfunc="sum", fill_value=0.0)
                     for m in ("bot", "co", "bot_co_rejected")}
            rejected = flows["bot_co_rejected"].reindex_like(flows["bot"]).fillna(0.0)
            variants = {"bot": flows["bot"], "co": flows["co"],
                        "bot_E_retained": (flows["bot"] - rejected).clip(lower=0.0), "bot_E_rejected": rejected}
            source, target = pid.split("__")
            e1s1 = source.endswith("_E1S1") and target.endswith("_E1S1")
            for variant, flow in variants.items():
                value, used, dropped = score(flow, allowed)
                rows.append({"run": name, "pair_id": pid, "group": group, "e1s1_moscot_sections": e1s1,
                             "variant": variant, "accuracy": value, "source_annotations_scored": used,
                             "source_annotations_without_mass": dropped,
                             "moscot_spatiotemporal": MOSCOT.get(group, (np.nan, np.nan))[0] if e1s1 else np.nan,
                             "moscot_temporal": MOSCOT.get(group, (np.nan, np.nan))[1] if e1s1 else np.nan})
    table = pd.DataFrame(rows)
    out.table(table, "analysis/literature/CURATED_ACCURACY_BY_PAIR.tsv")
    summary = (table.groupby(["run", "group", "variant"])["accuracy"]
               .agg(n_pairs="size", mean="mean", sd="std", median="median").reset_index())
    out.table(summary, "analysis/literature/CURATED_ACCURACY_SUMMARY.tsv")
    wide = table[table["e1s1_moscot_sections"]].pivot_table(
        index=["run", "group", "moscot_spatiotemporal", "moscot_temporal"], columns="variant", values="accuracy").reset_index()
    out.table(wide, "analysis/literature/CURATED_ACCURACY_E1S1_VS_MOSCOT.tsv")
    print(wide.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
