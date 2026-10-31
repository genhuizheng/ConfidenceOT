"""Every annotation of every pair, from the runs' own per-annotation tables (audit/annotation_fractions.csv.gz).

    python mouse_embryo/biology/34_systematic_annotations.py --out <analysis root> --run ds=<root> [--run logcpm=<root>]

No annotation is chosen: each annotation x pair x side row of 23_'s
annotation_fractions (hard M4-E / M4-R rejected fractions, partner presence,
M4-E coupling same-annotation share, balanced-OT same-annotation share on all
bins and on the M4-E-rejected bins) is kept, joined to the pair's calibration
and convergence flags (audit/METHOD_VALIDITY_AUDIT.tsv of 31_), and summarised
per annotation x transition (adjacent) or stage (same-stage control) x side,
split by partner presence.  Writes analysis/systematic/ORGAN_COVERAGE.tsv,
ANNOTATION_PAIR_TABLE.tsv.gz, ANNOTATION_SUMMARY.tsv and
ADJACENT_VS_SAME_STAGE.tsv (each adjacent transition against the same-stage
pairs of its source stage and of its target stage, per annotation and side).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bio_common import Outputs, stage_of  # noqa: E402

VALUES = ["m4e_rejected_fraction", "m4r_rejected_fraction", "co_same_share", "bot_same_share", "bot_co_rejected_same_share"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="name=root")
    args = parser.parse_args()
    out = Outputs(args.out)
    frames = []
    for item in args.run:
        name, root = item.split("=", 1)
        path = Path(root) / "audit" / "annotation_fractions.csv.gz"
        if not path.is_file():
            print(f"{name}: no {path}; left out", flush=True)
            continue
        t = pd.read_csv(path)
        pairs = pd.read_csv(Path(root) / "manifest" / "pairs.csv")[["pair_id", "source_sample", "target_sample"]]
        t = t.merge(pairs, on="pair_id", how="left")
        t.insert(0, "run", name)
        t["own_sample"] = np.where(t["side"] == "source", t["source_sample"], t["target_sample"])
        t["own_stage"] = t["own_sample"].map(stage_of)
        frames.append(t)
    table = pd.concat(frames, ignore_index=True)
    validity_path = args.out / "audit" / "METHOD_VALIDITY_AUDIT.tsv"
    if validity_path.is_file():
        v = pd.read_csv(validity_path, sep="\t")
        flags = v.drop_duplicates(["run", "pair_id"]).set_index(["run", "pair_id"])[["m4e_inference_valid"]]
        r = v[v["method"] == "M4-R"].set_index(["run", "pair_id"])[["outer_converged", "cycle_detected"]]
        flags = flags.join(r.rename(columns=lambda c: f"m4r_{c}")).reset_index()
        table = table.merge(flags, on=["run", "pair_id"], how="left")
    table["label"] = np.where(table["kind"] == "adjacent", table["group"], table["own_stage"] + " same-stage")
    out.table(table, "analysis/systematic/ANNOTATION_PAIR_TABLE.tsv.gz")

    coverage = (table.groupby(["run", "annotation", "kind", "side"])
                .agg(pair_sides=("pair_id", "size"), bins_total=("n_bins", "sum"),
                     partner_present_pair_sides=("partner_present", "sum"),
                     stages=("own_stage", lambda s: ",".join(sorted(set(s)))))
                .reset_index())
    out.table(coverage, "analysis/systematic/ORGAN_COVERAGE.tsv")

    summary = (table.groupby(["run", "annotation", "kind", "label", "side", "partner_present"])[VALUES]
               .agg(["size", "mean", "std", "median"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary["n_pairs"] = summary["m4e_rejected_fraction_size"]
    summary = summary.drop(columns=[c for c in summary.columns if c.endswith("_size")])
    out.table(summary, "analysis/systematic/ANNOTATION_SUMMARY.tsv")

    rows = []
    for (run, annotation, side), d in table.groupby(["run", "annotation", "side"]):
        same = d[d["kind"] == "same_stage"]
        for group, g in d[d["kind"] == "adjacent"].groupby("group"):
            a, b = group.split("->")
            for ref, stage in (("source stage", a), ("target stage", b)):
                c = same[same["own_stage"] == stage]
                for value in ("m4e_rejected_fraction", "m4r_rejected_fraction"):
                    rows.append({"run": run, "annotation": annotation, "side": side, "transition": group,
                                 "control": f"same-stage {stage} ({ref})", "value": value,
                                 "adjacent_n": len(g), "adjacent_mean": g[value].mean(),
                                 "adjacent_partner_present_mean": g.loc[g["partner_present"], value].mean(),
                                 "control_n": len(c), "control_mean": c[value].mean() if len(c) else np.nan,
                                 "control_partner_present_mean": c.loc[c["partner_present"], value].mean() if len(c) else np.nan,
                                 "difference": g[value].mean() - (c[value].mean() if len(c) else np.nan)})
    out.table(pd.DataFrame(rows), "analysis/systematic/ADJACENT_VS_SAME_STAGE.tsv")
    print(f"{len(table)} annotation x pair x side rows, {table['annotation'].nunique()} annotations", flush=True)


if __name__ == "__main__":
    main()
