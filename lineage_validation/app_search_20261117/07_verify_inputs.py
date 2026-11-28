"""Check every OT input written by 06_prepare_tacc.py before any OT run (PLAN_TACC.md, order step 2).

For each dataset directory under --app with a manifest.csv, and each manifest row and side:
  - the h5ad opens; n_obs / n_vars; cell count equals units.tsv;
  - every count is a non-negative integer (whole matrix, not a sample); cells with zero total counts; min / median total;
  - obs has sample_id (one value, equal to the manifest's sample) and malignant; obs names unique within and across sides;
  - source and target have identical var names, in the same order, unique;
  - missing values in each obs column.
Unit-level label tables are written per dataset: value counts of the label columns per unit and side.
Writes APP/verify/inputs.tsv, APP/verify/<dataset>_labels.tsv and prints one line per problem. Reads only.

    python -I 07_verify_inputs.py --app $SCRATCH/OT_project/app_search_20261117
    python -I 07_verify_inputs.py --app $SCRATCH/OT_project/app_search_20261117 --only splintr zebrafish --tag _20261127
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {"kang": ["cell_type", "donor"], "sciplex2": ["drug", "dose"], "schiebinger": ["stage", "condition"],
          "celegans": ["time_bin", "cell_type"], "pancreas": ["day", "cell_type"], "larry_invivo": ["time", "cell_type"],
          "splintr": ["condition", "mouse", "n_barcodes"], "zebrafish": ["orig.ident", "Cell_type", "tree_leaf"]}


def main() -> None:
    import anndata as ad
    import scipy.sparse as sp
    p = argparse.ArgumentParser()
    p.add_argument("--app", type=Path, required=True)
    p.add_argument("--only", nargs="+", default=None, help="dataset directories to check (default: every one with a manifest)")
    p.add_argument("--tag", default="", help="suffix of the summary file: verify/inputs<tag>.tsv (default keeps inputs.tsv)")
    a = p.parse_args()
    out = a.app / "verify"; out.mkdir(exist_ok=True)
    rows, problems = [], []
    for man_f in sorted(a.app.glob("*/manifest.csv")):
        ds = man_f.parent.name
        if a.only and ds not in a.only:
            continue
        man = pd.read_csv(man_f)
        units = pd.read_csv(man_f.parent / "prep" / "units.tsv", sep="\t").set_index("pair_id")
        lab_rows = []
        for _, r in man.iterrows():
            sides = {}
            for side in ("source", "target"):
                f = Path(r[f"{side}_h5ad"])
                A = ad.read_h5ad(f)
                X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
                d = X.data
                tot = np.asarray(X.sum(1)).ravel()
                rec = {"dataset": ds, "pair_id": r.pair_id, "side": side, "n_obs": A.n_obs, "n_vars": A.n_vars,
                       "units_cells": int(units.loc[r.pair_id, f"{side}_cells"]),
                       "integer": bool(np.all(d == np.round(d))), "negative": int((d < 0).sum()),
                       "zero_total_cells": int((tot == 0).sum()), "min_total": float(tot.min()), "median_total": float(np.median(tot)),
                       "sample_id_ok": bool((A.obs["sample_id"].astype(str) == str(r[f"{side}_sample"])).all()),
                       "malignant_ok": bool((A.obs["malignant"].astype(str) == "malignant").all()),
                       "obs_unique": bool(A.obs_names.is_unique), "var_unique": bool(A.var_names.is_unique),
                       "obs_missing": ";".join(f"{c}:{int(n)}" for c, n in A.obs.isna().sum().items() if n) or "none"}
                sides[side] = A
                rows.append(rec)
                for c in LABELS.get(ds, []):
                    if c in A.obs:
                        for v, n in A.obs[c].astype(str).value_counts().items():
                            lab_rows.append({"pair_id": r.pair_id, "side": side, "column": c, "value": v, "cells": int(n)})
                bad = [k for k in ("integer", "sample_id_ok", "malignant_ok", "obs_unique", "var_unique") if not rec[k]]
                if rec["n_obs"] != rec["units_cells"]: bad.append("n_obs != units.tsv")
                if rec["negative"]: bad.append("negative counts")
                if rec["zero_total_cells"]: bad.append(f"{rec['zero_total_cells']} zero-total cells")
                if rec["obs_missing"] != "none": bad.append("missing obs " + rec["obs_missing"])
                for b in bad:
                    problems.append(f"{ds} {r.pair_id} {side}: {b}")
            S, T = sides["source"], sides["target"]
            if not (S.var_names.equals(T.var_names)):
                problems.append(f"{ds} {r.pair_id}: source and target var names differ")
            if len(S.obs_names.intersection(T.obs_names)):
                problems.append(f"{ds} {r.pair_id}: obs names shared between sides")
        pd.DataFrame(lab_rows).to_csv(out / f"{ds}_labels.tsv", sep="\t", index=False)
        print(ds, "rows", len(man), flush=True)
    pd.DataFrame(rows).to_csv(out / f"inputs{a.tag}.tsv", sep="\t", index=False)
    print("problems:", len(problems))
    for x in problems:
        print("  " + x)


if __name__ == "__main__":
    main()
