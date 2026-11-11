"""Check that the character states rebuilt from the allele tables equal the authors' integer character matrices.

    python lineage_validation/crispr_matrix_benchmark_20261108/02_matrix_concordance.py --inputs INPUTS --out OUT

For every authors' matrix (KP families, Quinn clones) and every column k, the allele-table character (intBC, cut site)
it encodes is found by matching on the matrix's cells:
  missing agreement  share of cells where (k missing) == (intBC not observed in the allele table);
  bijection          on cells observed in both, the integer code determines the state string and vice versa
                     (share of cells whose code -> string and string -> code maps are both one-to-one);
  uncut agreement    code 0 <-> 'None'.
The best-matching character per column is reported with these three shares; a column without a match >= 0.95 is
listed as unmatched. Read only. Writes OUT/matrix_concordance.tsv (one row per column) and prints a summary.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import matrix_core as mc  # noqa: E402

KP_MAT = REPO / "results" / "kp_tracer_lineage_validation" / "export" / "trees" / "character_matrices"
QX = REPO / "lineage_validation" / "local_lineage_visualization" / "data" / "local_lineage_visualization_export"
MISSING = {"-", "-1", "", "nan", "NaN"}


def read_codes(path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).set_index("cellBC")
    return raw.apply(lambda c: c.map(lambda x: -1 if x.strip() in MISSING else int(x)))


def match_column(code: pd.Series, chars: dict) -> dict:
    best = {"character": None, "missing_agreement": 0.0, "bijection": 0.0, "uncut_agreement": np.nan, "cells_both": 0}
    for key, st in chars.items():
        s = st.reindex(code.index)
        miss_a, miss_b = code < 0, s.isna()
        ma = float((miss_a == miss_b).mean())
        if ma < best["missing_agreement"] - 1e-12 or ma < 0.5:
            continue
        both = (~miss_a) & (~miss_b)
        if both.sum() == 0:
            continue
        c, v = code[both], s[both]
        f = pd.DataFrame({"c": c, "v": v})
        c2v = f.groupby("c").v.nunique(); v2c = f.groupby("v").c.nunique()
        ok = f.c.map(c2v).eq(1) & f.v.map(v2c).eq(1)
        bij = float(ok.mean())
        unc = float(((c == 0) == (v == "None")).mean())
        if (ma, bij) > (best["missing_agreement"], best["bijection"]):
            best = {"character": key, "missing_agreement": ma, "bijection": bij, "uncut_agreement": unc, "cells_both": int(both.sum())}
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    I = a.inputs / "data"
    fin = mc.read_allele_table(I / "kp_tracer/KPTracer-Data/KPTracer.alleleTable.FINAL.txt", "KPTracer")
    unf = mc.read_allele_table(I / "kp_tracer/KPTracer-Data/KPTracer.alleleTable.unfiltered.txt", "KPTracer")
    tabs = {"KPTracer": pd.concat([fin, unf[~unf.cell.isin(set(fin.cell))]])}
    for f in glob.glob(str(I / "quinn2021_GSE161363/converted/lineage/*/*alleleTable*.txt.gz")):
        tabs[Path(f).parent.name] = mc.read_allele_table(f, "Quinn2021")
    mats = [("KPTracer", f) for f in sorted(KP_MAT.glob("*_character_matrix.txt"))]
    mats += [(Path(f).parents[2].name, Path(f)) for f in sorted(glob.glob(str(QX / "lineage/Quinn2021/*/character_matrices/character_matrices/*_character_matrix.alleleThresh.txt")))]
    rows = []
    for ds, f in mats:
        codes = read_codes(f)
        t = tabs[ds][tabs[ds].cell.isin(codes.index)]
        chars = {}
        for b, g in t.groupby("intBC"):
            for s in mc.SITES:
                chars[(b, s)] = g.set_index("cell")[s]
        for col in codes.columns:
            r = match_column(codes[col], chars)
            rows.append({"dataset": ds, "matrix": f.name.split("_character_matrix")[0], "column": col, "cells": len(codes),
                         "cells_in_allele_table": int(codes.index.isin(t.cell).sum()), **r})
        print(f"{f.name}: {codes.shape[1]} columns", flush=True)
    R = pd.DataFrame(rows)
    R["matched"] = (R.missing_agreement >= 0.95) & (R.bijection >= 0.95)
    R.to_csv(a.out / "matrix_concordance.tsv", sep="\t", index=False)
    s = R.groupby("dataset").agg(matrices=("matrix", "nunique"), columns=("column", "size"), matched=("matched", "mean"),
                                 median_missing_agreement=("missing_agreement", "median"), median_bijection=("bijection", "median"),
                                 min_bijection=("bijection", "min"), median_uncut_agreement=("uncut_agreement", "median"))
    print(s.round(4).to_string())
    print(R[~R.matched].head(20).to_string())


if __name__ == "__main__":
    main()
