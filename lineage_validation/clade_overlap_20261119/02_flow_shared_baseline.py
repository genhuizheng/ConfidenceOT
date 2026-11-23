"""How much of the grey 'Shared' in the RNA-OT vs CRISPR-OT flow figures would a structureless coupling also get?

    python lineage_validation/clade_overlap_20261119/02_flow_shared_baseline.py \
        --flows results/rna_lineage_discordance_20261110/flow/flows --out results/clade_overlap_20261119/flow_shared_baseline.tsv

Per unit, from the existing clade x clade flow matrices of 92_flow_figure.py (common support, each coupling normalised
to mass 1):
  shared_rna_crispr    sum min(F_rna, F_crispr)  (= bar 1 Shared of the figure)
  shared_indep_crispr  sum min(F_ind, F_crispr), F_ind = row marginal x column marginal of F_rna, i.e. a coupling with
                       RNA-OT's clade marginals and no structure inside them
  shared_rna_indep     sum min(F_rna, F_ind): how close RNA-OT's clade flow is to that structureless coupling
  excess               shared_rna_crispr - shared_indep_crispr
Read-only; refuses to overwrite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; not overwritten")
    rows = []
    for f in sorted(a.flows.glob("*.tsv")):
        d = pd.read_csv(f, sep="\t")
        R = d.pivot_table(index="source_clade", columns="target_clade", values="F_rna", aggfunc="sum", fill_value=0)
        C = d.pivot_table(index="source_clade", columns="target_clade", values="F_crispr", aggfunc="sum",
                          fill_value=0).reindex_like(R).fillna(0)
        R, C = R.to_numpy() / R.to_numpy().sum(), C.to_numpy() / C.to_numpy().sum()
        I = np.outer(R.sum(1), R.sum(0))
        rows.append({"unit": f.stem, "shared_rna_crispr": np.minimum(R, C).sum(),
                     "shared_indep_crispr": np.minimum(I, C).sum(), "shared_rna_indep": np.minimum(R, I).sum()})
    t = pd.DataFrame(rows)
    t["excess"] = t["shared_rna_crispr"] - t["shared_indep_crispr"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, sep="\t", index=False)
    print(t[["shared_rna_crispr", "shared_indep_crispr", "shared_rna_indep", "excess"]].median().round(3).to_string())
    print(f"units with excess > 0.05: {(t.excess > 0.05).sum()} of {len(t)}")


if __name__ == "__main__":
    main()
