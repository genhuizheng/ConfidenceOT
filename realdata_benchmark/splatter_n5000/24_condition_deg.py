"""Condition-level DEG contrasts of the N = 5,000 benchmark (no gate involved; one run per condition pair).

    python realdata_benchmark/splatter_n5000/24_condition_deg.py --manifest CONDITION_MANIFEST --pair-index I \
        --truth TRUTH_DIR --out OUT

Original counts, library size 1e4 + log1p, scanpy Wilcoxon, Benjamini-Hochberg; apparent DE = FDR < 0.05 and
|log2FC| > 0.25. Cells are not biological replicates: these are descriptive and the uncertainty comes from the
independent simulated replicates (rep 1-3), aggregated downstream.
  DEG-A  per side holding an unmatched population: true unmatched vs true shared cells, against deg_truth (AUROC /
         AUPRC of |score| for true_de, direction agreement, effect Spearman): how detectable the reference signature is.
  DEG-C  all source vs all target cells: apparent DE count; it mixes composition, batch and perturbation and is
         descriptive only.
  DEG-D  per shared group, source vs target cells of that group: apparent DE count, and AUROC of |score| for the
         biological truth (perturbed genes of that group) and for the technical truth (|log2 batch ratio| > 0.25),
         kept apart. In L0-L2 a shared group has no true shift (apparent DE is the noise floor); in L3 any shift is
         batch; from L4 Group1 also carries the perturbation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import average_precision_score, roc_auc_score


def wilcoxon(data: ad.AnnData, mask_a: np.ndarray) -> pd.DataFrame:
    d = data.copy(); d.obs["g"] = np.where(mask_a, "a", "b")
    sc.tl.rank_genes_groups(d, "g", groups=["a"], reference="b", method="wilcoxon")
    return sc.get.rank_genes_groups_df(d, "a").set_index("names")


def against(res: pd.DataFrame, truth_bool: pd.Series, prefix: str) -> dict:
    y = truth_bool.reindex(res.index).fillna(False).to_numpy(bool); s = res.scores.abs().to_numpy()
    if not 0 < y.mean() < 1:
        return {f"{prefix}_status": "no true signal" if y.sum() == 0 else "all genes true"}
    return {f"{prefix}_true_genes": int(y.sum()), f"{prefix}_auroc": float(roc_auc_score(y, s)), f"{prefix}_auprc": float(average_precision_score(y, s)),
            f"{prefix}_auprc_baseline": float(y.mean())}


def apparent(res: pd.DataFrame) -> int:
    return int(((res.pvals_adj < 0.05) & (res.logfoldchanges.abs() > 0.25)).sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--pair-index", type=int, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if (a.out / "deg_condition.tsv").exists():
        sys.exit(f"{a.out}/deg_condition.tsv exists; not overwritten")
    a.out.mkdir(parents=True, exist_ok=True)
    row = pd.read_csv(a.manifest).iloc[a.pair_index]
    truth = pd.read_csv(a.truth / "population_truth.tsv", sep="\t")
    deg = pd.read_csv(a.truth / "deg_truth.tsv", sep="\t")
    shift = pd.read_csv(a.truth / "shared_shift_truth.tsv", sep="\t")
    sides = {}
    for s in ("source", "target"):
        d = ad.read_h5ad(row[f"{s}_h5ad"]); sc.pp.normalize_total(d, target_sum=1e4); sc.pp.log1p(d)
        t = truth[truth.side == s].set_index("observation_id").loc[d.obs_names.astype(str)]
        d.obs["group"] = t.group.to_numpy(); d.obs["shared"] = t.shared.to_numpy(bool); d.obs["side"] = s
        sides[s] = d
    rows = []
    for s, d in sides.items():
        tr = deg[deg.side == s]
        if len(tr) and d.obs.shared.any() and (~d.obs.shared).any():
            res = wilcoxon(d, ~d.obs.shared.to_numpy(bool))
            t = tr.set_index("gene_id").reindex(res.index)
            y = t.true_de.fillna(False).to_numpy(bool)
            rows.append({"contrast": "DEG-A unmatched vs shared", "side": s, "group": ";".join(sorted(tr.unmatched_group.unique())),
                         "cells_a": int((~d.obs.shared).sum()), "cells_b": int(d.obs.shared.sum()), "apparent_de": apparent(res),
                         **against(res, t.true_de.fillna(False), "truth"),
                         "direction_agreement": float(np.mean(np.sign(res.logfoldchanges.to_numpy()[y]) == np.sign(t.true_log2fc.to_numpy()[y]))) if y.any() else np.nan})
    both = ad.concat([sides["source"], sides["target"]], join="inner")
    res = wilcoxon(both, (both.obs.side == "target").to_numpy())
    rows.append({"contrast": "DEG-C target vs source (all cells)", "side": "both", "cells_a": int((both.obs.side == "target").sum()),
                 "cells_b": int((both.obs.side == "source").sum()), "apparent_de": apparent(res)})
    for g in sorted(shift.group.unique()):
        sub = both[(both.obs.group == g).to_numpy()]
        mask = (sub.obs.side == "target").to_numpy()
        if mask.sum() < 20 or (~mask).sum() < 20:
            rows.append({"contrast": "DEG-D shared group target vs source", "group": g, "status": "not_estimable: < 20 cells"}); continue
        res = wilcoxon(sub, mask)
        st = shift[shift.group == g].set_index("gene_id")
        rows.append({"contrast": "DEG-D shared group target vs source", "side": "both", "group": g, "cells_a": int(mask.sum()),
                     "cells_b": int((~mask).sum()), "apparent_de": apparent(res),
                     **against(res, st.biological_log2fc.abs() > 0, "biological"), **against(res, st.technical_log2fc.abs() > 0.25, "technical")})
    pd.DataFrame(rows).to_csv(a.out / "deg_condition.tsv", sep="\t", index=False)
    print(f"{len(rows)} contrasts -> {a.out}")


if __name__ == "__main__":
    main()
