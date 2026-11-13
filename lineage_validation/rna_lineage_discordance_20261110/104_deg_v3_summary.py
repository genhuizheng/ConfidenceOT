"""PLAN_v3_deg.md reading rules (with the 2026-11-13 operational definitions) applied to the raw and thinned DEG / GSEA
runs. Reads deg_v3/{contrasts.tsv, deg_gsea_raw, deg_gsea_ds}; writes deg_v3/summary/:

  nes_long.tsv          real contrast x hallmark set: NES raw / thinned / IEG-removed, random-split range, paired baseline
  contrast_qc.tsv       per real contrast: groups, depth, edits, states, gates, IEG share of the top 50, baseline agreement
  baseline_pairs.tsv    A contrast vs its RNA-proximity baseline: Spearman of the 50 NES, top-100 DEG overlap
  mouse_nes.tsv         type x set x mouse: mean NES (raw, thinned, IEG-removed)
  consistency.tsv       A types and baselines: the sign rule, both runs, random split, baseline, IEG; with / without 3724
  bc_null.tsv           B1 / C1: share of contrasts with |NES| outside the random-split |NES| range
  b1_pairs.tsv          B1 per recurring state pair (alphabetical orientation), the sign rule within the pair

    python lineage_validation/rna_lineage_discordance_20261110/104_deg_v3_summary.py --root results/rna_lineage_discordance_20261110
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IEG = {"Fos", "Fosb", "Jun", "Junb", "Jund", "Egr1", "Egr2", "Egr3", "Ier2", "Ier3", "Ier5", "Atf3", "Dusp1", "Nr4a1",
       "Klf2", "Klf4", "Klf6", "Zfp36", "Btg2", "Cyr61", "Ccn1", "Hspa1a", "Hspa1b"}
SIGN_TYPES = ["A1_rna_vs_crispr", "A1_rna_vs_both", "A1_crispr_vs_both", "A2_cross_vs_own", "A1_baseline", "A2_baseline"]
A_TYPES = ["A1_rna_vs_crispr", "A1_rna_vs_both", "A1_crispr_vs_both", "A2_cross_vs_own"]
NULL_TYPES = {"A1_rna_vs_crispr", "A2_cross_vs_own", "B1_state", "C1_clade"}
BASE_OF = {"A1_rna_vs_crispr": "A1_baseline", "A1_rna_vs_both": "A1_baseline", "A1_crispr_vs_both": "A1_baseline",
           "A2_cross_vs_own": "A2_baseline"}


def pivot(f):
    g = pd.read_csv(f, sep="\t")
    return g.pivot_table(index="contrast_id", columns="term", values="nes", aggfunc="first")


def majority(v):
    v = v.dropna()
    n = len(v); pos = int((v > 0).sum()); neg = int((v < 0).sum())
    if n == 0:
        return 0, np.nan, 0
    if pos == neg:
        return 0, pos / n, n
    return (1 if pos > neg else -1), max(pos, neg) / n, n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    D = a.root / "deg_v3"; out = D / "summary"; out.mkdir(exist_ok=True)
    C = pd.read_csv(D / "contrasts.tsv", sep="\t", dtype={"mouse": str})
    rand = C[C.type.str.contains("_rand")]; real = C[~C.type.str.contains("_rand")].copy()
    N = {k: pivot(D / f"deg_gsea_{k}" / "gsea_hallmark.tsv") for k in ("raw", "ds")}
    for k in ("raw", "ds"):
        f = D / f"deg_gsea_{k}" / "gsea_hallmark_noieg.tsv"
        N[f"noieg_{k}"] = pivot(f) if f.exists() else None
    terms = sorted(N["raw"].columns)
    # random-split range per parent contrast and set
    rng = {}
    for k in ("raw", "ds"):
        R = N[k].reindex(rand.contrast_id).copy(); R["parent"] = rand.set_index("contrast_id").parent.reindex(R.index).to_numpy()
        rng[k] = (R.groupby("parent")[terms].min(), R.groupby("parent")[terms].max(),
                  R[terms].abs().assign(parent=R.parent).groupby("parent")[terms].max())
    # paired baseline: same unit, clade, site, side
    key = lambda d: d.unit.astype(str) + "|" + d.clade.astype(str) + "|" + d.site.astype(str) + "|" + d.side.astype(str)  # noqa: E731
    real["key"] = key(real)
    base_id = {}
    for t, bt in BASE_OF.items():
        bmap = real[real.type == bt].set_index("key").contrast_id
        for r in real[real.type == t].itertuples():
            base_id[r.contrast_id] = bmap.get(r.key)
    real["baseline_id"] = real.contrast_id.map(base_id)
    # long NES table
    rows = []
    for r in real.itertuples():
        for t in terms:
            rec = {"contrast_id": r.contrast_id, "type": r.type, "mouse": r.mouse, "term": t}
            for k in ("raw", "ds", "noieg_raw", "noieg_ds"):
                M = N[k]
                rec[f"nes_{k}"] = M.at[r.contrast_id, t] if M is not None and r.contrast_id in M.index and t in M.columns else np.nan
            for k in ("raw", "ds"):
                lo, hi, ab = rng[k]
                ok = r.contrast_id in lo.index
                rec[f"null_min_{k}"] = lo.at[r.contrast_id, t] if ok else np.nan
                rec[f"null_max_{k}"] = hi.at[r.contrast_id, t] if ok else np.nan
                rec[f"null_absmax_{k}"] = ab.at[r.contrast_id, t] if ok else np.nan
            b = base_id.get(r.contrast_id)
            rec["baseline_id"] = b
            rec["nes_baseline_raw"] = N["raw"].at[b, t] if isinstance(b, str) and b in N["raw"].index else np.nan
            rows.append(rec)
    L = pd.DataFrame(rows); L.to_csv(out / "nes_long.tsv", sep="\t", index=False)

    # contrast QC + IEG share + baseline agreement
    def deg(k, cid):
        return pd.read_csv(D / f"deg_gsea_{k}" / "deg" / f"{cid}.tsv.gz", sep="\t")
    qc = real.drop(columns=["key"]).copy()
    bp = []
    for k in ("raw", "ds"):
        sh = []
        for cid in qc.contrast_id:
            g = deg(k, cid); top = g.reindex(g.scores.abs().sort_values(ascending=False).index).head(50)
            sh.append(float(top.names.isin(IEG).mean()))
        qc[f"ieg_top50_share_{k}"] = sh
    for r in real[real.type.isin(BASE_OF)].itertuples():
        b = base_id.get(r.contrast_id)
        if not isinstance(b, str):
            continue
        rec = {"contrast_id": r.contrast_id, "type": r.type, "mouse": r.mouse, "baseline_id": b}
        for k in ("raw", "ds"):
            x, y = N[k].loc[r.contrast_id, terms], N[k].loc[b, terms]; m = x.notna() & y.notna()
            rec[f"nes_spearman_{k}"] = spearmanr(x[m], y[m])[0] if m.sum() >= 5 else np.nan
        g1, g2 = deg("raw", r.contrast_id), deg("raw", b)
        t1 = g1.reindex(g1.scores.abs().sort_values(ascending=False).index).head(100).set_index("names").scores
        t2 = g2.reindex(g2.scores.abs().sort_values(ascending=False).index).head(100).set_index("names").scores
        sh_ = t1.index.intersection(t2.index)
        rec["top100_overlap_same_sign"] = float((np.sign(t1[sh_]) == np.sign(t2[sh_])).sum() / 100)
        bp.append(rec)
    BP = pd.DataFrame(bp); BP.to_csv(out / "baseline_pairs.tsv", sep="\t", index=False)
    qc = qc.merge(BP[["contrast_id", "nes_spearman_raw", "nes_spearman_ds", "top100_overlap_same_sign"]], on="contrast_id", how="left")
    qc.to_csv(out / "contrast_qc.tsv", sep="\t", index=False)

    # mouse-level NES
    MN = L.groupby(["type", "term", "mouse"])[["nes_raw", "nes_ds", "nes_noieg_raw", "nes_noieg_ds"]].mean().reset_index()
    MN.to_csv(out / "mouse_nes.tsv", sep="\t", index=False)

    # consistency (A types, baselines)
    cons = []
    for excl in (False, True):
        Ls = L[L.mouse != "3724"] if excl else L
        for t in SIGN_TYPES:
            for term in terms:
                s = Ls[(Ls.type == t) & (Ls.term == term)]
                if s.empty:
                    continue
                mm = s.groupby("mouse")[["nes_raw", "nes_ds", "nes_noieg_raw", "nes_noieg_ds"]].mean()
                sr, shr, n = majority(mm.nes_raw); sd, shd, nd = majority(mm.nes_ds)
                rec = {"without_3724": excl, "type": t, "term": term, "contrasts": len(s), "mice": n,
                       "sign_raw": sr, "share_raw": shr, "sign_ds": sd, "share_ds": shd}
                rec["sign_ok"] = bool(n >= 2 and sr != 0 and sr == sd and shr >= 0.75 and shd >= 0.75)
                if t in NULL_TYPES:
                    v = s.dropna(subset=["null_min_raw"])
                    out_ = (v.nes_raw > v.null_max_raw) if sr > 0 else (v.nes_raw < v.null_min_raw) if sr < 0 else pd.Series(False, index=v.index)
                    rec["outside_null_share"] = float(out_.mean()) if len(v) else np.nan
                    rec["null_ok"] = bool(len(v) and out_.mean() >= 0.5)
                else:
                    rec["outside_null_share"] = np.nan; rec["null_ok"] = False
                if t in A_TYPES:
                    v = s.dropna(subset=["nes_baseline_raw"])
                    ex = (np.sign(v.nes_baseline_raw) == np.sign(v.nes_raw)) & (v.nes_baseline_raw.abs() >= 1)
                    rec["baseline_explained_share"] = float(ex.mean()) if len(v) else np.nan
                    rec["baseline_ok"] = bool(len(v) == 0 or ex.mean() < 0.5)
                else:
                    rec["baseline_explained_share"] = np.nan; rec["baseline_ok"] = True
                rec["consistent"] = bool(rec["sign_ok"] and rec["null_ok"] and rec["baseline_ok"] and t in A_TYPES)
                for k in ("noieg_raw", "noieg_ds"):
                    sg, shg, _ = majority(mm[f"nes_{k}"]); rec[f"sign_{k}"] = sg; rec[f"share_{k}"] = shg
                cons.append(rec)
    CO = pd.DataFrame(cons); CO.to_csv(out / "consistency.tsv", sep="\t", index=False)

    # B1 / C1: |NES| outside the random |NES| range
    bc = []
    for excl in (False, True):
        Ls = L[L.mouse != "3724"] if excl else L
        for t in ("B1_state", "C1_clade"):
            for term in terms:
                s = Ls[(Ls.type == t) & (Ls.term == term)]
                rec = {"without_3724": excl, "type": t, "term": term, "contrasts": len(s), "mice": s.mouse.nunique()}
                for k in ("raw", "ds"):
                    v = s.dropna(subset=[f"null_absmax_{k}", f"nes_{k}"])
                    rec[f"outside_abs_null_share_{k}"] = float((v[f"nes_{k}"].abs() > v[f"null_absmax_{k}"]).mean()) if len(v) else np.nan
                bc.append(rec)
    pd.DataFrame(bc).to_csv(out / "bc_null.tsv", sep="\t", index=False)

    # B1 per recurring state pair, alphabetical orientation
    b1 = real[real.type == "B1_state"].copy()
    b1["flip"] = np.where(b1.case_group.astype(str) > b1.reference_group.astype(str), -1, 1)
    b1["pair"] = [" vs ".join(sorted([x, y])) for x, y in zip(b1.case_group.astype(str), b1.reference_group.astype(str))]
    Lb = L[L.type == "B1_state"].merge(b1[["contrast_id", "flip", "pair"]], on="contrast_id")
    for k in ("raw", "ds"):
        Lb[f"o_{k}"] = Lb[f"nes_{k}"] * Lb.flip
    pr = []
    for (pair, term), s in Lb.groupby(["pair", "term"]):
        mm = s.groupby("mouse")[["o_raw", "o_ds"]].mean()
        if len(mm) < 2:
            continue
        sr, shr, n = majority(mm.o_raw); sd, shd, _ = majority(mm.o_ds)
        pr.append({"pair": pair, "term": term, "mice": n, "contrasts": len(s), "sign_raw": sr, "share_raw": shr, "sign_ds": sd, "share_ds": shd,
                   "sign_ok": bool(n >= 2 and sr != 0 and sr == sd and shr >= 0.75 and shd >= 0.75)})
    pd.DataFrame(pr).to_csv(out / "b1_pairs.tsv", sep="\t", index=False)

    # console digest
    for excl in (False, True):
        c = CO[CO.without_3724 == excl]
        print(f"--- without 3724: {excl}")
        for t in SIGN_TYPES:
            ct = c[c.type == t]
            print(f"{t}: sets {len(ct)}, sign rule {int(ct.sign_ok.sum())}, + null {int((ct.sign_ok & ct.null_ok).sum())}, "
                  f"consistent {int(ct.consistent.sum())}: {', '.join(ct[ct.consistent].term.str.replace('HALLMARK_', ''))}")
    print("baseline Spearman (raw) median by type:", BP.groupby("type").nes_spearman_raw.median().round(2).to_dict())
    print("IEG top-50 share median raw / ds:", qc.ieg_top50_share_raw.median(), qc.ieg_top50_share_ds.median())


if __name__ == "__main__":
    main()
