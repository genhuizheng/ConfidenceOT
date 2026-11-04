"""Strict pairing audit of a generated N = 5,000 root (read only), before any fit is run on it.

    python realdata_benchmark/splatter_n5000/29_pairing_audit.py --root ROOT --out OUT [--replicates 1 2 3]

For each replicate, level pairs that should differ by one factor only:
  batch only          L2_depth_dropout -> L3_batch, A4_perturb_only -> L4_batch_perturb, A5_imbalance_only -> L5_...
  perturbation only   L3_batch -> L4_batch_perturb, L2_depth_dropout -> A4_perturb_only
  proportions only    L2_depth_dropout -> A5_imbalance_only, L4_batch_perturb -> L5_batch_perturb_imbalance
Checks (each a row of PAIRING_AUDIT.tsv with expected / observed and holds True / False):
  gene truth          GeneMean, BaseGeneMean, OutlierFactor, DEFacGroup*, BatchFacDrawn* identical; the applied
                      BatchFacBatch* identical except across a batch-only pair; PerturbationFactor identical except
                      across a perturbation-only pair
  cells               cell ids, sides and expected library sizes identical; groups identical except the target side of
                      a proportions-only pair
  counts              per side and population block: share of identical entries. Expected bit-identical: every block
                      of a perturbation-only pair except target Group1, and there every gene with PerturbationFactor 1;
                      every block of a proportions-only pair, over the cells whose group is the same in both levels.
                      Batch-only pairs: reported (every mean moves), with
                      the Pearson r of the per-gene pseudobulk log ratio against the true batch log ratio of that side
  sampler             sampler_check.csv when present (paired sampler vs Splatter's own counts of the same call): median
                      |log2 mean ratio| over genes with mean > 0.5, maximum |zero-fraction difference|
Also every level of a replicate must share the drawn batch factors. On a root generated without --pairing paired the
same checks show what Splatter's own sampler does to unaffected blocks (the RNG-consumption question).
Writes OUT/PAIRING_AUDIT.tsv, OUT/PAIRING_AUDIT.md and OUT/PASSED or OUT/FAILED (refuses if OUT exists); exits 1 when
any check does not hold, so a fit chained afterok on it (submit_n5000.sh chain --pairing paired) does not start.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

PAIRS = [("batch", "L2_depth_dropout", "L3_batch"), ("batch", "A4_perturb_only", "L4_batch_perturb"),
         ("batch", "A5_imbalance_only", "L5_batch_perturb_imbalance"), ("perturbation", "L3_batch", "L4_batch_perturb"),
         ("perturbation", "L2_depth_dropout", "A4_perturb_only"), ("proportions", "L2_depth_dropout", "A5_imbalance_only"),
         ("proportions", "L4_batch_perturb", "L5_batch_perturb_imbalance")]
BASE_TRUTH = ("GeneMean", "BaseGeneMean", "OutlierFactor", "DEFacGroup", "BatchFacDrawn")


def load(gen: Path) -> dict:
    X = sp.csr_matrix(scipy.io.mmread(str(gen / "counts.mtx")).T)  # cells x genes
    cells = pd.read_csv(gen / "cells_truth.csv"); genes = pd.read_csv(gen / "genes_truth.csv")
    rec = json.loads((gen / "generation.json").read_text())
    chk = pd.read_csv(gen / "sampler_check.csv") if (gen / "sampler_check.csv").exists() else None
    return {"X": X, "cells": cells, "genes": genes, "rec": rec, "check": chk}


def same_cols(a: pd.DataFrame, b: pd.DataFrame, prefixes) -> tuple[bool, str]:
    cols = [c for c in a.columns if c.startswith(prefixes)]
    bad = [c for c in cols if c not in b or not np.array_equal(a[c].to_numpy(), b[c].to_numpy())]
    return (len(cols) > 0 and not bad), ";".join(bad) if bad else f"{len(cols)} columns"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replicates", type=int, nargs="*", default=[1, 2, 3])
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    a.out.mkdir(parents=True)
    rows = []
    def add(rep, kind, pair, check, expected, observed, holds, detail=""):
        rows.append({"replicate": rep, "pair_kind": kind, "pair": pair, "check": check, "expected": expected,
                     "observed": observed, "holds": bool(holds), "detail": detail})
    for rep in a.replicates:
        base = a.root / "generated" / "N5000" / f"rep{rep}"
        levels = sorted({x for _, l1, l2 in PAIRS for x in (l1, l2)})
        data = {}
        for lv in levels:
            if (base / lv / "GENERATION_DONE").exists():
                data[lv] = load(base / lv)
                if data[lv]["check"] is not None:
                    c = data[lv]["check"]; k = c.mean_splatter > 0.5
                    add(rep, "sampler", lv, "paired sampler vs Splatter (median |log2 mean ratio|, max |zero diff|)",
                        "small (same model)", f"{np.median(np.abs(np.log2(c.mean_paired[k] / c.mean_splatter[k]))):.4f} / "
                        f"{np.max(np.abs(c.zero_paired - c.zero_splatter)):.4f}", True, f"{int(k.sum())} genes")
            print("loaded", rep, lv, flush=True)
        drawn = [c for c in next(iter(data.values()))["genes"].columns if c.startswith("BatchFacDrawn")] if data else []
        if drawn:
            ref = next(iter(data.values()))["genes"][drawn]
            ok = all(np.array_equal(d["genes"][drawn].to_numpy(), ref.to_numpy()) for d in data.values())
            add(rep, "all", "every level", "drawn batch factors identical", "identical", "identical" if ok else "differ", ok)
        else:
            add(rep, "all", "every level", "drawn batch factors recorded", "BatchFacDrawn* columns", "absent (generated without the hook)", False)
        for kind, l1, l2 in PAIRS:
            if l1 not in data or l2 not in data:
                continue
            A, B = data[l1], data[l2]; pair = f"{l1} -> {l2}"
            ok, d = same_cols(A["genes"], B["genes"], BASE_TRUTH)
            add(rep, kind, pair, "gene means, outliers, group DE, drawn batch factors", "identical", "identical" if ok else "differ", ok, d)
            okb, d = same_cols(A["genes"], B["genes"], ("BatchFacBatch",))
            add(rep, kind, pair, "applied batch factors", "differ" if kind == "batch" else "identical",
                "identical" if okb else "differ", okb != (kind == "batch"), d)
            okp = np.array_equal(A["genes"].PerturbationFactor.to_numpy(), B["genes"].PerturbationFactor.to_numpy())
            add(rep, kind, pair, "perturbation factors", "differ" if kind == "perturbation" else "identical",
                "identical" if okp else "differ", okp != (kind == "perturbation"))
            ca, cb = A["cells"], B["cells"]
            same_ids = ca.cell_id.equals(cb.cell_id) and ca.side.equals(cb.side)
            add(rep, kind, pair, "cell ids and sides", "identical", "identical" if same_ids else "differ", same_ids)
            lib = np.array_equal(ca.ExpLibSize.to_numpy(), cb.ExpLibSize.to_numpy())
            add(rep, kind, pair, "expected library sizes", "identical", "identical" if lib else "differ", lib)
            for side in ("source", "target"):
                m = (ca.side == side).to_numpy()
                g_same = float((ca.group[m].to_numpy() == cb.group[m].to_numpy()).mean())
                exp_same = not (kind == "proportions" and side == "target")
                add(rep, kind, pair, f"{side} group labels (share identical)", "1" if exp_same else "< 1 (by design)",
                    f"{g_same:.4f}", (g_same == 1.0) == exp_same)
            if not same_ids:
                continue
            pert = (A["genes"].PerturbationFactor.to_numpy() != 1) | (B["genes"].PerturbationFactor.to_numpy() != 1)
            for side in ("source", "target"):
                for g in sorted(ca.group.unique()):
                    m = ((ca.side == side) & (ca.group == g) & (cb.group == g)).to_numpy()
                    if not m.any():
                        continue
                    xa, xb = A["X"][m], B["X"][m]
                    diff = (xa != xb)
                    share = 1.0 - diff.nnz / (xa.shape[0] * xa.shape[1])
                    # a proportions-only pair is compared on the cells whose group is the same in both levels: their
                    # latent means are the same, so their counts must be too
                    affected = (kind == "batch") or (kind == "perturbation" and side == "target" and g == "Group1")
                    detail = ""
                    if kind == "perturbation" and side == "target" and g == "Group1":
                        dcols = np.unique(diff.tocoo().col)
                        outside = int((~pert[dcols]).sum())
                        add(rep, kind, pair, f"{side} {g}: entries differing outside perturbed genes", "0", str(outside), outside == 0,
                            f"{len(dcols)} genes differ, {int(pert.sum())} perturbed")
                    if kind == "batch":
                        bcol = "BatchFacBatch1" if side == "source" else "BatchFacBatch2"
                        tb = np.log2(B["genes"][bcol].to_numpy(float) / A["genes"][bcol].to_numpy(float))
                        pa = np.asarray(xa.sum(0)).ravel(); pb = np.asarray(xb.sum(0)).ravel()
                        k = (pa + pb) > 50
                        lr = np.log2((pb[k] + 1) / pb.sum() * pa.sum() / (pa[k] + 1))
                        r = float(np.corrcoef(lr, tb[k])[0, 1]) if tb[k].std() > 0 else np.nan
                        detail = f"pseudobulk log2 ratio vs true batch log2 ratio r = {r:.3f}"
                    add(rep, kind, pair, f"{side} {g}: share of identical count entries", "reported (mean moves)" if affected else "1",
                        f"{share:.6f}", True if affected else share == 1.0, detail)
    t = pd.DataFrame(rows)
    t.to_csv(a.out / "PAIRING_AUDIT.tsv", sep="\t", index=False)
    bad = t[~t.holds]
    lines = [f"# Pairing audit of {a.root}", "", f"{len(t)} checks, {int(t.holds.sum())} hold, {len(bad)} do not.", ""]
    lines += ["| replicate | pair | check | expected | observed | detail |", "|---|---|---|---|---|---|"]
    for r in (bad if len(bad) else t.head(0)).itertuples():
        lines.append(f"| {r.replicate} | {r.pair} | {r.check} | {r.expected} | {r.observed} | {r.detail} |")
    (a.out / "PAIRING_AUDIT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    # a non-zero exit stops a fit chained afterok on this audit (submit_n5000.sh chain --pairing paired)
    (a.out / ("PASSED" if bad.empty else "FAILED")).write_text(f"{len(bad)} checks do not hold\n")
    sys.exit(0 if bad.empty else 1)


if __name__ == "__main__":
    main()
