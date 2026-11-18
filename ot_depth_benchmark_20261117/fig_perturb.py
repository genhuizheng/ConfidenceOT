"""Dataset B figures and tables (PLAN.md), one PDF + PNG per panel, nbt style.

    python ot_depth_benchmark_20261117/fig_perturb.py --root results/ot_depth_benchmark_20261117/norman [--tag ""]

Reads METRICS<tag>.tsv and GATES<tag>.tsv.gz of run_perturb.py; writes <root>/figures and <root>/tables.
Units are perturbations; the control-vs-control unit is the technical reference and is drawn as a line, not a point.
No coupling overlap is reported as accuracy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "realdata_benchmark" / "v3_grid" / "figures"))
sys.path.insert(0, str(HERE))
import nbt_style as ns  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from fig_splatter import COL, LABEL, SOLVERS  # noqa: E402

COSTNAME = {"C1": "C1 rank-cosine", "C2": "C2 log1p-PCA"}


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def strip(d, q, ylab, title, out, name, ref=None, solvers=SOLVERS, hline=None):
    """Per-perturbation values of one quantity, one column per cost x solver; the negative control as a short bar."""
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    rng = np.random.default_rng(0)
    pos, labels = 0, []
    for cost in ("C1", "C2"):
        for s in solvers:
            v = d[(d.cost == cost) & (d.solver == s) & (d.kind == "perturbation")][q].dropna().to_numpy()
            if len(v):
                ax.scatter(pos + rng.uniform(-0.18, 0.18, len(v)), v, s=2.5, color=COL[s], alpha=0.55, lw=0)
                ax.plot([pos - 0.28, pos + 0.28], [np.median(v)] * 2, color=ns.INK, lw=ns.LW_MAIN)
            if ref is not None:
                r = ref[(ref.cost == cost) & (ref.solver == s)][q].dropna()
                if len(r):
                    ax.plot([pos - 0.3, pos + 0.3], [r.mean()] * 2, color=ns.RULE, lw=0.9, ls=ns.DASH)
            labels.append(f"{cost} {LABEL[s].replace('Independent a×b', 'Indep.')}")
            pos += 1
        pos += 0.6
    ticks = [i + (0.6 if i >= len(solvers) else 0) for i in range(2 * len(solvers))]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=60, ha="right")
    if hline is not None:
        ax.axhline(hline, **ns.REF_KW)
    ax.set_ylabel(ylab); ax.set_title(title); ns.tidy(ax)
    save(fig, out, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--name", default=None)
    a = p.parse_args()
    ns.apply()
    tag = f"_{a.tag}" if a.tag else ""
    name = a.name or a.root.name
    m = pd.read_csv(a.root / f"METRICS{tag}.tsv", sep="\t")
    out, tab = a.root / "figures", a.root / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    ok = m[m.status == "ok"]
    neg = ok[ok.kind == "negative_control"]
    pert = ok[ok.kind == "perturbation"]
    # failures are listed, never dropped silently
    m[m.status != "ok"][["unit", "control_ref", "cost", "solver", "status"]].to_csv(tab / "FAILED_FITS.tsv", sep="\t", index=False)
    ref_a = pert[pert.control_ref == "A"]
    cols = ["transported_mass", "depth_corr_count", "depth_corr_feature", "depth_gap_ratio", "ite_depth_r2", "ite_owndepth_corr",
            "heldout_energy", "heldout_energy_sampled", "heldout_de_corr", "rejected_fraction_source", "rejected_fraction_target",
            "rejection_depth_corr_target", "rejection_score_corr_target"]
    cols = [c for c in cols if c in ok]
    summ = ref_a.groupby(["cost", "solver"])[cols].median().round(4)
    summ.to_csv(tab / "SUMMARY_median_over_perturbations_refA.tsv", sep="\t")
    neg.groupby(["cost", "solver"])[cols].mean().round(4).to_csv(tab / "NEGATIVE_CONTROL.tsv", sep="\t")
    # 1 coupling-depth association
    for q, ylab in (("depth_corr_count", "Coupling-weighted corr.\nof log nCount"), ("depth_corr_feature", "Coupling-weighted corr.\nof log nFeature")):
        strip(ref_a, q, ylab, f"{name}: control → perturbed (dashed: control vs control)", out, f"B_{q}", ref=neg, hline=0)
    # 2 depth-biology association in the data (one value per perturbation; solver-independent)
    u = ref_a[(ref_a.cost == "C2") & (ref_a.solver == "independent")].drop_duplicates("unit")
    u[["unit", "n_target_train", "shift_norm", "depth_shift_log2", "nfeature_shift_log2", "depth_state_corr", "nfeature_state_corr"] +
      [c for c in ("target_gene_log2fc", "target_gene_depth_corr", "target_gene_score_corr") if c in u]].to_csv(tab / "DEPTH_BIOLOGY_per_perturbation.tsv", sep="\t", index=False)
    for q, xl in (("depth_shift_log2", "log2 median nCount, perturbed / control"), ("depth_state_corr", "Spearman: log nCount vs perturbation score"),
                  ("target_gene_depth_corr", "Spearman: log nCount vs target-gene expression")):
        if q not in u or u[q].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(2.2, 1.7))
        ax.hist(u[q].dropna(), bins=30, color=ns.NEUTRAL, edgecolor="white", lw=0.3)
        ax.axvline(0, **ns.REF_KW)
        ax.set_xlabel(xl); ax.set_ylabel("Perturbations"); ax.set_title(f"{name}: depth vs biology"); ns.tidy(ax)
        save(fig, out, f"B_data_{q}")
    # depth shift vs effect size
    fig, ax = plt.subplots(figsize=(2.2, 1.9))
    ax.scatter(u.shift_norm, u.depth_shift_log2, s=4, color=ns.INK, lw=0, alpha=0.7)
    ax.axhline(0, **ns.REF_KW)
    ax.set_xlabel("Perturbation effect size (|mean shift|, PCA)"); ax.set_ylabel("log2 depth ratio\nperturbed / control")
    ax.set_title(f"{name}: strong perturbations change depth"); ns.tidy(ax)
    save(fig, out, "B_data_effect_vs_depth_shift")
    # 3 technical leakage into per-cell treatment effects
    strip(ref_a, "ite_depth_r2", "R² of ITE explained by\nnCount / nFeature terms", f"{name}: technical leakage into ITE", out, "B_ite_depth_r2",
          ref=neg, solvers=[s for s in SOLVERS])
    # 4 held-out distribution metrics relative to the mean-shift baseline (lower is better; 1 = mean shift)
    bl = ref_a[ref_a.cost == "none"].pivot_table(index="unit", columns="solver", values="heldout_energy")
    rel = ref_a[ref_a.cost != "none"].merge(bl["baseline_mean_shift"].rename("ms"), left_on="unit", right_index=True)
    for q in ("heldout_energy", "heldout_energy_sampled"):
        if q in rel:
            rel[f"{q}_rel"] = rel[q] / rel.ms
            strip(rel, f"{q}_rel", "Held-out energy distance\n/ mean-shift baseline", f"{name}: held-out prediction ({q.split('_')[-1]})",
                  out, f"B_{q}_relative", hline=1)
    bls = ref_a[ref_a.cost == "none"].groupby("solver")[["heldout_energy", "heldout_de_corr"]].median().round(4)
    bls.to_csv(tab / "HELDOUT_BASELINES_median.tsv", sep="\t")
    # 5 gate behaviour (ConfidenceOT only), one quantity per panel
    g = ref_a[ref_a.solver == "confidenceot"]
    for q, yl in (("rejected_fraction_target", "Rejected fraction of\nperturbed cells"), ("rejected_fraction_source", "Rejected fraction of\ncontrol cells"),
                  ("rejection_score_corr_target", "Spearman: rejection vs\nperturbation score"), ("rejection_depth_corr_target", "Spearman: rejection vs\nlog nCount (perturbed)")):
        if q in g:
            strip(g, q, yl, f"{name}: ConfidenceOT gate", out, f"B_gate_{q}", ref=neg, solvers=["confidenceot"], hline=0 if "corr" in q else None)
    # does the gate reject strong perturbations (no comparable control state) or shallow cells?
    if "rejection_score_corr_target" in g:
        fig, ax = plt.subplots(figsize=(2.2, 1.9))
        for cost, mk in (("C1", "o"), ("C2", "s")):
            d = g[g.cost == cost]
            ax.scatter(d.shift_norm, d.rejected_fraction_target, s=4, marker=mk, color=COL["confidenceot"] if cost == "C1" else ns.NEUTRAL, lw=0, label=COSTNAME[cost])
        ax.set_xlabel("Perturbation effect size (|mean shift|, PCA)"); ax.set_ylabel("Rejected fraction of\nperturbed cells")
        ax.legend(frameon=False); ax.set_title(f"{name}: gate vs effect size"); ns.tidy(ax)
        save(fig, out, "B_gate_vs_effect_size")
    # 6 replicate consistency: control reference A vs B, per perturbation
    rows = []
    for (cost, s), d in pert.groupby(["cost", "solver"]):
        w = d.pivot_table(index="unit", columns="control_ref", values="depth_corr_count")
        if {"A", "B"} <= set(w.columns) and len(w.dropna()) > 3:
            from scipy.stats import spearmanr
            rows.append({"cost": cost, "solver": s, "spearman_depth_corr_A_vs_B": spearmanr(w.A, w.B, nan_policy="omit").statistic,
                         "median_abs_diff": float((w.A - w.B).abs().median())})
    pd.DataFrame(rows).round(4).to_csv(tab / "REPLICATE_CONSISTENCY_depth_corr.tsv", sep="\t", index=False)
    gf = a.root / f"GATES{tag}.tsv.gz"
    if gf.exists():
        G = pd.read_csv(gf, sep="\t")
        G = G[(G.side == "target") & (G.unit != "control_vs_control")]
        jr = []
        for (unit, cost), d in G.groupby(["unit", "cost"]):
            w = d.pivot_table(index="cell_index", columns="control_ref", values="rejected", aggfunc="first")
            if {"A", "B"} <= set(w.columns):
                A_, B_ = w.A.astype(bool), w.B.astype(bool)
                un = (A_ | B_).sum()
                jr.append({"unit": unit, "cost": cost, "rejected_A": int(A_.sum()), "rejected_B": int(B_.sum()),
                           "jaccard": float((A_ & B_).sum() / un) if un else np.nan})
        J = pd.DataFrame(jr); J.to_csv(tab / "REPLICATE_CONSISTENCY_gate_jaccard.tsv", sep="\t", index=False)
    # 7 association vs harm across perturbations: depth corr vs held-out DE corr (C1 / C2, every solver)
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    for s in SOLVERS[1:]:
        d = ref_a[(ref_a.solver == s)]
        ax.scatter(d.depth_corr_count, d.heldout_de_corr, s=2.5, color=COL[s], lw=0, alpha=0.5)
    ax.set_xlabel("Coupling-weighted corr. of log nCount"); ax.set_ylabel("Held-out DE-shift corr.")
    ax.set_title(f"{name}: association vs prediction"); ns.tidy(ax)
    save(fig, out, "B_association_vs_heldout_de")
    print("->", out, tab)


if __name__ == "__main__":
    main()
