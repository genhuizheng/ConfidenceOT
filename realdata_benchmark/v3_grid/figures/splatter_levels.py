"""Splatter levels L1-L5 drawn from the generated data: what each level adds, and the evaluation cases.

    python realdata_benchmark/v3_grid/figures/splatter_levels.py \
        --root results/splatter_local_20261103 --replicate 1 --out results/v3_grid_20261104/figures/splatter

The redraw of scripts/make_nuisance_umap.py's problem.png and benchmark.png (deck 2026-10-15, slides 2-3) for the
L1-L5 design. Those were drawn from a toy sampler with three levels; these read the actual Splatter output of the
local root (20_generate_n5000.R --param-set splatter_default --pairing paired, N = 200 per side; the N = 5,000 TACC
run uses the same generator and parameters), the all_shared pair of each level, replicate 1.

Writes:
  levels_umap.{png,pdf}       per level, source + target embedded together (logcpm -> PCA -> UMAP, the project's
                              Preprocessing, no correction): coloured by side, by Splatter group (perturbed target
                              Group1 cells ringed), by genes detected per cell
  levels_benchmark.{png,pdf}  per level, 30 source | 30 target cells ordered by group then nCount: group, nCount and
                              nFeature strips, the counts of 60 fixed genes; below, the three cases (gates drawn, not run)
  levels_ratios.tsv           per level, target / source medians of nCount and nFeature over all cells of the pair
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
from make_journal_figure import STYLE  # noqa: E402

LEVELS = ["L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance"]
TITLE = ["L1  depth", "L2  + dropout", "L3  + batch", "L4  + perturbation", "L5  + imbalance"]
GROUPS = [f"Group{i}" for i in range(1, 6)]
G_COLOUR = dict(zip(GROUPS, ["#d95f02", "#1f78b4", "#1b9e77", "#e6ab02", "#e7298a"]))
C_SOURCE, C_TARGET = "#1f6f8b", "#b07d2b"
C_COUNT, C_FEATURE = "#b07d2b", "#1f6f8b"
C_KEEP, C_DROP = "#2f7d4f", "#c4553b"
FRAME = "#d8d8d2"; INK2 = "#55555a"


def load(root: Path, rep: int, level: str):
    d = next((root / "conditions" / f"rep{rep}" / level).glob("*_all_shared"))
    truth = pd.read_csv(root / "generated" / f"rep{rep}" / level / "cells_truth.csv").set_index("cell_id")
    out = {}
    for side in ("source", "target"):
        a = ad.read_h5ad(d / f"{side}.h5ad")
        X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        obs = a.obs[["cell_id", "group"]].copy()
        obs["perturbed"] = truth.reindex(obs.cell_id).perturbed.astype(str).str.upper().eq("TRUE").to_numpy()
        out[side] = (X, obs)
    return out


def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(FRAME); s.set_linewidth(0.6)


def embed(src: np.ndarray, tgt: np.ndarray, seed: int) -> np.ndarray:
    import umap
    from confidenceot import Preprocessing
    rep = Preprocessing.from_label("logcpm").representation(src, tgt, seed=seed)
    joint = np.vstack([np.asarray(rep.source), np.asarray(rep.target)])
    return umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=seed, verbose=False).fit_transform(joint)


def umap_figure(data, out: Path, seed: int) -> None:
    det_all = np.concatenate([np.log10((data[l][s][0] > 0).sum(1)) for l in LEVELS for s in ("source", "target")])
    vmin, vmax = np.percentile(det_all, [1, 99])
    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(11.0, 6.6))
        gs = fig.add_gridspec(3, 5, left=0.06, right=0.90, top=0.92, bottom=0.03, wspace=0.06, hspace=0.10)
        for j, lv in enumerate(LEVELS):
            (xs, os_), (xt, ot) = data[lv]["source"], data[lv]["target"]
            P = embed(xs, xt, seed)
            side = np.r_[np.zeros(len(xs), bool), np.ones(len(xt), bool)]
            grp = np.r_[os_.group.to_numpy(), ot.group.to_numpy()]
            pert = np.r_[os_.perturbed.to_numpy(), ot.perturbed.to_numpy()]
            det = np.log10(np.r_[(xs > 0).sum(1), (xt > 0).sum(1)])
            order = np.random.default_rng(seed).permutation(len(P))  # neither side drawn on top of the other
            for i in range(3):
                ax = fig.add_subplot(gs[i, j]); frame(ax)
                if i == 0:
                    ax.scatter(P[order, 0], P[order, 1], s=7, lw=0, c=np.where(side[order], C_TARGET, C_SOURCE))
                    ax.set_title(TITLE[j], fontsize=9, fontweight="semibold", pad=5)
                elif i == 1:
                    ax.scatter(P[order, 0], P[order, 1], s=7, lw=0, c=[G_COLOUR[g] for g in grp[order]])
                    if pert.any():
                        ax.scatter(P[pert, 0], P[pert, 1], s=16, facecolors="none", edgecolors="#222222", lw=0.45)
                else:
                    sc = ax.scatter(P[order, 0], P[order, 1], s=7, lw=0, c=det[order], cmap="viridis", vmin=vmin, vmax=vmax)
                if j == 0:
                    ax.set_ylabel(["side", "Splatter group", "genes detected"][i], fontsize=8.5, color=INK2, labelpad=6)
        # legends
        lx = 0.915
        for k, (lab, c) in enumerate((("source", C_SOURCE), ("target", C_TARGET))):
            fig.text(lx + 0.018, 0.86 - k * 0.03, lab, fontsize=7.5, va="center")
            fig.patches.append(mpl.patches.Circle((lx + 0.007, 0.86 - k * 0.03), 0.004, color=c, transform=fig.transFigure))
        for k, g in enumerate(GROUPS):
            fig.text(lx + 0.018, 0.60 - k * 0.028, g, fontsize=7.5, va="center")
            fig.patches.append(mpl.patches.Circle((lx + 0.007, 0.60 - k * 0.028), 0.004, color=G_COLOUR[g], transform=fig.transFigure))
        fig.patches.append(mpl.patches.Circle((lx + 0.007, 0.60 - 5 * 0.028), 0.005, fill=False, ec="#222222", lw=0.6, transform=fig.transFigure))
        fig.text(lx + 0.018, 0.60 - 5 * 0.028, "perturbed\n(target Group1)", fontsize=7.0, va="center")
        cax = fig.add_axes([lx + 0.004, 0.06, 0.010, 0.22])
        cb = fig.colorbar(sc, cax=cax); cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=6.5)
        cb.set_label("log10 genes detected", fontsize=7)
        for suf in ("png", "pdf"):
            fig.savefig(out / f"levels_umap.{suf}", bbox_inches="tight", dpi=300)
        plt.close(fig)
    print("  wrote levels_umap")


def pick(X: np.ndarray, obs: pd.DataFrame, n: int, rng) -> np.ndarray:
    """n cells drawn at random (so the group mix follows the side's proportions), ordered by group then nCount."""
    ix = rng.choice(len(X), n, replace=False)
    tot = X[ix].sum(1)
    g = obs.group.to_numpy()[ix]
    return ix[np.lexsort((tot, [GROUPS.index(v) for v in g]))]


def benchmark_figure(data, out: Path, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_show, n_genes = 30, 60
    xs1 = data[LEVELS[0]]["source"][0]
    genes = np.argsort(-xs1.mean(0))[:n_genes * 16:16]
    panels, ratios = {}, []
    for lv in LEVELS:
        (xs, os_), (xt, ot) = data[lv]["source"], data[lv]["target"]
        si, ti = pick(xs, os_, n_show, rng), pick(xt, ot, n_show, rng)
        panels[lv] = {
            "matrix": [np.log1p(xs[np.ix_(si, genes)]).T, np.log1p(xt[np.ix_(ti, genes)]).T],
            "total": [np.log10(xs[si].sum(1)), np.log10(xt[ti].sum(1))],
            "detected": [np.log10((xs[si] > 0).sum(1)), np.log10((xt[ti] > 0).sum(1))],
            "group": [os_.group.to_numpy()[si], ot.group.to_numpy()[ti]]}
        ratios.append({"level": lv,
                       "nCount_target_over_source": float(np.median(xt.sum(1)) / np.median(xs.sum(1))),
                       "nFeature_target_over_source": float(np.median((xt > 0).sum(1)) / np.median((xs > 0).sum(1))),
                       "target_group_share": " / ".join(f"{(ot.group == g).mean():.2f}" for g in GROUPS)})
    lim = {k: (min(min(v.min() for v in p[k]) for p in panels.values()), max(max(v.max() for v in p[k]) for p in panels.values()))
           for k in ("total", "detected")}
    top = float(np.percentile(np.concatenate([m.ravel() for p in panels.values() for m in p["matrix"]]), 99.5))
    gcmap = mpl.colors.ListedColormap([G_COLOUR[g] for g in GROUPS])

    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(11.0, 6.6))
        cw, gap, x0s = 0.158, 0.006, [0.105 + j * 0.178 for j in range(5)]
        hw = (cw - gap) / 2
        for j, lv in enumerate(LEVELS):
            p, x0 = panels[lv], x0s[j]
            r = ratios[j]
            fig.text(x0 + cw / 2, 0.975, TITLE[j], fontsize=9, fontweight="semibold", ha="center", va="top")
            fig.text(x0 + cw / 2, 0.945, f"target / source  nCount ×{r['nCount_target_over_source']:.2f}",
                     fontsize=6.8, color=C_COUNT, ha="center", va="top")
            fig.text(x0 + cw / 2, 0.925, f"nFeature ×{r['nFeature_target_over_source']:.2f}",
                     fontsize=6.8, color=C_FEATURE, ha="center", va="top")
            for h in range(2):
                xx = x0 + h * (hw + gap)
                fig.text(xx + hw / 2, 0.900, ("source", "target")[h], fontsize=7, color=(C_SOURCE, C_TARGET)[h], ha="center", va="top")
                rows = (("group", None), ("total", C_COUNT), ("detected", C_FEATURE))
                for line, (key, colour) in enumerate(rows):
                    ax = fig.add_axes([xx, 0.856 - 0.028 * line, hw, 0.018]); frame(ax)
                    if key == "group":
                        ax.imshow(np.array([GROUPS.index(g) for g in p["group"][h]])[None, :], aspect="auto", cmap=gcmap,
                                  vmin=-0.5, vmax=4.5, interpolation="nearest")
                    else:
                        ax.imshow(p[key][h][None, :], aspect="auto", cmap="viridis", vmin=lim[key][0], vmax=lim[key][1],
                                  interpolation="nearest")
                    if j == 0 and h == 0:
                        ax.set_ylabel(["group", "nCount", "nFeature"][line], fontsize=7, color=colour or INK2, rotation=0,
                                      ha="right", va="center", labelpad=6)
                ax = fig.add_axes([xx, 0.548, hw, 0.235]); frame(ax)
                img = ax.imshow(p["matrix"][h], aspect="auto", cmap="Blues", vmin=0, vmax=top, interpolation="nearest")
                if j == 0 and h == 0:
                    ax.set_ylabel("expression, 60 genes", fontsize=7, color=INK2, labelpad=6)
                    ax.set_xlabel("by group, then nCount", fontsize=6.6, color=INK2, labelpad=3)

        # the three cases (gates drawn, not run), as in the 2026-10-15 figure
        n = 48
        case_w, case_x = 0.27, (0.105, 0.405, 0.705)
        gate_map = mpl.colors.ListedColormap([C_DROP, C_KEEP])
        unmatched = np.zeros(n, dtype=bool); unmatched[8:16] = True
        shared_gate = np.ones(n, dtype=bool); shared_gate[[2, 3, 29, 41]] = False
        unmatched_gate = ~unmatched; unmatched_gate[[6, 7]] = False; unmatched_gate[15] = True
        disjoint_gate = np.zeros(n, dtype=bool); disjoint_gate[20:28] = True; disjoint_gate[[5, 38]] = True
        for c, (title, gate, truth) in enumerate((
                ("Case 1: all populations shared", shared_gate, np.ones(n, dtype=bool)),
                ("Case 2: one population unmatched", unmatched_gate, ~unmatched),
                ("Case 3: no population shared", disjoint_gate, np.zeros(n, dtype=bool)))):
            x0 = case_x[c]
            fig.text(x0, 0.455, title, fontsize=8, fontweight="semibold", ha="left", va="top")
            for line, v in enumerate((gate, truth)):
                y = 0.390 - 0.050 * line
                ax = fig.add_axes([x0, y, case_w, 0.030]); frame(ax)
                ax.imshow(v.astype(float)[None, :], aspect="auto", cmap=gate_map, vmin=0, vmax=1, interpolation="nearest")
                if c == 0:
                    fig.text(x0 - 0.008, y + 0.015, ("example gate", "ground truth")[line], fontsize=7.5, color=INK2,
                             ha="right", va="center")

        # legends
        cb = fig.colorbar(img, cax=fig.add_axes([0.105, 0.215, 0.10, 0.016]), orientation="horizontal")
        ticks = [v for v in (0, 1, 3, 10, 30, 100) if v <= float(np.expm1(top))]
        cb.set_ticks(np.log1p(ticks)); cb.set_ticklabels([str(v) for v in ticks]); cb.ax.tick_params(labelsize=6.5, length=2)
        cb.outline.set_linewidth(0.5)
        fig.text(0.215, 0.223, "counts (white = 0)", fontsize=7.5, color=INK2, va="center")
        for k, (c, t) in enumerate(((C_KEEP, "kept"), (C_DROP, "rejected"))):
            ax = fig.add_axes([0.36 + 0.09 * k, 0.215, 0.014, 0.016]); frame(ax); ax.set_facecolor(c)
            fig.text(0.378 + 0.09 * k, 0.223, t, fontsize=7.5, color=INK2, va="center")
        for k, g in enumerate(GROUPS):
            ax = fig.add_axes([0.56 + 0.07 * k, 0.215, 0.014, 0.016]); frame(ax); ax.set_facecolor(G_COLOUR[g])
            fig.text(0.578 + 0.07 * k, 0.223, g, fontsize=7.5, color=INK2, va="center")
        cb2 = fig.colorbar(mpl.cm.ScalarMappable(cmap="viridis"), cax=fig.add_axes([0.105, 0.150, 0.10, 0.016]),
                           orientation="horizontal")
        cb2.set_ticks([]); cb2.outline.set_linewidth(0.5)
        fig.text(0.215, 0.158, "nCount / nFeature (log10, low to high, one scale across levels)", fontsize=7.5, color=INK2,
                 va="center")
        for suf in ("png", "pdf"):
            fig.savefig(out / f"levels_benchmark.{suf}", bbox_inches="tight", dpi=300)
        plt.close(fig)
    print("  wrote levels_benchmark")
    return pd.DataFrame(ratios)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--replicate", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261105)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = {lv: load(a.root, a.replicate, lv) for lv in LEVELS}
    umap_figure(data, a.out, a.seed)
    r = benchmark_figure(data, a.out, a.seed)
    r.to_csv(a.out / "levels_ratios.tsv", sep="\t", index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
