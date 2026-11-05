"""Splatter N = 5,000 v3 grid: heatmaps of the three readouts, one quantity per heatmap, best configuration first.

    python realdata_benchmark/v3_grid/figures/splatter_heatmaps.py --summary GRID_SUMMARY_v3tacc.tsv --out OUTDIR

Rows are preprocessing configurations, ranked by Case 2 Rejection F1 at L5 (batch + perturbation + imbalance, the
hardest level; user 2026-11-05), ties broken by the mean Case 2 F1 over L1-L5. Best at the top. Columns
are the five Splatter levels plus that mean. Every cell is the mean over 3 replicates (and, for Case 2, over the
source-unmatched and target-unmatched pairs) of the effective metric of summarize_local.py.

Panels (one PDF + PNG each):
  splatter_f1_top30       Case 2 Rejection F1, the 30 best configurations and the production baseline
  splatter_frr_top30      Case 1 false rejection rate, same rows
  splatter_recall_top30   Case 3 unmatched recall, same rows
  splatter_f1_all296      Case 2 F1 of every configuration, same ranking, with the four grid factors as strips
Also writes splatter_ranking.tsv (every configuration, its rank, L5 and L1-L5 mean of the three metrics).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbt_style as st  # noqa: E402

LEVELS = ["L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance"]
LEVEL_LABEL = ["L1\ndepth", "L2\n+dropout", "L3\n+batch", "L4\n+perturb", "L5\n+imbalance"]
BASELINE = "ranknm256-production|none|noscale|cos|pca30"
METRICS = {  # column, label, colour ramp end, higher is better
    "f1": ("case2_f1_eff", "Case 2 Rejection F1", "#1F4E79", True),
    "frr": ("case1_frr_eff", "Case 1 false rejection rate", "#8E4A12", False),
    "recall": ("case3_recall_eff", "Case 3 unmatched recall", "#2F6B4F", True),
}


def ramp(end: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("r", ["#F6F7F8", end])


def label(cid: str) -> str:
    enc, _, scale, cost, pca = cid.split("|")
    key, _, ties = enc.partition("-")
    parts = [key] + ([ties] if ties else []) + [scale, cost, pca.replace("pca", "PCA ")]
    return "  ".join(parts)


# label table: (header, x in inches from the left edge, alignment)
LABEL_COLS = [("rank", 0.34, "right"), ("encoding", 0.44, "left"), ("ties", 1.30, "left"), ("scaling", 1.86, "left"),
              ("cost", 2.33, "left"), ("PCA", 2.62, "left")]
LAB_W = 2.92


def label_fields(cid: str, rank: int) -> list[str]:
    enc, _, scale, cost, pca = cid.split("|")
    key, _, ties = enc.partition("-")
    return [str(rank), key, ties or "-", scale, cost, pca.replace("pca", "")]


def label_table(sh, y0: float, height: float, n: int, rows: list, colour_rows: set, top_y: float) -> None:
    """Row labels as aligned columns beside the heatmap; row i at the heatmap's row i (None = blank separator)."""
    lx = sh.ax(0, y0, LAB_W, height)
    lx.set_xlim(0, LAB_W); lx.set_ylim(n - 0.5, -0.5); lx.axis("off")
    for i, r in enumerate(rows):
        if r is None:
            continue
        cid, rank = r
        colour = st.PAL["model"] if cid in colour_rows else st.INK
        for (_, x, ha), s in zip(LABEL_COLS, label_fields(cid, rank)):
            lx.text(x, i, s, ha=ha, va="center", fontsize=st.FS_MIN, color=colour)
    for name, x, ha in LABEL_COLS:
        sh.text(x, top_y, name, ha=ha, va="bottom", fontsize=st.FS_SUB, color=st.SUBTLE)


def colorbar(sh, cmap, vmax: float, x: float, y: float, title: str, w: float = 1.2) -> None:
    cb = sh.ax(x, y, w, 0.07)
    cb.imshow(np.linspace(0, vmax, 256)[None, :], cmap=cmap, vmin=0, vmax=vmax, aspect="auto",
              extent=(0, vmax, 0, 1))
    cb.set_yticks([]); cb.set_xticks([0, vmax / 2, vmax]); cb.set_xticklabels([f"{v:g}" for v in (0, vmax / 2, vmax)],
                                                                             fontsize=st.FS_MIN)
    for s in cb.spines.values():
        s.set_visible(False)
    cb.tick_params(length=1.5, width=st.LW_AXIS)
    sh.text(x, y + 0.11, title, va="bottom", ha="left", fontsize=st.FS_MIN, color=st.SUBTLE)


def outline(ax, n: int) -> None:
    """Frame the L5 column, the one the rows are ranked by."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((3.5, -0.5), 1, n, fill=False, ec=st.PAL["peptide"], lw=1.2, clip_on=False, zorder=5))
    ax.text(4, -0.5, "ranked by\n\n\n", ha="center", va="bottom", fontsize=st.FS_MIN, color=st.PAL["peptide"],
            transform=ax.transData, clip_on=False)


def heat(ax, M: np.ndarray, cmap, vmax: float, annotate: bool, fs: float) -> None:
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                dark = v / vmax > 0.6
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=fs, color="white" if dark else st.INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def top_panel(t: pd.DataFrame, rows: list[str], metric: str, out: Path, n_top: int) -> None:
    col, title, end, _ = METRICS[metric]
    p = t.pivot(index="config_id", columns="level", values=col)[LEVELS]
    p["mean"] = p.mean(1)
    M = p.loc[rows].to_numpy()
    vmax = 1.0 if metric != "frr" else max(0.3, float(np.ceil(np.nanmax(M) * 10) / 10))
    rh, cw = 0.118, 0.46
    # the top block, a blank separator row, then the baseline row; the mean column set apart by a blank column
    img = np.vstack([M[:n_top], np.full((1, 6), np.nan), M[n_top:]]) if len(rows) > n_top else M
    img = np.insert(img, 5, np.nan, axis=1)
    n = img.shape[0]
    y0, top = 0.66, 0.62
    H = y0 + rh * n + top
    W = LAB_W + cw * 7 + 0.10
    sh = st.Sheet(W, H)
    ax = sh.ax(LAB_W, y0, cw * 7, rh * n)
    cm = ramp(end).copy(); cm.set_bad("white")
    heat(ax, np.ma.masked_invalid(img), cm, vmax, False, st.FS_MIN)
    for i in range(n):
        for j in range(img.shape[1]):
            v = img[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=st.FS_MIN,
                        color="white" if v / vmax > 0.6 else st.INK)
    if metric == "f1":
        outline(ax, n)
    ax.set_yticks([])
    ax.set_xticks([0, 1, 2, 3, 4, 6]); ax.set_xticklabels(LEVEL_LABEL + ["mean\nL1-L5"], fontsize=st.FS_SUB)
    ax.xaxis.tick_top()
    lab_rows = [(r, t_rank[r]) for r in rows[:n_top]] + ([None] + [(r, t_rank[r]) for r in rows[n_top:]] if len(rows) > n_top else [])
    label_table(sh, y0, rh * n, n, lab_rows, {BASELINE}, y0 + rh * n + 0.06)
    sh.text(LAB_W + cw * 3.5, H - 0.08, title, ha="center", va="top", fontsize=st.FS_TITLE)
    colorbar(sh, cm, vmax, LAB_W + cw * 3.6, 0.42, title + (" (lower is better)" if metric == "frr" else ""))
    sh.text(0.10, 0.24, "Rows ranked by Case 2 F1 at L5 (outlined column; rank 1 = best of 296), ties by the L1-L5 "
            "mean; red row: production baseline.", ha="left", va="top", fontsize=st.FS_SUB, color=st.SUBTLE)
    sh.text(0.10, 0.24 - st.leading(st.FS_SUB), "Each cell: mean over 3 replicates (Case 2 also over the "
            "source- and target-unmatched pairs).", ha="left", va="top", fontsize=st.FS_SUB, color=st.SUBTLE)
    sh.save(out, f"splatter_{metric}_top{n_top}")


t_rank: dict[str, int] = {}


def cases_panel(t: pd.DataFrame, rows: list[str], out: Path, n_top: int) -> None:
    """Case 1, Case 2 and Case 3 side by side, L1-L5 each, same rows; one colour ramp and colour bar per case."""
    rh, cw, gap = 0.118, 0.40, 0.22
    blocks = [("frr", "Case 1: every population shared", "false rejection rate (lower is better)"),
              ("f1", "Case 2: one side has an unmatched population", "Rejection F1"),
              ("recall", "Case 3: no population shared", "unmatched recall")]
    n = len(rows) + (1 if len(rows) > n_top else 0)
    y0, top = 0.78, 0.78
    W = LAB_W + 3 * cw * 5 + 2 * gap + 0.10
    H = y0 + rh * n + top
    sh = st.Sheet(W, H)
    for k, (metric, head, sub) in enumerate(blocks):
        col, _, end, _ = METRICS[metric]
        M = t.pivot(index="config_id", columns="level", values=col)[LEVELS].loc[rows].to_numpy()
        vmax = 1.0 if metric != "frr" else 0.4
        img = np.vstack([M[:n_top], np.full((1, 5), np.nan), M[n_top:]]) if len(rows) > n_top else M
        x = LAB_W + k * (cw * 5 + gap)
        ax = sh.ax(x, y0, cw * 5, rh * n)
        cm = ramp(end).copy(); cm.set_bad("white")
        heat(ax, np.ma.masked_invalid(img), cm, vmax, False, st.FS_MIN)
        for i in range(n):
            for j in range(5):
                v = img[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=st.FS_MIN,
                            color="white" if v / vmax > 0.6 else st.INK)
        if metric == "f1":
            outline(ax, n)
        ax.set_yticks([]); ax.set_xticks(range(5)); ax.set_xticklabels(LEVEL_LABEL, fontsize=st.FS_SUB); ax.xaxis.tick_top()
        sh.text(x + cw * 2.5, H - 0.08, head, ha="center", va="top", fontsize=st.FS_TITLE)
        sh.text(x + cw * 2.5, H - 0.08 - st.leading(st.FS_TITLE), sub, ha="center", va="top", fontsize=st.FS_SUB,
                color=st.SUBTLE)
        colorbar(sh, cm, vmax, x + cw * 1.0, 0.48, sub, w=cw * 3)
    lab_rows = [(r, t_rank[r]) for r in rows[:n_top]] + ([None] + [(r, t_rank[r]) for r in rows[n_top:]] if len(rows) > n_top else [])
    label_table(sh, y0, rh * n, n, lab_rows, {BASELINE}, y0 + rh * n + 0.06)
    sh.text(0.10, 0.26, "Rows ranked by Case 2 F1 at L5 (outlined; rank 1 = best of 296), ties by the L1-L5 mean Case 2 "
            "F1; red row: production baseline.", ha="left", va="top", fontsize=st.FS_SUB, color=st.SUBTLE)
    sh.text(0.10, 0.26 - st.leading(st.FS_SUB), "Each cell: mean over 3 replicates; Case 2 also over its source-unmatched "
            "and target-unmatched pairs.", ha="left", va="top", fontsize=st.FS_SUB, color=st.SUBTLE)
    sh.save(out, f"splatter_all_cases_top{n_top}")


def all_panel(t: pd.DataFrame, order: list[str], out: Path) -> None:
    p = t.pivot(index="config_id", columns="level", values="case2_f1_eff")[LEVELS]
    p["mean"] = p.mean(1)
    M = p.loc[order].to_numpy()
    rh, cw = 0.016, 0.46
    x0, y0 = 1.75, 0.80
    W, H = x0 + cw * 7 + 0.75, y0 + rh * len(order) + 0.62
    sh = st.Sheet(W, H)
    ax = sh.ax(x0, y0, cw * 7, rh * len(order))
    img = np.insert(M, 5, np.nan, axis=1)
    cm = ramp(METRICS["f1"][2]).copy(); cm.set_bad("white")
    heat(ax, np.ma.masked_invalid(img), cm, 1.0, False, st.FS_MIN)
    outline(ax, len(order))
    ax.set_xticks([0, 1, 2, 3, 4, 6]); ax.set_xticklabels(LEVEL_LABEL + ["mean\nL1-L5"], fontsize=st.FS_SUB)
    ax.xaxis.tick_top()
    ticks = [0, 49, 99, 149, 199, 249, len(order) - 1]
    ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
    ax.set_yticks(ticks); ax.set_yticklabels([str(k + 1) for k in ticks], fontsize=st.FS_TICK)
    ax.set_ylabel("rank (Case 2 F1 at L5)", fontsize=st.FS_AXLAB)
    b = order.index(BASELINE)
    ax.plot([-0.75], [b], marker=">", ms=4, color=st.PAL["model"], clip_on=False)
    colorbar(sh, cm, 1.0, x0 + cw * 2, 0.30, METRICS["f1"][1])
    # factor strips: cost, PCA, scaling, encoding family
    parts = [c.split("|") for c in order]
    fam = [x[0].split("-")[0].rstrip("0123456789") for x in parts]
    strips = [("cost", [x[3] for x in parts], {"cos": "#4A6FA5", "sqeuc": "#D9DDE1"}),
              ("PCA", [x[4] for x in parts], {"pca30": "#4A6FA5", "pca50": "#D9DDE1"}),
              ("scaling", [x[2] for x in parts], {"noscale": "#4A6FA5", "scale": "#D9DDE1"}),
              ("encoding", fam, {"logcpm": "#2C6E9B", "ranknm": "#B4702A", "rank_value": "#4E8A6B", "rankhvg": "#7D3C98"})]
    for k, (name, vals, colours) in enumerate(strips):
        sx = sh.ax(0.20 + k * 0.36, y0, 0.24, rh * len(order))
        keys = list(colours)
        sx.imshow(np.array([[keys.index(v)] for v in vals]), cmap=ListedColormap([colours[x] for x in keys]),
                  vmin=-0.5, vmax=len(keys) - 0.5, aspect="auto", interpolation="nearest")
        sx.set_xticks([0]); sx.set_xticklabels([name], rotation=90, fontsize=st.FS_MIN); sx.xaxis.tick_top()
        sx.set_yticks([])
        for s in sx.spines.values():
            s.set_visible(False)
        sx.tick_params(length=0)
    legend = ["cost: dark cos / light sqeuc", "PCA: dark 30 / light 50", "scaling: dark noscale / light scale",
              "encoding: blue logcpm, orange ranknm,", "green rank_value, purple rankhvg",
              f"red triangle: production baseline (rank {b + 1})"]
    for k, s in enumerate(legend):
        sh.text(0.20, y0 - 0.08 - k * st.leading(st.FS_MIN), s, fontsize=st.FS_MIN, color=st.SUBTLE, va="top")
    sh.text(x0 + cw * 3.5, H - 0.10, "Case 2 Rejection F1, all 296 configurations", ha="center", va="top",
            fontsize=st.FS_TITLE)
    sh.save(out, "splatter_f1_all296")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top", type=int, default=30)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st.apply()
    t = pd.read_csv(a.summary, sep="\t")
    t = t[t.level.isin(LEVELS)]
    cols = ["case2_f1_eff", "case1_frr_eff", "case3_recall_eff"]
    l5 = t[t.level == LEVELS[-1]].set_index("config_id")[cols].add_prefix("L5_")
    m = l5.join(t.groupby("config_id")[cols].mean().add_prefix("mean_"))
    m = m.sort_values(["L5_case2_f1_eff", "mean_case2_f1_eff"], ascending=False)
    order = list(m.index)
    t_rank.update({c: k + 1 for k, c in enumerate(order)})
    m.insert(0, "rank", range(1, len(m) + 1))
    m.round(4).to_csv(a.out / "splatter_ranking.tsv", sep="\t")
    rows = order[:a.top] + ([BASELINE] if BASELINE not in order[:a.top] else [])
    for metric in METRICS:
        top_panel(t, rows, metric, a.out, a.top)
    cases_panel(t, rows, a.out, a.top)
    all_panel(t, order, a.out)


if __name__ == "__main__":
    main()
