"""LTL331 (PDX) v3 grid figures from pdx_prepare.py's tables, one PDF + PNG per panel (+ one interactive HTML).

    python realdata_benchmark/v3_grid/figures/pdx_figures.py --data results/v3_grid_20261104/figures/pdx/data \
        --out results/v3_grid_20261104/figures/pdx [--config baseline splatter_l5_best]

Per shown configuration (M4-E):
  pdx_slices_<cfg>          8 time points side by side on the authors' UMAP; lines: retained source cell -> its
                            retained target of largest coupling mass (a weighted sample per pair)
  pdx_slices3d_<cfg>        the same, time points stacked in 3D
  pdx_slices3d_<cfg>.html   the same, interactive (rotate, zoom, hover)
  pdx_flow_<cfg>            state flow over time: per time point two bars of all its cells (as target, as source),
                            states split retained / rejected; bands: coupling mass of the retained block
  pdx_transitions_<cfg>     per pair: share of each source state's retained mass going to each target state
  pdx_rejected_<cfg>        rejected fraction per state at each time point, as source (forward pair) and as target
Over every configuration:
  pdx_rejected_median296 / pdx_rejected_iqr296   median and interquartile range over the 296 configurations
  pdx_composition           cell-state composition of each time point (input data)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbt_style as st  # noqa: E402
from pdx_prepare import STATES  # noqa: E402

SAMPLES = ["w00_preCX_1", "w02_2wkCX_2", "w04_4wkCX_1", "w08_8wkCX_1", "w12_12wkCX_2", "w16_16wkCX_1",
           "w20_20wkCX_1_R", "w22_22wkCX_2_R"]
TLABEL = ["wk 0\npre-CX", "wk 2", "wk 4", "wk 8", "wk 12", "wk 16", "wk 20\nregrowth", "wk 22\nregrowth"]
PAIRS = [f"LTL331__{a}__{b}" for a, b in zip(SAMPLES[:-1], SAMPLES[1:])]
# checked in OKLab under Machado CVD simulation: minimum pair distance 13.7 normal, 8.0 deutan, 11.0 protan,
# 13.9 tritan; every panel also names the states (legend), so colour is never the only key
COLOUR = dict(zip(STATES, ["#0B3C5D", "#7FB8E0", "#1B7F5B", "#C2DF8A", "#E8A33D", "#C0392B", "#3D1236", "#A884C9"]))
REJECT = "#D5D8DB"
NAME = {"baseline": "production baseline (ranknm256 production noscale cos PCA30)",
        "splatter_l5_best": "Splatter L5 best (ranknm256 midrank scale cos PCA50)",
        "logcpm": "logCPM (logcpm noscale cos PCA30; the baseline with only the encoding changed)"}
CID = {"baseline": "ranknm256-production|none|noscale|cos|pca30", "splatter_l5_best": "ranknm256-midrank|none|scale|cos|pca50",
       "logcpm": "logcpm|none|noscale|cos|pca30"}


def legend(sh, x, y, fs=st.FS_MIN, extra=()):
    step = st.leading(fs) * 1.15
    items = [(s, COLOUR[s]) for s in STATES] + list(extra)
    for k, (s, c) in enumerate(items):
        yy = y - k * step
        sh.fig.patches.append(Rectangle((x / sh.W, (yy - 0.03) / sh.H), 0.07 / sh.W, 0.06 / sh.H, color=c,
                                        transform=sh.fig.transFigure, figure=sh.fig))
        sh.text(x + 0.10, yy, s, va="center", fontsize=fs)


def cell_status(cells: pd.DataFrame, gates: pd.DataFrame) -> pd.Series:
    """Cells of the OT run only (QC-passed): True when retained in at least one pair it is in (target of the previous,
    source of the next)."""
    return gates.groupby("cell").retained.any().astype(bool)


def sample_links(links: pd.DataFrame, n: int, seed: int = 20261105) -> pd.DataFrame:
    rng = np.random.default_rng(seed); out = []
    for pair, g in links.groupby("pair"):
        w = g.row_mass.to_numpy(); w = w / w.sum()
        k = min(n, len(g))
        out.append(g.iloc[rng.choice(len(g), k, replace=False, p=w)])
    return pd.concat(out) if out else links.head(0)


def slices_2d(cells, kept, links, cfg, out, n_links):
    x, y = cells.umap_x.to_numpy(), cells.umap_y.to_numpy()
    xmin, xmax, ymin, ymax = np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)
    span, gap = xmax - xmin, 0.35 * (xmax - xmin)
    off = {s: k * (span + gap) - xmin for k, s in enumerate(SAMPLES)}
    W, H = 7.20, 2.55
    sh = st.Sheet(W, H)
    ax = sh.ax(0.08, 0.50, 5.95, 1.55)
    ax.set_xlim(-0.02 * span, len(SAMPLES) * (span + gap) - gap + 0.02 * span); ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.axis("off")
    for k, s in enumerate(SAMPLES):
        ax.scatter(x + off[s], y, s=0.25, c="#EEF0F2", lw=0, rasterized=True, zorder=1)  # every LTL331 cell, as map
        c = cells[(cells.sample_id == s) & cells.index.isin(kept.index)]
        r = kept.reindex(c.index).to_numpy()
        ax.scatter(c.umap_x[~r] + off[s], c.umap_y[~r], s=0.8, c=REJECT, lw=0, rasterized=True, zorder=2)
        ax.scatter(c.umap_x[r] + off[s], c.umap_y[r], s=0.9, c=c.state[r].map(COLOUR), lw=0, rasterized=True, zorder=3)
        ax.text(k * (span + gap) + span / 2 - 0.0, ymax + 0.6, TLABEL[k], ha="center", va="bottom", fontsize=st.FS_TICK)
        ax.text(k * (span + gap) + span / 2, ymin - 0.6, f"n = {len(c):,}\nretained {r.mean():.0%}", ha="center", va="top",
                fontsize=st.FS_MIN, color=st.SUBTLE)
    L = sample_links(links, n_links)
    pos = cells[["umap_x", "umap_y", "sample_id"]]
    a, b = pos.reindex(L.source), pos.reindex(L.target)
    xa, ya = a.umap_x.to_numpy() + a.sample_id.map(off).to_numpy(), a.umap_y.to_numpy()
    xb, yb = b.umap_x.to_numpy() + b.sample_id.map(off).to_numpy(), b.umap_y.to_numpy()
    cols = cells.state.reindex(L.source).map(COLOUR).to_numpy()
    for i in range(len(L)):
        ax.plot([xa[i], xb[i]], [ya[i], yb[i]], color=cols[i], lw=0.25, alpha=0.35, zorder=4, solid_capstyle="round")
    legend(sh, 6.15, 2.15, extra=[("rejected in every pair", REJECT)])
    sh.text(0.08, H - 0.06, f"LTL331 cell matching over time, {NAME[cfg]}", fontsize=st.FS_TITLE, va="top")
    sh.text(0.08, 0.16, f"Authors' UMAP, same coordinates in every slice (grey: all LTL331 cells). Lines: {n_links} retained "
            "source cells per pair, sampled by coupling mass, each to its retained target of largest mass (M4-E).",
            fontsize=st.FS_MIN, color=st.SUBTLE, va="bottom")
    sh.text(0.08, 0.16 - st.leading(st.FS_MIN), "Coloured: cell retained in at least one of its pairs; retained % below each "
            "slice counts the same way. QC-passed cells of the OT run only.", fontsize=st.FS_MIN, color=st.SUBTLE, va="bottom")
    sh.save(out, f"pdx_slices_{cfg}")


def slices_3d(cells, kept, links, cfg, out, n_links):
    """Time points as horizontal sheets stacked upwards (time on the vertical axis), each sheet the authors' UMAP."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    W, H = 5.2, 6.4
    sh = st.Sheet(W, H)
    ax = sh.fig.add_axes([-0.10, 0.02, 0.92, 0.94], projection="3d")
    x, y = cells.umap_x.to_numpy(), cells.umap_y.to_numpy()
    xmin, xmax, ymin, ymax = np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)
    z = {s: k for k, s in enumerate(SAMPLES)}
    for k, s in enumerate(SAMPLES):
        ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], [k] * 5, color=st.GUIDE, lw=0.5)
        c = cells[(cells.sample_id == s) & cells.index.isin(kept.index)]; r = kept.reindex(c.index).to_numpy()
        ax.scatter(c.umap_x[~r], c.umap_y[~r], [k] * int((~r).sum()), s=0.4, c=REJECT, lw=0, depthshade=False, rasterized=True)
        ax.scatter(c.umap_x[r], c.umap_y[r], [k] * int(r.sum()), s=0.6, c=c.state[r].map(COLOUR), lw=0, depthshade=False,
                   rasterized=True)
    L = sample_links(links, n_links)
    pos = cells[["umap_x", "umap_y", "sample_id"]]
    a, b = pos.reindex(L.source), pos.reindex(L.target)
    cols = cells.state.reindex(L.source).map(COLOUR).to_numpy()
    za, zb = a.sample_id.map(z).to_numpy(), b.sample_id.map(z).to_numpy()
    for i in range(len(L)):
        ax.plot([a.umap_x.iat[i], b.umap_x.iat[i]], [a.umap_y.iat[i], b.umap_y.iat[i]], [za[i], zb[i]], color=cols[i],
                lw=0.25, alpha=0.30)
    ax.set_zticks(range(len(SAMPLES))); ax.set_zticklabels([t.replace("\n", " ") for t in TLABEL], fontsize=st.FS_MIN)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_box_aspect((1.0, 1.0, 2.4), zoom=1.05)
    ax.view_init(elev=24, azim=-58)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.grid(False)
    legend(sh, 3.75, 5.6, extra=[("rejected in every pair", REJECT)])
    sh.text(0.15, H - 0.08, f"LTL331 cell matching, time points stacked", fontsize=st.FS_TITLE, va="top")
    sh.text(0.15, H - 0.08 - st.leading(st.FS_TITLE), NAME[cfg], fontsize=st.FS_SUB, color=st.SUBTLE, va="top")
    sh.text(0.15, 0.20, "Each sheet: the authors' UMAP of one time point (time upwards).", fontsize=st.FS_MIN,
            color=st.SUBTLE, va="bottom")
    sh.text(0.15, 0.20 - st.leading(st.FS_MIN), f"Lines: {n_links} retained source cells per pair, sampled by coupling mass, "
            "to their retained target of largest mass (M4-E).", fontsize=st.FS_MIN, color=st.SUBTLE, va="bottom")
    sh.save(out, f"pdx_slices3d_{cfg}")


def slices_html(cells, kept, links, cfg, out, n_links):
    import plotly.graph_objects as go
    z = {s: k for k, s in enumerate(SAMPLES)}
    fig = go.Figure()
    c = cells[cells.index.isin(kept.index)].dropna(subset=["umap_x"]).copy(); c["t"] = c.sample_id.map(z); c["kept"] = kept.reindex(c.index)
    rej = c[~c.kept]
    fig.add_trace(go.Scatter3d(x=rej.umap_x, y=rej.t, z=rej.umap_y, mode="markers", name="rejected in every pair",
                               marker=dict(size=1.6, color=REJECT), hovertext=rej.index, hoverinfo="text"))
    for s in STATES:
        g = c[c.kept & (c.state == s)]
        fig.add_trace(go.Scatter3d(x=g.umap_x, y=g.t, z=g.umap_y, mode="markers", name=s,
                                   marker=dict(size=1.8, color=COLOUR[s]),
                                   hovertext=[f"{i}<br>{s}<br>{TLABEL[t].replace(chr(10), ' ')}" for i, t in zip(g.index, g.t)],
                                   hoverinfo="text"))
    L = sample_links(links, n_links)
    pos = c[["umap_x", "umap_y", "t"]]
    a, b = pos.reindex(L.source), pos.reindex(L.target)
    src_state = cells.state.reindex(L.source).to_numpy()
    for s in STATES:
        m = src_state == s
        if not m.any():
            continue
        xs, ys, zs = [], [], []
        for i in np.flatnonzero(m):
            xs += [a.umap_x.iat[i], b.umap_x.iat[i], None]; ys += [a.t.iat[i], b.t.iat[i], None]
            zs += [a.umap_y.iat[i], b.umap_y.iat[i], None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=f"links from {s}", line=dict(color=COLOUR[s], width=1),
                                   opacity=0.45, hoverinfo="skip"))
    fig.update_layout(title=f"LTL331 cell matching over time: {NAME[cfg]}", template="plotly_white",
                      scene=dict(xaxis=dict(title="UMAP 1", showticklabels=False),
                                 yaxis=dict(title="", tickvals=list(range(len(SAMPLES))),
                                            ticktext=[t.replace("\n", " ") for t in TLABEL]),
                                 zaxis=dict(title="UMAP 2", showticklabels=False), aspectmode="manual",
                                 aspectratio=dict(x=1, y=2.6, z=0.8)),
                      legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(out / f"pdx_slices3d_{cfg}.html", include_plotlyjs="cdn")
    print(f"  wrote pdx_slices3d_{cfg}.html")


def tint(c: str, f: float = 0.30) -> tuple:
    r, g, b = (int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


def flow(tr, srej, cfg, out):
    """Every time point has two bars of all its OT cells: as target of the pair from the previous time point (left)
    and as source of the pair to the next (right). The cells are the same; each pair gates them separately, so only
    the retained / rejected split differs. A state's segment is its share of the cells, solid = retained, hatched =
    rejected. Bands join retained segments; at the source end a band's height is the source state's retained share
    times the share of its retained mass going to that target state (row-normalised), at the target end the target
    state's retained share times the share of its received mass coming from that source state (column-normalised)."""
    t = None if tr is None else tr[tr.block == "retained"]
    r = srej[srej.config_id == CID[cfg]]
    W, H = 7.20, 3.25
    sh = st.Sheet(W, H)
    ax = sh.ax(0.45, 0.50, 5.5, 2.25)
    bw, gap = 0.085, 0.05
    hatch_lw = plt.rcParams["hatch.linewidth"]; plt.rcParams["hatch.linewidth"] = 0.45

    def bar(x0, pair, side):
        g = r[(r.pair == pair) & (r.side == side)].set_index("state").reindex(STATES).fillna(0)
        n = g.cells.sum()
        share, ret = (g.cells / n).to_numpy(), ((g.cells - g.rejected) / n).to_numpy()
        y = np.concatenate([[0], np.cumsum(share)])[:-1]
        for i, s in enumerate(STATES):
            if share[i] <= 0:
                continue
            ax.add_patch(Rectangle((x0, y[i]), bw, ret[i], facecolor=COLOUR[s], lw=0))
            ax.add_patch(Rectangle((x0, y[i] + ret[i]), bw, share[i] - ret[i], facecolor=tint(COLOUR[s]),
                                   edgecolor=COLOUR[s], hatch="//////", lw=0))
        rej = 1 - ret.sum()
        # the two bars of a time point sit close together: the target bar's number ends at its right edge, the
        # source bar's starts at its left edge, so they never overlap
        ax.text(x0 + (bw if side == "target" else 0), -0.015, f"{100 * rej:.0f}%", ha="right" if side == "target" else "left",
                va="bottom", fontsize=st.FS_MIN, color=st.INK)
        return y, ret

    for k, pair in enumerate(PAIRS):
        xs_, xt_ = k + gap / 2, k + 1 - gap / 2 - bw
        ys, rs = bar(xs_, pair, "source")
        yt, rt = bar(xt_, pair, "target")
        if t is None:
            continue
        M = t[t.pair == pair].pivot(index="source_state", columns="target_state", values="mass").reindex(
            index=STATES, columns=STATES).fillna(0).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.nan_to_num(M / M.sum(1, keepdims=True)); C = np.nan_to_num(M / M.sum(0, keepdims=True))
        o_src, o_tgt = ys.copy(), yt.copy()
        xl, xr = xs_ + bw, xt_
        xx = np.linspace(xl, xr, 40); w = (1 - np.cos(np.pi * (xx - xl) / (xr - xl))) / 2
        for i, s in enumerate(STATES):
            for j in range(len(STATES)):
                ha, hb = rs[i] * R[i, j], rt[j] * C[i, j]
                if ha < 2e-4 and hb < 2e-4:
                    continue
                top = o_src[i] + (o_tgt[j] - o_src[i]) * w
                bot = o_src[i] + ha + (o_tgt[j] + hb - o_src[i] - ha) * w
                o_src[i] += ha; o_tgt[j] += hb
                ax.add_patch(Polygon(np.c_[np.r_[xx, xx[::-1]], np.r_[top, bot[::-1]]], closed=True,
                                     facecolor=COLOUR[s], alpha=0.40, lw=0))
    plt.rcParams["hatch.linewidth"] = hatch_lw
    ax.set_xlim(-0.05, len(PAIRS) + 0.05); ax.set_ylim(1.0, -0.06)
    ax.set_xticks(range(len(SAMPLES))); ax.set_xticklabels(TLABEL, fontsize=st.FS_TICK)
    ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("share of the time point's cells", fontsize=st.FS_AXLAB)
    ax.spines["left"].set_bounds(0, 1)
    st.tidy(ax)
    legend(sh, 6.10, 2.70)
    y0 = 2.70 - 8 * st.leading(st.FS_MIN) * 1.15 - 0.10
    for k, (lab, solid) in enumerate((("retained", True), ("rejected", False))):
        yy = y0 - k * st.leading(st.FS_MIN) * 1.15
        sh.fig.patches.append(Rectangle((6.10 / W, (yy - 0.03) / H), 0.07 / W, 0.06 / H, transform=sh.fig.transFigure,
                                        figure=sh.fig, facecolor="#6B7177" if solid else tint("#6B7177"),
                                        edgecolor="#6B7177", hatch=None if solid else "//////", lw=0))
        sh.text(6.20, yy, lab, va="center", fontsize=st.FS_MIN)
    sh.text(0.45, H - 0.06, f"Cell states and their M4-E gates over time, {NAME[cfg]}", fontsize=st.FS_TITLE, va="top")
    sh.text(0.45, H - 0.22, "Each time point: left bar = its cells as target of the previous pair, right bar = the same cells "
            "as source of the next pair; number above a bar: rejected share of all its cells.",
            fontsize=st.FS_MIN, color=st.SUBTLE, va="top")
    sh.text(0.45, 0.06, "Segment height: the state's share of the cells (solid retained, hatched rejected). " + (
            "Bands: coupling mass on the retained x retained block; each end is scaled to the retained segment it joins."
            if t is not None else "No bands: this configuration's coupling is not local; the gates are."),
            fontsize=st.FS_MIN, color=st.SUBTLE, va="bottom")
    sh.save(out, f"pdx_flow_{cfg}")


def transitions(tr, cfg, out):
    t = tr[tr.block == "retained"]
    W, H = 7.20, 1.78
    sh = st.Sheet(W, H)
    cm = LinearSegmentedColormap.from_list("r", ["#F6F7F8", "#1F4E79"]); cm.set_bad("white")
    cw = 0.80
    for k, pair in enumerate(PAIRS):
        M = t[t.pair == pair].pivot(index="source_state", columns="target_state", values="mass").reindex(index=STATES, columns=STATES)
        R = M.div(M.sum(1), axis=0).to_numpy()
        R[~np.isfinite(R)] = np.nan
        ax = sh.ax(1.10 + k * (cw + 0.06), 0.36, cw, cw * 1.05)
        ax.imshow(np.ma.masked_invalid(R), cmap=cm, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for i in range(8):
            for j in range(8):
                if np.isfinite(R[i, j]) and R[i, j] >= 0.15:
                    v = "1" if R[i, j] >= 0.95 else f"{R[i, j]:.1f}".lstrip("0")
                    ax.text(j, i, v, ha="center", va="center", fontsize=st.FS_MIN, color="white" if R[i, j] > 0.6 else st.INK)
        ax.set_xticks(range(8)); ax.set_xticklabels([str(i + 1) for i in range(8)], fontsize=st.FS_MIN)
        ax.set_yticks(range(8)); ax.set_yticklabels([f"{i + 1} {s}" for i, s in enumerate(STATES)] if k == 0 else [], fontsize=st.FS_MIN)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        a, b = TLABEL[k].split("\n")[0], TLABEL[k + 1].split("\n")[0]
        ax.set_title(f"{a} -> {b}", fontsize=st.FS_SUB, pad=2)
    sh.text(1.10, H - 0.06, f"Where each source state's retained mass goes (row-normalised), {NAME[cfg]}",
            fontsize=st.FS_TITLE, va="top")
    sh.text(1.10, 0.04, "Rows: source state; columns: target state (same numbering). Shares >= 0.15 printed (1 = all); "
            "blank: state absent or fully rejected.", fontsize=st.FS_MIN, color=st.SUBTLE, va="bottom")
    sh.save(out, f"pdx_transitions_{cfg}")


def rejected_matrix(srej: pd.DataFrame, cid: str | None, stat: str) -> tuple[np.ndarray, np.ndarray]:
    """states x 14 columns (7 pairs as source, 7 as target): rejected fraction (one config) or its median / IQR over all."""
    s = srej.copy(); s["frac"] = s.rejected / s.cells
    if cid is not None:
        s = s[s.config_id == cid]
    cols = [(p, "source") for p in PAIRS] + [(p, "target") for p in PAIRS]
    V = np.full((len(STATES), len(cols)), np.nan); N = np.zeros_like(V)
    for j, (p, side) in enumerate(cols):
        g = s[(s.pair == p) & (s.side == side)]
        for i, x in enumerate(STATES):
            h = g[g.state == x]
            if len(h) == 0:
                continue
            N[i, j] = h.cells.iloc[0]
            if stat == "value":
                V[i, j] = h.frac.iloc[0]
            elif stat == "median":
                V[i, j] = h.frac.median()
            else:
                V[i, j] = h.frac.quantile(0.75) - h.frac.quantile(0.25)
    return V, N


def rejected_panel(srej, cid, stat, title, stem, out, vmax=1.0):
    V, N = rejected_matrix(srej, cid, stat)
    W, H = 7.20, 2.30
    sh = st.Sheet(W, H)
    cm = LinearSegmentedColormap.from_list("r", ["#F6F7F8", "#8E4A12"]); cm.set_bad("white")
    for half, (lab, cols) in enumerate((("as source (pair to the next time point)", range(0, 7)),
                                        ("as target (pair from the previous time point)", range(7, 14)))):
        ax = sh.ax(1.45 + half * 2.85, 0.42, 2.65, 1.45)
        M = V[:, list(cols)]
        ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    n = int(N[i, list(cols)[j]])
                    ax.text(j, i, f"{M[i, j]:.2f}" if n >= 20 else f"{M[i, j]:.2f}*", ha="center", va="center",
                            fontsize=st.FS_MIN, color="white" if M[i, j] / vmax > 0.6 else st.INK)
        times = SAMPLES[:-1] if half == 0 else SAMPLES[1:]
        ax.set_xticks(range(7)); ax.set_xticklabels([TLABEL[SAMPLES.index(s)].split("\n")[0] for s in times], fontsize=st.FS_MIN)
        ax.set_yticks(range(len(STATES))); ax.set_yticklabels(STATES if half == 0 else [], fontsize=st.FS_MIN)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(lab, fontsize=st.FS_SUB, pad=3)
    sh.text(1.45, H - 0.06, title, fontsize=st.FS_TITLE, va="top")
    sh.text(1.45, 0.10, "Each cell: one state at one time point. * fewer than 20 cells; blank: no cell of that state.",
            fontsize=st.FS_MIN, color=st.SUBTLE, va="bottom")
    sh.save(out, stem)


def composition(cells, out):
    W, H = 4.6, 2.4
    sh = st.Sheet(W, H)
    ax = sh.ax(0.45, 0.45, 2.6, 1.6)
    bottom = np.zeros(len(SAMPLES))
    tab = cells.groupby(["sample_id", "state"]).size().unstack(fill_value=0).reindex(index=SAMPLES, columns=STATES, fill_value=0)
    frac = tab.div(tab.sum(1), axis=0)
    for s in STATES:
        ax.bar(range(len(SAMPLES)), frac[s], bottom=bottom, color=COLOUR[s], width=0.72, lw=0)
        bottom += frac[s].to_numpy()
    ax.set_xticks(range(len(SAMPLES))); ax.set_xticklabels(TLABEL, fontsize=st.FS_MIN)
    ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("share of cells", fontsize=st.FS_AXLAB)
    for k, n in enumerate(tab.sum(1)):
        ax.text(k, 1.02, f"{n:,}", ha="center", va="bottom", fontsize=st.FS_MIN, color=st.SUBTLE)
    st.tidy(ax)
    legend(sh, 3.25, 1.95)
    sh.text(0.45, H - 0.06, "LTL331 cell states by time point (authors' clusters; QC-passed cells of the OT run)", fontsize=st.FS_TITLE, va="top")
    sh.save(out, "pdx_composition")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", nargs="*", default=["splatter_l5_best", "baseline", "logcpm"])
    p.add_argument("--links", type=int, default=250, help="links drawn per pair")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st.apply()
    cells = pd.read_csv(a.data / "cells.tsv.gz", sep="\t", index_col="cell")
    srej = pd.read_csv(a.data / "state_rejected_all.tsv", sep="\t")
    first = pd.read_csv(a.data / f"gates_{a.config[0]}.tsv.gz", sep="	", dtype={"cell": str}, usecols=["cell"])
    composition(cells[cells.index.isin(set(first.cell))], a.out)  # the cells of the OT run (QC-passed)
    for cfg in a.config:
        if not (a.data / f"transitions_{cfg}.tsv").exists():  # no local coupling: the gate panels only
            print(f"  {cfg}: no coupling tables, gate panels only")
            flow(None, srej, cfg, a.out)
        else:
            gates = pd.read_csv(a.data / f"gates_{cfg}.tsv.gz", sep="\t", dtype={"cell": str})
            links = pd.read_csv(a.data / f"links_{cfg}.tsv.gz", sep="\t", dtype={"source": str, "target": str})
            tr = pd.read_csv(a.data / f"transitions_{cfg}.tsv", sep="\t")
            kept = cell_status(cells, gates)
            slices_2d(cells, kept, links, cfg, a.out, a.links)
            slices_3d(cells, kept, links, cfg, a.out, a.links)
            slices_html(cells, kept, links, cfg, a.out, 2 * a.links)
            flow(tr, srej, cfg, a.out)
            transitions(tr, cfg, a.out)
        rejected_panel(srej, CID[cfg], "value", f"Rejected fraction by state and time point, {NAME[cfg]}", f"pdx_rejected_{cfg}", a.out)
    n_cfg = srej.config_id.nunique()
    rejected_panel(srej, None, "median", f"Rejected fraction, median over all {n_cfg} preprocessing configurations",
                   "pdx_rejected_median296", a.out)
    rejected_panel(srej, None, "iqr", f"Rejected fraction, interquartile range over all {n_cfg} configurations",
                   "pdx_rejected_iqr296", a.out, vmax=0.5)


if __name__ == "__main__":
    main()
