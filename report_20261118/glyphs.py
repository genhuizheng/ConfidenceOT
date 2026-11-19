"""Schematic glyph language shared by every figure (after the user's reference style, 2026-11-18):
small cell / matrix / chart icons, tinted pills for method steps, thin elbow connectors with arrowheads, grey italic
panel references on the right, thin rules between rows. All glyphs are matplotlib primitives in data coordinates.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Ellipse

import figstyle as fs

LINE = "#5F666B"


def tint(hex_color, a):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (1 - a * (1 - r), 1 - a * (1 - g), 1 - a * (1 - b))


def cell(ax, x, y, r, color, z=4, lw=0.5, nucleus=True):
    ax.add_patch(Circle((x, y), r, fc=tint(color, 0.45), ec=color, lw=lw, zorder=z))
    if nucleus:
        ax.add_patch(Circle((x + 0.18 * r, y + 0.12 * r), 0.38 * r, fc=color, ec="none", zorder=z + 0.1))


def cell_cluster(ax, x, y, color, n=5, r=0.16, spread=0.32, seed=0, z=4):
    rng = np.random.default_rng(seed)
    pts = [(0.0, 0.0)]
    while len(pts) < n:
        p = rng.uniform(-1, 1, 2) * spread
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 1.9 * r for q in pts):
            pts.append(tuple(p))
    for dx, dy in pts:
        cell(ax, x + dx, y + dy, r, color, z=z)


def pill(ax, x, y, w, h, text, color, fs_=None, tc=None, z=3, alpha=0.16, ec=None, ls="-", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={min(h, w) * 0.45}",
                                fc=tint(color, alpha), ec=ec if ec else "none", lw=0.6 if ec else 0, ls=ls, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_ or fs.FS_ANNOT, color=tc or color,
            zorder=z + 1, fontweight=weight, linespacing=1.05)


def frame(ax, x, y, w, h, color=fs.NEUTRAL, fc="white", lw=0.6, z=1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={min(w, h) * 0.08}",
                                fc=fc, ec=color, lw=lw, ls=ls, zorder=z))


def arrow(ax, x0, y0, x1, y1, color=LINE, lw=0.6, ms=5, z=2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z, clip_on=False))


def elbow(ax, starts, end, xmid=None, color=LINE, lw=0.6, ms=5, z=2):
    """Bracket: several start points joined at x = xmid, one arrow to `end`."""
    ex, ey = end
    xm = xmid if xmid is not None else (max(s[0] for s in starts) + ex) / 2
    ys = [s[1] for s in starts] + [ey]
    for sx, sy in starts:
        ax.plot([sx, xm], [sy, sy], color=color, lw=lw, zorder=z, solid_capstyle="butt")
    ax.plot([xm, xm], [min(ys), max(ys)], color=color, lw=lw, zorder=z)
    arrow(ax, xm, ey, ex, ey, color=color, lw=lw, ms=ms, z=z)


def ref(ax, x, y, text):
    ax.text(x, y, text, ha="right", va="center", fontsize=fs.FS_SMALL, color=fs.SUBTLE, style="italic")


def rule(ax, x0, x1, y):
    ax.plot([x0, x1], [y, y], color=fs.GUIDE, lw=0.6, zorder=0)


def label(ax, x, y, s, color=fs.INK, size=None, ha="center", va="center", **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size or fs.FS_SMALL, color=color, linespacing=1.08, **kw)


# ------------------------------------------------------------------ icons (centred at x, y; size s = half width)
def matrix_icon(ax, x, y, s, color, rows=5, cols=6, seed=1, holes=0.0):
    rng = np.random.default_rng(seed)
    cw, ch = 2 * s / cols, 1.6 * s / rows
    for i in range(rows):
        for j in range(cols):
            v = rng.uniform(0.15, 0.95)
            fc = "white" if rng.uniform() < holes else tint(color, v)
            ax.add_patch(Rectangle((x - s + j * cw, y - 0.8 * s + i * ch), cw * 0.92, ch * 0.88, fc=fc, ec="none", zorder=3))
    ax.add_patch(Rectangle((x - s, y - 0.8 * s), 2 * s, 1.6 * s, fc="none", ec=color, lw=0.5, zorder=4))


def heat_icon(ax, x, y, s, color, n=4, diag=True):
    c = 2 * s / n
    for i in range(n):
        for j in range(n):
            v = 0.85 if (diag and i == j) else 0.12 + 0.1 * ((i + j) % 2)
            ax.add_patch(Rectangle((x - s + j * c, y + s - (i + 1) * c), c * 0.92, c * 0.92, fc=tint(color, v), ec="none", zorder=3))


def bars_icon(ax, x, y, s, color, vals=(0.9, 0.7, 0.5, 0.35, 0.2), axes=True):
    n = len(vals); w = 2 * s / n
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x - s + i * w + 0.12 * w, y - s), 0.76 * w, 2 * s * v, fc=tint(color, 0.8), ec="none", zorder=3))
    if axes:
        ax.plot([x - s, x - s, x + s], [y + s, y - s, y - s], color=fs.INK, lw=0.45, zorder=4)


def line_icon(ax, x, y, s, color, ys=(0.2, 0.35, 0.3, 0.6, 0.85), dots=True, axes=True):
    xs = np.linspace(x - 0.85 * s, x + 0.85 * s, len(ys)); yy = y - s + 2 * s * np.asarray(ys)
    ax.plot(xs, yy, color=color, lw=0.8, zorder=4)
    if dots:
        ax.scatter(xs, yy, s=3, color=color, zorder=5, lw=0)
    if axes:
        ax.plot([x - s, x - s, x + s], [y + s, y - s, y - s], color=fs.INK, lw=0.45, zorder=4)


def scatter_icon(ax, x, y, s, colors, seed=2, n=9):
    rng = np.random.default_rng(seed)
    for k, c in enumerate(colors):
        cx, cy = x + rng.uniform(-0.5, 0.5) * s, y + rng.uniform(-0.5, 0.5) * s
        ax.scatter(cx + rng.normal(0, 0.16 * s, n), cy + rng.normal(0, 0.16 * s, n), s=1.6, color=c, lw=0, zorder=4)
    ax.plot([x - s, x - s, x + s], [y + s, y - s, y - s], color=fs.INK, lw=0.45, zorder=4)


def rank_icon(ax, x, y, s, color):
    """Per-cell rank encoding: one cell's genes sorted, top bars kept."""
    vals = np.linspace(1, 0.08, 8)
    w = 2 * s / 8
    for i, v in enumerate(vals):
        keep = i < 5
        ax.add_patch(Rectangle((x - s + i * w + 0.1 * w, y - s), 0.8 * w, 2 * s * v,
                               fc=tint(color, 0.8 if keep else 0.2), ec="none", zorder=3))
    ax.plot([x - s, x + s], [y - s, y - s], color=fs.INK, lw=0.45, zorder=4)


def coupling_icon(ax, x, y, s, kind="cot"):
    """Two columns of cells; lines = coupling; x = rejected."""
    xs, xt = x - 0.75 * s, x + 0.75 * s
    ys = np.linspace(y + 0.8 * s, y - 0.8 * s, 4)
    cols = [fs.POP[0], fs.POP[1], fs.POP[2], fs.POP[3]]
    for i, yy in enumerate(ys):
        cs = cols[i] if i < 3 else cols[2]
        ct = cols[i] if i < 2 else (cols[3] if i == 3 else cols[3])
    pairs = [(0, 0, fs.RETAIN, 1.0), (1, 1, fs.RETAIN, 1.0)]
    if kind == "balanced":
        pairs += [(2, 2, fs.REJECT, 1.0), (2, 3, fs.REJECT, 0.6), (3, 3, fs.REJECT, 0.9)]
    elif kind == "uot":
        pairs += [(2, 2, fs.REJECT, 0.4), (3, 3, fs.REJECT, 0.3)]
    for i, j, c, w in pairs:
        ax.plot([xs, xt], [ys[i], ys[j]], color=c, lw=0.4 + 0.8 * w, zorder=3, solid_capstyle="round")
    src_cols = [fs.POP[0], fs.POP[1], fs.POP[2], fs.POP[2]]
    tgt_cols = [fs.POP[0], fs.POP[1], fs.POP[3], fs.POP[3]]
    r = 0.13 * s
    for i, yy in enumerate(ys):
        cell(ax, xs, yy, r, src_cols[i], z=5, nucleus=False)
        cell(ax, xt, yy, r, tgt_cols[i], z=5, nucleus=False)
        if kind == "cot" and i >= 2:
            ax.scatter(xs - 0.38 * s, yy, marker="x", s=9, color=fs.REJECT, lw=0.8, zorder=6)
            ax.scatter(xt + 0.38 * s, yy, marker="x", s=9, color=fs.REJECT, lw=0.8, zorder=6)


def edit_array(ax, x, y, w, h, states, palette, missing=None):
    """CRISPR target array: one row of cut sites, coloured by state (0 = uncut)."""
    n = len(states); cw = w / n
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=fs.INK, lw=0.4, zorder=3))
    for i, st in enumerate(states):
        fc = "white" if st is None else ("#EEF0F1" if st == 0 else palette[st % len(palette)])
        ax.add_patch(Rectangle((x + i * cw + 0.08 * cw, y + 0.15 * h), 0.84 * cw, 0.7 * h, fc=fc, ec="none", zorder=4))


def tree_icon(ax, x, y, s, colors):
    """Small lineage tree: root at top, leaves coloured by clade."""
    ax.plot([x, x], [y + s, y + 0.5 * s], color=fs.INK, lw=0.5, zorder=3)
    ax.plot([x - 0.6 * s, x + 0.6 * s], [y + 0.5 * s] * 2, color=fs.INK, lw=0.5, zorder=3)
    leaves = np.linspace(x - 0.9 * s, x + 0.9 * s, len(colors))
    for i, (lx, c) in enumerate(zip(leaves, colors)):
        px = x - 0.6 * s if lx < x else x + 0.6 * s
        ax.plot([px, px], [y + 0.5 * s, y], color=fs.INK, lw=0.5, zorder=3)
        ax.plot([px, lx], [y, y], color=fs.INK, lw=0.5, zorder=3)
        ax.plot([lx, lx], [y, y - 0.5 * s], color=fs.INK, lw=0.5, zorder=3)
        cell(ax, lx, y - 0.62 * s, 0.13 * s, c, z=5, nucleus=False)


def mouse_icon(ax, x, y, s, color=fs.SUBTLE):
    """Side-view mouse silhouette from ellipses (body, head, ear, tail)."""
    ax.add_patch(Ellipse((x, y), 1.6 * s, 0.9 * s, fc=tint(color, 0.25), ec=color, lw=0.5, zorder=3))
    ax.add_patch(Ellipse((x + 0.85 * s, y + 0.12 * s), 0.6 * s, 0.48 * s, fc=tint(color, 0.25), ec=color, lw=0.5, zorder=3))
    ax.add_patch(Circle((x + 0.78 * s, y + 0.42 * s), 0.14 * s, fc=tint(color, 0.35), ec=color, lw=0.5, zorder=4))
    ax.add_patch(Circle((x + 1.0 * s, y + 0.17 * s), 0.035 * s, fc=color, ec="none", zorder=5))
    t = np.linspace(0, 1, 20)
    ax.plot(x - 0.8 * s - 0.7 * s * t, y - 0.1 * s + 0.25 * s * np.sin(t * 3), color=color, lw=0.6, zorder=3)


def tumour(ax, x, y, s, colors, seed=3):
    """Irregular tumour outline filled with cells of the given colours."""
    th = np.linspace(0, 2 * np.pi, 40)
    rng = np.random.default_rng(seed)
    rr = s * (1 + 0.08 * np.sin(5 * th + rng.uniform(0, 6)))
    ax.add_patch(Polygon(np.c_[x + rr * np.cos(th), y + 0.8 * rr * np.sin(th)], closed=True, fc=tint(fs.NEUTRAL, 0.15),
                         ec=fs.NEUTRAL, lw=0.5, zorder=2))
    pts = []
    while len(pts) < 14:
        p = rng.uniform(-0.72, 0.72, 2) * s
        if (p[0] / s) ** 2 + (p[1] / (0.8 * s)) ** 2 < 0.6 and all(np.hypot(*(p - q)) > 0.3 * s for q in pts):
            pts.append(p)
    for k, p in enumerate(pts):
        cell(ax, x + p[0], y + p[1], 0.13 * s, colors[k % len(colors)], z=4, nucleus=False)
