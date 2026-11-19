"""Shared figure style for the 2026-11-18 report: NBT-style tokens (user skill nbt-figure-style) sized for projection.

Every main figure is drawn ONCE as a single vector sheet; panel files are bbox crops of that same sheet
(`export_panels`), so the overview slide and the panel slides show the same rendering, data, colours and labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# ------------------------------------------------------------------ palette (one hue, one meaning, across all figures)
INK = "#333333"
SUBTLE = "#6B7280"
RULE = "#5F666B"
NEUTRAL = "#B0B5B9"
GUIDE = "#E3E6E8"
PALE = "#F3F4F5"

SRC = "#2C6E9B"        # source side
TGT = "#B4702A"        # target side
RETAIN = "#4E8A6B"     # retained / has a counterpart
REJECT = "#C0392B"     # rejected / no counterpart (accent)
FALSE_REJ = "#E0A458"  # falsely rejected (has a counterpart)
FALSE_RET = "#8C7196"  # falsely retained (no counterpart)

METHOD = {"Balanced OT": "#9A9A9A", "UOT": "#37536B", "ConfidenceOT": REJECT}

# biological populations in schematics / Splatter (muted, disjoint from the side colours)
POP = ["#5B8DB8", "#7FB07F", "#C9A54B", "#B07AA1", "#7A9E9F", "#D08770", "#8F8F5A", "#6C6FA8"]

# ------------------------------------------------------------------ type (projection-sized; floor 5.5 pt at print size)
FS_PANEL = 9.0
FS_TITLE = 6.8
FS_AXLAB = 6.6
FS_TICK = 6.0
FS_LEGEND = 6.0
FS_ANNOT = 5.8
FS_SMALL = 5.5

LW_AXIS = 0.6
DASH = (0, (3.2, 2.0))
REF_KW = dict(color=RULE, ls=DASH, lw=0.6)

RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS_TICK, "axes.labelsize": FS_AXLAB, "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
    "axes.linewidth": LW_AXIS, "xtick.major.width": LW_AXIS, "ytick.major.width": LW_AXIS,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.direction": "out", "ytick.direction": "out",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.titlecolor": INK, "legend.frameon": False, "axes.grid": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.titlepad": 3.0, "axes.labelpad": 2.0,
}


def apply():
    plt.rcParams.update(RC)


def tidy(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.set_yticks([])
    if not bottom:
        ax.set_xticks([])
    ax.grid(False)


def blank(ax):
    ax.set_axis_off()


class Sheet:
    """Absolute-inch layout (x, y from bottom-left). Panels are registered with their crop rectangle."""

    def __init__(self, w, h, name):
        self.W, self.H, self.name = float(w), float(h), name
        self.fig = plt.figure(figsize=(self.W, self.H))
        self.fig.patch.set_facecolor("white")
        self.panels: dict[str, tuple[float, float, float, float]] = {}

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H], **kw)

    def text(self, x, y, s, **kw):
        kw.setdefault("transform", self.fig.transFigure)
        return self.fig.text(x / self.W, y / self.H, s, **kw)

    def panel(self, letter, x0, y0, x1, y1, title=None):
        """Register panel `letter` over the rectangle (x0, y0)-(x1, y1) in inches; the letter sits at its top-left."""
        self.panels[letter] = (x0, y0, x1, y1)
        self.text(x0 + 0.02, y1 - 0.02, letter, fontsize=FS_PANEL, fontweight="bold", va="top", ha="left", color="#1A1A1A")
        if title:
            self.text(x0 + 0.22, y1 - 0.035, title, fontsize=FS_TITLE, va="top", ha="left", color=INK)

    def save(self, outdir: Path | None = None, dpi=450):
        outdir = (outdir or OUT) / self.name
        outdir.mkdir(parents=True, exist_ok=True)
        stem = outdir / self.name
        for ext in ("png", "pdf", "svg"):
            self.fig.savefig(f"{stem}.{ext}", dpi=dpi, facecolor="white")
        # panel crops of the SAME rendering
        meta = {"figure": self.name, "size_in": [self.W, self.H], "panels": {}}
        for k, (x0, y0, x1, y1) in self.panels.items():
            bb = Bbox.from_extents(x0, y0, x1, y1)
            for ext, d in (("png", 600), ("pdf", None), ("svg", None)):
                kw = {"dpi": d} if d else {}
                self.fig.savefig(f"{stem}_panel_{k}.{ext}", bbox_inches=bb, facecolor="white", **kw)
            meta["panels"][k] = [x0, y0, x1, y1]
        (outdir / f"{self.name}_panels.json").write_text(json.dumps(meta, indent=1))
        plt.close(self.fig)
        print(f"{self.name}: {self.W:.2f} x {self.H:.2f} in = {self.W*25.4:.0f} x {self.H*25.4:.0f} mm; panels {list(self.panels)}")


def box(ax, x, y, w, h, text, fc="white", ec=INK, tc=INK, fs=FS_ANNOT, lw=0.6, style="round,pad=0.02,rounding_size=0.04", **kw):
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, transform=ax.transData, clip_on=False)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, **kw)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.7, ms=6, style="-|>", **kw):
    from matplotlib.patches import FancyArrowPatch
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        shrinkA=0, shrinkB=0, clip_on=False, **kw)
    ax.add_patch(a)
    return a


def schematic_tag(ax, x=0.99, y=0.99, text="schematic", **kw):
    """Mark a panel (or sub-panel) as illustrative, not data."""
    ax.text(x, y, text, transform=ax.transAxes, ha="right", va="top", fontsize=FS_SMALL, color=SUBTLE, style="italic", **kw)
