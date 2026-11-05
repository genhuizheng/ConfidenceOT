"""
Style tokens for the NBT-style polish pass.

Calibrated against DIRECT VISUAL REFERENCE: the rendered main figures of
  * Kohlgruber et al., Nat Biotechnol 43, 623-634 (2025)  -- Figs 1 and 4
  * Huber et al., Nat Biotechnol 43, 1360-1372 (2025)     -- Fig 1
viewed at publisher resolution (~2170 px for a ~180 mm figure, ~306 dpi).

Observations that drove these tokens (full list in NBT_STYLE_REVIEW.md):
  1. Left and bottom spines only; NO gridlines anywhere in point / bar /
     scatter panels -- not faint grey gridlines, none at all.
  2. Spines and ticks are moderate, not hairline: about 0.7 pt at 306 dpi.
  3. Ticks point outward, short, few, at round numbers only.
  4. Panel letters are bold and markedly larger than every other element,
     set flush top-left, outside the y-axis label.
  5. Legends are frameless, untitled, small swatches.
  6. Light-grey secondary points plus a saturated filled highlight is the
     standard emphasis idiom.
  7. In schematics, an entity's text label carries the entity's own colour.
  8. Bold is reserved for panel letters; italic for biological conventions.

READ-ONLY with respect to every existing project file.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palette
# Semantic hues carried over unchanged from the Phase 2A integrated
# comparison: they are already muted, and re-cutting them would break the
# documented mapping for no visible gain.
PAL = {
    "tcr":      "#2C6E9B",   # TCR / CDR3beta
    "peptide":  "#B4702A",   # peptide / antigen
    "mhc":      "#4E8A6B",   # MHC / HLA / presentation
    "model":    "#C0392B",   # NetImmuno / model          (accent)
    "mutation": "#7D3C98",   # mutated residue
}
# Non-semantic greys. Kept out of PAL so the disjointness assertion below
# tests only hues that carry biological meaning.
INK = "#333333"
RULE = "#5F666B"         # reference lines; darkened from #6B7280 so the
                         # dashed chance line separates from the lightest
                         # comparator curve
NEUTRAL = "#B0B5B9"      # comparator dots in the dot matrices
GUIDE = "#E3E6E8"        # dot-matrix guide line
SUBTLE = "#6B7280"       # subordinate-but-readable text (n columns, notes)
BOXLINE = "#6E7478"

# Comparator curve colours, REBUILT for this pass.
#
# Five hues are reserved for biological entities, and seven comparator
# methods still need individual identity in the PR/ROC strips. These are
# deliberately desaturated so the set reads as a secondary family against the
# single saturated accent, and luminance-staggered so they stay partly
# separable in greyscale. ERGO-II (VDJdb) was moved off its previous amber
# #E69F00, which sat only ~11 degrees of hue from the peptide colour.
COMPARATOR = {
    "epiTCR-BH":        "#37536B",   # dark steel      L~32
    "ERGO-II (McPAS)":  "#6E8E7D",   # sage            L~54
    "ERGO-II (VDJdb)":  "#BCA177",   # light ochre     L~64
    "PISTE-Random":     "#8C7196",   # mauve           L~47
    "PISTE-RefTCR":     "#8FB0C6",   # pale steel      L~68
    "PISTE-UniPep":     "#444444",   # near-black      L~27
    "PMTnet":           "#9A9A9A",   # mid grey        L~60
}

# "One hue, one meaning" is enforced here rather than asserted in a comment:
# no comparator method may reuse a biological entity's colour.
_clash = set(c.lower() for c in COMPARATOR.values()) & set(
    c.lower() for c in PAL.values())
assert not _clash, f"comparator hue reused for a biological entity: {_clash}"

CURVE_COL = dict(COMPARATOR)
CURVE_COL["NetImmuno"] = PAL["model"]

# ---------------------------------------------------------------- type
# Hard floor: nothing in either figure is set below 5.0 pt. The Phase 2A
# review measured elements at 4.6-4.8 pt, under the journal minimum and with
# no headroom for the reduction production applies to open legend space.
FS_MIN = 5.0

FS_PANEL = 8.0       # panel letter -- strongest, and the only bold element
FS_AXLAB = 6.4       # axis label
FS_TITLE = 6.4       # panel / facet title
FS_TICK = 5.8        # tick label
FS_SUB = 5.0         # facet subtitle (n, prevalence)
FS_ANNOT = 5.2       # in-panel annotation (n columns)
FS_LEGEND = 5.6
FS_TABLE = 5.4
FS_TABLE_HEAD = 5.2
FS_CARTOON = 5.0     # cartoon body label
FS_CARTOON_SM = 5.0  # cartoon secondary label -- also at the floor


def leading(fs, factor=1.32):
    """Line step in INCHES for stacked text, never tighter than 1.32 x size.

    The Phase 2A facet titles used a fixed 0.085 in step for every line,
    which is 1.11x at 5.5 pt and put descenders 0.38 mm from the ascenders
    below.
    """
    return factor * fs / 72.0


# ---------------------------------------------------------------- strokes
LW_AXIS = 0.55       # spine and tick weight
LW_MAIN = 1.15       # NetImmuno curve
LW_COMP = 0.50       # comparator curve
LW_ERR = 0.65        # error bar
LW_CAP = 0.85        # error bar cap
LW_REF = 0.60        # reference line (prevalence / chance / zero)
LW_GUIDE = 0.40      # dot-matrix guide line
LW_BOX = 0.55        # cartoon box outline
LW_ARROW = 0.55      # cartoon arrow
ARROW_MS = 6

# One reference-line style, shared by both figures. Figure 4's zero line
# previously used a locally declared solid #333333, which read as a second
# data axis rather than as annotation.
DASH = (0, (3.2, 2.0))
REF_KW = dict(color=RULE, ls=DASH, lw=LW_REF)

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS_TICK,
    "axes.labelsize": FS_AXLAB, "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "axes.linewidth": LW_AXIS,
    "xtick.major.width": LW_AXIS, "ytick.major.width": LW_AXIS,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.direction": "out", "ytick.direction": "out",
    # Declared ink was previously never applied: matplotlib kept pure #000000
    # for text, labels, ticks and spines while PAL["ink"] claimed #333333.
    "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.titlecolor": INK,
    "legend.frameon": False,
    "axes.grid": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

PANEL_KW = dict(fontsize=FS_PANEL, fontweight="bold", va="top", ha="left",
                color="#1A1A1A")


def apply():
    plt.rcParams.update(RC)


# ---------------------------------------------------------------- layout
class Sheet:
    """Absolute inch layout, so every declared dimension is the true page size."""

    def __init__(self, w, h):
        self.W, self.H = float(w), float(h)
        self.fig = plt.figure(figsize=(self.W, self.H))
        self.fig.patch.set_facecolor("white")

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, y / self.H,
                                  w / self.W, h / self.H], **kw)

    def text(self, x, y, s, **kw):
        return self.fig.text(x / self.W, y / self.H, s, **kw)

    def panel(self, letter, x, y):
        return self.text(x, y, letter, **PANEL_KW)

    def save(self, outdir, stem, dpi=400):
        self.fig.savefig(outdir / f"{stem}.png", dpi=dpi, facecolor="white")
        self.fig.savefig(outdir / f"{stem}.pdf", facecolor="white")
        plt.close(self.fig)
        print(f"  wrote {stem}.png/.pdf   {self.W:.2f} x {self.H:.2f} in "
              f"= {self.W * 25.4:.0f} x {self.H * 25.4:.0f} mm "
              f"({self.W * self.H:.2f} sq in)")


def tidy(ax, left=True, bottom=True):
    """Left and bottom spines only, no grid -- the reference treatment."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.grid(False)
