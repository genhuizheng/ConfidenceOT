"""The Confidence-UOT algorithm deck: Figure 1 (input -> model -> output), the method step by step
with its production parameters, and the proposal's Area A and Area B as written.

    python cancer_metastasis/tools/build_algorithm_deck.py [OUT.pptx]

Every parameter on a slide is the production value read from the code (02_run_pair.py,
src/confidenceot/cuda.py, _cpu_uot.py, calibration.py), and every benchmark number is from
benchmark_results/native (method_comparison_full, M4-E, ranknm256_noscale_ds_cos). Area A, Area B
and the method positioning are copied from D:/Xia_lab/Proposal/CSEM_Proposal_Abstract.tex without
any change to the wording; inline math is typeset with sub/superscripts, display math as rendered
equations. Figure 1 is drawn with native shapes so it stays editable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "progress_report7" / "ConfidenceOT_algorithm.pptx"
ASSETS = OUT.parent / "algorithm_deck_assets"

NAVY, INK, MUTED, RULE = "1F2A44", "2B2B2B", "6B7280", "C9CED6"
PANEL, WHITE = "F4F6F9", "FFFFFF"
RETAIN, REJECT, CALIB, UNKNOWN = "0072B2", "D55E00", "009E73", "9AA3AE"
TITLE_FONT, BODY_FONT = "Cambria", "Calibri"


def rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_)


# --- text ------------------------------------------------------------------------------

TOKEN = re.compile(r"(_\{[^}]*\}|\^\{[^}]*\}|\*[^*]+\*|\*\*[^*]+\*\*)")


def rich(paragraph, text: str, size: float, colour: str = INK, bold: bool = False,
         font: str = BODY_FONT) -> None:
    """Text with _{sub}, ^{sup} and *italic* runs (inline math without an equation editor)."""
    for piece in TOKEN.split(text):
        if not piece:
            continue
        run = paragraph.add_run()
        baseline = None
        italic = False
        if piece.startswith("_{"):
            piece, baseline = piece[2:-1], "-25000"
        elif piece.startswith("^{"):
            piece, baseline = piece[2:-1], "30000"
        elif piece.startswith("**"):
            piece, bold_here = piece[2:-2], True
            run.font.bold = True
        elif piece.startswith("*"):
            piece, italic = piece[1:-1], True
        run.text = piece
        run.font.size = Pt(size)
        run.font.name = font
        run.font.color.rgb = rgb(colour)
        if bold:
            run.font.bold = True
        if italic:
            run.font.italic = True
        if baseline:
            run.font._element.set("baseline", baseline)


def textbox(slide, x, y, w, h, paragraphs, size=16, colour=INK, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, spacing=1.0, after=4, font=BODY_FONT, bold=False):
    """paragraphs: list of str or (str, dict) with overrides size / colour / bold / bullet."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.02)
    frame.margin_top = frame.margin_bottom = Inches(0.02)
    frame.vertical_anchor = anchor
    for k, item in enumerate(paragraphs):
        text, opts = (item, {}) if isinstance(item, str) else item
        p = frame.paragraphs[0] if k == 0 else frame.add_paragraph()
        p.alignment = opts.get("align", align)
        p.line_spacing = spacing
        p.space_after = Pt(opts.get("after", after))
        if opts.get("bullet"):
            text = "\u2022  " + text
        rich(p, text, opts.get("size", size), opts.get("colour", colour), opts.get("bold", bold),
             opts.get("font", font))
    return box


def title(slide, text, kicker=None):
    textbox(slide, 0.55, 0.32, 12.2, 0.7, [text], size=28, colour=NAVY, font=TITLE_FONT, bold=True)
    if kicker:
        textbox(slide, 0.57, 0.98, 12.2, 0.45, [kicker], size=15, colour=MUTED)
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.57), Inches(1.42),
                                      Inches(12.78), Inches(1.42))
    line.line.color.rgb = rgb(RULE)
    line.line.width = Pt(0.75)


def box(slide, x, y, w, h, fill=PANEL, border=RULE, width=1.0, dashed=False, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        radius=0.06):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    if border is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = rgb(border)
        s.line.width = Pt(width)
        if dashed:
            s.line.dash_style = 7  # MSO_LINE.DASH
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def dot(slide, x, y, r, colour):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - r), Inches(y - r), Inches(2 * r), Inches(2 * r))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(colour)
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def line(slide, x1, y1, x2, y2, colour=RULE, width=0.75, arrow=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(colour)
    c.line.width = Pt(width)
    if arrow:
        ln = c.line._get_or_add_ln()
        tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
        ln.append(tail)
    return c


def block_arrow(slide, x, y, w, h, colour="B8C0CC", direction=MSO_SHAPE.RIGHT_ARROW):
    s = slide.shapes.add_shape(direction, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(colour)
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def equation(latex: str, name: str, size: float = 22, colour: str = NAVY) -> tuple[Path, float, float]:
    """Render display math with matplotlib mathtext; returns the PNG and its size in inches."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"{name}.png"
    plt.rcParams["mathtext.fontset"] = "cm"
    fig = plt.figure(figsize=(0.01, 0.01))
    text = fig.text(0, 0, f"${latex}$", fontsize=size, color="#" + colour)
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    from PIL import Image

    with Image.open(path) as image:
        w, h = image.size
    return path, w / 300, h / 300


def place_equation(slide, latex, name, x, y, size=22, max_w=None, colour=NAVY, centre_w=None):
    path, w, h = equation(latex, name, size, colour)
    if max_w and w > max_w:
        h, w = h * max_w / w, max_w
    if centre_w:
        x = x + (centre_w - w) / 2
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    return h


def callout(slide, x, y, w, tag, text):
    """A one-line link from an algorithm step to the proposal Area that studies it."""
    pill = box(slide, x, y, 0.95, 0.4, fill=NAVY, border=None, radius=0.5)
    textbox(slide, x, y + 0.02, 0.95, 0.36, [(tag, {"size": 12.5, "bold": True, "colour": WHITE, "align": PP_ALIGN.CENTER})],
            anchor=MSO_ANCHOR.MIDDLE)
    textbox(slide, x + 1.08, y + 0.03, w - 1.1, 0.4, [(text, {"size": 12.5, "colour": NAVY})], anchor=MSO_ANCHOR.MIDDLE)
    return pill


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# --- Figure 1 ----------------------------------------------------------------------------

def clusters(seed=7):
    """Two shared populations, one only in the source, one only in the target (unit square)."""
    rng = np.random.default_rng(seed)

    def blob(cx, cy, n, sx=0.08, sy=0.11):
        return np.c_[rng.normal(cx, sx, n), rng.normal(cy, sy, n)]

    source = {"shared1": blob(0.25, 0.55, 14), "shared2": blob(0.55, 0.42, 14), "private": blob(0.85, 0.6, 8, 0.05, 0.08)}
    target = {"shared1": blob(0.22, 0.48, 13), "shared2": blob(0.52, 0.55, 15), "emerged": blob(0.85, 0.4, 8, 0.05, 0.08)}
    return source, target


def draw_cloud(slide, pts, x, y, w, h, colour_of, r=0.045):
    centres = {}
    for group, xy in pts.items():
        xs = x + np.clip(xy[:, 0], 0.03, 0.97) * w
        ys = y + (1 - np.clip(xy[:, 1], 0.05, 0.95)) * h
        centres[group] = list(zip(xs, ys))
        for px, py in centres[group]:
            dot(slide, px, py, r, colour_of(group))
    return centres


def figure_one(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "Figure 1 | Confidence-UOT: what goes in, what comes out",
          "Two unpaired cell populations in; every cell labelled retain or reject, and a coupling only between retained cells out")
    source, target = clusters()
    head, y0 = 1.55, 1.9

    for x, label in ((0.55, "INPUT"), (4.62, "MODEL"), (9.42, "OUTPUT")):
        textbox(slide, x, head, 3.5, 0.32, [label], size=13, colour=MUTED, bold=True)

    # INPUT: two unlabelled samples (grey: nobody knows yet which cells have a partner)
    for k, (name, sub, pts) in enumerate((("Source sample", "e.g. primary tumour cells", source),
                                           ("Target sample", "e.g. metastasis cells", target))):
        y = y0 + k * 2.35
        box(slide, 0.55, y, 3.55, 2.15)
        textbox(slide, 0.72, y + 0.08, 3.3, 0.36, [name], size=16, colour=NAVY, bold=True)
        textbox(slide, 0.72, y + 0.42, 3.3, 0.3, [sub], size=11.5, colour=MUTED)
        draw_cloud(slide, pts, 0.75, y + 0.72, 3.15, 1.32, lambda g: UNKNOWN)
    textbox(slide, 0.55, 6.5, 3.6, 0.75, [("raw counts, cells × genes; no cell is measured twice and none carries a label",
                                         {"size": 11.5, "colour": MUTED})])

    block_arrow(slide, 4.17, 3.85, 0.38, 0.45)

    # MODEL: three steps, the third a loop
    mx, mw = 4.62, 4.38
    box(slide, mx, y0, mw, 0.95, fill=WHITE)
    textbox(slide, mx + 0.15, y0 + 0.07, mw - 0.3, 0.85, [
        ("1  One shared space, one cost", {"size": 15, "colour": NAVY, "bold": True, "after": 2}),
        ("depth-equalise → rank top 256 → PCA 30 → cosine", {"size": 12, "after": 0}),
        ("C_{ij} = ‖x_{i} − y_{j}‖^{2} / median pair cost", {"size": 12})])
    block_arrow(slide, mx + mw / 2 - 0.13, y0 + 0.97, 0.26, 0.2, direction=MSO_SHAPE.DOWN_ARROW)
    box(slide, mx, y0 + 1.2, mw, 1.05, fill=WHITE, border=CALIB, width=1.5, dashed=True)
    textbox(slide, mx + 0.15, y0 + 1.27, mw - 0.3, 0.95, [
        ("2  Price of abstention c, calibrated", {"size": 15, "colour": CALIB, "bold": True, "after": 2}),
        ("split one sample into halves, where every cell surely has a partner; c = the smallest price that keeps ≥ 99% of them",
         {"size": 12})])
    block_arrow(slide, mx + mw / 2 - 0.13, y0 + 2.27, 0.26, 0.2, direction=MSO_SHAPE.DOWN_ARROW)
    box(slide, mx, y0 + 2.5, mw, 2.25, fill="EEF4FA", border=RETAIN, width=1.5)
    textbox(slide, mx + 0.15, y0 + 2.56, mw - 0.3, 0.36,
            [("3  Confidence-UOT, fitted jointly", {"size": 15, "colour": RETAIN, "bold": True})])
    box(slide, mx + 0.3, y0 + 2.95, mw - 0.6, 0.62, fill=WHITE)
    textbox(slide, mx + 0.38, y0 + 2.98, mw - 0.76, 0.58, [
        ("transport plan P", {"size": 13.5, "bold": True, "colour": NAVY, "align": PP_ALIGN.CENTER, "after": 0}),
        ("entropic, unbalanced, under the current gates", {"size": 11.5, "colour": MUTED, "align": PP_ALIGN.CENTER})])
    box(slide, mx + 0.3, y0 + 4.0, mw - 0.6, 0.62, fill=WHITE)
    textbox(slide, mx + 0.38, y0 + 4.03, mw - 0.76, 0.58, [
        ("gates δ_{i}, η_{j} ∈ {0, 1}", {"size": 13.5, "bold": True, "colour": NAVY, "align": PP_ALIGN.CENTER, "after": 0}),
        ("keep cell i  ⇔  its cost to partners d_{i} < c", {"size": 11.5, "colour": MUTED, "align": PP_ALIGN.CENTER})])
    block_arrow(slide, mx + 1.15, y0 + 3.6, 0.28, 0.37, colour=RETAIN, direction=MSO_SHAPE.DOWN_ARROW)
    block_arrow(slide, mx + mw - 1.43, y0 + 3.6, 0.28, 0.37, colour=RETAIN, direction=MSO_SHAPE.UP_ARROW)
    textbox(slide, mx + 0.2, y0 + 3.62, 0.9, 0.32, [("score", {"size": 11, "colour": RETAIN, "align": PP_ALIGN.RIGHT})])
    textbox(slide, mx + mw - 1.1, y0 + 3.62, 0.95, 0.32, [("re-solve", {"size": 11, "colour": RETAIN})])
    textbox(slide, mx, y0 + 4.8, mw, 0.32,
            [("repeat until no gate changes", {"size": 11.5, "colour": MUTED, "align": PP_ALIGN.CENTER})])

    block_arrow(slide, 9.03, 3.85, 0.38, 0.45)

    # OUTPUT: the same cells, labelled, and the coupling between retained cells
    ox, ow = 9.42, 3.4
    box(slide, ox, y0, ow, 4.5)
    colour = lambda g: REJECT if g in ("private", "emerged") else RETAIN  # noqa: E731
    textbox(slide, ox + 0.15, y0 + 0.07, 1.5, 0.3, [("source", {"size": 12, "colour": MUTED, "bold": True})])
    s_centres = draw_cloud(slide, source, ox + 0.2, y0 + 0.35, ow - 0.4, 1.3, colour)
    t_centres = draw_cloud(slide, target, ox + 0.2, y0 + 2.85, ow - 0.4, 1.3, colour)
    textbox(slide, ox + 0.15, y0 + 4.15, 1.5, 0.3, [("target", {"size": 12, "colour": MUTED, "bold": True})])
    rng = np.random.default_rng(3)
    for group in ("shared1", "shared2"):
        for i in rng.choice(len(s_centres[group]), 4, replace=False):
            j = rng.integers(len(t_centres[group]))
            (x1, y1), (x2, y2) = s_centres[group][i], t_centres[group][j]
            line(slide, x1, y1 + 0.05, x2, y2 - 0.05, colour="A9C1DA", width=0.75)
    textbox(slide, ox + ow - 1.45, y0 + 1.95, 1.35, 0.75, [
        ("coupling P only between retained cells", {"size": 11, "colour": RETAIN, "align": PP_ALIGN.RIGHT})],
        anchor=MSO_ANCHOR.MIDDLE)
    legend_y = y0 + 4.6
    dot(slide, ox + 0.12, legend_y + 0.15, 0.07, RETAIN)
    textbox(slide, ox + 0.27, legend_y, 3.2, 0.3, [("retain: correspondence trusted", {"size": 12})])
    dot(slide, ox + 0.12, legend_y + 0.45, 0.07, REJECT)
    textbox(slide, ox + 0.27, legend_y + 0.3, 3.2, 0.3, [("reject: abstain, no supported partner", {"size": 12})])
    textbox(slide, ox, legend_y + 0.64, ow, 0.45,
            [("per cell also the margin d_{i} − c; the rejected fraction is estimated, not set in advance",
              {"size": 10.5, "colour": MUTED})])
    notes(slide, "Input: two unpaired snapshots, raw counts, no labels (grey). Model: one shared representation and cost; "
                 "a price c for abstaining, calibrated on within-sample splits at acceptance 0.99; then Confidence-UOT "
                 "alternates the transport plan and binary gates. Output: the same cells, each retained (blue) or "
                 "rejected (orange), and a coupling only among retained cells. Rejection means abstaining from a "
                 "correspondence, not that a population disappeared or emerged.")


# --- the method -------------------------------------------------------------------------

def step_space(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "Step 1 | One shared space, one cost matrix",
          "The same fixed preprocessing for every pair (label ranknm256_noscale_ds_cos); nothing in it sees which cells match")
    chips = [("raw counts", "cells \u00d7 genes, both samples"),
             ("depth-equalise", "subsample reads to one global depth"),
             ("rank top 256", "each cell keeps its 256 highest genes, by rank"),
             ("2,000 HVGs \u2192 PCA 30", "joint over both samples; genes not z-scored"),
             ("unit length", "rows L2-normalised (cosine geometry)")]
    x = 0.55
    for k, (head, sub) in enumerate(chips):
        box(slide, x, 1.95, 2.25, 1.1, fill=WHITE)
        textbox(slide, x + 0.1, 2.05, 2.05, 1.2, [(head, {"size": 15, "bold": True, "colour": NAVY, "align": PP_ALIGN.CENTER, "after": 4}),
                                                  (sub, {"size": 11.5, "colour": MUTED, "align": PP_ALIGN.CENTER})])
        if k < len(chips) - 1:
            block_arrow(slide, x + 2.27, 2.34, 0.22, 0.32)
        x += 2.5
    textbox(slide, 0.55, 3.75, 6, 0.4, [("Cost between source cell i and target cell j", {"size": 16, "bold": True, "colour": NAVY})])
    place_equation(slide, r"C_{ij}=\frac{\|x_i-y_j\|_2^2}{\mathrm{median}_{(k,l)}\ \|x_k-y_l\|_2^2},\qquad (k,l)\ \mathrm{sampled\ source\ /\ target\ pairs}",
                   "cost", 0.75, 4.2, size=22)
    textbox(slide, 0.55, 5.45, 12.2, 1.6, [
        ("The median sampled cross-side pair cost is 1, so the price c and the entropy \u03b5 are in units of a typical pair.", {"bullet": True, "size": 14}),
        ("Up to 10\u2076 pairs are sampled to estimate the median (02_run_pair.py; preprocessing.median_pair_scale).", {"bullet": True, "size": 14}),
        ("Rank and depth equalisation remove sequencing depth before the cost; cosine removes the remaining scale.", {"bullet": True, "size": 14})])
    notes(slide, "Production preprocessing: depth equalisation is done once at file level (27_downsample_counts.py), then "
                 "rank_no_median top 256, 2,000 HVGs, PCA 30, no per-gene z-score, L2 rows. Cost = squared Euclidean "
                 "divided by the median of sampled cross-side pairs.")


def step_objective(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "Step 2 | The objective: transport with a price for abstaining",
          "Binary gates on both sides; a gated-off pair pays the constant c instead of its cost")
    textbox(slide, 0.55, 1.75, 6.5, 0.4, [("Gated cost", {"size": 16, "bold": True, "colour": NAVY})])
    place_equation(slide, r"\widetilde C_{ij}=\delta_i\eta_jC_{ij}+(1-\delta_i\eta_j)\,c,\qquad \delta_i,\eta_j\in\{0,1\}",
                   "gated", 0.75, 2.2, size=24)
    textbox(slide, 0.55, 3.05, 6.5, 0.4, [("Objective (Aim 2)", {"size": 16, "bold": True, "colour": NAVY})])
    place_equation(slide, r"\mathcal{L}(P,\delta,\eta)=\langle P,\widetilde C\rangle+\varepsilon\,\mathrm{KL}(P\,\|\,ab^{\mathsf{T}})"
                          r"+\lambda_s\,\mathrm{KL}(P\mathbf{1}\,\|\,a)+\lambda_t\,\mathrm{KL}(P^{\mathsf{T}}\mathbf{1}\,\|\,b)",
                   "objective", 0.75, 3.5, size=24, max_w=11.8)
    rows = [("a_{i} = 1/n,  b_{j} = 1/m", "uniform mass on the source and target cells"),
            ("\u03b5 = 0.1", "entropic smoothing, in median-pair units"),
            ("\u03bb_{s} = \u03bb_{t} = 1.0", "soft marginals: abundance may change between samples (KL-UOT)"),
            ("c", "the price of abstaining; calibrated in Step 5, one value for both sides"),
            ("KL(x\u2016y) = \u03a3 x log(x/y) \u2212 x + y", "generalised KL")]
    y = 4.55
    for left, right in rows:
        textbox(slide, 0.75, y, 3.6, 0.36, [(left, {"size": 14, "colour": NAVY, "bold": True})])
        textbox(slide, 4.4, y, 8.3, 0.36, [(right, {"size": 14})])
        y += 0.42
    textbox(slide, 0.55, 6.72, 12.2, 0.5, [("The KL marginal penalties model graded abundance change, whereas gates encode whether "
                                            "endpoint-specific correspondence geometry is trusted.", {"size": 13, "colour": MUTED})])
    notes(slide, "Objective exactly as in the proposal Aim 2. Production values: epsilon 0.1, lambda_s = lambda_t = 1.0, "
                 "uniform a and b, c from the within-side calibration.")


def step_solver(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "Step 3 | Solving it: alternate the plan and the gates",
          "For fixed gates the problem is strictly convex entropic UOT; for a fixed plan the best gates have a closed form")
    # left: Sinkhorn
    box(slide, 0.55, 1.7, 6.0, 3.15, fill=WHITE)
    textbox(slide, 0.75, 1.8, 5.6, 0.4, [("(a) gates fixed \u2192 generalised Sinkhorn", {"size": 16, "bold": True, "colour": NAVY})])
    place_equation(slide, r"K_{ij}=a_ib_j\,e^{-\widetilde C_{ij}/\varepsilon},\qquad \alpha=\frac{\lambda_s}{\lambda_s+\varepsilon},\ \ \beta=\frac{\lambda_t}{\lambda_t+\varepsilon}",
                   "sinkhorn1", 0.8, 2.3, size=19, max_w=5.6)
    place_equation(slide, r"u\leftarrow\left(\frac{a}{Kv}\right)^{\alpha},\qquad v\leftarrow\left(\frac{b}{K^{\mathsf{T}}u}\right)^{\beta},\qquad P=\mathrm{diag}(u)\,K\,\mathrm{diag}(v)",
                   "sinkhorn2", 0.8, 3.1, size=19, max_w=5.6)
    textbox(slide, 0.75, 3.95, 5.7, 0.85, [
        ("log domain; stop when max |\u0394log u|, |\u0394log v| < 10^{\u22124} (\u2264 20,000 iterations), warm-started", {"size": 12.5, "colour": MUTED})])
    # right: gate update
    box(slide, 6.8, 1.7, 6.0, 3.15, fill=WHITE)
    textbox(slide, 7.0, 1.8, 5.6, 0.4, [("(b) plan fixed \u2192 exact gate update", {"size": 16, "bold": True, "colour": NAVY})])
    place_equation(slide, r"s_i=\sum_jP_{ij}\,\eta_j\,(C_{ij}-c),\qquad \delta_i=\mathbf{1}[s_i<0]", "gate_s", 7.05, 2.3, size=19, max_w=5.6)
    place_equation(slide, r"t_j=\sum_iP_{ij}\,\delta_i\,(C_{ij}-c),\qquad \eta_j=\mathbf{1}[t_j<0]", "gate_t", 7.05, 3.1, size=19, max_w=5.6)
    textbox(slide, 7.0, 3.95, 5.7, 0.85, [
        ("source first, then target with the new \u03b4 (Gauss\u2013Seidel); exact because entropy and KL do not depend on the gates", {"size": 12.5, "colour": MUTED})])
    block_arrow(slide, 6.57, 2.9, 0.22, 0.4)
    # loop summary
    box(slide, 0.55, 5.1, 12.25, 1.65, fill=PANEL)
    textbox(slide, 0.75, 5.2, 11.9, 1.5, [
        ("start with every gate open (\u03b4 = \u03b7 = 1) \u2192 (a) \u2192 (b) \u2192 (a) \u2192 \u2026  until no gate changes, at most 30 rounds", {"size": 15, "bold": True, "colour": NAVY, "after": 6}),
        ("then one final Sinkhorn solve with the final gates; scores and margins are read from that plan", {"bullet": True, "size": 13.5}),
        ("M4-E (\u201cexact\u201d): the update above, a block minimiser, so the objective never increases; used for every result", {"bullet": True, "size": 13.5}),
        ("M4-R (\u201creversible\u201d): scores each cell by its ungated counterfactual cost, so rejected cells can return; fitted on the same c as a cross-check", {"bullet": True, "size": 13.5})])
    callout(slide, 0.55, 6.88, 12.25, "Area B",
            "Sinkhorn stops at a tolerance, so P and the scores s, t are approximate: when are the gate signs still certified?")
    notes(slide, "Production: tolerance 1e-4, 20,000 inner iterations, 30 outer, gate tolerance 0. A repeated gate pattern is "
                 "flagged as a cycle. Area B is about exactly this step: the Sinkhorn plan is approximate, so the gate "
                 "scores s and t carry numerical error.")


def step_decision(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "Step 4 | Retain or reject: one comparison per cell",
          "A cell is kept while transporting it to its current partners costs less than the price of leaving it out")
    place_equation(slide, r"s_i=\sum_jP_{ij}\eta_j(C_{ij}-c)=m_i\,(d_i-c),\qquad m_i=\sum_jP_{ij}\eta_j,\qquad d_i=\frac{\sum_jP_{ij}\eta_jC_{ij}}{m_i}",
                   "decision", 0.75, 1.75, size=22, max_w=11.8)
    textbox(slide, 0.55, 3.05, 12.2, 0.5, [("m_{i}: the cell's mass sent to open partners;  d_{i}: its average cost to them.   "
                                           "Since m_{i} > 0,  retain i  \u21d4  s_{i} < 0  \u21d4  d_{i} < c", {"size": 15})])
    # number line
    y = 4.55
    line(slide, 1.2, y, 11.9, y, colour="8A94A3", width=1.5, arrow=True)
    cx = 7.3
    line(slide, cx, y - 0.55, cx, y + 0.45, colour=CALIB, width=2.25)
    textbox(slide, cx - 0.6, y + 0.48, 1.2, 0.35, [("c", {"size": 18, "colour": CALIB, "bold": True, "align": PP_ALIGN.CENTER})])
    textbox(slide, 11.2, y + 0.12, 1.4, 0.35, [("d_{i}", {"size": 16, "colour": MUTED})])
    rng = np.random.default_rng(5)
    for v in rng.normal(4.5, 1.1, 22):
        dot(slide, float(np.clip(v, 1.4, cx - 0.2)), y - 0.25, 0.08, RETAIN)
    for v in rng.normal(9.6, 0.8, 7):
        dot(slide, float(np.clip(v, cx + 0.25, 11.6)), y - 0.25, 0.08, REJECT)
    textbox(slide, 1.2, y - 0.85, 5.5, 0.4, [("retain: d_{i} < c, margin d_{i} \u2212 c < 0", {"size": 14, "colour": RETAIN, "bold": True})])
    textbox(slide, cx + 0.3, y - 0.85, 4.6, 0.4, [("reject: d_{i} > c, margin > 0", {"size": 14, "colour": REJECT, "bold": True})])
    callout(slide, 0.55, 6.88, 12.25, "Area A",
            "Abundance change reshapes P itself: when does a shared population still get s_{k} < 0 and an unsupported one "
            "s_{u} > 0? Proposed: max_{k} L_{k} < c < d_{priv}")
    textbox(slide, 0.55, 5.42, 12.2, 1.45, [
        ("Target cells are decided the same way with t_{j} and the new source gates.", {"bullet": True, "size": 14}),
        ("Output per cell: retained (the final gate) and the signed margin d_{i} \u2212 c (cell_confidence.csv); the coupling P among retained cells.", {"bullet": True, "size": 14}),
        ("No quota: the rejection budget 0.95 is recorded but not enforced, so the rejected fraction is an estimate, not a setting.", {"bullet": True, "size": 14}),
        ("Rejection means abstaining from a correspondence, not that a population disappeared or emerged.", {"bullet": True, "size": 14})])
    notes(slide, "The dots on the line are a schematic of the rule, not data. In the code: retained is the final gate; "
                 "signed_rejection_margin = decision_cost - c; budget 0.95 is not enforced (enforce_rejection_budget False), "
                 "so the gate is unconstrained.")


def step_calibration(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "Step 5 | What 0.99 means: calibrating the price c",
          "c is not chosen by hand: it is read off a null in which every cell certainly has a counterpart")
    steps = [("split", "take up to 2,000 cells of one sample; split them at random into two halves"),
             ("why", "both halves come from one population: every cell has a true partner, so any rejection is a false one"),
             ("fit", "run M4-E on half-vs-half at a grid of c (5 values, 10 nulls: 5 from source splits, 5 from target splits)"),
             ("pick", "c* = the smallest c whose mean acceptance is \u2265 0.99 on both rows and columns; bisect to 0.1%"),
             ("check", "re-fit M4-R at c* on 10 held-out nulls; the same \u2265 0.99 test")]
    y = 1.7
    for k, (head, text) in enumerate(steps):
        dot(slide, 0.85, y + 0.24, 0.2, CALIB)
        textbox(slide, 0.65, y + 0.06, 0.4, 0.36, [(str(k + 1), {"size": 14, "bold": True, "colour": WHITE, "align": PP_ALIGN.CENTER})])
        textbox(slide, 1.25, y, 1.0, 0.45, [(head, {"size": 15, "bold": True, "colour": CALIB})])
        textbox(slide, 2.25, y, 5.0, 0.6, [(text, {"size": 13})])
        y += 0.78
    textbox(slide, 0.65, 5.68, 6.7, 1.3, [
        ("0.99 = at the chosen price, at least 99% of cells that surely have a partner are kept: the gate is tuned on false rejections only.", {"size": 14, "bold": True, "colour": NAVY, "after": 4}),
        ("One c for both samples and both gate variants; the cost of every real pair is then judged against it.", {"size": 13, "colour": MUTED})])
    # right: measured effect of the acceptance minimum
    box(slide, 7.65, 1.7, 5.15, 5.25, fill=PANEL)
    textbox(slide, 7.85, 1.8, 4.8, 0.7, [("Native simulation, M4-E, ranknm256_noscale_ds_cos", {"size": 13.5, "bold": True, "colour": NAVY, "after": 0}),
                                         ("mean over 3 sizes \u00d7 5 replicates (benchmark_results/native)", {"size": 11, "colour": MUTED})])
    header = ["acceptance \u2265", "0.90", "0.95", "0.99"]
    data = [("directional F1", ["0.917", "0.958", "0.992"]),
            ("shared cells rejected, source", ["5.6\u20137.8%", "2.0\u20133.5%", "0.2\u20130.5%"]),
            ("shared cells rejected, target", ["6.5\u20139.1%", "3.3\u20134.9%", "0.6\u20131.1%"]),
            ("unmatched population rejected", ["100%", "99.9\u2013100%", "99.7\u2013100%"]),
            ("emerged case: source rejected", ["33\u201345%", "20\u201334%", "4.8\u201315%"])]
    table = slide.shapes.add_table(len(data) + 1, 4, Inches(7.85), Inches(2.6), Inches(4.75), Inches(3.0)).table
    widths = [2.15, 0.86, 0.87, 0.87]
    for k, w in enumerate(widths):
        table.columns[k].width = Inches(w)
    for r, row in enumerate([header] + [[a] + b for a, b in data]):
        for c_, value in enumerate(row):
            cell = table.cell(r, c_)
            cell.text = ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(NAVY if r == 0 else (WHITE if r % 2 else "EDF0F4"))
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c_ == 0 else PP_ALIGN.CENTER
            rich(p, value, 11.5, WHITE if r == 0 else (CALIB if c_ == 3 and r > 0 else INK), bold=(r == 0 or c_ == 3))
            cell.margin_left = cell.margin_right = Inches(0.05)
    textbox(slide, 7.85, 5.75, 4.8, 1.1, [("Ranges are over the three technical levels (matched depth, depth mismatch, depth + dropout). "
                                           "0.99 is used for every real-data result.", {"size": 11, "colour": MUTED})])
    notes(slide, "Code: within_side_null_costs and calibrate_confidence_cost (calibration.py); smallest jointly feasible c, "
                 "geometric bisection to relative 1e-3, M4-R validation on held-out nulls. The table is "
                 "method_comparison_heatmap_values.csv and method_comparison_f1.csv for M4-E on ranknm256_noscale_ds_cos.")


# --- the proposal's Areas, as written ---------------------------------------------------

def area_a(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "Area A: Applied Mathematics", "From the proposal abstract, unchanged")
    textbox(slide, 0.55, 1.65, 12.2, 1.4, [
        ("*Problem.* Area A asks one question: under what conditions does Confidence-UOT preserve shared-population gates under "
         "abundance variation while rejecting a geometrically unsupported population? The difficulty is that abundance mismatch "
         "changes the UOT coupling itself, so a valid shared population can acquire transport to geometrically incorrect "
         "populations and appear unreliable.", {"size": 14})])
    textbox(slide, 0.55, 2.62, 12.2, 1.1, [
        ("*Model and proposed claim.* I will study the anchored block model from Aim 2: *K* shared populations, optionally one "
         "source-private population *u*, active target gates, low within-population cost, separated cross/private costs, and "
         "bounded abundance ratios. The average transport cost paid by shared population *k*, including leakage into incorrect "
         "populations, will be bounded as", {"size": 14})])
    place_equation(slide, r"\bar D_k\leq L_k", "areaA_bound", 0.55, 3.72, size=22, centre_w=12.2)
    textbox(slide, 0.55, 4.2, 12.2, 0.4, [("using the UOT scaling equations. The proposed result establishes conditions under which", {"size": 14})])
    place_equation(slide, r"\max_kL_k<c<d_{\mathrm{priv}}\quad\Longrightarrow\quad s_k<0\ \mathrm{for\ shared\ populations},\qquad s_u>0\ \mathrm{for\ the\ private\ population},",
                   "areaA_claim", 0.55, 4.62, size=20, centre_w=12.2, max_w=12.0)
    textbox(slide, 0.55, 5.18, 12.2, 1.9, [
        ("with the signs stable after re-solving. Thus, shared populations remain favored for acceptance while the unsupported "
         "private population is favored for rejection.", {"size": 14, "after": 6}),
        ("*Establishment and scope.* The bound will be derived from off-block leakage in the UOT scaling equations. Balanced "
         "transport removes marginal relaxation, making it a useful control for gating when abundance cannot be remodeled. "
         "Exact gate updates, finite termination, rejected-row structure, and source/target ambiguity remain supporting facts "
         "and limitations. An empty sufficient interval means only that the certificate fails.", {"size": 14})])
    notes(slide, "Text copied from CSEM_Proposal_Abstract.tex, section Area A. In the algorithm this is Step 4: s_k is the "
                 "gate score, c the calibrated price; the claim is that abundance change alone cannot push a shared "
                 "population over the price.")


def area_b(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "Area B: Numerical Analysis and Scientific Computation", "From the proposal abstract, unchanged")
    textbox(slide, 0.55, 1.65, 12.2, 1.1, [
        ("*Problem.* Area B asks when the gate set from an approximate Confidence-UOT computation can be certified to match the "
         "exact fixed-gate decision. This is a numerical-analysis problem because even a small error in the coupling can change "
         "the gate-score signs or rankings and therefore change which endpoints are rejected.", {"size": 14})])
    textbox(slide, 0.55, 2.78, 12.2, 0.4, [("*Error quantities.* I will derive bounds", {"size": 14})])
    place_equation(slide, r"E_s\geq\|\widehat s-s\|_\infty,\qquad E_t\geq\|\widehat t-t\|_\infty,", "areaB_bounds", 0.55, 3.18, size=22, centre_w=12.2)
    textbox(slide, 0.55, 3.72, 12.2, 2.9, [
        ("where *E*_{s} and *E*_{t} bound numerical perturbations of the source- and target-side gate scores.", {"size": 14, "after": 8}),
        ("*Certification claim.* Error below a sign margin certifies the corresponding sign decision. Under a binding rejection "
         "budget, error below half the ranking gap at the selection boundary certifies the selected gate set. These margin "
         "implications are elementary.", {"size": 14, "after": 8}),
        ("*How it will be established.* The proposed numerical contribution is to turn runtime solver or kernel information into "
         "a coupling-error bound and then into *E*_{s}, *E*_{t}, using quantities such as a primal-dual gap, justified fixed-point "
         "residual, or cost/kernel approximation error. When certification fails, the method reports ambiguous endpoints and an "
         "upper bound on how much objective value is lost by using the uncertified gate block. Log-domain Sinkhorn, batching, "
         "GPU execution, blockwise kernels, and unit tests are supporting engineering, not the Area B contribution.", {"size": 14})])
    notes(slide, "Text copied from CSEM_Proposal_Abstract.tex, section Area B. In the algorithm this is Step 3: Sinkhorn stops at "
                 "tolerance 1e-4, so the plan and therefore s and t are approximate; a sign is certified when the score's "
                 "margin exceeds E.")


def positioning(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "Method positioning", "From the proposal abstract, unchanged")
    textbox(slide, 0.55, 1.75, 12.2, 3.2, [
        ("*Method positioning.* Partial OT handles unmatched mass directly. Robust OT and truncated-cost methods avoid expensive "
         "correspondences by modifying the objective. IC-POT uses endpoint-specific unmatched costs and slack variables. "
         "Confidence-UOT instead keeps KL-relaxed transport and adds a binary endpoint-level decision about whether "
         "correspondence geometry is trusted. It is not claimed as the first robust or rejection-based method. If partial "
         "transport assigns zero mass to an endpoint, its confidence score can also collapse to zero, making the mechanisms "
         "redundant.", {"size": 16})])
    notes(slide, "Text copied from CSEM_Proposal_Abstract.tex, Area B method positioning.")


def cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = rgb(NAVY)
    bg.line.fill.background()
    textbox(slide, 0.9, 2.3, 11.5, 1.0, [("Confidence-UOT", {"size": 44, "bold": True, "colour": WHITE, "font": TITLE_FONT})])
    textbox(slide, 0.9, 3.35, 11.5, 1.0, [("Unbalanced optimal transport with an explicit retain / reject decision for every cell",
                                          {"size": 22, "colour": "D6DEEA"})])
    textbox(slide, 0.9, 4.6, 11.5, 0.5, [("Aim 2 \u00b7 the algorithm, its parameters, and the questions of Areas A and B",
                                          {"size": 16, "colour": "9FB0C8"})])
    for k, colour in enumerate((RETAIN, RETAIN, REJECT)):
        dot(slide, 0.98 + k * 0.32, 1.95, 0.1, colour)


def main() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    cover(prs)
    figure_one(prs)
    step_space(prs)
    step_objective(prs)
    step_solver(prs)
    step_decision(prs)
    step_calibration(prs)
    area_a(prs)
    area_b(prs)
    positioning(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"wrote {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
