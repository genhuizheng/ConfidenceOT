"""ConfidenceOT_20261015_v3.pptx: v2 plus the full KP-Tracer tumour-level section.

Starts from ConfidenceOT_20261015_v2.pptx (only read). Final order of the KP-Tracer part:

20 KP-Tracer: Independent Lineage Evidence            (kept from v2)
21 How CRISPR Lineage Tracing Works                   (new; schematic, verified against Yang 2022)
22 Tumor-Level Validation: Which Primary Matches Each Metastasis?   (v2 strategy slide, redrawn)
23 Retained Fractions Reveal Candidate Primary Matches (new; heatmap, every mouse, real data)
24 Tumor-Level Concordance Across Lesions             (v2, retitled)
25 Tumor-Level Consistency Across Mice                (v2, retitled)
26 Candidate Ranking Beyond d_M                       (new; exploratory ranking metrics, both labels)
27 Does RNA Preprocessing Change Tumor-Family Discrimination?  (new; configuration table, paired mice)
28 Raw vs Preprocessed RNA: Matched Heatmaps          (new)
29 CNV and Multimodal Representations: Not Yet Evaluated       (new; feasibility only)
30 Summary: What the Tumor-Level Benchmark Supports   (new)
31 Summary and next steps                             (kept)

Every number and every heatmap cell is read from the verified export
(results/kp_tracer_lineage_validation/export) or from the exploratory tables written by
kp_lineage/tumour_level/01_ranking_and_representations.py. Schematics say so on the slide.
"""
import importlib.util
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib
import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.dml import MSO_PATTERN
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("deck_v2", HERE / "build_deck_20261015_v2.py")
v2 = importlib.util.module_from_spec(_spec)
sys.modules["deck_v2"] = v2
_spec.loader.exec_module(v2)
text, box, arrow, segment, dots, notes = v2.text, v2.box, v2.arrow, v2.segment, v2.dots, v2.notes
ORANGE, LIGHT, FILL, INK, MUTED, GREY, PALE = v2.ORANGE, v2.LIGHT, v2.FILL, v2.INK, v2.MUTED, v2.GREY, v2.PALE
RETAINED, REJECTED, FONT = v2.RETAINED, v2.REJECTED, v2.FONT
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SRC = Path(r"D:\Xia_lab\Progress_report\progress_report7\ConfidenceOT_20261015_v2.pptx")
OUT = Path(r"D:\Xia_lab\Progress_report\progress_report7\ConfidenceOT_20261015_v3.pptx")
TMP = OUT.with_name("_tmp_v3.pptx")
EXPORT = Path(r"D:\Xia_lab\CellOT\results\kp_tracer_lineage_validation\export")
EXPLORE = Path(r"D:\Xia_lab\CellOT\results\kp_tracer_lineage_validation\tumour_level_exploratory")
MAIN, RAW = "ranknm256_noscale_ds_cos", "raw"
NEW_SLIDES = 7
BLUES = matplotlib.colormaps["Blues"]


def colour(value: float) -> RGBColor:
    r, g, b, _ = BLUES(0.04 + 0.96 * float(value))
    return RGBColor(int(r * 255), int(g * 255), int(b * 255))


def natural(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


# --- data ----------------------------------------------------------------------------

def heatmap_data(label: str) -> dict:
    """Per mouse: primaries, lesions, value per cell (r, or None when not run), family flag."""
    pairs = pd.read_csv(EXPORT / "readout_tumour_level" / f"pairs_{label}.csv")
    pairs = pairs[pairs["source_sample"].notna() & pairs["source_sample"].astype(str).ne("")]
    pairs["same_met_family"] = pairs["same_met_family"].fillna(False).astype(bool)
    out = {}
    for mouse, group in pairs.groupby("patient_id"):
        done = group[group["status"].eq("completed")]
        if done.empty:
            continue
        primaries = sorted(group["source_sample"].astype(str).unique(), key=natural)
        lesions = sorted(done["target_sample"].astype(str).unique(), key=natural)
        cells, informative = {}, {}
        for lesion in lesions:
            row = done[done["target_sample"].eq(lesion)]
            informative[lesion] = bool(row["same_met_family"].any() and (~row["same_met_family"]).any())
            for primary in primaries:
                hit = group[group["target_sample"].eq(lesion) & group["source_sample"].eq(primary)]
                if hit.empty:
                    cells[(lesion, primary)] = (None, False)
                    continue
                record = hit.iloc[0]
                value = float(record["source_retained_fraction"]) if record["status"] == "completed" else None
                cells[(lesion, primary)] = (value, bool(record["same_met_family"]))
        genotype = primaries[0].split("_")[1]
        out[str(mouse)] = {"primaries": primaries, "lesions": lesions, "cells": cells, "informative": informative,
                           "genotype": genotype}
    return out


def short(name: str) -> str:
    return name.split("_", 2)[2]


# --- heatmap drawing ---------------------------------------------------------------------

def draw_block(slide, mouse: str, data: dict, x: float, y: float, cell: float, label_w: float, font=10) -> tuple:
    block = data[mouse]
    text(slide, x, y, label_w + cell * len(block["primaries"]) + 0.1, 0.28,
         [[(mouse, {"bold": True}), (f"  {block['genotype']}", {"color": MUTED})]], size=font + 1)
    top = y + 0.30
    for c, primary in enumerate(block["primaries"]):
        text(slide, x + label_w + c * cell, top, cell, 0.24, short(primary), size=font - 1, color=MUTED,
             align=PP_ALIGN.CENTER)
    grid_top = top + 0.26
    for r, lesion in enumerate(block["lesions"]):
        yy = grid_top + r * cell
        text(slide, x, yy, label_w - 0.04, cell, short(lesion), size=font - 1,
             color=INK if block["informative"][lesion] else GREY, italic=not block["informative"][lesion],
             align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
        for c, primary in enumerate(block["primaries"]):
            value, family = block["cells"][(lesion, primary)]
            square = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + label_w + c * cell + 0.015),
                                            Inches(yy + 0.015), Inches(cell - 0.03), Inches(cell - 0.03))
            square.shadow.inherit = False
            if value is None:
                square.fill.patterned()
                square.fill.pattern = MSO_PATTERN.LIGHT_UPWARD_DIAGONAL
                square.fill.fore_color.rgb = GREY
                square.fill.back_color.rgb = WHITE
            else:
                square.fill.solid()
                square.fill.fore_color.rgb = colour(value)
            if family:
                square.line.color.rgb = INK
                square.line.width = Pt(2.25)
            else:
                square.line.fill.background()
    width = label_w + cell * len(block["primaries"])
    height = 0.56 + cell * len(block["lesions"])
    return width, height


def draw_heatmaps(slide, data: dict, mice: list, x0: float, y0: float, x_max: float, cell: float, label_w: float,
                  gap: float = 0.3, font=10) -> float:
    x, y, row_height = x0, y0, 0.0
    for mouse in mice:
        block = data[mouse]
        width = label_w + cell * len(block["primaries"])
        if x + width > x_max and x > x0:
            x, y, row_height = x0, y + row_height + 0.2, 0.0
        w, h = draw_block(slide, mouse, data, x, y, cell, label_w, font)
        x += w + gap
        row_height = max(row_height, h)
    return y + row_height


def draw_legend(slide, x: float, y: float, vertical=True) -> None:
    text(slide, x, y, 2.4, 0.3, "retained fraction", size=12, color=MUTED)
    for k in range(11):
        square = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + k * 0.2), Inches(y + 0.32), Inches(0.2),
                                        Inches(0.22))
        square.fill.solid()
        square.fill.fore_color.rgb = colour(k / 10)
        square.line.fill.background()
        square.shadow.inherit = False
    text(slide, x - 0.05, y + 0.56, 0.4, 0.25, "0", size=11, color=MUTED)
    text(slide, x + 0.95, y + 0.56, 0.4, 0.25, "0.5", size=11, color=MUTED)
    text(slide, x + 2.05, y + 0.56, 0.4, 0.25, "1", size=11, color=MUTED)
    items = y + 0.95
    outline = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(items), Inches(0.24), Inches(0.24))
    outline.fill.solid()
    outline.fill.fore_color.rgb = colour(0.5)
    outline.line.color.rgb = INK
    outline.line.width = Pt(2.25)
    outline.shadow.inherit = False
    text(slide, x + 0.34, items - 0.02, 2.6, 0.3, "same family (lineage labels)", size=12, color=MUTED)
    hatch = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(items + 0.4), Inches(0.24), Inches(0.24))
    hatch.fill.patterned()
    hatch.fill.pattern = MSO_PATTERN.LIGHT_UPWARD_DIAGONAL
    hatch.fill.fore_color.rgb = GREY
    hatch.fill.back_color.rgb = WHITE
    hatch.line.fill.background()
    hatch.shadow.inherit = False
    text(slide, x + 0.34, items + 0.38, 2.6, 0.3, "not run (< 20 malignant cells)", size=12, color=MUTED)
    text(slide, x, items + 0.8, 2.9, 0.5, [[("K1", {"italic": True, "color": GREY}),
                                            ("  lesion without both family groups (not in d", {"color": MUTED}),
                                            ("M", {"color": MUTED, "baseline": -25000}), (")", {"color": MUTED})]],
         size=12)


# --- slides ------------------------------------------------------------------------------

def slide_crispr(slide) -> None:
    columns = [0.45, 3.6, 6.75, 9.9]
    heads = ["1  Recorder", "2  Growth and metastasis", "3  Single-cell readout", "4  Lineage reconstruction"]
    for x, head in zip(columns, heads):
        text(slide, x, 0.95, 3.0, 0.35, head, size=15, bold=True, color=ORANGE)
    for x in columns[1:]:
        arrow(slide, x - 0.32, 2.6, x - 0.08, 2.6)

    # 1. The recorder in one cell.
    x = columns[0]
    box(slide, x, 1.45, 2.9, 2.35, None, line=GREY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="Cell")
    text(slide, x + 0.12, 1.5, 2.6, 0.3, "one engineered cell", size=12, color=MUTED)
    for k in range(2):
        yy = 1.95 + k * 0.6
        box(slide, x + 0.15, yy, 0.6, 0.32, "intBC", fill=PALE, line=None, size=10, shape=MSO_SHAPE.RECTANGLE)
        for j in range(3):
            site = box(slide, x + 0.85 + j * 0.62, yy, 0.52, 0.32, f"cut {j + 1}", fill=WHITE, line=INK, size=9,
                       shape=MSO_SHAPE.RECTANGLE)
    text(slide, x + 0.15, 3.1, 2.7, 0.6, ["10–30 target sites per ES clone,", "3 cut sites each"], size=11,
         color=MUTED)
    text(slide, x, 3.95, 3.0, 1.5, ["Lenti-Cre starts the tumour and switches on Cas9: editing begins at initiation",
                                    "",
                                    "Indels are heritable; descendants add new ones"], size=12)

    # 2. A founder expanding into subclones at two sites.
    x = columns[1]
    founder = box(slide, x + 1.25, 1.5, 0.45, 0.45, None, fill=FILL, line=ORANGE, shape=MSO_SHAPE.OVAL)
    for (dx, dy, fill) in ((0.35, 2.35, FILL), (1.25, 2.35, FILL), (2.15, 2.35, FILL)):
        segment(slide, x + 1.47, 1.95, x + dx + 0.2, dy, color=GREY)
        box(slide, x + dx, dy, 0.4, 0.4, None, fill=fill, line=ORANGE, shape=MSO_SHAPE.OVAL)
    for (px, dx, dy, met) in ((0.55, 0.15, 3.2, False), (0.55, 0.65, 3.2, False), (1.45, 1.2, 3.2, False),
                              (2.35, 1.95, 3.2, True), (2.35, 2.45, 3.2, True)):
        segment(slide, x + px, 2.75, x + dx + 0.17, dy, color=GREY)
        box(slide, x + dx, dy, 0.34, 0.34, None, fill=RGBColor(0x59, 0x59, 0x59) if met else FILL,
            line=INK if met else ORANGE, shape=MSO_SHAPE.OVAL)
    box(slide, x + 0.05, 3.7, 0.3, 0.22, None, fill=FILL, line=ORANGE, shape=MSO_SHAPE.OVAL)
    text(slide, x + 0.42, 3.66, 1.2, 0.3, "primary", size=11, color=MUTED)
    box(slide, x + 1.5, 3.7, 0.3, 0.22, None, fill=RGBColor(0x59, 0x59, 0x59), line=INK, shape=MSO_SHAPE.OVAL)
    text(slide, x + 1.87, 3.66, 1.3, 0.3, "metastasis", size=11, color=MUTED)
    text(slide, x, 3.95, 3.0, 1.5, ["Subclones grow at the primary site; some seed metastases",
                                    "",
                                    "Sampled cells are relatives, not ancestors, of one another"], size=12)

    # 3. The character matrix (conceptual).
    x = columns[2]
    rows = [("Cell", "Site 1", "Site 2", "Site 3", "Sample"), ("P1", "A", "0", "0", "Primary"),
            ("P2", "A", "B", "0", "Primary"), ("M1", "A", "B", "C", "Metastasis"), ("M2", "A", "B", "D", "Metastasis"),
            ("M3", "A", "–", "C", "Metastasis")]
    table = slide.shapes.add_table(len(rows), 5, Inches(x), Inches(1.45), Inches(2.8), Inches(2.3)).table
    widths = [0.42, 0.5, 0.5, 0.5, 0.88]
    for c, w in enumerate(widths):
        table.columns[c].width = Inches(w)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = value
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.CENTER
            run = paragraph.runs[0]
            run.font.name, run.font.size, run.font.bold = FONT, Pt(11), r == 0
            run.font.color.rgb = INK if value not in ("0", "–") else MUTED
            cell.fill.solid()
            cell.fill.fore_color.rgb = PALE if r == 0 else (FILL if r in (1, 2) else WHITE)
            cell.margin_left = cell.margin_right = Inches(0.02)
    text(slide, x, 3.85, 3.0, 1.6, [[("0", {"bold": True}), (" unedited   ", {}), ("–", {"bold": True}),
                                     (" missing (not detected)", {})],
                                    "",
                                    "Letters: indel states. Shared states are evidence of common ancestry"], size=12)

    # 4. The inferred tree for that matrix, with sites.
    x = columns[3]
    box(slide, x, 1.45, 2.95, 0.45, "character matrix → Cassiopeia-Hybrid", fill=PALE, line=None, size=11)
    base = 1.95
    segment(slide, x + 1.45, base + 0.05, x + 1.45, base + 0.3)          # root
    segment(slide, x + 0.6, base + 0.3, x + 2.05, base + 0.3)
    segment(slide, x + 0.6, base + 0.3, x + 0.6, base + 1.35)            # to P1
    segment(slide, x + 2.05, base + 0.3, x + 2.05, base + 0.65)          # A,B node
    segment(slide, x + 1.45, base + 0.65, x + 2.6, base + 0.65)
    segment(slide, x + 1.45, base + 0.65, x + 1.45, base + 1.35)         # to P2
    segment(slide, x + 2.6, base + 0.65, x + 2.6, base + 0.95)           # A,B,* node
    segment(slide, x + 2.25, base + 0.95, x + 2.9, base + 0.95)
    for (lx, name, met) in ((0.6, "P1", False), (1.45, "P2", False), (2.25, "M1", True), (2.9, "M2", True)):
        if met:
            segment(slide, x + lx, base + 0.95, x + lx, base + 1.35)
        leaf = box(slide, x + lx - 0.25, base + 1.33, 0.5, 0.5, name, fill=RGBColor(0x59, 0x59, 0x59) if met else FILL,
                   line=INK if met else ORANGE, shape=MSO_SHAPE.OVAL, size=10, bold=True,
                   color=WHITE if met else INK)
        leaf.text_frame.margin_left = leaf.text_frame.margin_right = 0
        leaf.text_frame.word_wrap = False
    text(slide, x + 1.55, base + 0.36, 0.9, 0.25, "+B", size=10, color=MUTED)
    text(slide, x + 2.65, base + 0.62, 0.5, 0.25, "+C/D", size=10, color=MUTED)
    text(slide, x, 3.95, 3.0, 1.6, ["The tree is inferred, not observed",
                                    "",
                                    "Metastatic families: a primary and its related metastases, annotated by the "
                                    "authors from the lineage data"], size=12)

    text(slide, 0.45, 6.55, 12.4, 0.4, "Schematic; the matrix is illustrative, not KP-Tracer data. After Yang, Jones "
         "et al., Cell 2022 (STAR Methods).", size=11, color=MUTED, italic=True)
    notes(slide, "\n".join([
        "Verified against Yang, Jones et al., Cell 2022, STAR Methods (PMC9452598).",
        "Recorder: each target site carries three Cas9 cut sites in the 3' UTR of a mCherry reporter and a 14-bp "
        "integration barcode (intBC); ES clones with 10–30 integrated target sites were selected; three sgRNAs are "
        "expressed constitutively and Cas9 sits behind a lox-stop-lox (Rosa26 LSL-Cas9).",
        "Intratracheal lentiviral Cre initiates the KP tumour and switches on Cas9 at the same time, so editing "
        "begins at tumour initiation. Lenti-Cre carries a static 16-nt barcode (lenti-Cre-BC), one per tumour, used "
        "to check clonality.",
        "Readout: scRNA-seq plus target-site amplicons per cell; tumour identity from MULTI-seq and non-overlapping "
        "intBC sets (Cassiopeia call-lineages).",
        "Character matrix: one row per cell, one column per cut site; 0 unedited, a missing entry where the site was "
        "not recovered; cut sites with more than 80% missing or a state in more than 98% of cells were removed.",
        "Trees: Cassiopeia-Hybrid (greedy splits, then ILP within clusters), with indel priors; no branch lengths in "
        "the released trees.",
        "Metastatic families: the authors define a tumour-metastasis family as a primary tumour with a large enough "
        "metastatic population; the methods give no numeric rule. The released MetFamily labels are used as given; "
        "they are an annotation built on the lineage data, not something read off a single tree here.",
        "The matrix and tree on the slide are a conceptual illustration (M3 shows a missing entry), not KP-Tracer "
        "characters.",
    ]))


def slide_strategy(slide) -> None:
    text(slide, 0.45, 0.9, 8.0, 0.35, "For one metastasis M: which of the mouse's primary tumours does it match?",
         size=15, bold=True)
    box(slide, 0.45, 2.0, 1.75, 1.9, None, line=ORANGE, name="M box")
    text(slide, 0.55, 2.05, 1.4, 0.3, "M", size=14, bold=True)
    dots(slide, 0.65, 2.5, 6, 5, [GREY] * 30, size=0.13, gap=0.06)
    text(slide, 0.35, 3.95, 1.95, 0.55, ["target: metastatic", "malignant cells"], size=11, color=MUTED,
         align=PP_ALIGN.CENTER)
    candidates = (("P1", 0.72, True), ("P4", 0.64, True), ("P2", 0.35, False), ("P3", 0.28, False))
    text(slide, 3.0, 1.35, 4.2, 0.3, "candidate primaries, ranked by r(P, M)", size=12, color=MUTED)
    for k, (name, r, same) in enumerate(candidates):
        y = 1.75 + k * 0.85
        arrow(slide, 2.25, 2.95, 2.95, y + 0.3, color=GREY, width=1.25)
        text(slide, 3.0, y + 0.1, 0.5, 0.4, name, size=14, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        n = 20
        dots(slide, 3.5, y + 0.12, 10, 2, [RETAINED] * round(r * n) + [REJECTED] * (n - round(r * n)), size=0.12,
             gap=0.05)
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(5.35), Inches(y + 0.14), Inches(1.4 * r), Inches(0.32))
        bar.fill.solid()
        bar.fill.fore_color.rgb = colour(r)
        bar.line.fill.background()
        bar.shadow.inherit = False
        text(slide, 5.42 + 1.4 * r, y + 0.12, 0.6, 0.35, f"{r:.2f}", size=12, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
        box(slide, 7.2, y + 0.12, 1.45, 0.36, "same family" if same else "other family",
            fill=FILL if same else None, line=ORANGE, dashed=True, size=11)
    text(slide, 7.1, 1.35, 1.7, 0.3, "lineage label", size=12, color=MUTED, align=PP_ALIGN.CENTER)

    box(slide, 8.8, 1.45, 4.05, 4.45, None, fill=FILL, line=None, name="Readout panel")
    text(slide, 9.0, 1.6, 3.7, 4.2, [
        [("Each pair is one ConfidenceOT run", {"bold": True})],
        [("source: the primary's malignant cells", {"color": MUTED})],
        [("r(P, M) = retained / analysed source cells", {})],
        "",
        [("Several primaries can share M's family, so this is a ranking, not one-label classification", {})],
        "",
        [("Prespecified summary", {"bold": True})],
        [("d", {}), ("M", {"baseline": -25000}), (" = mean r(same family) − mean r(other family)", {})],
        "",
        [("Lineage labels are attached only after the runs", {"color": REJECTED, "bold": True})],
    ], size=13)
    dots(slide, 0.5, 6.12, 1, 1, [RETAINED])
    text(slide, 0.7, 6.05, 2.2, 0.3, "retained source cell", size=12, color=MUTED)
    dots(slide, 2.9, 6.12, 1, 1, [REJECTED])
    text(slide, 3.1, 6.05, 2.2, 0.3, "rejected source cell", size=12, color=MUTED)
    text(slide, 0.45, 6.55, 12.4, 0.35, "Schematic: dots and values are illustrative, not data.", size=11,
         color=MUTED, italic=True)
    notes(slide, "\n".join([
        "For each metastatic lesion M, every eligible primary tumour P of the same mouse (at least 20 malignant cells "
        "per side) is a candidate. 104 eligible pairs of 131.",
        "Each pair is a separate ConfidenceOT run (ranknm256_noscale_ds_cos, M4-E, within-side acceptance 0.99, no "
        "cell cap): source = P's malignant cells, target = M's malignant cells, malignant = not is_normal (authors).",
        "r(P, M) = the share of P's analysed malignant cells the gate retains against M.",
        "Only afterwards the authors' MetFamily labels mark each candidate as same-family or other-family. Several "
        "candidates can be same-family, so the question is a ranking with possibly several positives.",
        "The prespecified readout (READOUT_PRESPECIFICATION_2026-10-14.md) summarises each lesion by d_M = mean r of "
        "same-family candidates minus mean r of other-family candidates, and tests d_M > 0 across lesions.",
        "Not implied: that a sampled primary cell seeded M, or the route of spread.",
    ]))


def slide_heatmap_all(slide, data: dict) -> None:
    mice = sorted(data, key=int)
    draw_heatmaps(slide, data, mice, 0.45, 0.95, 9.9, 0.3, 0.55, gap=0.32)
    draw_legend(slide, 10.2, 1.0)
    text(slide, 10.2, 3.65, 2.8, 2.6, ["Rows: metastatic lesions",
                                       "Columns: the mouse's candidate primaries, in tumour order"], size=12, color=MUTED)
    text(slide, 0.45, 6.75, 12.4, 0.3, f"ranknm256_noscale_ds_cos, M4-E. Source: readout/pairs_{MAIN}.csv "
         "(104 completed pairs, 11 mice).", size=10, color=MUTED, italic=True)
    informative = sum(v for m in data.values() for v in m["informative"].values())
    lesions = sum(len(m["lesions"]) for m in data.values())
    notes(slide, "\n".join([
        "Every completed primary-metastasis pair of the main representation, one block per mouse. Colour: source "
        "retained fraction r, one scale 0–1 for every mouse. Black outline: the candidate shares the lesion's "
        "metastatic family (authors' MetFamily); no outline: other family.",
        "Hatched: the pair was not run because one side has fewer than 20 malignant cells; distinct from r = 0.",
        f"{lesions} lesions with at least one completed pair; {informative} have both a same-family and an "
        "other-family candidate and enter d_M (black labels). Grey italic lesions have candidates of one kind only "
        "(mice 3465 and 3724 have a single primary; in 3467 the family primary was not sampled).",
        "Columns are in tumour order (T1, T2, ...), not ordered by the labels.",
        "Read: within a row, are the outlined cells darker than the others?",
    ]))


def add_error_bars(series, lower: list, upper: list) -> None:
    c = "http://schemas.openxmlformats.org/drawingml/2006/chart"
    a = "http://schemas.openxmlformats.org/drawingml/2006/main"

    def lit(values):
        points = "".join(f'<c:pt idx="{i}"><c:v>{v:.6f}</c:v></c:pt>' for i, v in enumerate(values))
        return f'<c:numLit><c:formatCode>General</c:formatCode><c:ptCount val="{len(values)}"/>{points}</c:numLit>'
    xml = (f'<c:errBars xmlns:c="{c}" xmlns:a="{a}"><c:errBarType val="both"/><c:errValType val="cust"/>'
           f'<c:noEndCap val="0"/><c:plus>{lit(upper)}</c:plus><c:minus>{lit(lower)}</c:minus>'
           f'<c:spPr><a:ln w="12700"><a:solidFill><a:srgbClr val="000000"/></a:solidFill></a:ln></c:spPr></c:errBars>')
    element = etree.fromstring(xml)
    series._element.find(f"{{{c}}}cat").addprevious(element)


def style_value_axis(axis, title_text=None, maximum=None):
    axis.has_major_gridlines = True
    axis.major_gridlines.format.line.color.rgb = RGBColor(0xE6, 0xE6, 0xE6)
    axis.format.line.fill.background()
    axis.tick_labels.font.size = Pt(12)
    axis.tick_labels.font.color.rgb = MUTED
    if maximum is not None:
        axis.maximum_scale, axis.minimum_scale = maximum, 0
    if title_text:
        axis.has_title = True
        paragraph = axis.axis_title.text_frame.paragraphs[0]
        run = paragraph.add_run()
        run.text = title_text
        run.font.size, run.font.bold, run.font.color.rgb, run.font.name = Pt(13), False, INK, FONT


def slide_ranking(slide, ranking: dict) -> None:
    metrics = (("top1", "Top-1 hit"), ("rr", "Reciprocal rank"), ("auroc", "Pairwise AUROC"))
    data = CategoryChartData()
    data.categories = [name for _, name in metrics]
    series_spec = ((MAIN, "ranknm256_noscale_ds_cos", ORANGE), (RAW, "raw", GREY))
    for label, name, _ in series_spec:
        data.add_series(name, [ranking["labels"][label][m]["mean_of_mouse_means"] for m, _ in metrics])
    data.add_series("random ranking", [ranking["labels"][MAIN][m]["random_mean_of_mouse_means"] for m, _ in metrics])
    frame = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.45), Inches(0.95), Inches(8.2),
                                   Inches(5.5), data)
    chart = frame.chart
    v2.chart_style(chart)
    chart.has_title = False
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(12)
    plot = chart.plots[0]
    plot.gap_width = 60
    plot.overlap = -10
    fills = [ORANGE, GREY, RGBColor(0xD9, 0xD9, 0xD9)]
    for k, series in enumerate(plot.series):
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = fills[k]
        series.format.line.fill.background()
        series.invert_if_negative = False
        if k < 2:
            label = series_spec[k][0]
            lows = [ranking["labels"][label][m]["mean_of_mouse_means"] - ranking["labels"][label][m]["bootstrap_95"][0]
                    for m, _ in metrics]
            highs = [ranking["labels"][label][m]["bootstrap_95"][1] - ranking["labels"][label][m]["mean_of_mouse_means"]
                     for m, _ in metrics]
            add_error_bars(series, lows, highs)
    category = chart.category_axis
    category.tick_labels.font.size = Pt(13)
    category.major_tick_mark = XL_TICK_MARK.NONE
    category.format.line.color.rgb = INK
    style_value_axis(chart.value_axis, "mean of mouse means", maximum=1.0)
    text(slide, 0.6, 6.45, 8.0, 0.3, "bars: mean over the 8 mice of the per-mouse mean; whiskers: 95% bootstrap "
         "over mice", size=12, color=MUTED, align=PP_ALIGN.CENTER)
    main = ranking["labels"][MAIN]
    raw = ranking["labels"][RAW]
    x = 9.0
    text(slide, x, 1.0, 3.9, 0.9, [[(f"{main['top1']['mice_above_random']} / {main['top1']['mice']}",
                                     {"bold": True, "size": 40, "color": ORANGE})]])
    text(slide, x, 1.85, 3.9, 0.9, "mice above random on top-1, reciprocal rank and AUROC (ranknm256_noscale_ds_cos)",
         size=13, color=MUTED)
    text(slide, x, 2.95, 3.9, 0.9, [[(f"{raw['top1']['mice_above_random']} / {raw['top1']['mice']}",
                                     {"bold": True, "size": 28, "color": GREY})]])
    text(slide, x, 3.6, 3.9, 0.6, "mice above random on each, raw", size=13, color=MUTED)
    text(slide, x, 4.4, 3.9, 1.8, ["Exploratory: computed after the prespecified d_M result was known",
                                   "31 lesions in 8 mice; ties split at random"], size=12, color=MUTED)
    text(slide, 0.45, 6.75, 12.4, 0.3, "Source: tumour_level_exploratory/ranking_summary.json "
         "(kp_lineage/tumour_level/01_ranking_and_representations.py).", size=10, color=MUTED, italic=True)
    notes(slide, "\n".join([
        "Exploratory, added after the prespecified d_M result was known; it does not replace d_M.",
        "For each of the 31 lesions with both same-family and other-family candidates, candidates are ranked by r. "
        "Top-1 hit: a same-family primary is ranked first. Reciprocal rank: 1 / rank of the best-ranked same-family "
        "primary. Pairwise AUROC: the share of same-family vs other-family candidate pairs ordered correctly. Ties "
        "are split at random (exact expectation). Several same-family candidates are allowed.",
        "Random ranking: the exact expectation for each lesion's numbers of candidates and positives (top-1 = p/n), "
        "averaged the same way.",
        f"ranknm256_noscale_ds_cos, mean of mouse means: top-1 {main['top1']['mean_of_mouse_means']:.2f} "
        f"(random {main['top1']['random_mean_of_mouse_means']:.2f}), reciprocal rank {main['rr']['mean_of_mouse_means']:.2f} "
        f"(random {main['rr']['random_mean_of_mouse_means']:.2f}), AUROC {main['auroc']['mean_of_mouse_means']:.2f} (random 0.5); "
        f"{main['top1']['mice_above_random']} of 8 mice above random on each.",
        f"raw: top-1 {raw['top1']['mean_of_mouse_means']:.2f}, reciprocal rank {raw['rr']['mean_of_mouse_means']:.2f}, AUROC "
        f"{raw['auroc']['mean_of_mouse_means']:.2f}; {raw['top1']['mice_above_random']} of 8 mice above random.",
        f"Top-2 (lesions with at least 3 candidates: {main['top2']['lesions']} lesions, {main['top2']['mice']} mice): "
        f"{main['top2']['mean_of_mouse_means']:.2f} vs random {main['top2']['random_mean_of_mouse_means']:.2f}.",
        "AUROC and d_M use the same scores and are not independent evidence.",
    ]))


def slide_representations(slide, configs: pd.DataFrame, mice: pd.DataFrame, comparison: dict) -> None:
    rows = [("", "raw", "ranknm256_noscale_ds_cos"),
            ("Input counts", "original", "equalised to one depth"),
            ("Per-cell transform", "none", "rank of top 256 genes"),
            ("Gene scaling", "z-score", "none"),
            ("Cost", "squared Euclidean", "cosine"),
            ("Same cells, pairs, gate", "yes", "yes")]
    table = slide.shapes.add_table(len(rows), 3, Inches(0.45), Inches(1.0), Inches(6.0), Inches(2.6)).table
    for c, w in enumerate((1.9, 1.8, 2.3)):
        table.columns[c].width = Inches(w)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = value
            for run in cell.text_frame.paragraphs[0].runs:
                run.font.name, run.font.size, run.font.bold = FONT, Pt(12), r == 0 or c == 0
                run.font.color.rgb = INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = PALE if r == 0 else WHITE
    text(slide, 0.45, 3.7, 6.0, 0.6, "The two differ in several settings at once: no single step is identified",
         size=12, color=REJECTED, bold=True)
    x = 0.45
    stats_lines = [
        [("median retained fraction  ", {"color": MUTED}),
         (f"{comparison['source_retained_fraction_median'][RAW]:.2f}  vs  "
          f"{comparison['source_retained_fraction_median'][MAIN]:.2f}", {"bold": True})],
        [("source cells changing label  ", {"color": MUTED}),
         (f"{100 * comparison['label_changed_share_median_over_pairs']:.0f}%", {"bold": True}),
         (" (median per pair)", {"color": MUTED})],
        [("lesions with d", {"color": MUTED}), ("M", {"color": MUTED, "baseline": -25000}), (" > 0  ", {"color": MUTED}),
         (f"{comparison['positive_lesions'][RAW]} vs {comparison['positive_lesions'][MAIN]} of {comparison['paired_lesions']}",
          {"bold": True})],
        [("mice with median d", {"color": MUTED}), ("M", {"color": MUTED, "baseline": -25000}), (" > 0  ", {"color": MUTED}),
         (f"{comparison['mice_median_d_above_0'][RAW]} vs {comparison['mice_median_d_above_0'][MAIN]} of "
          f"{comparison['paired_mice']}", {"bold": True})],
    ]
    text(slide, x, 4.3, 6.0, 2.0, stats_lines, size=14)
    text(slide, x, 6.0, 6.0, 0.3, "raw vs ranknm256_noscale_ds_cos; same 31 lesions, 8 mice", size=11, color=MUTED)

    data = CategoryChartData()
    order = sorted(mice["patient_id"].astype(str), key=int)
    mice = mice.set_index(mice["patient_id"].astype(str))
    data.categories = order
    data.add_series("ranknm256_noscale_ds_cos", [round(float(mice.loc[m, f"median_d_{MAIN}"]), 4) for m in order])
    data.add_series("raw", [round(float(mice.loc[m, f"median_d_{RAW}"]), 4) for m in order])
    frame = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(6.8), Inches(0.95), Inches(6.1),
                                   Inches(5.4), data)
    chart = frame.chart
    v2.chart_style(chart)
    chart.has_title = False
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(12)
    for k, series in enumerate(chart.plots[0].series):
        series.format.line.fill.background()
        series.smooth = False
        marker = series.marker
        marker.style = XL_MARKER_STYLE.CIRCLE if k == 0 else XL_MARKER_STYLE.DIAMOND
        marker.size = 11 if k == 0 else 10
        marker.format.fill.solid()
        marker.format.fill.fore_color.rgb = ORANGE if k == 0 else GREY
        marker.format.line.fill.background()
    category = chart.category_axis
    category.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    category.major_tick_mark = XL_TICK_MARK.NONE
    category.tick_labels.font.size = Pt(12)
    category.format.line.color.rgb = INK
    category.format.line.width = Pt(1.25)
    v2.axis_title(chart.value_axis, "mouse median")
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE6, 0xE6, 0xE6)
    chart.value_axis.format.line.fill.background()
    chart.value_axis.tick_labels.font.size = Pt(12)
    chart.value_axis.tick_labels.font.color.rgb = MUTED
    text(slide, 6.8, 6.4, 6.1, 0.3, "mouse", size=12, color=MUTED, align=PP_ALIGN.CENTER)
    text(slide, 0.45, 6.75, 12.4, 0.3, "Source: tumour_level_exploratory/representation_mice.tsv, "
         "representation_pairs.tsv, preprocessing_configurations.tsv.", size=10, color=MUTED, italic=True)
    notes(slide, "\n".join([
        "Only two KP-Tracer representations exist: raw and ranknm256_noscale_ds_cos. Both use the same 131-row "
        "manifest, the same 104 eligible pairs, the same malignant call, the same MetFamily labels and gate settings, "
        "and identical cell IDs in all 208 pair-sides (checked).",
        "raw: original counts, no library-size normalisation, HVG on raw values, genes z-scored, 30 PCs, squared "
        "Euclidean cost divided by the median random pair. ranknm256_noscale_ds_cos: counts equalised to the pooled "
        "10th-percentile depth, top-256 rank encoding per cell (no gene-median division), HVG on the encoding, no "
        "gene scaling, 30 PCs, cosine cost, same median scaling.",
        "They differ in depth equalisation, per-cell transform, HVG basis, gene scaling and cost geometry at once, so "
        "the difference cannot be attributed to one step. One-step arms on KP-Tracer have not been run.",
        f"raw retains a median {comparison['source_retained_fraction_median'][RAW]:.2f} of source cells, leaving little "
        f"room for a contrast; the main label retains {comparison['source_retained_fraction_median'][MAIN]:.2f}. A median "
        f"{100 * comparison['label_changed_share_median_over_pairs']:.0f}% of source cells per pair change gate label "
        "between the two (cells scored in both). That is a change in which cells are retained, not in any cell-to-cell "
        "correspondence (no coupling was saved).",
        f"Paired over the same {comparison['paired_lesions']} lesions: d_M > 0 in {comparison['positive_lesions'][MAIN]} "
        f"(main) vs {comparison['positive_lesions'][RAW]} (raw); median d_M {comparison['median_d'][MAIN]:+.3f} vs "
        f"{comparison['median_d'][RAW]:+.3f}; the main median exceeds raw in "
        f"{comparison['mice_where_main_median_exceeds_raw']} of 8 mice.",
        "The main representation was chosen on the technical benchmark before KP-Tracer was run; raw was added on "
        "2026-10-14 before any gate was seen, as a comparison, never as a candidate to pick after the fact.",
    ]))


def slide_heatmap_pair(slide, data_main: dict, data_raw: dict, mice: list) -> None:
    text(slide, 0.45, 0.88, 6.0, 0.3, "ranknm256_noscale_ds_cos", size=14, bold=True, color=ORANGE)
    bottom = draw_heatmaps(slide, data_main, mice, 0.45, 1.16, 13.0, 0.24, 0.46, gap=0.3, font=9)
    text(slide, 0.45, bottom + 0.06, 6.0, 0.3, "raw", size=14, bold=True, color=MUTED)
    draw_heatmaps(slide, data_raw, mice, 0.45, bottom + 0.34, 13.0, 0.24, 0.46, gap=0.3, font=9)
    text(slide, 0.45, 6.75, 12.4, 0.3, "Same mice, lesions, candidate order, colour scale and outlines as slide 23; "
         "the 8 mice with an informative lesion. Source: readout/pairs_*.csv.", size=10, color=MUTED, italic=True)
    notes(slide, "\n".join([
        "The same heatmap for the two representations, restricted to the 8 mice whose lesions enter d_M, with identical "
        "ordering, colour scale (retained fraction 0–1), same-family outlines and not-run hatching.",
        "raw retains nearly every source cell in nearly every pair (median 0.99), so its rows are uniformly dark and the "
        "outlined cells do not stand out; the main representation retains a median 0.60, and within a row the outlined "
        "cells are usually darker.",
        "Legend as on slide 23.",
    ]))


def slide_cnv(slide) -> None:
    rows = [("Representation", "KP-Tracer status", "What it needs"),
            ("R1  raw RNA", "done: 104 pairs", "—"),
            ("R2  ranknm256_noscale_ds_cos", "done: 104 pairs (main)", "—"),
            ("R3  other RNA preprocessing", "not run on KP-Tracer", "one-step arms on the same 104 pairs"),
            ("R4  CNV only", "not evaluated", "CNV profiles: none released; inferCNV not yet run"),
            ("R5  RNA + CNV", "not evaluated", "R4 first, then the combined arm")]
    table = slide.shapes.add_table(len(rows), 3, Inches(0.45), Inches(1.0), Inches(12.4), Inches(3.0)).table
    for c, w in enumerate((3.4, 3.4, 5.6)):
        table.columns[c].width = Inches(w)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = value
            run = cell.text_frame.paragraphs[0].runs[0]
            run.font.name, run.font.size, run.font.bold = FONT, Pt(13), r == 0 or c == 0
            run.font.color.rgb = INK if not (c == 1 and "not" in value) else REJECTED
            cell.fill.solid()
            cell.fill.fore_color.rgb = PALE if r == 0 else WHITE
    box(slide, 0.45, 4.35, 12.4, 1.85, None, fill=FILL, line=None, name="CNV needs")
    text(slide, 0.65, 4.45, 12.0, 1.7, [
        [("For a CNV arm on KP-Tracer", {"bold": True})],
        "a mouse gene-position file; inferCNV per mouse with the authors' normal cells as reference, expression only "
        "(no tree, barcode or family input, so no label leakage); per-cell coverage checked; then cnv and rna_cnv arms "
        "on the same 104 pairs and gate",
        [("KP tumours carry relatively few large CNVs, so informativeness has to be shown first", {"color": MUTED})],
    ], size=13)
    text(slide, 0.45, 6.55, 12.4, 0.35, "No CNV-based KP-Tracer result exists; nothing on this slide is a result.",
         size=11, color=MUTED, italic=True)
    notes(slide, "\n".join([
        "Feasibility only (kp_lineage/tumour_level/TUMOUR_LEVEL_REPRESENTATION_AUDIT.md).",
        "KPTracer-Data releases expression, trees, character matrices, metadata, fitness, expansions and plasticity "
        "scores, but no CNV profiles (README; the running audit v3 inventory will confirm).",
        "Yang et al. ran inferCNV per tumour from expression with MULTI-seq-identified normal lung cells as reference, "
        "to corroborate subclones; expression-derived CNV is an inference from the same RNA.",
        "Our human-data pipeline (R inferCNV per patient, 44_ export, cnv / rna_cnv arms) has never been run on "
        "KP-Tracer. It needs a mouse gene-position file (not in the repository), inferCNV per mouse with is_normal "
        "cells as reference and no lineage input, coverage checks, and then the arms on the same pairs.",
        "None of this has been submitted; it waits on a plan and on the running validation jobs.",
    ]))


def slide_summary(slide, ranking: dict, comparison: dict) -> None:
    columns = (("Supported", ORANGE, [
        "Same-family primaries retain more source cells: 27 of 31 lesions, prespecified one-sided Wilcoxon "
        "p = 5.0 × 10⁻⁷",
        "Consistent across mice: median d_M > 0 in 8 of 8 (sign test p = 0.0039, exploratory)",
        "Lineage labels never entered ConfidenceOT",
    ]), ("Exploratory", MUTED, [
        f"Candidate ranking above random in 8 of 8 mice (top-1 {ranking['labels'][MAIN]['top1']['mean_of_mouse_means']:.2f} "
        f"vs {ranking['labels'][MAIN]['top1']['random_mean_of_mouse_means']:.2f})",
        f"raw is weaker (median d_M +{comparison['median_d'][RAW]:.3f}; 5 of 8 mice), but it retains a median "
        f"{comparison['source_retained_fraction_median'][RAW]:.2f} of cells; the two differ in several steps",
    ]), ("Not shown here", REJECTED, [
        "Which primary cells seeded a metastasis, or the route",
        "Cell-level lineage concordance: E1 inconclusive; Level 2 and 3 running",
        "One-step RNA preprocessing, CNV and RNA+CNV on KP-Tracer: not run",
    ]))
    for k, (head, colour_value, items) in enumerate(columns):
        x = 0.45 + k * 4.2
        box(slide, x, 1.0, 3.95, 5.3, None, fill=PALE if k else FILL, line=None)
        text(slide, x + 0.2, 1.15, 3.6, 0.4, head, size=17, bold=True, color=colour_value)
        text(slide, x + 0.2, 1.7, 3.55, 4.5, [item for pair in zip(items, [""] * len(items)) for item in pair][:-1],
             size=13)
    text(slide, 0.45, 6.55, 12.4, 0.35, "Tumour level: ranknm256_noscale_ds_cos unless stated; mouse = independent unit.",
         size=11, color=MUTED, italic=True)
    notes(slide, "\n".join([
        "What the KP-Tracer tumour-level benchmark supports, as of 2026-10-15.",
        "Supported (prespecified): d_M > 0 across lesions, 27 of 31, one-sided Wilcoxon p = 4.99e-7; every mouse "
        "positive. Lesions are nested in mice; the mouse-level sign test is exploratory.",
        "Exploratory: candidate-ranking metrics and the raw comparison; computed after the d_M result was known.",
        "Not shown: seeding cells or routes (same family is a tumour-level relation); the cell-level analyses (E1 "
        "inconclusive, Level 2/3 still running on TACC); other preprocessing, CNV and RNA+CNV, which were never run on "
        "KP-Tracer.",
    ]))


# --- assembly ------------------------------------------------------------------------------

def main():
    ranking = json.loads((EXPLORE / "ranking_summary.json").read_text())
    comparison = ranking["representation_comparison"]
    configs = pd.read_csv(EXPLORE / "preprocessing_configurations.tsv", sep="\t")
    mice_table = pd.read_csv(EXPLORE / "representation_mice.tsv", sep="\t")
    data_main, data_raw = heatmap_data(MAIN), heatmap_data(RAW)
    informative_mice = sorted({m for m, d in data_main.items() if any(d["informative"].values())}, key=int)
    assert len(informative_mice) == 8

    script = v2.SKILL / "scripts" / "add_slide.py"
    # Seven copies of the title-only slide, appended; the order is set below.
    subprocess.run([sys.executable, str(script), str(SRC), "slide13.xml", "-o", str(TMP)],
                   check=True, capture_output=True, text=True, encoding="utf-8")
    for _ in range(NEW_SLIDES - 1):
        subprocess.run([sys.executable, str(script), str(TMP), "slide13.xml"],
                       check=True, capture_output=True, text=True, encoding="utf-8")
    prs = Presentation(str(TMP))
    slides = list(prs.slides)
    old, new = slides[:24], slides[24:]
    assert len(new) == NEW_SLIDES
    titles = {}
    for slide in slides:
        box_ = next(s for s in slide.shapes if s.name == "TextBox 3")
        titles[id(slide)] = box_.text_frame.text
    assert titles[id(old[19])].startswith("KP-Tracer") and titles[id(old[23])].startswith("Summary")

    # New and redrawn content.
    plan = [
        (new[0], "How CRISPR Lineage Tracing Works", slide_crispr, ()),
        (old[20], "Which Primary Matches Each Metastasis?", slide_strategy, ()),
        (new[1], "Retained Fractions Reveal Candidate Primary Matches", slide_heatmap_all, (data_main,)),
        (new[2], "Candidate-Primary Ranking (Exploratory)", slide_ranking, (ranking,)),
        (new[3], "RNA Preprocessing and Family Discrimination", slide_representations,
         (configs, mice_table, comparison)),
        (new[4], "Raw vs Preprocessed RNA: Matched Heatmaps", slide_heatmap_pair, (data_main, data_raw, informative_mice)),
        (new[5], "CNV and RNA+CNV: Not Yet Evaluated", slide_cnv, ()),
        (new[6], "What the Tumor-Level Benchmark Supports", slide_summary, (ranking, comparison)),
    ]
    for slide, title, builder, extra in plan:
        v2.clear_content(slide)
        v2.set_title(slide, title)
        builder(slide, *extra)
    v2.set_title(old[21], "Tumor-Level Concordance Across Lesions")
    v2.set_title(old[22], "Tumor-Level Consistency Across Mice")

    # Order.
    id_list = prs.slides._sldIdLst
    entries = list(id_list)
    by_slide = dict(zip([id(s) for s in slides], entries))
    order = old[:20] + [new[0], old[20], new[1], old[21], old[22], new[2], new[3], new[4], new[5], new[6], old[23]]
    for entry in entries:
        id_list.remove(entry)
    for slide in order:
        id_list.append(by_slide[id(slide)])
    for number, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if shape.name == "Rectangle 6":
                for field in shape._element.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}fld"):
                    field.find("{http://schemas.openxmlformats.org/drawingml/2006/main}t").text = str(number)
    prs.save(str(OUT))
    TMP.unlink()
    print(OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
