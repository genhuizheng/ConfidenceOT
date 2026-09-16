"""Assemble the PI report deck from the generated figures.

Thirteen slides in the order the work happened: what we set out to find, the
result that did not hold, the causes we traced, and what the prostate dataset
added. The audience is biological, so every number on a slide is one a
biologist can act on and the wording avoids the method's vocabulary.

Every figure is one of the PNGs from ``make_report_figures.py`` and every number
written here was read back out of the collected summaries rather than carried
over from a previous draft. Where a slide quotes a statistic, the arm it came
from is named, because the retained fraction differs tenfold between the free
and capped rules and a number without its arm is not checkable.

The chrome is deliberately colourless -- slate and white -- because the figures
already carry a validated blue and orange, and a second palette competing with
them would make identity ambiguous across slides. The one accent is the same
orange the figures use for "the cells the method kept", so a reader who learns
it on slide 2 keeps it for the rest of the deck.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

SLATE = RGBColor(0x1F, 0x29, 0x33)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xDE, 0xDE, 0xD8)
DIM = RGBColor(0x9A, 0x9A, 0x92)
INK = RGBColor(0x14, 0x18, 0x1C)
BODY = RGBColor(0x45, 0x4C, 0x52)
MUTED = RGBColor(0x8A, 0x8A, 0x85)
ACCENT = RGBColor(0xEB, 0x68, 0x34)
BLUE = RGBColor(0x2A, 0x78, 0xD6)

TITLE_FONT = "Cambria"
BODY_FONT = "Calibri"

WIDE, TALL = 13.333, 7.5
MARGIN = 0.62


def add_slide(presentation: Presentation, dark: bool = False):
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    if dark:
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = SLATE
    return slide


def text_box(slide, left, top, width, height, *, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    frame.paragraphs[0].alignment = align
    return frame


def write(frame, runs, *, space_after=0, line_spacing=None, first=False):
    """Append a paragraph built from (text, size, bold, colour, font) runs."""
    paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
    paragraph.space_after = Pt(space_after)
    if line_spacing:
        paragraph.line_spacing = line_spacing
    for text, size, bold, colour, font in runs:
        run = paragraph.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = colour
        run.font.name = font
    return paragraph


def slide_title(slide, title, subtitle=None):
    frame = text_box(slide, MARGIN, 0.40, WIDE - 2 * MARGIN, 0.9)
    write(frame, [(title, 30, True, INK, TITLE_FONT)], first=True)
    if subtitle:
        sub = text_box(slide, MARGIN, 1.26, WIDE - 2 * MARGIN, 0.55)
        write(sub, [(subtitle, 15, False, BODY, BODY_FONT)], first=True,
              line_spacing=1.15)


def place_image(slide, path: Path, left, top, width, height):
    """Fit an image inside a box, preserving its aspect ratio."""
    with Image.open(path) as image:
        ratio = image.width / image.height
    if width / height > ratio:
        drawn_height, drawn_width = height, height * ratio
    else:
        drawn_width, drawn_height = width, width / ratio
    slide.shapes.add_picture(
        str(path), Inches(left + (width - drawn_width) / 2),
        Inches(top + (height - drawn_height) / 2),
        width=Inches(drawn_width), height=Inches(drawn_height),
    )


def stat(slide, left, top, width, number, label, colour=ACCENT, size=44):
    frame = text_box(slide, left, top, width, 1.6)
    write(frame, [(number, size, True, colour, TITLE_FONT)], first=True,
          space_after=3)
    write(frame, [(label, 13, False, BODY, BODY_FONT)], line_spacing=1.12)


def bullets(slide, left, top, width, height, items, size=15, gap=11):
    frame = text_box(slide, left, top, width, height)
    for index, item in enumerate(items):
        runs = []
        if isinstance(item, tuple):
            head, rest = item
            runs.append((head, size, True, INK, BODY_FONT))
            runs.append((rest, size, False, BODY, BODY_FONT))
        else:
            runs.append((item, size, False, BODY, BODY_FONT))
        write(frame, runs, space_after=gap, line_spacing=1.18, first=index == 0)
    return frame


def caption(slide, text, top=None):
    frame = text_box(slide, MARGIN, top if top is not None else TALL - 0.66,
                     WIDE - 2 * MARGIN, 0.48)
    write(frame, [(text, 11, False, MUTED, BODY_FONT)], first=True,
          line_spacing=1.12)


def figure_slide(presentation, title, subtitle, image: Path, note=None,
                 image_top=1.95):
    slide = add_slide(presentation)
    slide_title(slide, title, subtitle)
    bottom = TALL - (0.82 if note else 0.42)
    place_image(slide, image, MARGIN, image_top, WIDE - 2 * MARGIN,
                bottom - image_top)
    if note:
        caption(slide, note)
    return slide


def build(figures: Path, output: Path, reference: bool) -> None:
    presentation = Presentation()
    presentation.slide_width = Inches(WIDE)
    presentation.slide_height = Inches(TALL)

    # 1 -- title
    slide = add_slide(presentation, dark=True)
    frame = text_box(slide, 1.0, 2.3, WIDE - 2.0, 2.5)
    write(frame, [("Which primary tumour cells can metastasise?", 40, True,
                   WHITE, TITLE_FONT)], first=True, space_after=14,
          line_spacing=1.05)
    write(frame, [("What we found, why the first answer did not hold, and what "
                   "the prostate data added", 18, False,
                   RGBColor(0xC3, 0xC2, 0xB7), BODY_FONT)], line_spacing=1.2)
    footer = text_box(slide, 1.0, 5.65, WIDE - 2.0, 0.5)
    write(footer, [("Ovarian, 29 patients  ·  head and neck  ·  "
                    "colorectal  ·  prostate, 4 patients", 13, False, DIM,
                    BODY_FONT)], first=True)

    # 2 -- the question
    figure_slide(
        presentation,
        "The question",
        "Given a patient's primary tumour and their metastasis, which primary "
        "cells look like they could have seeded it?",
        figures / "fig_umap_ovarian.png",
        "Ovarian cancer, 187,383 cells. Left: the two tissues overlap almost "
        "completely. Middle: the cells the method kept. Right: how fast those "
        "cells are dividing.",
        image_top=2.02)

    # 3 -- the negative result
    slide = add_slide(presentation)
    slide_title(slide, "The first answer did not hold",
                "In all three original datasets the biology came out backwards")
    stat(slide, MARGIN, 2.15, 3.6, "0.005",
         "Difference in metastasis-signature score between the two groups of "
         "primary cells")
    stat(slide, MARGIN + 4.05, 2.15, 3.6, "2 × 10⁻⁸",
         "p value for that same difference", colour=BLUE)
    stat(slide, MARGIN + 8.1, 2.15, 3.6, "3.1×",
         "Sequencing depth between the two groups, in the worst dataset")
    bullets(slide, MARGIN, 4.55, WIDE - 2 * MARGIN, 2.1, [
        ("A tiny effect with an overwhelming p value is the signature of a "
         "systematic artefact, ", "not of a small biological difference."),
        ("The genes separating the two groups were keratins and SPRR family "
         "members in all three cancers ",
         "— the same answer regardless of the biology, which is what a "
         "technical confounder looks like."),
    ])
    caption(slide, "Ovarian (GSE180661), head and neck (GSE181919) and "
                   "colorectal (GSE225857). Signature scoring: UCell on "
                   "metastasis-derived gene sets. Depth ratio between the two "
                   "groups: 1.3×, 2.0× and 3.1×.")

    # 4 -- cause one
    figure_slide(
        presentation,
        "Cause 1: the method was reading sequencing depth",
        "Cells sequenced more deeply were systematically kept or dropped, in "
        "both cancers",
        figures / "fig1_depth_in_gate.png",
        "Equalising every cell's read count helped but did not finish the job: "
        "depth also changes which genes are detected at all, not only how many "
        "reads they get.")

    # 5 -- the fix. The callout gets its own column rather than floating over
    # the plot, which is where it landed when the image used the full width.
    slide = add_slide(presentation)
    slide_title(slide, "What fixed it: ranking genes inside each cell",
                "Tested on simulated tumours where the correct answer is known")
    column = 2.95
    place_image(slide, figures / "fig2_representation_benchmark.png",
                MARGIN, 2.0, WIDE - 2 * MARGIN - column - 0.35, TALL - 0.86 - 2.0)
    box = text_box(slide, WIDE - MARGIN - column, 2.35, column, 2.6)
    write(box, [("Ovarian, after the fix", 14, True, INK, BODY_FONT)],
          first=True, space_after=9)
    for head, value in (("read depth", "0.59 → 0.53"),
                        ("genes detected", "0.42 → 0.51")):
        write(box, [(head + "  ", 13, False, BODY, BODY_FONT),
                    (value, 13, True, ACCENT, BODY_FONT)],
              line_spacing=1.2, space_after=6)
    write(box, [("Neither is significant any more. 0.50 would mean no depth "
                 "effect at all.", 12, False, MUTED, BODY_FONT)],
          line_spacing=1.22)
    caption(slide,
            "Ranking each cell's genes by how unusual their level is for that "
            "gene removes the depth scale. Ranking without that per-gene step "
            "does nothing, and ranking cannot replace read equalisation "
            "— both are needed.")

    # 6 -- the control
    figure_slide(
        presentation,
        "A control the project had never run",
        "Give a patient's primary tumour somebody else's metastasis and see "
        "what happens",
        figures / "fig4_mismatched_control.png",
        "The method does use the metastasis it is given: with the wrong "
        "patient's metastasis it keeps essentially nothing (35% → 0%, "
        "p = 8 × 10⁻¹⁷), and the two kept sets overlap no "
        "more than chance. So the split is patient-specific.",
        image_top=2.0)

    # 7 -- what it measures
    figure_slide(
        presentation,
        "But what the kept fraction measures is similarity",
        "Not how many cells could metastasise, but how alike the two tissues "
        "are",
        figures / "fig3_similarity_readout.png",
        "Correlation 0.76 across 188 pairs, against a measure the method never "
        "sees and that uses no transport at all. It is not a cell-number "
        "effect: cell number predicts similarity but not the kept fraction "
        "(rho = −0.06, p = 0.43).",
        image_top=2.0)

    # 8 -- why that is a problem
    figure_slide(
        presentation,
        "Which is why the number cannot mean metastatic potential",
        "Every one of these 29 patients already had a metastasis",
        figures / "fig5_patient_spread.png",
        "The same measure reads 0% in one patient and 88% in another, median "
        "33%. Not a sample-size artefact: across the 26 patients with at least "
        "500 primary cells it still spans 0.5% to 87%, median 32%. Ascitic "
        "metastases are four of the seven lowest — free-floating tumour "
        "cells are transcriptionally far from the primary.",
        image_top=2.0)

    # 9 -- prostate dataset
    slide = add_slide(presentation)
    slide_title(slide, "Prostate cancer: a dataset where the two sides differ",
                "GSE271675, re-annotated by Faming — matched primary "
                "tumours and lymph-node metastases")
    bullets(slide, MARGIN, 2.05, 6.0, 4.4, [
        ("Why it matters: ", "in ovarian cancer the primary and the metastasis "
         "are nearly identical — correlation 0.94, and no genes separate "
         "the two tissues at all. There was almost nothing for the method to "
         "find."),
        ("In prostate they genuinely differ: ", "androgen signalling down, "
         "invasion and interferon up in the lymph node."),
        ("236,507 malignant cells across 38 specimens, 24 primary-metastasis "
         "pairs from 4 patients. ",
         "A fifth patient had no metastatic cells in the object."),
    ])
    stat(slide, MARGIN + 6.7, 2.2, 2.8, "29%",
         "of primary cells kept\novarian")
    stat(slide, MARGIN + 9.6, 2.2, 2.8, "1.6%",
         "of primary cells kept\nprostate")
    frame = text_box(slide, MARGIN + 6.7, 4.2, WIDE - MARGIN - 6.7 - MARGIN, 2.1)
    write(frame, [("The more the two tissues differ, the less the method keeps.",
                   16, True, INK, BODY_FONT)], first=True, space_after=9,
          line_spacing=1.14)
    write(frame, [("That is the wrong way round. Real divergence between a "
                   "primary and its metastasis is the interesting case, and it "
                   "is the case where the method returns almost nothing.",
                   14, False, BODY, BODY_FONT)], line_spacing=1.16)
    caption(slide, "Both percentages come from the same pipeline (gene ranking "
                   "plus read equalisation). 1.6% is too few cells for "
                   "gene-level statistics in most patients, so the next two "
                   "slides use the capped rule, which keeps 15%.")

    # 10 -- against the reference signature. The figure only carries the
    # reference bars when the workbook was available to make_report_figures.py,
    # so the wording follows the same switch rather than claiming a comparison
    # the slide does not show.
    if reference:
        figure_slide(
            presentation,
            "Checked against an independent metastasis signature",
            "Faming's primary-versus-metastasis tissue comparison, against the "
            "cells our method kept",
            figures / "fig6_gsea_agreement.png",
            "Androgen response agrees. Proliferation is reversed, and it is by "
            "far the strongest signal on our side (E2F +3.4, G2M +2.8, both "
            "FDR < 0.001), which points at what the method is really "
            "separating.",
            image_top=2.0)
    else:
        figure_slide(
            presentation,
            "What actually separates the cells we kept",
            "Pathway enrichment on the prostate pairs, capped rule",
            figures / "fig6_gsea_agreement.png",
            "Proliferation dominates: E2F +3.4 and G2M +2.8, both "
            "FDR < 0.001, against androgen response at −1.4 (FDR 0.046) "
            "and nothing significant for invasion or interferon.",
            image_top=2.0)

    # 11 -- cause two
    figure_slide(
        presentation,
        "Cause 2: the kept cells are the dividing cells",
        "Of the 598 genes that separate the two groups, the strongest are the "
        "cell-division programme",
        figures / "fig7_cell_cycle.png",
        "DIAPH3, TOP2A, NUSAP1, KIF11, ASPM, CENPF and MKI67 are all enriched "
        "in the kept cells. Prostate, capped rule, paired across 4 patients. "
        "The mitochondrial genes in the list are not dying cells: these are "
        "isolated nuclei, median mitochondrial content 0.2%, and the gate's "
        "mitochondrial bias fell from 0.82 to 0.55 with the same fix that "
        "removed depth.",
        image_top=2.0)

    # 12 -- replication
    figure_slide(
        presentation,
        "And it happens in both cancers",
        "Kept versus rejected primary cells, within each patient — the "
        "kept cells divide faster",
        figures / "fig_hallmark_ovarian.png",
        "Ovarian, 26 patients: kept cells +0.089 against the rejected ones, in "
        "17 of 26 patients, while the metastasis moves only +0.015 on the same "
        "baseline. Prostate, 4 patients: +0.040 against +0.006, in 4 of 4. "
        "Interferon and invasion show no consistent shift in either cancer.",
        image_top=2.0)

    # 13 -- where we are
    slide = add_slide(presentation, dark=True)
    frame = text_box(slide, MARGIN + 0.3, 0.72, WIDE - 2 * MARGIN - 0.6, 1.0)
    write(frame, [("Where this leaves us", 34, True, WHITE, TITLE_FONT)],
          first=True)

    left = text_box(slide, MARGIN + 0.3, 1.95, 5.6, 4.5)
    write(left, [("Fixed", 20, True, ACCENT, BODY_FONT)], first=True,
          space_after=11)
    for item in ("Sequencing depth, in both cancers and against simulated "
                 "ground truth",
                 "The calibration that made the kept fraction meaningless",
                 "The cross-patient control the project had never run"):
        write(left, [("—  " + item, 14, False, PALE, BODY_FONT)],
              space_after=9, line_spacing=1.16)

    right = text_box(slide, MARGIN + 6.7, 1.95, 5.7, 4.5)
    write(right, [("Open", 20, True, ACCENT, BODY_FONT)], first=True,
          space_after=11)
    for item in ("Cell division is now the strongest confounder — it can "
                 "be regressed out, which is a standard step",
                 "The threshold tightens as the two tissues diverge, so the "
                 "rule needs a different comparison built into it",
                 "Transcriptional similarity to an adapted metastasis is not "
                 "evidence of ancestry — that limit no algorithm removes"):
        write(right, [("—  " + item, 14, False, PALE, BODY_FONT)],
              space_after=9, line_spacing=1.16)

    note = text_box(slide, MARGIN + 0.3, 6.6, WIDE - 2 * MARGIN - 0.6, 0.5)
    write(note, [("Next: rerun the ovarian differential expression on the "
                  "depth-corrected cells, with cell cycle removed.", 13, False,
                  DIM, BODY_FONT)], first=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(str(output))
    print(f"wrote {output}  ({len(presentation.slides._sldIdLst)} slides)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-gsea", action="store_true",
                        help="the enrichment figure carries the reference bars, "
                             "so slide 10 is written as a comparison")
    args = parser.parse_args()
    build(args.figures, args.output, args.reference_gsea)


if __name__ == "__main__":
    main()
