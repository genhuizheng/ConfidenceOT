"""Add one "LTL331 PDX: the data" slide to a deck that the user has edited by hand, without rebuilding it.

    python realdata_benchmark/v3_grid/slides/add_pdx_data_slide.py --deck DECK_WITH_A_COPY_OF_THE_STATES_SLIDE.pptx \
        --slide 10 --data results/v3_grid_20261104/figures/pdx/data \
        --depth results/ltl331_nepc/tacc_20261031/ltl331_confidenceot/inputs/ds/predownsample_depth.csv.gz --out OUT.pptx

The slide at --slide must be a copy of the states slide (made with the pptx skill's add_slide.py, so the decoration and
the logo come with it): its tables and figure are removed, the title is replaced, and a timeline of the 8 time points
(OT cells, median UMIs per cell) and a table describing the data are drawn. Every other slide is left as it is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_deck_20261105 import INK, LEFT, ORANGE, PEACH, RIGHT, SUBTLE, TOP, set_runs, table  # noqa: E402

SAMPLES = ["w00_preCX_1", "w02_2wkCX_2", "w04_4wkCX_1", "w08_8wkCX_1", "w12_12wkCX_2", "w16_16wkCX_1", "w20_20wkCX_1_R",
           "w22_22wkCX_2_R"]
WEEKS = [0, 2, 4, 8, 12, 16, 20, 22]


def label(s, x, y, w, h, lines, size=11, color=INK, bold=False, align=PP_ALIGN.CENTER):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    for k, line in enumerate(lines):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
    return tb


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--deck", type=Path, required=True)
    a.add_argument("--slide", type=int, required=True)
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--depth", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a = a.parse_args()

    cells = pd.read_csv(a.data / "cells.tsv.gz", sep="\t", usecols=["cell", "sample_id"])
    ot = set(pd.read_csv(a.data / "gates_logcpm.tsv.gz", sep="\t", usecols=["cell"], dtype={"cell": str}).cell)
    depth = pd.read_csv(a.depth).set_index("observation_id")["predownsample_total_counts"]
    c = cells[cells.cell.isin(ot)].assign(depth=lambda t: t.cell.map(depth))
    per = c.groupby("sample_id").agg(n=("cell", "size"), umi=("depth", "median")).reindex(SAMPLES)

    prs = Presentation(a.deck)
    s = prs.slides[a.slide - 1]
    for sh in list(s.shapes):
        if sh.has_table or (sh.shape_type == 13 and sh.left > Inches(4)):  # the copied tables and figure, not the logo
            sh._element.getparent().remove(sh._element)
        elif sh.has_text_frame and sh.text_frame.text.startswith("LTL331 PDX"):
            set_runs(sh.text_frame.paragraphs[0], "LTL331 PDX: the data")
    s.notes_slide.notes_text_frame.text = (
        "GSE297328, Sar et al., Genome Medicine 2026 (doi 10.1186/s13073-026-01806-8). The LTL331 patient-derived "
        "xenograft (prostate adenocarcinoma) grown under the kidney capsule of NOD-SCID mice; the host is castrated after "
        "the wk 0 sample. Weeks 2-16 are after castration; weeks 20 and 22 are the castration-resistant relapse LTL331R, "
        "which is neuroendocrine.\n"
        "One tumour per time point, so there are no biological replicates: every statement is about this one time course.\n"
        "Counts: rebuilt from the RAW CellRanger matrices, because the annotated h5ad holds only log-normalised values; "
        "every annotated cell is found by barcode and reproduces the authors' .raw to 1e-15.\n"
        "Timeline: OT cells per time point (the project QC, 50,492 of the authors' 51,726) and the median UMIs per cell "
        "before any equalisation. The OT runs on these counts without downsampling.")

    # timeline: weeks to scale
    x0, x1, yl = LEFT + 0.75, RIGHT - 0.55, TOP + 0.95
    xs = [x0 + (x1 - x0) * w / 22 for w in WEEKS]
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(xs[6] - 0.32), Inches(yl - 0.92), Inches(xs[7] - xs[6] + 0.64),
                              Inches(1.58))
    band.fill.solid(); band.fill.fore_color.rgb = PEACH; band.line.fill.background()
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0 - 0.3), Inches(yl), Inches(x1 + 0.3), Inches(yl))
    ln.line.color.rgb = SUBTLE; ln.line.width = Pt(1.5)
    for k, x in enumerate(xs):
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - 0.09), Inches(yl - 0.09), Inches(0.18), Inches(0.18))
        dot.fill.solid(); dot.fill.fore_color.rgb = ORANGE if k >= 6 else INK; dot.line.fill.background()
        top = ["wk 0", "pre-CX"] if k == 0 else [f"wk {WEEKS[k]}"]
        label(s, x - 0.5, yl - 0.50, 1.0, 0.36, top, size=12, bold=True)
        r = per.loc[SAMPLES[k]]
        label(s, x - 0.55, yl + 0.18, 1.1, 0.5, [f"{int(r.n):,} cells", f"{r.umi / 1000:.1f}k UMI"], size=11, color=SUBTLE)
    cx = (xs[0] + xs[1]) / 2
    label(s, cx - 0.75, yl - 0.86, 1.5, 0.25, ["castration ↓"], size=11, color=ORANGE, bold=True)
    label(s, xs[1] - 0.2, yl + 0.78, xs[5] - xs[1] + 0.4, 0.25, ["after castration: PRAD, AR-low, then the mesenchymal "
                                                                "gateway (wk 12-16)"], size=11, color=SUBTLE)
    label(s, xs[6] - 0.32, yl - 0.86, xs[7] - xs[6] + 0.64, 0.25, ["relapse (NEPC)"], size=11, color=ORANGE, bold=True)

    rows = [["Item", "Setting"],
            ["Study", "Sar et al., Genome Medicine 2026; GEO GSE297328"],
            ["Model", "LTL331 patient-derived prostate adenocarcinoma xenograft, kidney capsule of NOD-SCID mice; one "
                      "tumour per time point, no replicate"],
            ["Design", "host castration after wk 0; sampled 2, 4, 8, 12, 16 weeks after; wk 20 and 22 = the castration-"
                       "resistant relapse LTL331R (NEPC)"],
            ["Assay", "single-cell RNA-seq (10x Genomics, CellRanger matrices); human-specific reads only"],
            ["Cells", f"51,726 after the authors' QC; {len(c):,} after the project QC = the OT cells; median UMIs per cell "
                      f"{per.umi.min() / 1000:.1f}k-{per.umi.max() / 1000:.1f}k by time point"],
            ["Counts", "integer counts rebuilt from the RAW CellRanger matrices (reproduce the authors' .raw to 1e-15); "
                       "no downsampling in the OT"],
            ["Labels", "the authors' 18 Seurat clusters → 8 published states (next slide); for reading the results only, "
                       "the OT uses no label; the authors' UMAP for display only"],
            ["Published", "one clone reprogrammes PRAD → NEPC through a transient mesenchymal gateway at wk 12-16 "
                          "(~0.25% of cells), then mostly an ASCL1− fate"],
            ["OT here", "the 7 adjacent pairs (wk t → next), ConfidenceOT M4-E, acceptance 0.99; Splatter L5 best and "
                        "logCPM + PCA only"]]
    table(s, LEFT, TOP + 2.15, RIGHT - LEFT, rows, [1.6, 10.93], size=11, row_h=0.34)
    prs.save(a.out)
    print(f"wrote {a.out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
