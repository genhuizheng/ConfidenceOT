"""English progress deck: RNA-OT vs CRISPR-OT, built on the layout of ConfidenceOT_20261015.pptx (logo, title box,
bottom bar and slide-number box are copied from it; its slides are removed). One main figure per slide, a one-line
takeaway under it, details in the speaker notes.

    python lineage_validation/rna_lineage_discordance_20261110/110_make_pptx.py \
        --template D:/Xia_lab/Progress_report/progress_report7/ConfidenceOT_20261015.pptx \
        --figs results/rna_lineage_discordance_20261110/figures --out D:/Xia_lab/Progress_report/progress_report7/RNA_CRISPR_lineage_20261112.pptx
"""

from __future__ import annotations

import argparse
import copy
import io
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

A = "http://schemas.openxmlformats.org/drawingml/2006/main"
ORANGE, PEACH, GREYF, GREYL, INK = "BF5700", "FDEBDD", "F2F2F2", "A6A6A6", "262626"


def rgb(h):
    return RGBColor.from_string(h)


class Deck:
    def __init__(self, template):
        self.p = Presentation(template)
        s2 = self.p.slides[1]
        self.bar = [copy.deepcopy(sh._element) for sh in s2.shapes if sh.name in ("Rectangle 5", "Rectangle 6")]
        self.title_el = copy.deepcopy([sh for sh in s2.shapes if sh.name == "TextBox 3"][0]._element)
        logo = [sh for sh in s2.shapes if sh.name == "Picture 8"][0]
        self.logo = (logo.image.blob, logo.left, logo.top, logo.width, logo.height)
        sld = self.p.slides._sldIdLst
        for sid in list(sld):
            self.p.part.drop_rel(sid.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")); sld.remove(sid)
        self.blank = self.p.slide_layouts[6]

    def slide(self, title, notes=""):
        s = self.p.slides.add_slide(self.blank)
        for ph in list(s.placeholders):
            ph._element.getparent().remove(ph._element)
        for el in self.bar:
            s.shapes._spTree.append(copy.deepcopy(el))
        b, x, y, w, h = self.logo
        s.shapes.add_picture(io.BytesIO(b), x, y, w, h)
        t = copy.deepcopy(self.title_el); s.shapes._spTree.append(t)
        runs = t.findall(f".//{{{A}}}r")
        runs[0].find(f"{{{A}}}t").text = title
        for r in runs[1:]:
            r.getparent().remove(r)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    @staticmethod
    def image(s, path, x, y, w, h):
        """Fit the image into the box (inches), centred."""
        im = Image.open(path); ar = im.width / im.height
        bw, bh = w, h
        if bw / bh > ar:
            ww, hh = bh * ar, bh
        else:
            ww, hh = bw, bw / ar
        return s.shapes.add_picture(str(path), Inches(x + (bw - ww) / 2), Inches(y + (bh - hh) / 2), Inches(ww), Inches(hh))

    @staticmethod
    def text(s, x, y, w, h, paras, size=15, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
        tf.vertical_anchor = anchor
        for k, para in enumerate(paras if isinstance(paras, list) else [paras]):
            pp = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); pp.alignment = align
            items = para if isinstance(para, list) else [(para, {})]
            for txt, opt in items:
                r = pp.add_run(); r.text = txt; f = r.font
                f.size = Pt(opt.get("size", size)); f.bold = opt.get("bold", bold); f.italic = opt.get("italic", False)
                f.color.rgb = rgb(opt.get("color", color))
            pp.space_after = Pt(6)
        return tb

    @staticmethod
    def box(s, x, y, w, h, head, sub="", hi=False, head_size=13, sub_size=11):
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        sh.adjustments[0] = 0.14
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(PEACH if hi else GREYF)
        sh.line.color.rgb = rgb(ORANGE if hi else GREYL); sh.line.width = Pt(1)
        sh.shadow.inherit = False
        tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for m in ("margin_left", "margin_right"):
            setattr(tf, m, Inches(0.05))
        p0 = tf.paragraphs[0]; p0.alignment = PP_ALIGN.CENTER; r = p0.add_run(); r.text = head
        r.font.size = Pt(head_size); r.font.bold = True; r.font.color.rgb = rgb("000000")
        if sub:
            p1 = tf.add_paragraph(); p1.alignment = PP_ALIGN.CENTER; r = p1.add_run(); r.text = sub
            r.font.size = Pt(sub_size); r.font.color.rgb = rgb("595959")
        return sh

    @staticmethod
    def arrow(s, x, y, w=0.24, h=0.22):
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
        a.fill.solid(); a.fill.fore_color.rgb = rgb("8C8C8C"); a.line.fill.background(); a.shadow.inherit = False
        return a

    @staticmethod
    def table(s, x, y, colw, rows, size=13, rowh=0.36, first_bold=False):
        """Booktabs table as in the template: rules above and below the header and below the last row only."""
        n, m = len(rows), len(rows[0])
        gf = s.shapes.add_table(n, m, Inches(x), Inches(y), Inches(sum(colw)), Inches(rowh * n))
        tbl = gf.table
        tblPr = tbl._tbl.tblPr
        for att in ("firstRow", "bandRow"):
            tblPr.set(att, "0")
        sid = tblPr.find(f"{{{A}}}tableStyleId")
        if sid is not None:
            tblPr.remove(sid)
        for j, w in enumerate(colw):
            tbl.columns[j].width = Inches(w)
        for i in range(n):
            tbl.rows[i].height = Inches(rowh)
            for j in range(m):
                c = tbl.cell(i, j); c.text = ""
                pp = c.text_frame.paragraphs[0]; r = pp.add_run(); r.text = rows[i][j]
                r.font.size = Pt(size); r.font.color.rgb = rgb("000000"); r.font.bold = bool(first_bold and j == 0 and i > 0)
                c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.035)
                c.vertical_anchor = MSO_ANCHOR.MIDDLE
                tcPr = c._tc.get_or_add_tcPr()
                for tag in ("lnL", "lnR", "lnT", "lnB"):
                    el = tcPr.find(f"{{{A}}}{tag}")
                    if el is not None:
                        tcPr.remove(el)
                rules = {"lnT": i == 0, "lnB": i == 0 or i == n - 1}
                for tag in ("lnL", "lnR", "lnT", "lnB"):
                    ln = etree.SubElement(tcPr, f"{{{A}}}{tag}")
                    if rules.get(tag):
                        ln.set("w", "19050"); sf = etree.SubElement(ln, f"{{{A}}}solidFill"); etree.SubElement(sf, f"{{{A}}}srgbClr").set("val", "000000")
                    else:
                        etree.SubElement(ln, f"{{{A}}}noFill")
                for f_ in tcPr.findall(f"{{{A}}}solidFill") + tcPr.findall(f"{{{A}}}noFill"):
                    tcPr.remove(f_)
                etree.SubElement(tcPr, f"{{{A}}}noFill")
        return gf


def take(d, s, text):
    d.text(s, 0.55, 6.62, 11.6, 0.5, text, size=15, color=INK)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--figs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    F = a.figs; d = Deck(a.template); DF = F / "deck_20261112"    # slide versions (111_, 92_ --deck)
    X, Y, W, H = 0.5, 0.78, 12.3, 5.75     # main figure box

    # 1 cover
    s = d.slide("RNA-OT vs CRISPR-OT", "Progress report, 2026-11-12. KP-Tracer (Yang et al. 2022) and Quinn et al. 2021; ConfidenceOT on RNA and on CRISPR edits.")
    d.text(s, 0.6, 2.0, 5.6, 1.6, [[("RNA vs CRISPR lineage", {"size": 36, "bold": True})],
                                   [("in the same tumour cells", {"size": 28})]])
    d.text(s, 0.6, 3.9, 5.6, 1.4, ["KP-Tracer (lung tumours and metastases, 32 mice)", "Quinn 2021 (A549 xenografts, 4 experiments)",
                                   [("ConfidenceOT  ·  2026-11-12", {"color": "595959"})]], size=16)
    d.image(s, DF / "3724_NT_T1.png", 6.4, 0.8, 6.5, 5.7)

    # 2 why
    s = d.slide("Why compare RNA and CRISPR?",
                "RNA tells us how similar two cells look now. CRISPR edits tell us how closely two cells are related. "
                "If we run the same OT on both, any difference between the two matchings points to cells whose current state and "
                "family history disagree: relatives that look different, or unrelated cells that look alike. The aim is not to decide "
                "which OT is right.")
    d.box(s, 0.7, 1.3, 2.6, 1.1, "RNA counts", "how alike two cells look", hi=False)
    d.box(s, 0.7, 3.0, 2.6, 1.1, "CRISPR edits", "how closely two cells are related", hi=False)
    d.arrow(s, 3.45, 1.74); d.arrow(s, 3.45, 3.44)
    d.box(s, 3.85, 1.3, 2.2, 1.1, "RNA-OT", "cost = RNA distance", hi=True)
    d.box(s, 3.85, 3.0, 2.2, 1.1, "CRISPR-OT", "cost = edit distance", hi=True)
    d.arrow(s, 6.2, 2.59)
    d.box(s, 6.6, 1.9, 2.7, 1.5, "Same cells, same solver", "same source and target cells, same calibration, only the cost differs")
    d.arrow(s, 9.45, 2.59)
    d.box(s, 9.85, 1.6, 2.9, 2.1, "Compare", "which cells are kept or rejected, and who is matched to whom", hi=True)
    d.text(s, 0.7, 4.55, 11.8, 1.9, [[("Two kinds of difference to look for", {"bold": True, "size": 16})],
                                     "Same clade, different RNA states: relatives that now look different",
                                     "Different clades, similar RNA states: unrelated cells that look alike"], size=15)
    take(d, s, "The goal is to see where cell state and lineage agree and where they do not, not to pick a winner.")

    # 3 workflow
    s = d.slide("Data and analysis workflow",
                "RNA-OT: the v3 ConfidenceOT fits (raw counts, frozen calibration, Splatter L5-best preprocessing; M4-E). "
                "CRISPR-OT: new fits with the same solver; cost = 1 - prior-weighted edit agreement (missing intBCs excluded, indel "
                "priors, one event per intBC and string); within-side split calibration, acceptance 0.99, no MNN. "
                "Edit matrix: the allele-table states (equal to the authors' character matrices in 3,849 columns). "
                "Clades: UPGMA on edit distance within one family, cut at 0.5, at least 20 cells. KP states: authors' Cluster-Name.")
    steps = [("raw counts", "and CRISPR allele table", False, 1.35), ("RNA-OT", "RNA cost", True, 1.25), ("CRISPR-OT", "edit cost", True, 1.25),
             ("edit matrix", "no OT", True, 1.25), ("compare", "gates and couplings", False, 1.45), ("clades x states", "sites", False, 1.45),
             ("DEG / GSEA", "", False, 1.25)]
    x = 0.55
    for i, (h_, sb, hi, w) in enumerate(steps):
        d.box(s, x, 0.95, w, 0.85, h_, sb, hi=hi, head_size=12, sub_size=10.5); x += w
        if i < len(steps) - 1:
            d.arrow(s, x + 0.06, 1.27); x += 0.36
    d.table(s, 0.55, 2.2, [4.6, 1.6, 5.9], [["Data", "Cells", "Used for"],
                                             ["KP-Tracer, 45 OT units (10 mice)", "34,820", "RNA-OT vs CRISPR-OT, difference flow"],
                                             ["KP-Tracer, all eligible cells (32 mice)", "74,399", "clade x RNA state, primary vs metastasis"],
                                             ["Quinn 2021, 4 experiments (A549 xenograft)", "63,262", "RNA-OT vs CRISPR-OT (states = our Leiden clusters)"]])
    d.table(s, 0.55, 4.0, [3.0, 9.1], [["Setting", "Value"],
                                        ["OT", "ConfidenceOT, UOT backbone, M4-E gate, calibrated on within-side split nulls"],
                                        ["Same for both OTs", "cells, solver, calibration; only the cost matrix changes"],
                                        ["Clades", "UPGMA on edit distance within one tumour family, cut 0.5, >= 20 cells"],
                                        ["RNA states", "KP: authors' annotation; Quinn: Leiden clusters on RNA coordinates"]])
    take(d, s, "49 CRISPR-OT units were fitted; every comparison uses the same cells as the RNA-OT fit.")

    # 4 top sources
    s = d.slide("RNA-OT vs CRISPR-OT",
                "For every target cell we take the 15 source cells each OT matches it to most strongly. Left: how related those "
                "partners are (CRISPR edit similarity s, 0-1). Right: how alike they look (RNA similarity r). One dot per unit, x = "
                "RNA-OT partners, y = CRISPR-OT partners; dashed line = equal. KP RNA-OT partners: s 0.10-0.34, r 0.45-0.62; "
                "CRISPR-OT partners: s 0.47-0.86, r 0.01-0.13. Overlap of the two top-15 lists: 0-2%.")
    d.image(s, DF / "D1_what_each_OT_matches.png", X, Y, W, H)
    take(d, s, "RNA-OT picks sources that look alike but share few edits; CRISPR-OT picks close relatives that look different.")

    # 5 gates
    s = d.slide("The two gates keep different cells",
                "Cohen's kappa between the RNA-OT and CRISPR-OT keep / reject decisions, per unit and side. Near 0 almost everywhere: "
                "the two gates reject different cells. The higher source-side values are the KP 3724 pair units, where the edit "
                "distance separates cells weakly. Gate disagreement is reported apart from coupling disagreement.")
    d.image(s, DF / "D2_same_cells_kept.png", 1.3, Y, 10.7, H)
    take(d, s, "Agreement on which cells to keep is close to zero (kappa about 0), except a few KP 3724 units.")

    # 6 flow 3460
    s = d.slide("Where do their matches differ?",
                "Same source (primary) and target (metastasis); only cells kept by both gates. Each coupling is normalised to 1 and "
                "summed per source clade x target clade. Grey = matching both OTs agree on; blue = extra matching from RNA-OT; orange "
                "= extra matching from CRISPR-OT. Left tree: clade tree from the edits. Bars coloured by RNA state. Across 45 KP "
                "units, the orange part lies in the same clade (median 87%), the blue part almost never (median 0%); CRISPR-OT puts "
                "more mass on same-clade pairs in 45 of 45 units (0.675 vs 0.40). This is clade-level mass, not cell-to-cell matching. "
                "Full version with gate strips and overlap bars: figures/flow/.")
    d.image(s, DF / "flow_L1__3460__3460_Lkb1_T1__3460_Lkb1_N6.png", 1.6, Y, 10.1, H)
    take(d, s, "Extra CRISPR-OT matching (orange) stays inside a clade; extra RNA-OT matching (blue) crosses clades. 45 of 45 KP units.")

    # 7 flow 3724
    s = d.slide("Where do their matches differ? KP 3724",
                "Pooled case: primary tumour to all four metastases. Two clades: c5 (Mesenchymal-1) and c45 (Mesenchymal-2). Most "
                "mass is shared; RNA-OT adds some c5 <-> c45 mass. 3724 is not resolved by the frozen CRISPR rule (8% distinct edit "
                "sets), clades and sites are confounded, and its depth is the highest of the OT mice.")
    d.image(s, DF / "flow_L23__KPTracer_3724__3724_NT_T1__family_metastases.png", 1.6, Y, 10.1, H)
    take(d, s, "Two clades carry two mesenchymal states; both OTs follow them, RNA-OT adds a little cross-clade mass.")

    # 8 clade vs cell level
    s = d.slide("Clade-level vs cell-level overlap",
                "Medians over KP units. Left: clade-level overlap of RNA-OT and CRISPR-OT (share of clade x clade mass they agree "
                "on). Middle: the same overlap between RNA-OT and a coupling that matches cells only in proportion to clade sizes "
                "(closer than CRISPR-OT in 44 of 45 units). Right: for the same target cells, the overlap of the two OTs' top-15 "
                "partner lists.")
    d.image(s, DF / "D3_clade_vs_cell_level.png", 1.3, Y, 10.7, H)
    take(d, s, "High clade-level overlap mostly reflects clade sizes; the two OTs rarely pick the same partner cells (4%).")

    # 9 tree 3457
    s = d.slide("Same clade, different RNA states",
                "Cell-level CRISPR tree of family 3457_Apc_T1 (885 cells, resolved: 12 edits per cell). Leaves are cells; dashed line "
                "= clade cut. Tracks: clade, authors' RNA state, primary / metastasis; right: informative edits (black present, grey "
                "intBC missing). Clade c6 splits below the cut into an Apc Early branch and an Early EMT-1 / Mesenchymal-1 branch, "
                "each with its own edits. c1 is mostly metastatic Pre-EMT cells.")
    d.image(s, DF / "3457_Apc_T1.png", 1.3, Y, 10.7, H)
    take(d, s, "Inside one clade, sub-branches with their own edits carry different RNA states (Apc Early vs Early EMT-1).")

    # 10 tree 3724
    s = d.slide("CRISPR edits and RNA states in KP 3724",
                "948 of 20,992 cells drawn. c5 = Mesenchymal-1 and Mesenchymal-1 (Met); c45 = Mesenchymal-2. Inside c45, the "
                "sub-branch with extra edits is the metastatic Mesenchymal-2 (Met). Not resolved by the frozen rule (7 edits per "
                "cell, distinct edit sets 8%).")
    d.image(s, DF / "3724_NT_T1.png", 1.3, Y, 10.7, H)
    take(d, s, "Lineage branch, RNA state and site line up: a sub-branch with its own edits holds the metastatic Mesenchymal-2 cells.")

    # 11 shared states
    s = d.slide("Which RNA states are shared across clades?",
                "All eligible KP cells (74,399, 32 mice). For each mouse and RNA state (>= 50 cells) we ask whether the state's cells "
                "fall into fewer clades than chance, as many as chance, or more. Chance = cells shuffled among clades within each "
                "site (999 draws); 'fewer' / 'more' = Hill-number z <= -2 / >= 2. Clade x state coupling beyond the null: 0.149 (18 "
                "mice, resolved lineages), 0.120 without 3724; unchanged with merged (Met) labels.")
    d.image(s, DF / "D4_states_tied_to_few_clades.png", 1.0, Y, 11.3, H)
    take(d, s, "Early EMT-1, Apc Early, Mesenchymal-2 sit in fewer clades than chance; AT2-like often in more; Endoderm-like is mixed.")

    # 12 3513 families
    s = d.slide("Different clades, similar RNA states",
                "KP 3513: for each RNA state (>= 50 cells), the share of its cells that come from each independent primary tumour "
                "(each tumour is its own CRISPR family). Right: tumours with >= 5 cells of the state, and the largest share. "
                "Gastric-like: 5 tumours, 22 clades; High plasticity: 5 tumours, 21 clades. Shared states are not evidence of "
                "convergence on their own.")
    d.image(s, DF / "D5_states_across_tumours_3513.png", 1.0, Y, 11.3, H)
    take(d, s, "The same RNA state can come from several independent tumours (Gastric-like) or mostly from one (Endoderm-like).")

    # 13 primary vs met 3724
    s = d.slide("Primary vs metastasis",
                "KP 3724, all eligible cells. One block per site; within a site, one column per CRISPR clade (width = cells) coloured "
                "by RNA state. A site with fewer than 2 clades or 2 states (>= 5 cells) cannot test the association; that is not "
                "evidence of no association.")
    d.image(s, DF / "D6_primary_vs_metastasis_3724.png", 0.6, Y, 12.1, H)
    take(d, s, "In 3724 each metastasis carries one clade and its state, so single-clade sites cannot test the association.")

    # 14 primary vs met table
    s = d.slide("Primary vs metastasis across mice",
                "Clade x state coupling beyond the within-site null (Cramer's V minus null V), both sides subsampled to the same "
                "number of cells (200 draws), all eligible cells, authors' labels. Mean without 3724: primary 0.015, metastasis 0.037. "
                "The interim result 'metastases are less lineage-coupled' came from 3724 and does not generalise.")
    d.table(s, 0.55, 1.1, [1.1, 1.3, 1.5, 2.5], [["Mouse", "Primary", "Metastasis", "Metastatic cells in identifiable sites"],
                                                  ["3724", "0.198", "0.006", "69%"], ["3457", "0.047", "0.000", "0% (not identifiable)"],
                                                  ["3467", "0.007", "0.150", "94%"], ["3460", "-0.003", "0.078", "75%"],
                                                  ["3451", "0.019", "0.003", "97%"], ["3513", "0.005", "-0.044", "100%"]], size=14, rowh=0.5)
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
    cd = CategoryChartData(); cd.categories = ["3724", "3457", "3467", "3460", "3451", "3513"]
    cd.add_series("Primary", (0.198, 0.047, 0.007, -0.003, 0.019, 0.005)); cd.add_series("Metastasis", (0.006, 0.000, 0.150, 0.078, 0.003, -0.044))
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.1), Inches(1.0), Inches(5.7), Inches(5.3), cd).chart
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(13)
    for ser, colr in zip(ch.series, ("2C6E9B", "B4702A")):
        ser.invert_if_negative = False; ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(colr)
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    ch.value_axis.has_major_gridlines = False; ch.value_axis.tick_labels.font.size = Pt(12); ch.category_axis.tick_labels.font.size = Pt(12)
    ch.value_axis.has_title = True; ch.value_axis.axis_title.text_frame.text = "coupling beyond null (V - null V)"
    ch.value_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    ch.category_axis.has_title = True; ch.category_axis.axis_title.text_frame.text = "mouse"
    ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    take(d, s, "No common direction: weaker in 3724 metastases, stronger in 3460 and 3467, untestable in 3457.")

    # 15 DEG/GSEA design (RNA-OT vs CRISPR-OT preference within a clade); results pending
    s = d.slide("Next: do RNA-OT and CRISPR-OT prefer different cells?",
                "Plan fixed in PLAN_v3_deg.md before any result. Per clade, site and side, on cells kept by both gates: q_R(i) = share "
                "of the clade's RNA-OT mass carried by source cell i, q_C(i) the same for CRISPR-OT; d = n x (q_R - q_C). RNA-OT-"
                "preferred = top quartile of d, CRISPR-OT-preferred = bottom quartile, both supported = neither quartile and each "
                "OT gives the cell at least the uniform share. Wilcoxon on raw counts and on depth-thinned counts; hallmark prerank "
                "GSEA (50 sets, both directions). Controls: RNA-similarity baseline (RNA-OT preference is built from RNA, so part of "
                "the difference is expected), 5 random splits of the same cells, immediate-early-gene check. 257 contrasts, 10 mice.")
    d.box(s, 0.6, 1.25, 2.9, 1.3, "One CRISPR clade", "same site, same side, cells kept by both OTs")
    d.arrow(s, 3.62, 1.78)
    d.box(s, 4.0, 0.95, 3.6, 0.8, "RNA-OT-preferred cells", "RNA-OT gives them more mass", hi=True, head_size=13, sub_size=11)
    d.box(s, 4.0, 1.85, 3.6, 0.8, "both supported", "both OTs give them mass", head_size=13, sub_size=11)
    d.box(s, 4.0, 2.75, 3.6, 0.8, "CRISPR-OT-preferred cells", "CRISPR-OT gives them more mass", hi=True, head_size=13, sub_size=11)
    d.arrow(s, 7.75, 1.78)
    d.box(s, 8.15, 1.25, 4.6, 1.3, "Compare expression", "DEG (Wilcoxon) and hallmark GSEA: RNA-OT-preferred vs CRISPR-OT-preferred")
    d.table(s, 0.6, 3.95, [3.6, 8.5], [["Control", "Why"],
                                        ["RNA-similarity baseline", "RNA-OT preference comes from RNA: only differences beyond it count"],
                                        ["5 random splits", "how large differences are between random groups of the same cells"],
                                        ["Depth-thinned counts", "removes differences that only follow sequencing depth"],
                                        ["Immediate-early genes", "dissociation stress genes reported separately"]], size=13, rowh=0.38)
    take(d, s, "Results pending: 257 contrasts in 10 KP mice are running; no pathway-level claim yet.")

    # 16 controls
    s = d.slide("Controls and limitations",
                "Controls were fixed before the results were read; later additions are dated post-hoc sections in the plans.")
    d.table(s, 0.55, 0.95, [4.0, 8.1], [["Control", "What it showed"],
                                         ["Within-site permutation null", "state-lineage association measured beyond site composition"],
                                         ["Size-only (independent) coupling", "RNA-OT is close to it at clade level (0.98)"],
                                         ["Clade level vs cell level", "0.83 vs 0.04: clade overlap is not cell matching"],
                                         ["Without KP 3724", "state-lineage result holds (0.120); primary vs metastasis does not"],
                                         ["Merged (Met) labels", "no change"],
                                         ["Sequencing depth", "discordance unrelated to depth (rho -0.01); depth-thinned GSEA"]], size=14, rowh=0.4)
    d.table(s, 0.55, 4.0, [4.0, 8.1], [["Limitation", "Consequence"],
                                        ["One tumour per site", "no biological replicates; cell-level tests are exploratory"],
                                        ["KP 3724 unresolved, site-confounded", "reported separately, every pooled number also without it"],
                                        ["CRISPR-OT uses the same edits", "agreement with the edit matrix is not an independent check"],
                                        ["Single time point", "no direction or timing of state changes"]], size=14, rowh=0.4)
    take(d, s, "The main state-lineage result survives the controls; the primary-vs-metastasis claim did not.")

    # 17 findings
    s = d.slide("Main findings and next questions", "")
    d.box(s, 0.55, 0.95, 6.2, 0.55, "What we found", hi=True, head_size=15)
    d.text(s, 0.6, 1.6, 6.1, 4.8, ["1. RNA-OT and CRISPR-OT match different cells (kappa about 0, top-15 overlap 0-2%).",
                                    "2. CRISPR-OT adds same-clade matches; RNA-OT adds cross-clade matches (45 of 45 units).",
                                    "3. Clade-level overlap looks high but mostly reflects clade sizes.",
                                    "4. Some RNA states are tied to a few clades (Early EMT-1, Mesenchymal-2); others are shared by many (Pre-EMT, AT2-like).",
                                    "5. Primary vs metastasis differs by mouse; no common rule."], size=15)
    d.box(s, 7.0, 0.95, 5.8, 0.55, "Next questions", head_size=15)
    d.text(s, 7.05, 1.6, 5.7, 4.8, ["Within a clade, which cells does each OT prefer, and do they differ in expression? (DEG v3, running)",
                                     "Can RNA-OT recover true fates? Test on LARRY (clonal barcodes over days 2, 4, 6).",
                                     "More resolved mice with metastases, to test primary vs metastasis without 3724."], size=15)
    take(d, s, "State and lineage agree for some states and diverge for others; the two OTs see different parts of that structure.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    d.p.save(a.out); print(a.out, len(d.p.slides), "slides")


if __name__ == "__main__":
    main()
