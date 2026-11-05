"""Progress deck 2026-11-05: v3 grid on Splatter N = 5,000 and LTL331 (PDX), in the style of
Progress_report/progress_report7/ConfidenceOT_20261015.pptx (logo, title top right, orange bar with the slide number;
details in the speaker notes).

    python realdata_benchmark/v3_grid/slides/build_deck_20261105.py \
        --template D:/Xia_lab/Progress_report/progress_report7/ConfidenceOT_20261015.pptx \
        --figures results/v3_grid_20261104/figures --data results/v3_grid_20261104/figures/pdx/data \
        --grid-summary results/v3_grid_20261104/splatter/GRID_SUMMARY_v3tacc.tsv \
        --out results/v3_grid_20261104/slides/ConfidenceOT_20261105.pptx

The template's slide 1 (method schematic) and slide 4 (Case 2 schematic) are kept; every other template slide is
dropped. New slides copy the template's decoration (orange bar, number box, logo, title box) from its slide 7.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.opc.packuri import PackURI
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

ORANGE, PEACH, INK, SUBTLE = RGBColor(0xBF, 0x57, 0x00), RGBColor(0xFD, 0xEB, 0xDC), RGBColor(0, 0, 0), RGBColor(0x59, 0x59, 0x59)
TOP, BOTTOM, LEFT, RIGHT = 0.80, 6.55, 0.40, 12.93  # content area (inches); the orange bar starts at 6.64
CFG = {"splatter_l5_best": "ranknm256-midrank|none|scale|cos|pca50", "baseline": "ranknm256-production|none|noscale|cos|pca30",
       "logcpm": "logcpm|none|noscale|cos|pca30"}
LABEL = {"splatter_l5_best": "Splatter L5 best", "baseline": "Production baseline", "logcpm": "logCPM"}
SAMPLES = ["w00_preCX_1", "w02_2wkCX_2", "w04_4wkCX_1", "w08_8wkCX_1", "w12_12wkCX_2", "w16_16wkCX_1", "w20_20wkCX_1_R",
           "w22_22wkCX_2_R"]
WEEK = ["wk 0", "wk 2", "wk 4", "wk 8", "wk 12", "wk 16", "wk 20", "wk 22"]
PAIRS = [f"LTL331__{a}__{b}" for a, b in zip(SAMPLES[:-1], SAMPLES[1:])]


class Deck:
    def __init__(self, template: Path, keep: list[int]):
        self.prs = Presentation(str(template))
        proto = self.prs.slides[6]
        self.decor = [copy.deepcopy(s._element) for s in proto.shapes if s.name in ("Rectangle 5", "Rectangle 6", "Picture 8")]
        self.title_xml = copy.deepcopy(next(s._element for s in proto.shapes if s.name == "TextBox 3"))
        logo = next(s for s in proto.shapes if s.name == "Picture 8")
        self.logo_part = proto.part.related_part(logo._element.blipFill.find(qn("a:blip")).get(qn("r:embed")))
        self.layout = proto.slide_layout
        lst = self.prs.slides._sldIdLst
        for k, sid in reversed(list(enumerate(list(lst), start=1))):
            if k not in keep:
                self.prs.part.drop_rel(sid.get(qn("r:id"))); lst.remove(sid)
        self.kept = {k: self.prs.slides[i] for i, k in enumerate(keep)}
        # python-pptx names a new slide slide<count + 1>.xml, which would collide with a kept part of that name
        for k, sl in self.kept.items():
            sl.part.partname = PackURI(f"/ppt/slides/slide{900 + k}.xml")

    def slide(self, title: str, notes: str = ""):
        s = self.prs.slides.add_slide(self.layout)
        for ph in list(s.placeholders):
            ph._element.getparent().remove(ph._element)
        tree = s.shapes._spTree
        for el in self.decor:
            e = copy.deepcopy(el)
            blip = e.find(".//" + qn("a:blip"))
            if blip is not None:
                blip.set(qn("r:embed"), s.part.relate_to(self.logo_part, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"))
            tree.append(e)
        t = copy.deepcopy(self.title_xml); tree.append(t)
        sh = s.shapes[-1]
        set_runs(sh.text_frame.paragraphs[0], title)
        num = len(self.prs.slides)
        for fld in tree.iter(qn("a:fld")):
            tt = fld.find(qn("a:t"))
            if tt is not None:
                tt.text = str(num)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def save(self, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(out)); print(f"wrote {out} ({len(self.prs.slides)} slides)")


def set_runs(p, text: str):
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def picture(s, path: Path, x, y, w, h, align="center"):
    """Fit the image inside the box (inches), keeping its aspect ratio."""
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih); pw, ph = iw * sc, ih * sc
    px = x + (w - pw) / 2 if align == "center" else x
    return s.shapes.add_picture(str(path), Inches(px), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def text(s, x, y, w, h, paras, size=13, color=INK, fill=None, anchor=MSO_ANCHOR.TOP):
    """paras: list of str or (str, dict) with keys b (bold), color, size, bullet."""
    if fill is not None:
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        box.adjustments[0] = 0.06
        box.fill.solid(); box.fill.fore_color.rgb = fill; box.line.fill.background(); box.shadow.inherit = False
        tf = box.text_frame
        tf.margin_left = tf.margin_right = Inches(0.18); tf.margin_top = tf.margin_bottom = Inches(0.12)
    else:
        box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = box.text_frame
        tf.margin_left = tf.margin_right = Inches(0.02)
    tf.word_wrap = True; tf.vertical_anchor = anchor
    for k, item in enumerate(paras):
        t, o = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(o.get("after", 4))
        r = p.add_run(); r.text = t
        r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("b", False); r.font.name = "Calibri"
        r.font.color.rgb = o.get("color", color)
    return box


def _line(tcPr, tag, w):
    ln = etree.SubElement(tcPr, qn(tag), w=str(int(w)), cap="flat", cmpd="sng", algn="ctr")
    if w:
        sf = etree.SubElement(ln, qn("a:solidFill")); etree.SubElement(sf, qn("a:schemeClr"), val="tx1")
        etree.SubElement(ln, qn("a:prstDash"), val="solid")
    else:
        ln.clear(); ln.attrib.clear(); etree.SubElement(ln, qn("a:noFill"))


def table(s, x, y, w, rows, widths, size=12, header=True, bold_first_col=False, row_h=0.30, mono_cols=(), colour=None):
    """Template table style: no fill, a heavy rule above and below the header, a heavy rule under the last row."""
    nr, nc = len(rows), len(rows[0])
    gf = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow"):
        tblPr.set(a, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        tblPr.remove(sid)
    tot = sum(widths)
    for j, cw in enumerate(widths):
        tbl.columns[j].width = Emu(int(Inches(w) * cw / tot))
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, v in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Emu(54864); c.margin_top = c.margin_bottom = Emu(32004)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]; r = p.add_run(); r.text = str(v)
            r.font.size = Pt(size); r.font.name = "Courier New" if (j in mono_cols and i > 0) else "Calibri"
            r.font.bold = bool((bold_first_col and j == 0 and i > 0))
            r.font.color.rgb = colour(i, j) if colour else INK
            tcPr = c._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                for e in tcPr.findall(qn(tag)):
                    tcPr.remove(e)
            top = 19050 if (header and i in (0, 1)) or (not header and i == 0) else 0
            bot = 19050 if i == nr - 1 else 0
            _line(tcPr, "a:lnL", 0); _line(tcPr, "a:lnR", 0); _line(tcPr, "a:lnT", top); _line(tcPr, "a:lnB", bot)
            # schema order: lines before the fill
            for e in tcPr.findall(qn("a:noFill")):
                tcPr.remove(e)
            etree.SubElement(tcPr, qn("a:noFill"))
    return gf


def rejected_shares(srej: pd.DataFrame) -> pd.DataFrame:
    """Per configuration, time point and gate (as target / as source): rejected share of the time point's cells."""
    out = []
    for key, cid in CFG.items():
        r = srej[srej.config_id == cid]
        for k, pair in enumerate(PAIRS):
            for side, t in (("source", k), ("target", k + 1)):
                g = r[(r.pair == pair) & (r.side == side)]
                out.append({"cfg": key, "time": t, "as": side, "rejected": g.rejected.sum() / g.cells.sum()})
    return pd.DataFrame(out)


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--template", type=Path, required=True)
    a.add_argument("--figures", type=Path, required=True)
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--grid-summary", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a = a.parse_args()
    F, P = a.figures / "splatter", a.figures / "pdx"
    gs = pd.read_csv(a.grid_summary, sep="\t")
    srej = pd.read_csv(a.data / "state_rejected_all.tsv", sep="\t")
    rk = pd.read_csv(F / "splatter_ranking.tsv", sep="\t").set_index("config_id")["rank"]
    n_cfg = srej.config_id.nunique()

    d = Deck(a.template, keep=[1, 4])
    W = RIGHT - LEFT

    # ---- Splatter -------------------------------------------------------------------------------------------------
    s = d.slide("Splatter N = 5,000: experimental setting", notes=(
        "Table 1. The Splatter benchmark behind the v3 grid.\n"
        "Every Splatter parameter is the installed package default (newSplatParams(), read at run time). Custom design "
        "factors, which Splatter cannot set by itself: N per side, the target depth ratio 0.5, the Group1 perturbation on "
        "the target, the L5 target proportions, and the paired sampler (common uniforms, so every level shares the same "
        "group DE and batch draws).\n"
        "Levels are cumulative: each adds one factor to the one before.\n"
        "Effective readout: when one side retains no cell, the other side's retained cells count as unmatched."))
    rows = [["Category", "Setting"],
            ["Simulator", "Splatter 1.30.0, method = \"groups\"; every parameter at its package default"],
            ["Populations", "K = 5 groups; source proportions equal"],
            ["Cells per side", "N = 5,000 source + 5,000 target"],
            ["Replicates", "3, each its own seed; levels paired within a replicate (same group DE and batch draws)"],
            ["L1  depth", "target library size × 0.5 (lib.loc + log 0.5)"],
            ["L2  + dropout", "L1 + Splatter dropout (default mid / shape) on both sides"],
            ["L3  + batch", "L2 + Splatter batch effect between source and target (default batch.facLoc / facScale)"],
            ["L4  + perturbation", "L3 + transcriptional perturbation of Group1 on the target only (shared, label kept)"],
            ["L5  + imbalance", "L4 + target proportions 0.40 / 0.25 / 0.15 / 0.12 / 0.08 (Group1 to Group5)"],
            ["Cases", "Case 1: all 5 groups on both sides. Case 2: Group5 removed from the target (lost) or from the source "
                      "(emerged). Case 3: Groups 1-2 in the source, 3-5 in the target"],
            ["Calibration", "v3, frozen 2026-11-04: within-side split nulls + mnncovlocal batch correction; no downsampling"],
            ["OT gate", "ConfidenceOT M4-E, acceptance 0.99, no rejection cap"],
            ["Grid", "37 encodings × scaling × cost × PCA {30, 50} = 296 configurations (next slide)"],
            ["Runs", "5 levels × 4 pairs × 3 replicates = 60 pairs; × 296 = 17,760 fits, 0 failed"],
            ["Readout", "Case 1: false rejection rate; Case 2: Rejection F1 (mean of lost and emerged); Case 3: unmatched recall"]]
    table(s, LEFT, TOP + 0.10, W, rows, [2.2, 10.3], size=12, row_h=0.335)

    case2 = d.kept[4]
    case2.notes_slide.notes_text_frame.text = (
        "Case 2 is built in both directions.\n"
        "Top row, population lost: Group5 is in the source only. Correct gate: reject those source cells, keep the rest.\n"
        "Bottom row, population emerged: Group5 is in the target only. Correct gate: reject those target cells.\n"
        "The Case 2 F1 on the next slides is the mean over these two pairs; each pair's F1 is scored on the side that "
        "holds the unmatched population.")
    d.prs.slides._sldIdLst.append(d.prs.slides._sldIdLst[1])  # the Case 2 schematic after the setting table

    s = d.slide("Preprocessing grid: 296 configurations", notes=(
        "Every configuration = encoding × tie rule × scaling × cost × PCA. 37 encoding keys: ranknm and rank_value at "
        "K = 128, 256, 512, 1024, 2048 and rankhvg at H = 2,000, 3,000, each with three tie rules (36), plus logcpm.\n"
        "Ties: genes with equal counts in one cell. production = the stored entry order of the joint matrix (arbitrary but "
        "reproducible; what the production code does). expected = the average over random tie-breaking. midrank = every "
        "tied gene gets the value at the group's mean rank; equals expected unless the tie group straddles the top-K cut.\n"
        "The production baseline is ranknm256 production noscale cos PCA30."))
    rows = [["Factor", "Levels", "What is computed for each cell"],
            ["ranknm<K>", "K = 128 … 2048", "the cell's top K genes by count, value 1 − r / K; every other gene 0"],
            ["rank_value<K>", "K = 128 … 2048", "the same after dividing each gene by its nonzero median over both sides (Geneformer)"],
            ["rankhvg<H>", "H = 2,000, 3,000", "ranks within the H highest-variance log-CPM genes, no top-K cut"],
            ["logcpm", "—", "log(1 + 10⁴ × count / total)"],
            ["ties", "production, expected, midrank", "what value genes with equal counts in one cell get (next slide; not for logcpm)"],
            ["scaling", "scale, noscale", "scale: each gene standardised before PCA; noscale: centred only"],
            ["cost", "cos, sqeuc", "cos: PC vectors scaled to unit length (cosine distance); sqeuc: squared Euclidean"],
            ["PCA", "30, 50", "principal components on source and target together"]]
    table(s, LEFT, TOP + 0.10, W, rows, [1.8, 2.8, 7.9], size=13, row_h=0.42, mono_cols=(0,))
    text(s, LEFT, TOP + 4.25, W, 1.2, [
        ("37 encoding keys × 2 scalings × 2 costs × 2 PCA sizes = 296 configurations, every one fitted on every pair.", {"b": True}),
        "Production baseline: ranknm256 · production · noscale · cos · PCA30 (rank "
        f"{rk[CFG['baseline']]} of 296 by L5 Case 2 F1)."], size=14, fill=PEACH)

    s = d.slide("Tie rules: production, expected, midrank", notes=(
        "Why ties matter: counts are integers, so in a sparse cell many genes share a count of 1 or 2, and such a tie "
        "group often sits exactly at the top-K cut. The rank encoding then has to decide which of them get the last "
        "slots.\n"
        "Toy cell, K = 4: slot values 1, 0.75, 0.50, 0.25 for positions 0-3; everything from position 4 on is 0. "
        "A and B are untied (positions 0 and 1). C, D, E, F tie at count 2 and occupy positions 2-5, so the group "
        "straddles the cut: 4 genes compete for 2 slots.\n"
        "production: confidenceot.preprocessing.rank_value_encode, a stable argsort over the cell's entries in the order "
        "the joint sparse matrix stores them. That order comes from scipy's column indexing into the sorted gene keys: it "
        "is neither gene name nor gene position, but it is fixed, so the result is reproducible. Here the stored order "
        "is taken as E, C, F, D.\n"
        "expected: (0.50 + 0.25) x 2 slots / 4 genes = 0.1875 each. midrank: mean position 2 + (4 - 1) / 2 = 3.5, value "
        "1 - 3.5 / 4 = 0.125 each (0 if the mean position is >= K).\n"
        "A tie group wholly inside the top K (e.g. two genes at positions 2-3): expected = midrank = (0.50 + 0.25) / 2 "
        "= 0.375 each; production gives 0.50 and 0.25. rankhvg has no top-K cut, so there expected = midrank exactly.\n"
        "Implementation: realdata_benchmark/rb_common.py rank_encode."))
    rows = [["Gene", "Count", "Position", "production", "expected", "midrank"],
            ["A", "9", "0", "1.00", "1.00", "1.00"], ["B", "5", "1", "0.75", "0.75", "0.75"],
            ["C", "2", "2-5 (tied)", "0.25", "0.19", "0.13"], ["D", "2", "2-5 (tied)", "0", "0.19", "0.13"],
            ["E", "2", "2-5 (tied)", "0.50", "0.19", "0.13"], ["F", "2", "2-5 (tied)", "0", "0.19", "0.13"],
            ["Genes above 0", "", "", "4", "6", "6"]]
    text(s, LEFT, TOP + 0.05, 5.6, 0.75, [("Toy cell, K = 4: slot values 1, 0.75, 0.50, 0.25, then 0. C, D, E, F tie "
                                           "at count 2: four genes for the last two slots.", {"b": True})], size=13)
    table(s, LEFT, TOP + 0.85, 5.6, rows, [1.45, 0.75, 1.15, 1.25, 1.05, 1.0], size=13, row_h=0.40)
    text(s, LEFT, TOP + 4.25, 5.6, 1.2, [
        ("Why it matters", {"b": True}),
        "Counts are integers, so in a sparse cell many genes share a count of 1 or 2, and that tie group often sits "
        "exactly at the top-K cut."], size=13)
    x2, w2 = 6.35, RIGHT - 6.35
    for k, (head, body) in enumerate((
            ("production: the order the matrix happens to store", "Ties keep the order in which the joint count matrix "
             "stores the cell's entries. Fixed and reproducible, but which tied gene wins is arbitrary (here E, then C; "
             "D and F get 0). This is what the production code does."),
            ("expected: the average over random tie orders", "Every tied gene gets the mean of the slot values the "
             "group wins, shared over all its members: (0.50 + 0.25) × 2 / 4 = 0.19. No gene order involved; the "
             "total value 2.50 is the same as production."),
            ("midrank: the value at the group's mean rank", "Every tied gene gets 1 − p / K at the mean position "
             "p = 3.5: 0.13. Equal to expected when the group is wholly inside the top K; at the cut it gives all four "
             "tied genes a value, so more than K genes can be above 0 (here 6 > 4)."))):
        text(s, x2, TOP + 0.05 + k * 1.92, w2, 1.78, [(head, {"b": True, "size": 14}), body], size=13, fill=PEACH)

    s = d.slide("Splatter: all cases and levels, top 30 by L5 F1", notes=(
        "Rows: the 30 best configurations by Case 2 Rejection F1 at L5 (batch + perturbation + imbalance; outlined), ties "
        "broken by the L1-L5 mean F1; the production baseline in red at the bottom (rank 171).\n"
        "Each cell: mean over 3 replicates; Case 2 also over its lost and emerged pairs.\n"
        "Case 1 false rejection rate (lower is better), Case 2 Rejection F1, Case 3 unmatched recall; each block has its "
        "own colour scale.\n"
        "Rank 1, ranknm256 midrank scale cos PCA50, is also the best on the L1-L5 mean F1 (0.90). L5 Case 1 FRR stays at "
        "0.13-0.25 even at the top."))
    picture(s, F / "splatter_all_cases_top30.png", LEFT, TOP, W, BOTTOM - TOP)

    s = d.slide("Case 2 F1, all 296 configurations", notes=(
        "Left strips: the factors of each row (cost, PCA, scaling, encoding). Rows ranked by L5 Case 2 F1, as before.\n"
        "Table: per level, Case 1 FRR / Case 2 F1 / Case 3 recall (effective readout), mean over 3 replicates.\n"
        "logCPM row = the baseline with only the encoding changed; it is the logCPM configuration shown on the PDX slides."))
    picture(s, F / "splatter_f1_all296.png", LEFT, TOP, 5.9, BOTTOM - TOP)
    lv = ["L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance"]
    rows = [["Level", "Metric", f"L5 best (#{rk[CFG['splatter_l5_best']]})", f"Baseline (#{rk[CFG['baseline']]})",
             f"logCPM (#{rk[CFG['logcpm']]})"]]
    for k, l in enumerate(lv):
        for m, lab in (("case2_f1_eff", "Case 2 F1"), ("case1_frr_eff", "Case 1 FRR")):
            rows.append([f"L{k + 1}" if m == "case2_f1_eff" else "", lab] +
                        [f"{gs[(gs.config_id == CFG[c]) & (gs.level == l)][m].iloc[0]:.2f}" for c in ("splatter_l5_best", "baseline", "logcpm")])
    table(s, 6.55, TOP + 0.15, 6.35, rows, [0.8, 1.6, 1.5, 1.4, 1.3], size=12, row_h=0.33)
    text(s, 6.55, TOP + 4.0, 6.35, 1.6, [
        ("L5 separates the configurations; L1-L3 hardly do.", {"b": True}),
        "The top is almost all cosine cost; scaling and PCA50 are frequent. Case 3 recall is 1.00 for all three rows at "
        "L3-L5."], size=13, fill=PEACH)

    # ---- PDX ------------------------------------------------------------------------------------------------------
    cells = pd.read_csv(a.data / "cells.tsv.gz", sep="\t", usecols=["cell", "sample_id"])
    r0 = srej[srej.config_id == CFG["baseline"]]
    n_ot = [int(r0[(r0.pair == PAIRS[0]) & (r0.side == "source")].cells.sum())] + \
           [int(r0[(r0.pair == p) & (r0.side == "target")].cells.sum()) for p in PAIRS]
    s = d.slide("LTL331 PDX: eight time points, PRAD to NEPC", notes=(
        "GSE297328 (Sar et al. 2026): the LTL331 patient-derived xenograft before castration (wk 0) and 2-22 weeks after; "
        "wk 20 and 22 are castration-resistant regrowth, which transdifferentiates to NEPC.\n"
        "States: the authors' 18 seurat_clusters mapped to their 8 published states. No filter on states, clusters or "
        "cells beyond the project QC; nothing is fitted on the authors' UMAP or Harmony embedding (display only).\n"
        f"OT cells: QC-passed cells, {sum(n_ot):,} of {len(cells):,} converted cells. Raw counts, no downsampling, "
        "the same v3 calibration and grid as Splatter, M4-E at acceptance 0.99, 7 adjacent pairs × 296 configurations."))
    rows = [["State", "Authors' clusters"], ["AR-high PRAD", "11, 15"], ["AR-low PRAD", "0, 2, 4, 5, 8, 12, 16"],
            ["pre-EMT", "6"], ["intermediate EMT-like", "17"], ["NEPC ASCL1+", "10"], ["early NEPC ASCL1−", "13"],
            ["terminal NEPC ASCL1−", "1, 7, 9"], ["NEPC other", "3, 14"]]
    table(s, LEFT, TOP + 0.15, 4.6, rows, [2.5, 2.1], size=13, row_h=0.37)
    rows = [["Time point"] + WEEK, ["OT cells"] + [f"{n:,}" for n in n_ot]]
    table(s, LEFT, TOP + 4.0, 4.6, [[rows[0][0]] + rows[0][1:5], [rows[1][0]] + rows[1][1:5]], [1.4, 0.8, 0.8, 0.8, 0.8], size=12, row_h=0.32)
    table(s, LEFT, TOP + 4.8, 4.6, [[rows[0][0]] + rows[0][5:], [rows[1][0]] + rows[1][5:]], [1.4, 0.8, 0.8, 0.8, 0.8], size=12, row_h=0.32)
    picture(s, P / "pdx_composition.png", 5.3, TOP, RIGHT - 5.3, BOTTOM - TOP)

    s = d.slide("Which cell matches which: 3D slices over time", notes=(
        "Each sheet is one time point on the authors' UMAP (display only), time upwards; colour = state; grey = rejected "
        "in every pair the cell is in.\n"
        "A line joins a retained source cell to the retained target cell that receives its largest coupling mass. 250 "
        "lines per pair, sampled with weight equal to each row's retained mass (500 in the interactive HTML, "
        "pdx_slices3d_<config>.html).\n"
        "A cell's mass is spread over many targets; the line shows only the largest one. The full split is on the state "
        "flow and transition slides.\n"
        "Configuration: Splatter L5 best (ranknm256 midrank scale cos PCA50), chosen from Splatter alone."))
    picture(s, P / "pdx_slices3d_splatter_l5_best.png", LEFT, TOP, 4.9, BOTTOM - TOP)
    picture(s, P / "pdx_slices_splatter_l5_best.png", 5.4, TOP + 0.2, RIGHT - 5.4, 2.8)
    text(s, 5.5, TOP + 3.35, RIGHT - 5.5, 2.3, [
        ("Reading the lines", {"b": True}),
        "wk 0 → 2: AR-high and AR-low PRAD both land on AR-low PRAD.",
        "wk 2 → 12: AR-low PRAD maps to itself; pre-EMT partly to itself, partly to AR-low.",
        "wk 12 → 16 → 20: NEPC ASCL1+ carries the line into the regrowth.",
        "wk 20 → 22: terminal NEPC ASCL1− maps to itself; ASCL1+ goes to terminal."], size=13, fill=PEACH)

    s = d.slide("The same matching under the baseline and logCPM", notes=(
        "Left: production baseline (ranknm256 production noscale cos PCA30). Right: logCPM (logcpm noscale cos PCA30, "
        "the baseline with only the encoding changed). Same lines and sampling as the previous slide."))
    picture(s, P / "pdx_slices3d_baseline.png", 1.2, TOP, 5.0, BOTTOM - TOP - 0.3)
    picture(s, P / "pdx_slices3d_logcpm.png", 7.1, TOP, 5.0, BOTTOM - TOP - 0.3)
    for x, lab in ((1.2, "Production baseline"), (7.1, "logCPM")):
        text(s, x, BOTTOM - 0.32, 5.0, 0.3, [(lab, {"b": True})], size=13)

    s = d.slide("Cell states and their gates over time", notes=(
        "Every time point has two bars of the same cells: left = as target of the pair from the previous time point "
        "(does it have a counterpart before?), right = as source of the pair to the next (does it have one after?). Each "
        "pair gates the cells separately, so only the retained / rejected split differs, never the composition.\n"
        "Segment height = the state's share of the time point's cells; solid = retained, hatched = rejected. Number above "
        "a bar = rejected share of all its cells.\n"
        "Bands: coupling mass on the retained × retained block. At the source end a band is the source state's retained "
        "share × the share of its retained mass going to that target state (row-normalised); at the target end, the "
        "target state's retained share × the share of its received mass coming from that source state.\n"
        "Example wk 16: AR-low PRAD is 38% rejected as target of wk 12 → 16, but 95% as source of wk 16 → 20: wk 20 is "
        "almost all NEPC, so AR-low has nothing to go to."))
    picture(s, P / "pdx_flow_splatter_l5_best.png", LEFT, TOP, W, BOTTOM - TOP)

    rs = rejected_shares(srej)
    s = d.slide("Gate shares under the baseline and logCPM", notes=(
        "The same panel for the production baseline (top) and logCPM (bottom).\n"
        "Table: rejected share of all the time point's cells, as target (T) of the previous pair and as source (S) of the "
        "next, for the three configurations. Bold: differs from the L5 best by 10 points or more."))
    picture(s, P / "pdx_flow_baseline.png", LEFT, TOP, 7.0, 2.85)
    picture(s, P / "pdx_flow_logcpm.png", LEFT, TOP + 2.9, 7.0, 2.85)
    rows = [["Time", "Gate", "L5 best", "Baseline", "logCPM"]]
    bold = set()
    for t in range(8):
        for side, lab in (("target", "T"), ("source", "S")):
            g = rs[(rs.time == t) & (rs["as"] == side)].set_index("cfg").rejected
            if g.empty:
                continue
            for j, c in enumerate(("baseline", "logcpm")):
                if abs(g[c] - g["splatter_l5_best"]) >= 0.10:
                    bold.add((len(rows), 3 + j))
            rows.append([WEEK[t] if (side == "target" or t == 0) else "", lab] +
                        [f"{100 * g[c]:.0f}%" for c in ("splatter_l5_best", "baseline", "logcpm")])
    gf = table(s, 7.75, TOP + 0.05, RIGHT - 7.75, rows, [1.0, 0.8, 1.2, 1.2, 1.1], size=11, row_h=0.36)
    for (i, j) in bold:
        for r in gf.table.cell(i, j).text_frame.paragraphs[0].runs:
            r.font.bold = True; r.font.color.rgb = ORANGE

    s = d.slide("Where each state's retained mass goes", notes=(
        "Row-normalised state transition matrices per pair on the retained × retained block: the share of each source "
        "state's retained coupling mass going to each target state. Shares ≥ 0.15 printed (1 = all); blank = the state "
        "is absent or fully rejected. Top: Splatter L5 best; bottom: production baseline."))
    picture(s, P / "pdx_transitions_splatter_l5_best.png", LEFT, TOP, W, 2.85)
    picture(s, P / "pdx_transitions_baseline.png", LEFT, TOP + 2.9, W, 2.85)

    w16 = srej[srej.pair.str.contains("w16_16wkCX_1__w20") & (srej.side == "source")]
    w4s = srej[srej.pair.str.contains("w04_4wkCX_1__w08") & (srej.side == "source")]
    w8t = srej[srej.pair.str.contains("w04_4wkCX_1__w08") & (srej.side == "target")]

    def ret(df, state, c):
        g = df[(df.config_id == CFG[c]) & (df.state == state)]
        return int((g.cells - g.rejected).iloc[0]), int(g.cells.iloc[0])
    s = d.slide("logCPM: transitions and where it differs", notes=(
        "Top: logCPM transition matrices, read as on the previous slide.\n"
        "Table: retained cells (of all cells of that state) in the gates where logCPM differs most from the other two. "
        "At wk 16 → 20 logCPM rejects every AR-low PRAD cell as source, so its row is blank and only NEPC ASCL1+ carries "
        "mass into wk 20; at wk 4 → 8 it keeps more AR-low PRAD on both sides."))
    picture(s, P / "pdx_transitions_logcpm.png", LEFT, TOP, W, 2.85)
    rows = [["Pair, gate", "State", "L5 best", "Baseline", "logCPM", "Cells"]]
    for lab, df, st_ in (("wk 16 → 20, as source", w16, "AR-low PRAD"), ("wk 16 → 20, as source", w16, "NEPC ASCL1+"),
                         ("wk 4 → 8, as source", w4s, "AR-low PRAD"), ("wk 4 → 8, as target", w8t, "AR-low PRAD")):
        v = [ret(df, st_, c) for c in ("splatter_l5_best", "baseline", "logcpm")]
        rows.append([lab, st_] + [f"{x[0]:,}" for x in v] + [f"{v[0][1]:,}"])
    table(s, LEFT, TOP + 3.1, 8.2, rows, [2.4, 1.9, 1.0, 1.0, 1.0, 0.9], size=12, row_h=0.36)
    text(s, 8.9, TOP + 3.1, RIGHT - 8.9, 2.55, [
        ("Same in all three", {"b": True}), "PRAD holds to wk 12; NEPC ASCL1+ carries wk 16 into the regrowth.",
        ("Depends on preprocessing", {"b": True}), "Whether a few AR-low PRAD cells link wk 16 directly to NEPC."], size=13, fill=PEACH)

    s = d.slide("Rejected fraction per state and time point", notes=(
        "Each cell: the rejected fraction of one state at one time point, as source (pair to the next time point, left) "
        "and as target (pair from the previous time point, right). * fewer than 20 cells; blank: no cell of that state.\n"
        f"Top: Splatter L5 best. Bottom: median over all {n_cfg} preprocessing configurations: the same pattern, e.g. "
        "wk 16 AR-low PRAD as source, median 0.97; wk 0 AR-high PRAD, 0.64."))
    picture(s, P / "pdx_rejected_splatter_l5_best.png", LEFT, TOP, W, 2.85)
    picture(s, P / "pdx_rejected_median296.png", LEFT, TOP + 2.9, W, 2.85)

    s = d.slide("Rejected fraction: baseline and logCPM", notes=(
        "Read as the previous slide. Top: production baseline. Bottom: logCPM (the baseline with only the encoding "
        "changed)."))
    picture(s, P / "pdx_rejected_baseline.png", LEFT, TOP, W, 2.85)
    picture(s, P / "pdx_rejected_logcpm.png", LEFT, TOP + 2.9, W, 2.85)

    s = d.slide(f"How much preprocessing moves the gate", notes=(
        f"Interquartile range of each state's rejected fraction over all {n_cfg} preprocessing configurations (colour "
        "scale 0-0.5). Small = the gate is the same whatever the preprocessing; large = it depends on it. Cells with * "
        "have fewer than 20 cells."))
    picture(s, P / "pdx_rejected_iqr296.png", LEFT, TOP, W, 3.95)
    text(s, LEFT + 0.6, TOP + 4.15, W - 1.2, 1.4, [
        ("Median [IQR] of the rejected fraction over 296 configurations, as source", {"b": True}),
        "wk 16 AR-low PRAD 0.97 [0.94-1.00]: stable.   wk 0 AR-high PRAD 0.64 [0.44-0.80]; wk 12 AR-low PRAD "
        "0.73 [0.53-0.88]; wk 4 AR-low PRAD 0.45 [0.29-0.63]: these depend on preprocessing."], size=13, fill=PEACH)

    s = d.slide("Summary and next steps", notes="")
    text(s, LEFT + 0.2, TOP + 0.2, 6.0, 5.2, [
        ("Splatter N = 5,000", {"b": True, "size": 16}),
        "296 configurations × 60 pairs, 17,760 fits, none failed.",
        "Ranked by L5 Case 2 F1: ranknm256 midrank scale cos PCA50 is first (L5 F1 0.68, FRR 0.15) and also best on the "
        "L1-L5 mean (0.90). The production baseline is 171st (L5 F1 0.48, FRR 0.35).",
        "L5 (batch + perturbation + imbalance) is the only level that separates configurations; its FRR stays ≥ 0.13.",
        ("LTL331 PDX", {"b": True, "size": 16}),
        "Under three preprocessings: PRAD holds to wk 12, almost no wk 16 AR-low cell has a counterpart in the "
        "regrowth (rejected 0.97, median over 296), and NEPC ASCL1+ carries the line into wk 20-22.",
        "The median over 296 configurations shows the same gates."], size=14)
    text(s, 6.9, TOP + 0.2, RIGHT - 6.9, 2.6, [
        ("Next", {"b": True, "size": 16}),
        "PDX: DEG and hallmark GSEA (all 50 sets, both sides), rejected vs retained within each state.",
        "KP CRISPR: the last 21 rows, then the lineage scorers on the same grid.",
        "Copy the couplings off TACC scratch (Splatter about 3 TB) before the purge."], size=14, fill=PEACH)

    d.save(a.out)


if __name__ == "__main__":
    main()
