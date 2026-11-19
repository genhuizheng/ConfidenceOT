"""Build the 2026-11-18 report deck from the rendered figures (out/FigureN/...).

Every figure: one Full Figure Overview slide (FigureN.png), then one slide per panel showing the panel crop
(FigureN_panel_<k>.png, a bbox crop of the same rendering) with the panel notes (compared / observed / meaning) and
the panel's caption paragraph as speaker notes. Branding is copied from the 20261015 deck (logo, orange bar, number).

    python -I build_deck.py --template ConfidenceOT_20261015.pptx --out out/ConfidenceOT_20261118.pptx
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
INK = RGBColor(0x33, 0x33, 0x33); SUB = RGBColor(0x6B, 0x72, 0x80); ACC = RGBColor(0xC0, 0x39, 0x2B)
SRC = RGBColor(0x2C, 0x6E, 0x9B); GRN = RGBColor(0x4E, 0x8A, 0x6B); ORANGE = RGBColor(0xBF, 0x57, 0x00)
FONT = "Calibri"

FIGS = [
    ("Figure1", "Figure 1", "ConfidenceOT: why and how",
     "ConfidenceOT keeps the unbalanced OT backbone and adds one thing: a keep / reject decision for every cell, so cells "
     "without a counterpart are not forced into a match."),
    ("Figure2", "Figure 2", "Splatter: does ConfidenceOT work under technical variation?",
     "With known truth, rejection is reliable under depth, dropout, batch and a shifted population (F1 0.94-0.98); it "
     "weakens at L5, and a depth-sensitive cost or a very large depth gap breaks it."),
    ("Figure3", "Figure 3", "PDX: can ConfidenceOT capture changes in cell states?",
     "On a real PRAD -> NEPC time course without truth, retained matches follow the published order of states; "
     "rejection is high where composition changes, and some links rest on few cells."),
    ("Figure4", "Figure 4", "CRISPR: does RNA-based matching agree with lineage information?",
     "Independent CRISPR lineage supports RNA-only matching at the family and clade level, not at the level of exact "
     "cell-to-cell partners."),
    ("FigureFW", "Future work", "Where can we apply ConfidenceOT next?",
     "Ten collected data sets; six (68 units) are prepared and checked, none is evaluated yet. Next: evaluate the ones "
     "with independent reference information."),
]


def txt(slide, x, y, w, h, runs, size=14, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, wrap=True):
    """runs: list of paragraphs; each paragraph = list of (text, dict(bold, color, size, italic))."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = wrap; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if isinstance(para, tuple) and para and para[0] == "__space__":
            p.space_before = Pt(para[1]); para = para[2]
        for t, o in para:
            r = p.add_run(); r.text = t
            f = r.font; f.name = FONT; f.size = Pt(o.get("size", size)); f.bold = o.get("bold", False)
            f.italic = o.get("italic", False); f.color.rgb = o.get("color", INK)
    return tb


class Deck:
    def __init__(self, template: Path):
        self.prs = Presentation(str(template))
        ref = self.prs.slides[2]
        self.brand_xml, self.logo = [], None
        for sh in ref.shapes:
            if sh.name in ("Rectangle 5", "Rectangle 6"):
                self.brand_xml.append(copy.deepcopy(sh._element))
            if sh.name == "Picture 8":
                self.logo = (sh.image.blob, sh.left, sh.top, sh.width, sh.height)
        # drop every existing slide
        sld = self.prs.slides._sldIdLst
        for s in list(sld):
            self.prs.part.drop_rel(s.rId); sld.remove(s)
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, title, notes=None):
        s = self.prs.slides.add_slide(self.blank)
        for ph in list(s.placeholders):
            ph._element.getparent().remove(ph._element)
        for el in self.brand_xml:
            s.shapes._spTree.append(copy.deepcopy(el))
        blob, l, t, w, h = self.logo
        s.shapes.add_picture(io.BytesIO(blob), l, t, w, h)
        if title:
            txt(s, 4.45, 0.04, 8.75, 0.62, [[(title, {"size": 22 if len(title) <= 62 else 19})]], anchor=MSO_ANCHOR.MIDDLE)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        self.n += 1
        return s

    def picture(self, s, path: Path, box):
        """Fit the image in box (x, y, w, h inches), centred; returns placed (x, y, w, h)."""
        x, y, w, h = box
        with Image.open(path) as im:
            iw, ih = im.size
        r = iw / ih
        pw, ph = (w, w / r) if w / h < r else (h * r, h)
        px, py = x + (w - pw) / 2, y + (h - ph) / 2
        s.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
        return px, py, pw, ph

    def save(self, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(out))


def caption_paragraphs(path: Path) -> dict:
    t = path.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"\*\*([a-z])\*\*,\s*(.*?)(?=\n\n\*\*[a-z]\*\*,|\Z)", t, re.S):
        out[m.group(1)] = " ".join(m.group(2).split())
    return out


def notes_block(n):
    return [
        [("What is compared  ", {"bold": True, "color": SRC, "size": 13}), (n["compared"], {"size": 13})],
        ("__space__", 8, [("What we observe  ", {"bold": True, "color": GRN, "size": 13}), (n["observe"], {"size": 13})]),
        ("__space__", 8, [("What it means  ", {"bold": True, "color": ACC, "size": 13}), (n["meaning"], {"size": 13})]),
    ]


def figure_slides(D: Deck, root: Path, stem, label, question, message):
    d = root / stem
    meta = json.loads((d / f"{stem}_panels.json").read_text())
    notes = json.loads((d / "panel_notes.json").read_text(encoding="utf-8"))
    caps = caption_paragraphs(d / "caption.md")
    cap_title = (d / "caption.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    # overview
    s = D.slide(f"{label} | {question}", notes=cap_title + "\n\n" + "\n\n".join(f"{k}: {v}" for k, v in caps.items()))
    D.picture(s, d / f"{stem}.png", (0.25, 0.72, 8.75, 6.36))
    txt(s, 9.25, 0.95, 3.85, 2.6, [[("Main message", {"bold": True, "color": ORANGE, "size": 15})],
                                   ("__space__", 6, [(message, {"size": 15})])])
    plist = [[("Panels", {"bold": True, "color": SUB, "size": 12})]]
    for k in meta["panels"]:
        plist.append(("__space__", 3, [(f"{k}  ", {"bold": True, "size": 12}), (notes[k]["title"], {"size": 12, "color": SUB})]))
    txt(s, 9.25, 3.85, 3.85, 3.1, plist)
    # panels
    for k, (x0, y0, x1, y1) in meta["panels"].items():
        n = notes[k]
        tag = f"{label} {k}" if label == "Future work" else f"{label}{k}"
        s = D.slide(f"{tag} | {n['title']}", notes=f"{tag}. {caps.get(k, '')}")
        img = d / f"{stem}_panel_{k}.png"
        r = (x1 - x0) / (y1 - y0)
        if r >= 1.75:  # wide panel: image on top, three text columns below
            px, py, pw, ph = D.picture(s, img, (0.35, 0.78, 12.65, 4.55))
            ty = py + ph + 0.18
            cols = notes_block(n)
            w = 12.4 / 3
            for i, para in enumerate(cols):
                txt(s, 0.45 + i * w, ty, w - 0.2, 7.0 - ty, [para])
        else:
            px, py, pw, ph = D.picture(s, img, (0.3, 0.78, 8.6, 6.25))
            txt(s, 9.15, 1.0, 3.95, 5.8, notes_block(n))


def intro_slides(D: Deck):
    s = D.slide(None)
    txt(s, 0.9, 2.0, 11.6, 1.6, [[("ConfidenceOT", {"size": 40, "bold": True, "color": ORANGE})],
                                 ("__space__", 6, [("Matching cells between two samples, and saying when a cell has no match", {"size": 24})])])
    txt(s, 0.9, 4.1, 11.6, 1.4, [[("Method  >  controlled validation  >  real cell-state change  >  independent lineage evaluation  >  next applications",
                                    {"size": 16, "color": SUB})],
                                 ("__space__", 14, [("Research report, 2026-11-18", {"size": 16, "color": SUB})])])
    s = D.slide("Why do we need rejection?")
    txt(s, 0.6, 0.85, 12.1, 1.2, [[("Two samples rarely contain the same cell populations. A method that must match every cell "
                                    "creates false correspondences for cells that have no counterpart. ConfidenceOT matches cells "
                                    "and also decides, cell by cell, whether a match should be claimed.", {"size": 17})]])
    steps = [("Figure 1", "Method", "What does ConfidenceOT add to standard OT?", "schematic + real example"),
             ("Figure 2", "Controlled validation", "How well does it work, and when does it fail?", "simulation, known truth"),
             ("Figure 3", "Real cell-state change", "Can OT capture changes in cell states?", "PDX, no truth: consistency with the published biology"),
             ("Figure 4", "Independent lineage", "Does RNA matching agree with lineage information?", "CRISPR edits, not seen by the RNA-only OT"),
             ("Future work", "Applications", "Where can we apply ConfidenceOT next?", "collected data sets; not yet evaluated")]
    w, gap, x0, y0 = 2.25, 0.27, 0.55, 2.45
    for i, (f, h, q, ev) in enumerate(steps):
        x = x0 + i * (w + gap)
        b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y0), Inches(w), Inches(1.75))
        b.fill.solid(); b.fill.fore_color.rgb = RGBColor(0xF3, 0xF4, 0xF5); b.line.color.rgb = RGBColor(0xB0, 0xB5, 0xB9)
        b.adjustments[0] = 0.08
        txt(s, x + 0.08, y0 + 0.12, w - 0.16, 2.35, [[(f, {"bold": True, "color": ORANGE, "size": 15})],
                                                     ("__space__", 2, [(h, {"bold": True, "size": 15})]),
                                                     ("__space__", 10, [(q, {"size": 13})])])
        txt(s, x + 0.05, y0 + 1.88, w - 0.1, 0.9, [[(ev, {"size": 12, "color": SUB, "italic": True})]], align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + w + 0.04), Inches(y0 + 0.74), Inches(0.19), Inches(0.28))
            a.fill.solid(); a.fill.fore_color.rgb = RGBColor(0x6B, 0x72, 0x80); a.line.fill.background()
    txt(s, 0.6, 6.2, 12.1, 0.7, [[("The levels of evidence differ and are not merged: known truth (Figure 2), consistency with a "
                                   "paper (Figure 3), independent lineage (Figure 4), prepared but unevaluated data (Future work).",
                                   {"size": 13, "color": SUB})]])


def closing_slides(D: Deck):
    s = D.slide("Next steps")
    items = [
        ("Evaluate ConfidenceOT on existing biological applications with independent reference information where available.", True),
        ("Submit OT for the 68 prepared units (same frozen pipeline: main configuration, calibration v3, M4-E), with your go.", False),
        ("Score them with the definitions fixed in PLAN_TACC: LARRY in vivo clones, sciPlex2 dose, Kang control; then map each "
         "result to the original paper, in both directions.", False),
        ("Finish SPLINTR and zebrafish preparation after the three open decisions (clone rule, cell set, unit list).", False),
        ("Not planned: new algorithm development or a new preprocessing grid search.", False),
    ]
    paras = []
    for i, (t, hl) in enumerate(items):
        para = [("-  ", {"size": 18, "color": ORANGE, "bold": True}), (t, {"size": 18, "bold": hl, "color": ACC if hl else INK})]
        paras.append(para if i == 0 else ("__space__", 14, para))
    txt(s, 0.8, 1.2, 11.8, 5.5, paras)
    s = D.slide("Summary")
    rows = [("Figure 1", "ConfidenceOT = UOT matching + a per-cell keep / reject decision, calibrated without labels."),
            ("Figure 2", "Known truth: F1 0.94-0.98 at L1-L4, 0.68 at L5; failures: a depth-sensitive cost lets depth win over biology, "
                         "and a ~20x global depth gap makes ConfidenceOT reject shared cells."),
            ("Figure 3", "PDX: retained matches follow the published state order (intermediate -> intermediate, ASCL1+ -> ASCL1-); "
                         "consistency, not a reconstructed trajectory."),
            ("Figure 4", "CRISPR: RNA matching agrees with lineage at family (d = 0.23) and clade level, not at exact cell partners (0.04)."),
            ("Future work", "68 units prepared on real data with independent references where available; evaluation is the next step.")]
    paras = []
    for i, (h, t) in enumerate(rows):
        para = [(f"{h}   ", {"bold": True, "color": ORANGE, "size": 17}), (t, {"size": 17})]
        paras.append(para if i == 0 else ("__space__", 16, para))
    txt(s, 0.8, 1.1, 11.8, 5.8, paras)


def supplementary_slides(D: Deck, outroot: Path):
    import shutil
    from supplementary import SUPP, LEGEND_SOLVERS
    sdir = outroot / "Supplementary"; sdir.mkdir(parents=True, exist_ok=True)
    s = D.slide("Supplementary figures")
    paras = [[("Earlier figures of the current result sets, shown as they were made; they are not part of Figures 1-4.",
               {"size": 16})]]
    groups = {}
    for i, (grp, f, t, c, n) in enumerate(SUPP, 1):
        groups.setdefault(grp, []).append(f"S{i} {t}")
    for grp, items in groups.items():
        paras.append(("__space__", 12, [(f"{grp}  ", {"bold": True, "color": ORANGE, "size": 14}),
                                        ("; ".join(items), {"size": 12, "color": SUB})]))
    txt(s, 0.8, 1.0, 11.8, 5.9, paras)
    index = ["# Supplementary figures (copies of the original files)\n", "| S | Supports | Title | Original file |", "|---|---|---|---|"]
    for i, (grp, f, t, c, n) in enumerate(SUPP, 1):
        assert f.exists(), f
        dst = sdir / f"S{i:02d}_{f.name}"; shutil.copy2(f, dst)
        index.append(f"| S{i} | {grp} | {t} | `{f.relative_to(HERE.parent).as_posix()}` |")
        s = D.slide(f"Supplementary S{i} | {t}", notes=f"S{i} (supports {grp}). {c} {n or ''} Source: {f.relative_to(HERE.parent).as_posix()}")
        box = (0.3, 0.75, 12.7, 5.35) if n != "legend" else (0.3, 0.75, 10.2, 5.35)
        D.picture(s, f, box)
        if n == "legend":
            D.picture(s, LEGEND_SOLVERS, (10.6, 2.2, 2.5, 2.3))
        paras = [[(f"Supports {grp}.  ", {"bold": True, "color": ORANGE, "size": 12}), (c, {"size": 12})]]
        if n and n != "legend":
            paras.append(("__space__", 3, [(n, {"size": 12, "color": ACC, "bold": True})]))
        paras.append(("__space__", 3, [(f"Source: {f.relative_to(HERE.parent).as_posix()}", {"size": 10, "color": SUB})]))
        txt(s, 0.45, 6.15, 12.4, 0.95, paras)
    (sdir / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", type=Path, default=HERE / "out")
    p.add_argument("--fig1-root", type=Path, default=None, help="layout checks only")
    a = p.parse_args()
    D = Deck(a.template)
    intro_slides(D)
    for stem, label, q, msg in FIGS:
        root = a.fig1_root if (stem == "Figure1" and a.fig1_root) else a.root
        if not (root / stem / "panel_notes.json").exists():
            print("MISSING", stem, "- skipped (layout check only)")
            continue
        figure_slides(D, root, stem, label, q, msg)
    closing_slides(D)
    supplementary_slides(D, a.root)
    D.save(a.out)
    print("slides", D.n, "->", a.out)


if __name__ == "__main__":
    main()
