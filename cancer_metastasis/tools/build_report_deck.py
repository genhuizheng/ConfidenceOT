"""Assemble the PI report deck from the generated figures.

Sixteen slides in the order the work happened: the biological question, the
assumption the method rests on, the earlier three-dataset result and why it did
not hold, the two confounders behind it, the controls, and what the prostate
dataset added.

The question comes before any of the method's vocabulary, and the assumption is
stated as an assumption rather than left implicit -- that transcriptional
similarity to the metastasis marks the cells that could have seeded it is the
premise of the whole approach, so the later finding is that the premise is not
supported, not that a number came out wrong. A short glossary sits beside the
conclusions, because "effect", "p value", "artefact" and "confounder" all carry
the argument and none of them is self-explanatory to the audience.

The earlier results are carried as tables rather than summarised in a sentence.
A reader who is told "the direction came out backwards" has to take it on
trust; a reader who sees head and neck at -0.0099 with p = 0.375 beside ovarian
at +0.0063 with p = 1.9e-8 can see both that nothing replicated and that the one
significant cell is too small to be biology.

The layout follows the lab's own progress-report template, measured off
``Progress_report7.pptx`` rather than guessed: white page, the Oden Institute
logo at the top left, a running header, and the burnt-orange footer band with
the slide number in a lighter block at the right. Colours were sampled from a
render of that deck (#BF5700 band, #EB8F2F number block) and the band geometry
measured from the same image.

Each slide carries a text block in that deck's register -- a bulleted section
label, then numbered headings with an explanation line under each, English for
the technical terms and Chinese for the connective prose. The figures were
readable on their own but did not say what had been done or what followed from
it, which is what this rewrite adds.

Every number was read back out of the collected summaries rather than carried
over from a draft. Where a slide quotes a statistic the arm it came from is
named, because the retained fraction differs tenfold between the free and capped
rules and a number without its arm is not checkable.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

# Sampled from a render of Progress_report7.pptx.
ORANGE = RGBColor(0xBF, 0x57, 0x00)
ORANGE_LIGHT = RGBColor(0xEB, 0x8F, 0x2F)
SLATE = RGBColor(0x1F, 0x29, 0x33)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x14, 0x18, 0x1C)
BODY = RGBColor(0x33, 0x38, 0x3D)
MUTED = RGBColor(0x7A, 0x7A, 0x75)

LATIN = "Calibri"
EAST = "Microsoft YaHei"

WIDE, TALL = 13.333, 7.5
MARGIN = 0.5
BAND_TOP, BAND_HEIGHT, BAND_SPLIT = 7.162, 0.338, 11.94
HEADER = "Progress Report"
CONTENT_TOP = 0.78
CONTENT_BOTTOM = BAND_TOP - 0.12


def set_fonts(run) -> None:
    """Name the Latin and East Asian typefaces on one run.

    ``font.name`` only writes the Latin typeface, so mixed Chinese and English
    text would leave the Chinese characters on whatever the theme happens to
    supply. Both attributes are set so a run renders the same everywhere.
    """
    run.font.name = LATIN
    properties = run.font._rPr
    for tag, typeface in ((qn("a:ea"), EAST), (qn("a:cs"), EAST)):
        element = properties.find(tag)
        if element is None:
            element = properties.makeelement(tag, {})
            properties.append(element)
        element.set("typeface", typeface)


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


def write(frame, runs, *, space_after=0, line_spacing=None, first=False,
          indent=0.0, align=None):
    """Append a paragraph built from (text, size, bold, colour) runs."""
    paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
    paragraph.space_after = Pt(space_after)
    if line_spacing:
        paragraph.line_spacing = line_spacing
    if indent:
        # python-pptx exposes no paragraph_format on _Paragraph, so the left
        # margin is written straight onto the paragraph properties.
        paragraph._p.get_or_add_pPr().set("marL", str(int(Inches(indent))))
    if align is not None:
        paragraph.alignment = align
    for text, size, bold, colour in runs:
        run = paragraph.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = colour
        set_fonts(run)
    return paragraph


def strip_cell_borders(cell) -> None:
    """Remove the table style's per-cell borders.

    The reference deck's tables are ruled horizontally and nowhere else. A
    python-pptx table inherits a style that draws a box around every cell, and
    the only way to clear it is to write explicit no-fill lines onto the cell
    properties.
    """
    properties = cell._tc.get_or_add_tcPr()
    for edge in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        tag = qn(edge)
        for existing in properties.findall(tag):
            properties.remove(existing)
        line = properties.makeelement(tag, {"w": "0", "cap": "flat",
                                            "cmpd": "sng", "algn": "ctr"})
        line.append(line.makeelement(qn("a:noFill"), {}))
        properties.append(line)


def rectangle(slide, left, top, width, height, colour):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left),
                                   Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = colour
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


class Deck:
    """The template: page furniture, then one method per slide kind."""

    def __init__(self, logo: Path | None):
        self.presentation = Presentation()
        self.presentation.slide_width = Inches(WIDE)
        self.presentation.slide_height = Inches(TALL)
        self.logo = logo if logo and logo.exists() else None
        # The title slide carries no furniture but still counts as page one, so
        # the first content page is numbered 2 as in the reference deck.
        self.number = 1

    # -- page furniture ---------------------------------------------------
    def blank(self, dark: bool = False):
        slide = self.presentation.slides.add_slide(
            self.presentation.slide_layouts[6])
        if dark:
            fill = slide.background.fill
            fill.solid()
            fill.fore_color.rgb = SLATE
        return slide

    def page(self, header: str | None = None):
        """A content page: logo, running header, footer band, slide number."""
        slide = self.blank()
        self.number += 1
        if self.logo:
            slide.shapes.add_picture(str(self.logo), Inches(0.1), Inches(0.1),
                                     width=Inches(3.29), height=Inches(0.57))
        frame = text_box(slide, 5.2, 0.06, 8.2, 0.5, align=PP_ALIGN.CENTER)
        write(frame, [(header or HEADER, 22, False, INK)], first=True,
              align=PP_ALIGN.CENTER)
        rectangle(slide, 0, BAND_TOP, BAND_SPLIT, BAND_HEIGHT, ORANGE)
        rectangle(slide, BAND_SPLIT, BAND_TOP, WIDE - BAND_SPLIT, BAND_HEIGHT,
                  ORANGE_LIGHT)
        number = text_box(slide, BAND_SPLIT, BAND_TOP + 0.04,
                          WIDE - BAND_SPLIT - 0.18, BAND_HEIGHT,
                          align=PP_ALIGN.RIGHT)
        write(number, [(str(self.number), 13, False, WHITE)], first=True,
              align=PP_ALIGN.RIGHT)
        return slide

    # -- content ----------------------------------------------------------
    def label(self, slide, text, top, left=MARGIN, size=16):
        """The bulleted section label the template puts above each block."""
        frame = text_box(slide, left, top, WIDE - left - MARGIN, 0.34)
        write(frame, [("•  ", size, True, ORANGE), (text, size, True, INK)],
              first=True)
        return top + 0.4

    def block(self, slide, left, top, width, items, *, size=14.5,
              gap=9, lead=1.22):
        """Numbered headings, each with its explanation on the lines below."""
        frame = text_box(slide, left, top, width, CONTENT_BOTTOM - top)
        first = True
        for index, item in enumerate(items, 1):
            if isinstance(item, str):
                # A closing line, marked with an arrow instead of a number.
                write(frame, [("→  " + item, size, True, ORANGE)],
                      first=first, space_after=gap, line_spacing=lead)
                first = False
                continue
            heading, detail = item
            write(frame, [(f"{index}. ", size, True, INK),
                          (heading, size, True, INK)],
                  first=first, space_after=2, line_spacing=lead)
            first = False
            write(frame, [(detail, size, False, BODY)], space_after=gap,
                  line_spacing=lead, indent=0.2)
        return frame

    def table(self, slide, left, top, width, rows, *, widths=None, size=13,
              row_height=0.34, highlight=None):
        """A horizontally ruled table, as the reference deck draws them.

        `rows[0]` is the header. `highlight` is a set of row indices whose
        numbers carry the accent colour, for the one row a slide is arguing
        about.
        """
        columns = len(rows[0])
        shape = slide.shapes.add_table(len(rows), columns, Inches(left),
                                       Inches(top), Inches(width),
                                       Inches(row_height * len(rows)))
        table = shape.table
        table.first_row = False
        table.horz_banding = False
        if widths:
            total = sum(widths)
            for index, share in enumerate(widths):
                table.columns[index].width = Inches(width * share / total)
        for r, row in enumerate(rows):
            table.rows[r].height = Inches(row_height)
            for c, value in enumerate(row):
                cell = table.cell(r, c)
                cell.fill.background()
                strip_cell_borders(cell)
                cell.margin_left = Inches(0.05)
                cell.margin_right = Inches(0.05)
                cell.margin_top = cell.margin_bottom = Inches(0.02)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                frame = cell.text_frame
                frame.word_wrap = True
                header = r == 0
                colour = INK if header else BODY
                if highlight and r in highlight and c > 0:
                    colour = ORANGE
                write(frame, [(str(value), size, header or bool(
                    highlight and r in highlight and c > 0), colour)],
                    first=True,
                    align=PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT)
        # Rules above and below the header, and under the last row.
        for y in (top, top + row_height, top + row_height * len(rows)):
            rectangle(slide, left, y, width, 0.012, INK if y == top else MUTED)
        return top + row_height * len(rows) + 0.16

    def figure(self, slide, image: Path, left, top, width, height,
               anchor="middle"):
        """Fit an image in a box, preserving aspect. `anchor` sets the vertical
        placement: a wide figure in a tall box otherwise floats in the middle
        of it, which reads as a gap under the text rather than as a layout."""
        with Image.open(image) as handle:
            ratio = handle.width / handle.height
        if width / height > ratio:
            drawn_height, drawn_width = height, height * ratio
        else:
            drawn_width, drawn_height = width, width / ratio
        offset = 0.0 if anchor == "top" else (height - drawn_height) / 2
        slide.shapes.add_picture(
            str(image), Inches(left + (width - drawn_width) / 2),
            Inches(top + offset),
            width=Inches(drawn_width), height=Inches(drawn_height))

    def split(self, label, items, image: Path, *, text_width=4.5):
        """Explanation on the left, figure on the right."""
        slide = self.page()
        top = self.label(slide, label, CONTENT_TOP)
        self.block(slide, MARGIN, top, text_width, items)
        figure_left = MARGIN + text_width + 0.3
        self.figure(slide, image, figure_left, top - 0.05,
                    WIDE - MARGIN - figure_left, CONTENT_BOTTOM - top + 0.05,
                    anchor="top")
        return slide

    def stacked(self, label, items, image: Path, *, text_height=1.85):
        """Explanation across the top, wide figure underneath."""
        slide = self.page()
        top = self.label(slide, label, CONTENT_TOP)
        self.block(slide, MARGIN, top, WIDE - 2 * MARGIN, items, gap=4)
        image_top = top + text_height
        self.figure(slide, image, MARGIN, image_top, WIDE - 2 * MARGIN,
                    CONTENT_BOTTOM - image_top)
        return slide

    def save(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        self.presentation.save(str(output))
        print(f"wrote {output}  ({len(self.presentation.slides._sldIdLst)} slides)")


def build(figures: Path, output: Path, reference: bool, logo: Path | None,
          month: str) -> None:
    deck = Deck(logo)

    # 1 -- title
    slide = deck.blank(dark=True)
    if logo and logo.exists():
        white = logo.with_name("logo_white.png")
        if white.exists():
            slide.shapes.add_picture(str(white), Inches(0.5), Inches(0.4),
                                     width=Inches(3.29), height=Inches(0.57))
    frame = text_box(slide, 0.62, 2.85, 11.0, 1.1)
    write(frame, [("Progress Report", 36, True, WHITE)], first=True)
    line = rectangle(slide, 0.62, 4.05, 8.6, 0.03, ORANGE)
    line.line.fill.background()
    frame = text_box(slide, 0.62, 1.75, 6.0, 0.4)
    write(frame, [(month, 14, False, ORANGE_LIGHT)], first=True)
    frame = text_box(slide, 0.62, 4.3, 11.0, 0.9)
    write(frame, [("ConfidenceOT — which primary tumour cells can "
                   "metastasise?", 18, False, RGBColor(0xC3, 0xC2, 0xB7))],
          first=True, line_spacing=1.2)
    frame = text_box(slide, 0.62, 6.5, 9.0, 0.4)
    write(frame, [("Genhui Zheng, PhD candidate, The University of Texas at "
                   "Austin", 14, False, MUTED)], first=True)

    # 2 -- the biological question, before any method vocabulary
    slide = deck.page()
    top = deck.label(slide, "我们想回答的生物学问题", CONTENT_TOP)
    deck.block(slide, MARGIN, top, WIDE - 2 * MARGIN, [
        ("转移是从原发瘤的哪些细胞出去的",
         "同一个原发瘤里的肿瘤细胞并不一样。能离开原发部位、进入循环、在远处"
         "存活下来并长成转移灶的，只是其中极小一部分。问题是：在原发瘤里能不能"
         "把这一部分细胞指出来。"),
        ("指出来之后能做什么",
         "可以问它们有什么特征 —— 哪些通路、哪些基因在这批细胞里高；能不能作为"
         "“这个病人会不会转移”的早期判断依据；以及能不能针对这批细胞用药。"),
        ("为什么需要 primary–metastasis 配对数据",
         "同一个病人同时有原发瘤和转移灶的单细胞数据，才能把“转移灶长什么样”"
         "当作已知条件，反过来回到原发瘤里找来源。单看原发瘤是无从判断的。"),
    ], size=14.5, gap=13)

    after = deck.label(slide, "我们的假设，以及方法怎么把它变成可计算的", 4.5)
    deck.block(slide, MARGIN, after, WIDE - 2 * MARGIN, [
        ("假设（整个方法的前提）",
         "转移灶的细胞是从原发瘤里出去的后代，所以在转录上应该保留与来源细胞的"
         "相似性。反过来说：原发瘤里那些转录上更像 metastasis 的细胞，可能就是"
         "有转移潜力的那批。"),
        ("方法",
         "Optimal transport 把 primary 的细胞与 metastasis 的细胞做匹配，代价"
         "就是转录距离。代价低于阈值的 primary 细胞叫 retained，高于阈值的叫 "
         "rejected。对比始终在 primary 内部 —— retained vs rejected，两组都是"
         "原发瘤的细胞。"),
    ], size=14.5, gap=13)

    # 3 -- what this report concludes, with the words defined
    slide = deck.page()
    top = deck.label(slide, "本次汇报的核心", CONTENT_TOP)
    deck.block(slide, MARGIN, top, 7.5, [
        ("上次的结论不成立",
         "三个数据集里没有一个支持这个方向：卵巢的差值方向对但只有 0.0063，已"
         "确认是测序深度造成的；头颈方向相反（−0.0099）；结肠两组都是 0。"),
        ("找到了两个混杂",
         "Sequencing depth —— 已解决；cell division（细胞分裂）—— 新发现，且在"
         "卵巢和前列腺两种癌里一致。"),
        ("假设没有被数据支持",
         "retained 的比例基本就是两侧组织整体相似度的读数（rho = 0.76，188 "
         "pairs），所以它不能当“有多少细胞能转移”来读。"),
        ("前列腺的结果",
         "在两侧真有差异的数据集上方法只留下 1.6%，而卵巢留 29%。两侧越不同，"
         "它留得越少 —— 这是反的。后面基因层面的分析用 cap 操作（最多拒绝 85% "
         "细胞，即至少保留 15%），否则细胞太少没法做统计。"),
    ], size=14, gap=10)

    frame = text_box(slide, 8.4, top + 0.02, 4.45, 0.4)
    write(frame, [("名词说明", 15, True, ORANGE)], first=True)
    terms = [
        ("effect（效应量）",
         "两组之间某个指标的差值大小。这里是 retained 组与 rejected 组的 "
         "metastasis-signature 中位分数之差。"),
        ("p value",
         "在“两组其实没有差别”的前提下，观察到当前这个差值或更大的概率。越小 = "
         "越不像偶然。"),
        ("artefact（技术假象）",
         "不是生物学造成的差异，而是实验或测序流程造成的。这里是测序深度。"),
        ("混杂（confounder）",
         "同时影响分组和结果的第三个变量，会让人把技术差异误读成生物学差异。"),
    ]
    frame = text_box(slide, 8.4, top + 0.48, 4.45, 3.3)
    for index, (term, meaning) in enumerate(terms):
        write(frame, [(term, 13, True, INK)], first=index == 0, space_after=1,
              line_spacing=1.18)
        write(frame, [(meaning, 12.5, False, BODY)], space_after=7,
              line_spacing=1.18, indent=0.14)
    frame = text_box(slide, 8.4, top + 3.95, 4.45, 1.6)
    write(frame, [("为什么“极小 effect 加极显著 p”正好是 artefact 的特征",
                   13, True, ORANGE)], first=True, space_after=5,
          line_spacing=1.18)
    write(frame, [("生物学差异通常幅度大，但病人之间有波动。技术假象幅度小，"
                   "却在每个病人里方向完全一致，所以配对检验的 p 会非常小。"
                   "0.0063 配上 1.9 × 10⁻⁸，是后者的样子。", 12.5, False,
                   BODY)], line_spacing=1.18)

    # 3 -- the question and the data, on one patient
    slide = deck.page()
    top = deck.label(slide, "问题定义与数据："
                            "先看一个病人", CONTENT_TOP)
    deck.block(slide, MARGIN, top, WIDE - 2 * MARGIN, [
        ("数据",
         "GSE180661 HGSOC，29 patients，187,383 malignant cells"
         "（depth-equalised），94 个 primary–metastasis "
         "pairs。下图是 SPECTRUM-OV-083：4,726 个 "
         "primary、4,307 个 metastasis，留下 54%"
         "（挑它是因为两组都看得"
         "清）。"),
        ("三张图怎么读",
         "左：这个病人的 primary（灰"
         "）与 metastasis（蓝）混在一起"
         "。中：方法在这个病人"
         "内部留下了哪些 primary 细胞 "
         "—— 按位置分布，不是"
         "整块。右：同一批细胞的 "
         "cell division score，高分区域与留"
         "下的区域重合。"),
    ], size=14, gap=8)
    deck.figure(slide, figures / "fig_umap_ovarian_patient.png",
                MARGIN, 3.02, WIDE - 2 * MARGIN, CONTENT_BOTTOM - 3.02,
                anchor="top")

    # 4 -- is the split a batch effect, and how much of it is just the patient
    slide = deck.page()
    top = deck.label(slide, "这个划分是不是"
                            "批次效应？", CONTENT_TOP)
    deck.block(slide, MARGIN, top, 7.9, [
        ("为什么要问",
         "16 个病人的 primary 取自两个"
         "解剖位点，即两次独立"
         "取材、两个 library。如果 "
         "retain / reject 是批次造成的，"
         "划分就会跟着位点走。"),
        ("结果：不是",
         "位点只解释病人内部方"
         "差的 5.0%（中位 4.5%，16 人中"
         "无一超过 50%；最大的 OV-050 "
         "是 30%，OV-083 是 5.6%）。全部"
         "细胞同属一个研究 GSE180661。"),
        ("但确实有一半是“挑病"
         "人”",
         "右图：29 个病人合在一起"
         "时是整块橙 / 整块灰，"
         "因为这个 embedding 的 cluster 基本"
         "按病人分。所以后面所"
         "有统计都先在病人内部"
         "做差。"),
    ], size=14, gap=10)
    after = deck.table(slide, MARGIN, 4.62, 7.9, [
        ["retained 标签的方差分三层",
         "占比"],
        ["病人之间", "45.6%"],
        ["同一病人不同 primary 位点",
         "5.0%"],
        ["同一样本内、细胞与细"
         "胞之间", "49.4%"],
    ], widths=[5.6, 2.3], size=13.5, highlight={2})
    frame = text_box(slide, MARGIN, after + 0.06, 7.9, 0.4)
    write(frame, [("84,465 个 primary 细胞。批次"
                   "和位点在这里是绑在"
                   "一起的，所以 5.0% 是两"
                   "者合计的上界。", 12, False,
                   MUTED)], first=True, line_spacing=1.16)

    frame = text_box(slide, 8.65, top - 0.02, 4.2, 0.4)
    write(frame, [("29 个病人合在一起", 13.5,
                   True, ORANGE)], first=True)
    deck.figure(slide, figures / "fig_gate_ovarian_all.png",
                8.65, 1.42, 4.2, 4.3, anchor="top")

    # 4 -- the negative result, as the three datasets actually reported it
    slide = deck.page()
    top = deck.label(slide, "老版本的结论：三个数据集，没有一个支持这个方向", CONTENT_TOP)
    after = deck.table(slide, MARGIN, top, 12.3, [
        ["数据集", "病人", "细胞（primary / met）", "retained",
         "rejected", "差值", "Wilcoxon p"],
        ["GSE180661  卵巢", "29", "84k / 103k", "0.0664", "0.0600",
         "+0.0063", "1.9 × 10⁻⁸"],
        ["GSE181919  头颈", "4", "526 / 284", "0.0235", "0.0251",
         "−0.0099", "0.375"],
        ["GSE225857  结肠", "5", "9,585 / 13,987", "0.0000", "0.0000",
         "0.0000", "1.000"],
    ], widths=[2.4, 0.8, 2.2, 1.2, 1.2, 1.1, 1.3], highlight={1})
    deck.block(slide, MARGIN, after + 0.1, 12.3, [
        ("唯一“显著”的那一个是 artefact",
         "卵巢差值只有 0.0063 而 p = 1.9 × 10⁻⁸。极小 effect 加极显著 p 正是 "
         "systematic artefact 的特征 — 后来确认是 depth：retained 与 rejected 的"
         "测序深度差 1.32×，UCell 按细胞内排名打分，深度低的细胞检出基因少，"
         "signature 基因够不到 maxRank，分数就偏低。"),
        ("另外两个数据集没有复现",
         "头颈方向相反（−0.0099，p = 0.375），结肠两组都是 0.0000。结肠那个 0 "
         "不是生物学阴性：rejected 细胞中位只检出 1,689 个基因，signature 基因"
         "大多未被检出，分数是被构造成 0 的。"),
        "三个数据集里 top DEG 都是 keratins 和 SPRR family，与各自的生物学无关 — "
        "同一个答案反复出现，就是 technical confounder 的样子。",
    ], size=13.5, gap=7)

    after = deck.label(slide, "老版本的 retained 比例，以及它为什么不能当概率读",
                       5.16)
    deck.table(slide, MARGIN, after, 7.4, [
        ["数据集", "exact pairs", "rejected", "retained"],
        ["GSE180661  卵巢", "94", "0.814", "0.186"],
        ["GSE181919  头颈", "4", "0.844", "0.156"],
        ["GSE225857  结肠", "5", "0.652", "0.348"],
    ], widths=[2.6, 1.6, 1.6, 1.6], size=13, row_height=0.32)
    frame = text_box(slide, 8.3, after - 0.02, 4.5, 1.8)
    write(frame, [("这些比例由 cap 决定，不是测出来的概率。", 13.5, True, INK)],
          first=True, space_after=6, line_spacing=1.2)
    write(frame, [("source cap 设在 0.85，三个数据集的 rejected 都顶到 0.65–0.84 "
                   "附近 — 任何 cap 都会得到等于该 cap 的 rejection rate。这就是"
                   "后来必须先重算 c、再把 cap 去掉的原因。", 13, False, BODY)],
          line_spacing=1.2)

    # 5 -- how large the depth effect was in the old version
    slide = deck.page()
    top = deck.label(slide, "老版本里 depth 有多严重", CONTENT_TOP)
    after = deck.table(slide, MARGIN, top, 11.4, [
        ["", "GSE180661 卵巢", "GSE181919 头颈", "GSE225857 结肠"],
        ["rho（decision cost, depth）", "−0.330", "−0.593", "−0.568"],
        ["AUC（depth → retained）", "0.673", "0.767", "0.791"],
        ["retained 中位深度", "13,773", "37,447", "13,002"],
        ["rejected 中位深度", "10,433", "18,982", "4,243"],
        ["深度比值", "1.32×", "1.97×", "3.07×"],
    ], widths=[3.4, 2.7, 2.7, 2.7], size=13.5, highlight={2})
    deck.block(slide, MARGIN, after + 0.1, 12.3, [
        ("方向在三个数据集里完全一致",
         "retained 永远是测序更深的那一批。AUC 0.673 / 0.767 / 0.791 — 0.5 才是"
         "无关，所以 depth 一个变量就能把 gate 预测到七到八成。"),
        ("不是 budget、也不是 solver 的问题",
         "两个早期假设都测过并推翻了：rejection budget 最多只贡献 retained 集的"
         "3.8%，在结肠是 0；unbalanced transport 的 mass 项不改变 gate 的排序。"
         "gate 本身是 decision cost 上的干净阈值（与规则一致 99.7%）。"),
        "把 depth 成分从 cost 里去掉，retained 集会变掉 25–36% — 所以它不是边缘"
        "效应，而是决定了谁被留下。",
    ], size=13.5, gap=7)

    # 6 -- the mechanism, as a picture. The prose version of this did not land.
    slide = deck.page()
    top = deck.label(slide, "depth 是什么，以及它为什么会变成一个 gate 决定",
                     CONTENT_TOP)
    deck.figure(slide, figures / "fig_depth_cartoon.png", MARGIN, top,
                WIDE - 2 * MARGIN, CONTENT_BOTTOM - top - 0.5, anchor="top")
    frame = text_box(slide, MARGIN, CONTENT_BOTTOM - 0.44,
                     WIDE - 2 * MARGIN, 0.44)
    write(frame, [("→  ", 14.5, True, ORANGE),
                  ("A 和 B 的生物学完全相同，只是测序深度不同。gate 把 A 判为 "
                   "retained、B 判为 rejected —— 依据是深度，不是生物学。"
                   "这就是“极小 effect 加极显著 p”的来源。",
                   14.5, True, ORANGE)], first=True)

    # 7 -- what each preprocessing does, before the results that used them
    slide = deck.page()
    top = deck.label(slide, "我们试过的几种做法分别在做什么", CONTENT_TOP)
    deck.figure(slide, figures / "fig_methods_cartoon.png", MARGIN, top,
                WIDE - 2 * MARGIN, CONTENT_BOTTOM - top - 0.5, anchor="top")
    frame = text_box(slide, MARGIN, CONTENT_BOTTOM - 0.44,
                     WIDE - 2 * MARGIN, 0.44)
    write(frame, [("→  ", 14.5, True, ORANGE),
                  ("downsample 和 gene rank 都有效，但必须一起用；log-CPM 和 "
                   "Pearson residuals 都修不掉 dropout。下一页是这四种做法在"
                   "真实数据和模拟数据上的结果。", 14.5, True, ORANGE)],
          first=True)

    # 8 -- cause one
    deck.split(
        "原因一：gate 在读 sequencing depth",
        [("怎么量的",
          "用 depth 预测 retain / reject 的 AUC。"
          "0.5 = 完全无关，图上画的是"
          "偏离 0.5 的幅度，越短越好。"),
         ("结果",
          "卵巢 0.09，前列腺 0.19；前列腺完全不做 depth 处理时是 0.44，"
          "即 gate 几乎完全由 depth 决定。"),
         ("和上一页的数字怎么接上",
          "上页卵巢 AUC 0.673 是原始数据，偏离 0.5 有 0.17。这里的 0.09 已经做过 "
          "read equalisation，加 gene rank 之后再降到 0.03 — 同一个量的三个阶段。"),
         ("Read equalisation 有用但不够",
          "前列腺 0.44 → 0.19。depth 还会改变哪些基因能被检出，不只是 reads "
          "数量，所以没能收尾。")],
        figures / "fig1_depth_in_gate.png")

    # 6 -- the fix
    deck.split(
        "解决办法：gene rank 表示",
        [("做法",
          "每个 cell 内把基因按“相"
          "对该基因自身的异常程度"
          "”排序，取 top 256，depth 的尺"
          "度完全消掉。"),
         ("在 simulation 上验证",
          "模拟数据的正确答案已知"
          "。rank 是唯一把 depth effect 压到 0 "
          "的表示；去掉 per-gene median 那一"
          "步完全无效。"),
         ("两个都需要",
          "只做 rank 不做 read equalisation 时 AUC "
          "0.06，比不处理还差。"),
         ("结果",
          "卵巢 depth AUC 0.59 → 0.53、genes detected "
          "0.42 → 0.51，均不再显著；前"
          "列腺 0.31 → 0.49。")],
        figures / "fig2_representation_benchmark.png")

    # 7 -- the control
    deck.split(
        "对照：给错病人的 metastasis",
        [("做法",
          "把 A 病人的 primary 配 B 病人的 "
          "metastasis，其余 pipeline 一行不改。"),
         ("结果",
          "retained fraction 从 0.351 降到 0.000，"
          "p = 8 × 10⁻¹⁷；两次留下"
          "的细胞集 Jaccard = 0，等于随机"
          "水平。"),
         ("说明什么",
          "gate 确实用到了 metastasis 一侧，"
          "不是只看 primary 自己。所以"
          "这个划分是 patient-specific 的。"),
         "这个对照项目之前从没做"
         "过。"],
        figures / "fig4_mismatched_control.png")

    # 8 -- what it measures
    deck.split(
        "retained fraction 实际在量什么",
        [("检验方式",
          "用 pseudobulk correlation 去预测 retained "
          "fraction — 这个量不做 PCA、不"
          "做 transport，方法从来看不到"
          "它。"),
         ("结果",
          "rho = 0.76，188 pairs。只看同一病"
          "人的 pair、或只看实体转移"
          "，关系依然成立。"),
         ("不是细胞数效应",
          "cell number 能预测 similarity，却预测"
          "不了 retained fraction（rho = −0.06，"
          "p = 0.43）。"),
         "这个数字是“两侧组织有"
         "多像”，不是“有多少细"
         "胞能转移”。"],
        figures / "fig3_similarity_readout.png")

    # 9 -- patient spread
    deck.stacked(
        "所以它不能叫 metastatic potential",
        [("同一指标跳了 100 倍",
          "29 个病人全部已经发生转"
          "移，而这个数从 0% 跳到 88%"
          "，中位 33%。"),
         ("不是样本量问题",
          "只看 n ≥ 500 的 26 个病人仍"
          "然是 0.5%–87%，中位 32%；cell "
          "number 与这个比例的相关只"
          "有 −0.16。Ascites 占最低 7 个里"
          "的 4 个。")],
        figures / "fig5_patient_spread.png", text_height=1.32)

    # 10 -- prostate dataset
    slide = deck.page()
    top = deck.label(slide, "前列腺：两侧真有"
                            "差异的数据集", CONTENT_TOP)
    deck.block(slide, MARGIN, top, 7.7, [
        ("数据",
         "GSE271675，由 Faming 重新注释。"
         "236,507 malignant cells，38 specimens，4 个病"
         "人共 24 个 pairs；第五个病人"
         "对象里没有 metastatic cells。"),
        ("为什么需要它",
         "卵巢两侧 pseudobulk r = 0.94，组织"
         "间 zero DEG — 方法几乎没东西"
         "可找。前列腺是 AR 下调、"
         "invasion 和 interferon 上调，是真存"
         "在的差异。"),
        ("结果",
         "同一 pipeline 下卵巢留 29%，前"
         "列腺只留 1.6%。两侧越不同"
         "，方法留下的越少 — 而"
         "两侧真不同才是有意义的"
         "情形。"),
    ], size=14, gap=8)
    for x, number, text in ((8.5, "29%", "卵巢：留下"
                                         "的 primary 细胞"),
                            (10.9, "1.6%", "前列腺：同"
                                           "一 pipeline")):
        frame = text_box(slide, x, 1.15, 2.4, 1.5)
        write(frame, [(number, 42, True, ORANGE)], first=True, space_after=3)
        write(frame, [(text, 13, False, BODY)], line_spacing=1.14)
    # The embedding below was exported from the capped arm, which keeps 18%,
    # not from the free rule this slide's 1.6% comes from. Saying so here is
    # not optional: the panel prints its own retained percentage, so an
    # unlabelled figure would appear to contradict the headline number.
    frame = text_box(slide, 8.5, 2.62, 4.35, 1.6)
    write(frame, [("Arm 说明", 14, True, INK)], first=True,
          space_after=6)
    write(frame, [("free rule 留 1.6%，对大多数"
                   "病人不够做 gene-level 统计"
                   "。下方 UMAP 与后两页的 "
                   "DEG / GSEA 都用 cap 0.85 arm：每个"
                   "病人 15%，Patient5 24%，合计 "
                   "18%。", 13, False, BODY)], line_spacing=1.2)
    # The prostate embedding goes here rather than on a page of its own: it is
    # the same three panels as the ovarian one, so the two read together.
    deck.figure(slide, figures / "fig_umap_prostate.png", MARGIN, 4.3,
                WIDE - 2 * MARGIN, CONTENT_BOTTOM - 4.3, anchor="top")

    # 11 -- enrichment
    if reference:
        items = [
            ("对比对象",
             "师兄的 primary vs metastasis 组织比"
             "较（蓝）对我们的 retained "
             "cells（橙）。同向就说明 "
             "retained 像 metastasis。"),
            ("Proliferation 完全反向",
             "转移灶本身是低增殖的"
             "：G2M −2.5、E2F −2.4、mitotic spindle "
             "−2.0，p.adj ≤ 3 × 10⁻⁷。而"
             "我们留下的恰好是高增"
             "殖：+2.8 / +3.4 / +1.7，FDR ≤ 0.002。"),
            ("唯一显著一致的是 androgen",
             "双方都是负（−1.74 / −1.44"
             "，我们 FDR 0.046）。EMT 同向但"
             "我们这边不显著（FDR 0.15"
             "）。"),
            "在最强的那条轴上，方"
            "法挑出的恰好是 metastasis 的"
            "反面。",
        ]
        title = "富集分析：对比独立的 " \
                "metastasis signature"
    else:
        items = [
            ("做法",
             "对 primary rejected vs retained 的 preranked GSEA"
             "，正值 = 在 retained 中富集。"),
            ("Proliferation 完全主导",
             "E2F +3.4，G2M +2.8，mitotic spindle +1.7，"
             "FDR ≤ 0.002。"),
            ("其余通路很弱",
             "Androgen response −1.4（FDR 0.046，是显"
             "著项里最弱的）；EMT 和 "
             "interferon 不显著。"),
            "师兄那份 Hallmark GSEA 的 xlsx 本机"
            "未找到，蓝色对比柱待"
            "补。",
        ]
        title = "富集分析：到底是什么" \
                "把 retained 分出来的"
    deck.split(title, items, figures / "fig6_gsea_agreement.png")

    # 12 -- cause two
    deck.split(
        "原因二：retained 就是正在分"
        "裂的细胞",
        [("598 个显著基因里最强的那"
          "一批",
          "DIAPH3、TOP2A、NUSAP1、KIF11、ASPM、CENPF"
          "、MKI67 — 全是 cell-division programme。"),
         ("MT- 基因的说明",
          "这批是 multiome 细胞核，MT 中"
          "位数 0.2%，不是垓死细胞。"
          "MT 偏倒 AUC 从 0.82 降到 0.55，和 "
          "depth 被同一个修正解决。"),
         ("这也解释了 androgen 那一条",
          "AR 驱动分化，与 proliferation 反"
          "相关；挑出分裂细胞就"
          "自动得到低 AR。")],
        figures / "fig7_cell_cycle.png")

    # 13 -- replication
    deck.split(
        "两种癌里都一致",
        [("Contrast",
          "kept vs rejected primary cells，先在每个病"
          "人内部做差 — 与 DEG 的 paired "
          "design 一致。池在一起看会"
          "被病人间差异淹掉。"),
         ("卵巢 26 个病人",
          "kept +0.089，17/26 为正；metastasis 在同"
          "一基线上只 +0.015。"),
         ("前列腺 4 个病人",
          "kept +0.040，4/4 为正；metastasis +0.006。"),
         ("Interferon 和 mesenchymal",
          "两种癌里都没有一致位"
          "移。"),
         "cell division 是唯一在两种癌里"
         "都复现的信号，kept 的位移"
         "比 metastasis 大 6–7 倍。"],
        figures / "fig_hallmark_ovarian.png")

    # 14 -- status
    slide = deck.page()
    top = deck.label(slide, "现状与下一步", CONTENT_TOP)
    frame = text_box(slide, MARGIN, top + 0.05, 5.9, 2.2)
    write(frame, [("已解决", 17, True, ORANGE)], first=True,
          space_after=10)
    for item in ("Sequencing depth — 两种癌加 "
                 "simulation ground truth 上都验证过",
                 "让 retained fraction 失去意义的 "
                 "calibration",
                 "项目此前从未做过的 "
                 "cross-patient control"):
        write(frame, [("—  " + item, 14, False, BODY)], space_after=9,
              line_spacing=1.2)

    frame = text_box(slide, 6.9, top + 0.05, 6.0, 2.2)
    write(frame, [("未解决", 17, True, ORANGE)], first=True,
          space_after=10)
    for item in ("Cell division 现在是最强的混"
                 "杂 — 可以 regress out，是标"
                 "准做法",
                 "两侧越分歧 threshold 越紧"
                 "，criterion 需要换一个 null",
                 "与“已适应的 metastasis”"
                 "相似不等于同源 — "
                 "这一层算法解决不了"):
        write(frame, [("—  " + item, 14, False, BODY)], space_after=9,
              line_spacing=1.2)

    top = deck.label(slide, "下一步", 3.32)
    deck.block(slide, MARGIN, top, WIDE - 2 * MARGIN, [
        ("重跑卵巢 DEG",
         "在 depth-corrected gate 上重跑，并把 "
         "cell cycle 回归掉 — 这是判定"
         "剩下的信号是不是真的"
         "生物学的关键一步。"),
        ("补两个检查",
         "回 TACC 查 MT% 的上尾分布；"
         "把师兄 Hallmark GSEA 的对比柱补"
         "到富集那一页。"),
        ("换 criterion",
         "当前阈值是 within 除以 cross 的"
         "比值，所以两侧越分歧"
         "留得越少。正在试两种"
         "形式：用目标侧自己的"
         "散度，或 kNN 归属检验。"),
    ], size=14, gap=8)

    deck.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--logo", type=Path, default=None,
                        help="the lab logo for the top left of every page")
    parser.add_argument("--month", default="September 2026")
    parser.add_argument("--reference-gsea", action="store_true",
                        help="the enrichment figure carries the reference bars, "
                             "so its slide is written as a comparison")
    args = parser.parse_args()
    build(args.figures, args.output, args.reference_gsea, args.logo, args.month)


if __name__ == "__main__":
    main()
