"""Chinese / English version of the RNA-OT vs CRISPR-OT deck (110_): same 20 slides, same figures, tables and chart,
rewritten text only. Each result slide: short Chinese title, "怎么看图" and "主要发现" (observation vs interpretation
kept apart), a take-home bar, the part label on the bottom bar; method details in the speaker notes.
Chinese text in Microsoft YaHei (East Asian font), Latin text in the theme font.

    python lineage_validation/rna_lineage_discordance_20261110/113_make_pptx_cn.py \
        --template D:/Xia_lab/Progress_report/progress_report7/ConfidenceOT_20261015.pptx \
        --figs results/rna_lineage_discordance_20261110/figures \
        --out D:/Xia_lab/Progress_report/progress_report7/RNA_CRISPR_lineage_CN_EN.pptx
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from lxml import etree
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("deck110", HERE / "110_make_pptx.py"); D110 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D110)
Deck, rgb, A = D110.Deck, D110.rgb, D110.A
ORANGE, PEACH, INK, GREY = "BF5700", "FDEBDD", "262626", "595959"
EA = "Microsoft YaHei"
PART = {1: "Part 1 · RNA-OT vs CRISPR-OT", 2: "Part 2 · RNA state vs lineage", 3: "Part 3 · 两种 OT 偏好的细胞有什么不同", 0: "总结", 9: "背景与方法"}


def bullets(s, x, y, w, h, sections, size=12.5, head=14):
    """sections: [(heading, [item, ...]), ...]; an item is a str or (label, text) with a bold grey label."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for m in ("margin_left", "margin_right"):
        setattr(tf, m, Inches(0.04))
    first = True
    for hd, items in sections:
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        r = p.add_run(); r.text = hd; r.font.size = Pt(head); r.font.bold = True; r.font.color.rgb = rgb(ORANGE)
        p.space_before = Pt(0 if p is tf.paragraphs[0] else 7); p.space_after = Pt(3)
        for it in items:
            p = tf.add_paragraph(); p.space_after = Pt(3); p.line_spacing = 1.08
            if isinstance(it, tuple):
                r = p.add_run(); r.text = it[0]; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = rgb(GREY)
                txt = it[1]
            else:
                txt = it
            r = p.add_run(); r.text = txt; r.font.size = Pt(size); r.font.color.rgb = rgb(INK)
            pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(Emu(Inches(0.17)))); pPr.set("indent", str(-Emu(Inches(0.17))))
            etree.SubElement(pPr, f"{{{A}}}buFont").set("typeface", "Arial")
            etree.SubElement(pPr, f"{{{A}}}buChar").set("char", "•")
    return tb


def takehome(s, text, y=6.47):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(y), Inches(12.45), Inches(0.58))
    sh.adjustments[0] = 0.2; sh.fill.solid(); sh.fill.fore_color.rgb = rgb(PEACH); sh.line.color.rgb = rgb(ORANGE); sh.line.width = Pt(1)
    sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.15); tf.margin_right = Inches(0.1)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = "Take-home  "; r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = rgb(ORANGE)
    r = p.add_run(); r.text = text; r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = rgb(INK)


def part(s, k):
    tb = s.shapes.add_textbox(Inches(0.3), Inches(7.17), Inches(8), Inches(0.33)); tf = tb.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_top = tf.margin_bottom = Inches(0)
    r = tf.paragraphs[0].add_run(); r.text = PART[k]; r.font.size = Pt(12); r.font.color.rgb = rgb("FFFFFF")


def wide(d, s, img, look, found, take, k, fig_h=3.6, size=12.5):
    """Layout A: wide figure on top, two text columns below, take-home bar."""
    d.image(s, img, 0.4, 0.66, 12.5, fig_h)
    y = 0.66 + fig_h + 0.1; h = 6.4 - y
    bullets(s, 0.45, y, 6.1, h, look, size=size); bullets(s, 6.8, y, 6.1, h, found, size=size)
    takehome(s, take); part(s, k)


def side(d, s, img, sections, take, k, fig_w=8.3):
    """Layout B: figure on the left, text panel on the right, take-home bar."""
    d.image(s, img, 0.35, 0.66, fig_w, 5.72)
    bullets(s, 0.35 + fig_w + 0.25, 0.72, 13.0 - (0.35 + fig_w + 0.25), 5.65, sections)
    takehome(s, take); part(s, k)


def set_ea(prs):
    """Give every run property an East Asian typeface (the template theme has none)."""
    tags = ("rPr", "defRPr", "endParaRPr")
    after = ("cs", "sym", "hlinkClick", "hlinkMouseOver", "rtl", "extLst")
    parts = [sl.part for sl in prs.slides] + [sl.notes_slide.part for sl in prs.slides if sl.has_notes_slide]
    for pt in parts:
        root = pt._element
        for tag in tags:
            for rp in root.iter(f"{{{A}}}{tag}"):
                ea = rp.find(f"{{{A}}}ea")
                if ea is None:
                    ea = etree.Element(f"{{{A}}}ea")
                    lat = rp.find(f"{{{A}}}latin")
                    if lat is not None:
                        lat.addnext(ea)
                    else:
                        nxt = next((c for c in rp if etree.QName(c).localname in after), None)
                        if nxt is not None:
                            nxt.addprevious(ea)
                        else:
                            rp.append(ea)
                ea.set("typeface", EA)
        for para in root.iter(f"{{{A}}}p"):        # East Asian line breaking rules and hanging punctuation
            pPr = para.find(f"{{{A}}}pPr")
            if pPr is None:
                pPr = etree.Element(f"{{{A}}}pPr"); para.insert(0, pPr)
            pPr.set("eaLnBrk", "1"); pPr.set("hangingPunct", "1"); pPr.set("latinLnBrk", "0")
        for r in root.iter(f"{{{A}}}r"):          # runs with Chinese text are zh-CN
            t = r.find(f"{{{A}}}t")
            if t is not None and t.text and any("一" <= ch <= "鿿" or "　" <= ch <= "〿" or "＀" <= ch <= "￯" for ch in t.text):
                rp = r.find(f"{{{A}}}rPr")
                if rp is not None:
                    rp.set("lang", "zh-CN"); rp.set("altLang", "en-US")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--figs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    F = a.figs; DF = F / "deck_20261112"; d = Deck(a.template)

    # 1 cover
    s = d.slide("RNA-OT vs CRISPR-OT", "2026-11-12 进展汇报。数据：KP-Tracer（Yang et al. 2022）与 Quinn et al. 2021。"
                "方法：ConfidenceOT 分别用 RNA 和 CRISPR edits 作为 cost。")
    d.text(s, 0.6, 1.7, 5.7, 1.9, [[("同一批肿瘤细胞的", {"size": 30, "bold": True})], [("RNA 与 CRISPR lineage", {"size": 30, "bold": True})]])
    d.text(s, 0.6, 3.75, 5.7, 2.4, ["KP-Tracer：小鼠肺肿瘤 primary 与 metastasis，32 只 mice",
                                     "Quinn 2021：A549 xenograft，4 个实验",
                                     [("Part 1  RNA-OT vs CRISPR-OT", {"size": 14, "color": GREY})],
                                     [("Part 2  RNA state vs lineage", {"size": 14, "color": GREY})],
                                     [("Part 3  两种 OT 偏好的细胞有什么不同", {"size": 14, "color": GREY})],
                                     [("ConfidenceOT · 2026-11-12", {"size": 14, "color": GREY})]], size=16)
    d.image(s, DF / "3724_NT_T1.png", 6.4, 0.8, 6.5, 5.7)

    # 2 why
    s = d.slide("为什么要比较 RNA 和 CRISPR？",
                "RNA 告诉我们两个细胞现在有多像；CRISPR edits 告诉我们两个细胞在家系上有多近。对同一批细胞运行同一个 OT，"
                "只换 cost matrix，两种匹配的差别就指向 state 和 lineage 不一致的地方：亲缘近但看起来不同，或亲缘远但看起来相似。"
                "目的不是判断哪种 OT 正确。")
    d.box(s, 0.6, 0.95, 2.7, 1.05, "RNA counts", "两个细胞现在有多像")
    d.box(s, 0.6, 2.35, 2.7, 1.05, "CRISPR edits", "两个细胞在家系上有多近")
    d.arrow(s, 3.45, 1.36); d.arrow(s, 3.45, 2.76)
    d.box(s, 3.85, 0.95, 2.3, 1.05, "RNA-OT", "cost = RNA 距离", hi=True)
    d.box(s, 3.85, 2.35, 2.3, 1.05, "CRISPR-OT", "cost = edit 距离", hi=True)
    d.arrow(s, 6.3, 2.06)
    d.box(s, 6.7, 1.45, 2.75, 1.45, "同样的细胞、同样的 solver", "同样的 source / target cells 和 calibration，只换 cost matrix")
    d.arrow(s, 9.6, 2.06)
    d.box(s, 10.0, 1.2, 2.9, 1.95, "比较两种结果", "gate：哪些细胞被 keep / reject；OT coupling：谁和谁匹配", hi=True)
    bullets(s, 0.6, 3.75, 6.0, 2.6, [("两个概念", [("OT coupling：", "OT 给每一对 source cell 和 target cell 分配的匹配权重。"),
                                                 ("Gate：", "ConfidenceOT 判断每个细胞是 keep 还是 reject；找不到可信匹配的细胞被 reject。")]),
                                    ("值得关注的两种不一致", ["同一 clade、RNA state 不同：亲缘很近，但现在看起来不同。",
                                                         "不同 clade、RNA state 相似：亲缘很远，但看起来相似。"])])
    bullets(s, 6.9, 3.75, 6.0, 2.6, [("这份汇报的三个部分", [("Part 1：", "两种 OT 是否匹配、保留同样的细胞？"),
                                                          ("Part 2：", "不看 OT，直接看 lineage tree 和 RNA state 的关系。"),
                                                          ("Part 3：", "两种 OT 各自偏好的细胞，在表达上有什么不同？")])])
    takehome(s, "目的不是选出哪种 OT 更好，而是看 cell state 与 lineage 在哪里一致、在哪里不一致。"); part(s, 9)

    # 3 workflow
    s = d.slide("数据和分析流程",
                "RNA-OT：v3 ConfidenceOT fits（raw counts，冻结的 calibration，Splatter L5-best preprocessing；M4-E）。"
                "CRISPR-OT：同一个 solver 的新 fit；cost = 1 − prior-weighted edit agreement（排除缺失的 intBC，indel priors，"
                "每个 intBC 每条 string 一个 event）；within-side split calibration，acceptance 0.99，不用 MNN。"
                "Edit matrix：allele table 的 states（3,849 列与作者的 character matrices 相同）。"
                "Clade：同一 family 内按 edit distance 做 UPGMA，cut 0.5，≥ 20 cells。KP states：作者的 Cluster-Name。")
    steps = [("raw counts", "+ CRISPR allele table", False, 1.4), ("RNA-OT", "RNA cost", True, 1.2), ("CRISPR-OT", "edit cost", True, 1.2),
             ("edit matrix", "不做 OT", True, 1.2), ("比较", "gates 和 couplings", False, 1.45), ("clade × RNA state", "以及 sites", False, 1.55),
             ("DEG / GSEA", "", False, 1.2)]
    x = 0.5
    for i, (h_, sb, hi, w) in enumerate(steps):
        d.box(s, x, 0.85, w, 0.85, h_, sb, hi=hi, head_size=12, sub_size=10.5); x += w
        if i < len(steps) - 1:
            d.arrow(s, x + 0.06, 1.17); x += 0.36
    d.table(s, 0.55, 1.95, [4.6, 1.6, 6.15], [["数据", "Cells", "用途"],
                                             ["KP-Tracer，45 个 OT units（10 只 mice）", "34,820", "RNA-OT vs CRISPR-OT，difference flow"],
                                             ["KP-Tracer，全部可用细胞（32 只 mice）", "74,399", "clade × RNA state，primary vs metastasis"],
                                             ["Quinn 2021，4 个实验（A549 xenograft）", "63,262", "RNA-OT vs CRISPR-OT（states = 我们的 Leiden clusters）"]],
            size=12.5, rowh=0.34)
    d.table(s, 0.55, 3.55, [3.0, 9.35], [["设置", "内容"],
                                        ["OT", "ConfidenceOT，UOT backbone，M4-E gate，用 within-side split null 校准"],
                                        ["两种 OT 相同的部分", "细胞、solver、calibration；只有 cost matrix 不同"],
                                        ["Clade", "根据 CRISPR edits 划分的 lineage group：同一 tumour family 内 UPGMA，cut 0.5，≥ 20 cells"],
                                        ["RNA states", "KP：作者的注释；Quinn：在 RNA 坐标上做的 Leiden clusters"]], size=12.5, rowh=0.34)
    takehome(s, "共 49 个 CRISPR-OT units；每次比较都使用与 RNA-OT 完全相同的细胞。"); part(s, 9)

    # 4 D1
    s = d.slide("两种 OT 匹配的细胞不一样",
                "对每个 target cell，取每种 coupling 中权重最高的 15 个 source cells。左：它们的平均 edit agreement s；右：平均 RNA "
                "similarity r。每个点是一个 unit。KP RNA-OT partners：s 0.10–0.34，r 0.45–0.62；CRISPR-OT partners：s 0.47–0.86，"
                "r 0.01–0.13。两个 top-15 列表的重叠：0–2%。同一 target 上两种 coupling 的 total variation：0.71–0.83。")
    wide(d, s, DF / "D1_what_each_OT_matches.png",
         [("怎么看图", ["对每个 target cell，取每种 OT 权重最高的 15 个 source cells（top-15 partners）。",
                        "横轴 = RNA-OT 的 partners，纵轴 = CRISPR-OT 的 partners；每个点 = 一个 comparison unit；虚线 = 两者相同。",
                        "左图比较亲缘有多近（CRISPR edits），右图比较 RNA 有多像。"])],
         [("主要发现", ["左图几乎所有点在虚线上方：CRISPR-OT 的 partners 共享更多 CRISPR edits。",
                        "右图所有点在虚线下方：RNA-OT 的 partners RNA expression 更相似。",
                        "两种 OT 的 top-15 partners 重叠只有 0–2%。"])],
         "RNA 相似不代表 lineage 相近，因此两种 OT 会选择几乎完全不同的细胞。", 1)

    # 5 D2
    s = d.slide("两种 OT 保留的细胞也不一样",
                "RNA-OT 与 CRISPR-OT 的 keep / reject 决定之间的 Cohen's kappa，每个 unit、每一侧一个值。几乎所有值接近 0："
                "两个 gate reject 的是不同的细胞。较高的 source 侧数值来自 KP 3724 的 pair units，那里 edit distance 对细胞的区分较弱。"
                "Gate 的不一致与 coupling 的不一致分开报告。")
    wide(d, s, DF / "D2_same_cells_kept.png",
         [("怎么看图", [("Cohen's kappa：", "衡量两种 OT 的 keep / reject 结果有多一致；0 = 和随机一样，1 = 完全相同。"),
                        "每个点 = 一个 unit 的一侧：灰 = primary-tumour cells，黑 = metastasis cells；横线 = 中位数。",
                        "三组：KP 单个 metastasis、KP 合并全部 metastases、Quinn。"])],
         [("主要发现", ["绝大多数点在 0 附近：两个 gate reject 的基本是不同的细胞。",
                        "少数较高的点（主要是 source 侧）来自 KP 3724，那里 CRISPR edits 对细胞的区分较弱。"]),
          ("注意", ["Gate 的不一致与匹配（coupling）的不一致分开分析。"])],
         "不仅匹配对象不同，两种 OT 认为可信、应该 keep 的细胞也几乎不重叠。", 1)

    # 6 flow 3460
    s = d.slide("多出来的匹配落在哪里？",
                "同一个 source（primary）和 target（metastasis）；只用两个 gate 都 keep 的细胞。每个 coupling 归一化为 1，再按 "
                "source clade × target clade 求和。灰 = 两种 OT 都放在那里的质量；蓝 = RNA-OT 多出的质量；橙 = CRISPR-OT 多出的质量。"
                "45 个 KP units 中，橙色质量在同一 clade 的中位数为 87%，蓝色为 0%；CRISPR-OT 放在 same-clade pairs 上的质量在 "
                "45/45 units 中更多（0.675 vs 0.40）。这是 clade 层面的质量，不是逐个细胞的匹配。完整版（含 gate 条和 overlap bars）"
                "见 figures/flow/。")
    side(d, s, DF / "flow_L1__3460__3460_Lkb1_T1__3460_Lkb1_N6.png",
         [("怎么看图", ["左边 = primary（source）的 clades，右边 = metastasis（target）的 clades；最左侧是根据 CRISPR edits 画的 clade tree。",
                        "只用两个 gate 都 keep 的细胞；ribbon 越宽，两个 clade 之间的匹配越多。",
                        "灰 = 两种 OT 都有的匹配；蓝 = RNA-OT 多出的；橙 = CRISPR-OT 多出的。柱子颜色 = RNA state。"]),
          ("主要发现", ["橙色大多连回同一个 clade（45 个 KP units 中位 87%）；蓝色几乎都跨到别的 clade（同一 clade 的中位 0%）。",
                        "45/45 units 中，CRISPR-OT 放在同一 clade 的匹配更多（0.675 vs 0.40）。"]),
          ("注意", ["CRISPR-OT 的 cost 和 clade 都来自 CRISPR edits，所以这一点在意料之中，不能当作 CRISPR-OT 更准确的独立证据。"])],
         "CRISPR-OT 多出的匹配留在 clade 内，RNA-OT 多出的匹配跨 clade；这是 clade 层面的统计。", 1)

    # 7 flow 3724
    s = d.slide("KP 3724：两种 OT 都跟着两个 clade 走",
                "合并的情况：primary tumour 到全部 4 个 metastases。两个 clades：c5（Mesenchymal-1）和 c45（Mesenchymal-2）。大部分"
                "质量两种 OT 一致；RNA-OT 多出一些 c5 ↔ c45 的质量。3724 不满足冻结的 CRISPR resolution 规则（distinct edit sets "
                "8%），clade 与 site 混杂，而且它的测序深度是 OT mice 中最高的。")
    side(d, s, DF / "flow_L23__KPTracer_3724__3724_NT_T1__family_metastases.png",
         [("怎么看图", ["与上一页相同；source = primary T1，target = 4 个 metastases 合并。"]),
          ("主要发现", ["只有两个主要 clades：c5（Mesenchymal-1）和 c45（Mesenchymal-2）。",
                        "大部分匹配两种 OT 一致（灰）；RNA-OT 额外加了少量 c5 ↔ c45 的跨 clade 匹配（蓝）。"]),
          ("注意", ["按预先规则，3724 的 CRISPR 分辨率不足（distinct edit sets 8%）；clade 与 site 混在一起；测序深度也是 OT mice 中最高的。",
                    "所以 3724 单独报告，所有合并结果也给出去掉 3724 的版本。"])],
         "在 3724，两种 OT 大体一致，分歧主要是 RNA-OT 的少量跨 clade 匹配。", 1)

    # 8 D3
    s = d.slide("clade 层面一致 ≠ 细胞层面一致",
                "KP units 的中位数。左：RNA-OT 与 CRISPR-OT 在 clade 层面的 overlap（clade × clade 质量中两者一致的比例）。中：RNA-OT "
                "与一个只按 clade 大小分配的 coupling 的 overlap（44/45 units 中比 CRISPR-OT 更接近）。右：同一批 target cells 上，"
                "两种 OT 的 top-15 partner 列表的重叠。")
    wide(d, s, DF / "D3_clade_vs_cell_level.png",
         [("怎么看图", ["三根柱子都是 KP units 的中位数。",
                        "左：两种 OT 在 clade × clade 层面的重叠。中：RNA-OT 与一个只按 clade 大小分配的 coupling 的重叠。",
                        "右：同一批 target cells，两种 OT 的 top-15 partners 的重叠。"])],
         [("主要发现", ["clade 层面重叠很高（90%），但 RNA-OT 与只看 clade 大小的 coupling 更像（98%；44/45 units 比 CRISPR-OT 更接近）。",
                        "细胞层面，两种 OT 的 partners 只重叠 4%。"]),
          ("概念", [("Baseline / null model：", "完全不看细胞信息、只按 clade 大小分配时，会得到什么结果。")])],
         "clade 层面的高重叠主要来自 clade 大小；两种 OT 很少选到同一个 partner cell。", 1)

    # 9 tree 3457
    s = d.slide("同一个 clade 内可以有不同 RNA states",
                "Family 3457_Apc_T1 的细胞层面 CRISPR tree（885 cells，resolved：每个细胞 12 个 edits）。叶子是细胞；虚线 = clade cut。"
                "色带：clade、作者的 RNA state、primary / metastasis；右侧：informative edits（黑 = 有，灰 = intBC 缺失）。Clade c6 在 cut "
                "以下分成 Apc Early 一支和 Early EMT-1 / Mesenchymal-1 一支，各有自己的 edits。c1 主要是 metastatic Pre-EMT cells。")
    side(d, s, DF / "3457_Apc_T1.png",
         [("从这里开始：Part 2", ["这一部分不看 OT matching，直接看 CRISPR lineage tree 和 RNA state 的关系。"]),
          ("怎么看图", ["每片叶子是一个细胞；树按 CRISPR edits 的距离画；红色虚线 = clade 的切分阈值。",
                        "三条色带：CRISPR clade、RNA state（作者注释）、primary / metastasis。",
                        "右侧矩阵：每列一个 CRISPR edit，黑 = 细胞有这个 edit。"]),
          ("主要发现", ["3457_Apc_T1（885 cells，edits 分辨率足够）：clade c6 在阈值以下又分成两支，各有自己的 edits。",
                        "一支以 Apc Early 为主，另一支以 Early EMT-1 / Mesenchymal-1 为主；c1 主要是 metastasis 的 Pre-EMT cells。"]),
          ("注意", ["这说明亲缘很近的细胞可以处在不同 RNA states，但不能说明状态变化的方向。"])],
         "同一 clade 内带有不同 edits 的 sub-branches，可以对应不同的 RNA states。", 2)

    # 10 tree 3724
    s = d.slide("KP 3724：lineage、RNA state 和 site 对齐",
                "从 20,992 个细胞中画出 948 个。c5 = Mesenchymal-1 和 Mesenchymal-1 (Met)；c45 = Mesenchymal-2。c45 内带有额外 edits "
                "的 sub-branch 是 metastatic Mesenchymal-2 (Met)。按冻结规则不 resolved（每个细胞 7 个 edits，distinct edit sets 8%）。")
    side(d, s, DF / "3724_NT_T1.png",
         [("怎么看图", ["与上一页相同；从 20,992 个细胞中抽取 948 个画图。"]),
          ("主要发现", ["c5 = Mesenchymal-1 和 Mesenchymal-1 (Met)；c45 = Mesenchymal-2。",
                        "c45 内有自己额外 edits 的 sub-branch，正好是 metastatic 的 Mesenchymal-2 (Met) cells。"]),
          ("注意", ["3724 每个细胞约 7 个 edits，按预先规则 CRISPR 分辨率不足；只作为个案。"])],
         "在 3724，一个带有特有 edits 的 sub-branch 对应 metastatic Mesenchymal-2 细胞。", 2)

    # 11 D4
    s = d.slide("有些 RNA states 只出现在少数 clades",
                "全部可用的 KP 细胞（74,399 cells，32 只 mice）。对每只 mouse 和每个 RNA state（≥ 50 cells），判断这个 state 的细胞"
                "落在比随机更少、和随机一样、还是更多的 clades 中。随机 = 在每个 site 内把细胞的 clade 打乱（999 次）；fewer / more = "
                "Hill-number z ≤ −2 / ≥ 2。超出 null 的 clade × state coupling：0.149（18 mice，resolved lineages），去掉 3724 为 0.120；"
                "合并 (Met) labels 后不变。")
    wide(d, s, DF / "D4_states_tied_to_few_clades.png",
         [("怎么看图", ["每行一个 RNA state；横轴 = 这个 state 有 ≥ 50 cells 的 mice 数。",
                        "红 = 比随机更集中在少数 clades；灰 = 和随机一样；蓝 = 比随机更分散。"]),
          ("概念", [("Null model：", "在每个 site 内随机打乱 clade 标签（999 次），看随机时一个 state 会落在多少 clades。"),
                    ("Cramer's V：", "衡量 clade 与 RNA state 两个分类变量的关联强度。")])],
         [("主要发现", ["Early EMT-1（8/11 mice）、Apc Early（5/6）、Mesenchymal-2（4/6）、Lung progenitor-like（3/3）多集中在少数 clades。",
                        "AT2-like 常比随机更分散（5/12）；Pre-EMT 大多和随机一样；Endoderm-like 两种情况都有。",
                        "超出 null 的 clade × state 关联（Cramer's V）：0.149（18 mice），去掉 3724 为 0.120。"])],
         "RNA state 与 lineage 的关系因 state 而异：有的 state 跟着 lineage 走，有的分布在很多 lineages。", 2, size=12)

    # 12 D5
    s = d.slide("不同肿瘤可以有相似的 RNA state",
                "KP 3513：每个 RNA state（≥ 50 cells）的细胞来自各个独立 primary tumour 的比例（每个 tumour 是独立的 CRISPR family）。"
                "右侧：有 ≥ 5 个该 state 细胞的 tumours 数，以及最大占比。Gastric-like：5 个 tumours，22 个 clades；High plasticity："
                "5 个 tumours，21 个 clades。相同 state 本身不是 convergence 的证据。")
    wide(d, s, DF / "D5_states_across_tumours_3513.png",
         [("怎么看图", ["KP 3513；每行一个 RNA state，颜色 = 细胞来自哪个独立的 primary tumour（每个 tumour 是一个独立的 CRISPR family）。",
                        "右侧文字 = 有 ≥ 5 个该 state 细胞的 tumours 数，以及最大的占比。"])],
         [("主要发现", ["Gastric-like 来自 5 个 tumours（22 个 clades），High plasticity 来自 5 个 tumours（21 个 clades）。",
                        "Endoderm-like 95% 来自 T1。"]),
          ("注意", ["不同 lineages 出现相同 state，本身不能证明 convergence（例如它也可能只是常见的状态）。"])],
         "同一 RNA state 可以来自多个独立的 tumours，也可以几乎只来自一个。", 2)

    # 13 D6
    s = d.slide("KP 3724：每个 metastasis 只有一个 clade",
                "KP 3724，全部可用细胞。每个 site 一块；块内每列是一个 CRISPR clade（宽度 = 细胞数），颜色 = RNA state。一个 site 如果"
                "少于 2 个 clades 或 2 个 states（各 ≥ 5 cells），就无法检验 association；这不是没有 association 的证据。")
    wide(d, s, DF / "D6_primary_vs_metastasis_3724.png",
         [("怎么看图", ["每块是一个 site：T1 = primary，其余 = metastases。",
                        "块内每列是一个 CRISPR clade（宽度 = 细胞数），颜色 = RNA state。"])],
         [("主要发现", ["primary T1 有两个 clades（c45 和 c5）；L1、L2、L3 只有 c5，S1 只有 c45。",
                        "每个 metastasis 几乎只有一个 clade 和一种 state。"]),
          ("注意", ["site 内少于 2 个 clades 或 2 个 states 时无法检验 lineage-state association；这不是没有关联的证据。"])],
         "3724 的 metastases 无法检验 lineage-state association，下一页看其他 mice。", 2)

    # 14 table + chart
    s = d.slide("Primary vs metastasis：没有统一方向",
                "超出 within-site null 的 clade × state coupling（Cramer's V 减去 null V），两侧抽样到相同细胞数（200 次），全部可用细胞，"
                "作者的 labels。去掉 3724 的均值：primary 0.015，metastasis 0.037。中间结论'metastasis 的 lineage coupling 更弱'来自 3724，"
                "不能推广。")
    d.table(s, 0.55, 0.85, [1.0, 1.15, 1.35, 3.1], [["Mouse", "Primary", "Metastasis", "可检验 sites 中的 metastatic cells"],
                                                   ["3724", "0.198", "0.006", "69%"], ["3457", "0.047", "0.000", "0%（无法检验）"],
                                                   ["3467", "0.007", "0.150", "94%"], ["3460", "-0.003", "0.078", "75%"],
                                                   ["3451", "0.019", "0.003", "97%"], ["3513", "0.005", "-0.044", "100%"]], size=13, rowh=0.46)
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
    cd = CategoryChartData(); cd.categories = ["3724", "3457", "3467", "3460", "3451", "3513"]
    cd.add_series("Primary", (0.198, 0.047, 0.007, -0.003, 0.019, 0.005)); cd.add_series("Metastasis", (0.006, 0.000, 0.150, 0.078, 0.003, -0.044))
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.3), Inches(0.7), Inches(5.6), Inches(3.75), cd).chart
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(12)
    for ser, colr in zip(ch.series, ("2C6E9B", "B4702A")):
        ser.invert_if_negative = False; ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(colr)
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    ch.value_axis.has_major_gridlines = False; ch.value_axis.tick_labels.font.size = Pt(11); ch.category_axis.tick_labels.font.size = Pt(11)
    ch.value_axis.has_title = True; ch.value_axis.axis_title.text_frame.text = "coupling beyond null (V - null V)"
    ch.value_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
    ch.category_axis.has_title = True; ch.category_axis.axis_title.text_frame.text = "mouse"
    ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
    bullets(s, 0.45, 4.6, 6.1, 1.8, [("怎么看图", ["数值 = clade × RNA state 的 Cramer's V 减去 within-site null 的 V；越高 = lineage 与 state 的关联越强。",
                                                 "primary 和 metastasis 两侧抽样到相同细胞数（200 次），避免细胞数不同的影响。"])], size=12)
    bullets(s, 6.8, 4.6, 6.1, 1.8, [("主要发现", ["3724：primary 高、metastasis 低；3467 和 3460 则相反；3457 的 metastasis 无法检验。",
                                                "去掉 3724 后的均值：primary 0.015，metastasis 0.037。"])], size=12)
    takehome(s, "'metastasis 的 lineage-state 关联更弱'这一中间结论来自 3724，不能推广到其他 mice。"); part(s, 2)

    # 15 DEG design
    s = d.slide("两种 OT 在同一 clade 内偏好哪些细胞？",
                "计划写在 PLAN_v3_deg.md 中，在任何结果之前固定。每个 clade、site 和 side，只用两个 gate 都 keep 的细胞：q_R(i) = "
                "source cell i 承担的本 clade RNA-OT 质量份额，q_C(i) 同理用 CRISPR-OT；d = n × (q_R − q_C)。RNA-OT-preferred = d 的上"
                "四分位，CRISPR-OT-preferred = 下四分位，both supported = 不在这两个四分位且两种 OT 给它至少均匀份额。raw counts 与"
                "depth-thinned counts 上做 Wilcoxon；Hallmark prerank GSEA（50 sets，两个方向）。对照：RNA-similarity baseline、同一批"
                "细胞的 5 次 random splits、immediate-early genes 检查。257 个 contrasts，10 只 mice。")
    d.text(s, 0.6, 0.72, 12.2, 0.6, "Part 1 发现两种 OT 匹配不同的细胞。这里问：在同一个 clade 内，被 RNA-OT 偏好的细胞和被 CRISPR-OT 偏好的细胞，"
           "表达上有什么不同？", size=13.5, color=GREY)
    d.box(s, 0.6, 1.75, 2.9, 1.3, "一个 CRISPR clade", "同一 site、同一侧，两个 gate 都 keep 的细胞")
    d.arrow(s, 3.62, 2.28)
    d.box(s, 4.0, 1.45, 3.6, 0.6, "RNA-OT-preferred cells", "RNA-OT 给它们更多匹配权重", hi=True, head_size=12.5, sub_size=10.5)
    d.box(s, 4.0, 2.1, 3.6, 0.6, "both supported", "两种 OT 都给它们足够的权重", head_size=12.5, sub_size=10.5)
    d.box(s, 4.0, 2.75, 3.6, 0.6, "CRISPR-OT-preferred cells", "CRISPR-OT 给它们更多匹配权重", hi=True, head_size=12.5, sub_size=10.5)
    d.arrow(s, 7.75, 2.28)
    d.box(s, 8.15, 1.75, 4.7, 1.3, "比较表达", "DEG（Wilcoxon）+ Hallmark GSEA：RNA-OT-preferred vs CRISPR-OT-preferred")
    d.table(s, 0.6, 3.6, [3.4, 8.85], [["对照", "作用"],
                                        ["RNA-similarity baseline", "RNA-OT 的偏好本来就来自 RNA；只有超出它的差异才算 OT 特有的"],
                                        ["5 次 random splits", "把同一批细胞随机分成两组，看随机情况下差异有多大"],
                                        ["Depth-thinned counts", "把测序深度降到同一水平，排除只随 sequencing depth 变化的差异"],
                                        ["Immediate-early genes", "解离应激基因单独报告，并去掉后重跑 GSEA"]], size=12.5, rowh=0.34)
    bullets(s, 0.6, 5.42, 12.2, 0.95, [("概念", [("DEG：", "比较两组细胞，找出差异表达的基因。  "),
                                                ("GSEA：", "检验一组 genes（例如一条 Hallmark pathway）整体是否偏向某一组细胞，结果用 NES 表示。")])], size=12)
    takehome(s, "257 个 contrasts、10 只 KP mice，raw 与 depth-thinned 各一遍；判断规则在看结果前固定。"); part(s, 3)

    # 16 D7
    s = d.slide("RNA-OT 偏好 RNA 更近、测序更深的细胞",
                "左：第一组 / 第二组的 median sequencing depth，每个点一个 contrast（一个 clade、一个 site）。右：这个 contrast 的 50 个"
                "Hallmark NES 与对应 RNA-similarity 对照（离同一批 targets 最近 vs 最远的 source cells）的 Spearman 相关。RNA-OT- vs "
                "CRISPR-OT-preferred：深度 ×1.40（86% contrasts > 1），与对照相关 0.77（thinned 0.81），top-100 genes 重叠 45%。target 侧 "
                "0.91。CRISPR-OT-preferred 方向相反（−0.75，深度 ×0.78）。两组的 informative edits 相同（中位 4）。")
    wide(d, s, DF / "D7_preference_depth_and_rna.png",
         [("怎么看图", ["每个点 = 一个 contrast（一个 clade、一个 site）；横线 = 中位数。",
                        "左：两组的 median sequencing depth 之比（> 1 = 第一组测得更深）。",
                        "右：与 RNA-similarity baseline 的一致程度（Spearman，1 = 和只按 RNA 远近分组的结果一样）。"])],
         [("主要发现", ["RNA-OT- vs CRISPR-OT-preferred：深度 ×1.40，与 baseline 的相关 0.77；target 侧 0.91。",
                        "CRISPR-OT-preferred cells 方向相反（−0.75，深度 ×0.78）；两组的 CRISPR edits 数相同（中位 4）。"]),
          ("解释（Interpretation）", ["RNA-OT 的偏好大部分可以用 RNA 更接近 targets 和 sequencing depth 来解释。"])],
         "同一 clade 内，RNA-OT 多选的主要是 RNA 更近、测序更深的细胞；CRISPR-OT 正好相反。", 3)

    # 17 D8
    s = d.slide("没有 Hallmark pathway 通过所有检验",
                "全部 50 个 Hallmark sets。颜色：各 mice 的 mouse-mean NES 的中位数（上 raw counts，下 depth-thinned）。黑点：在两个 run "
                "中都有 ≥ 75% mice 同号（RNA-OT- vs CRISPR-OT-preferred 有 17 个 sets，例如 E2F、MYC、OXPHOS 更高）。结果前固定的规则："
                "还需在 ≥ 50% contrasts 中超出 5 次 random splits 的范围，并且不能被 RNA baseline 解释。没有 set 满足（最高：MYC V1，0.49）；"
                "17 个中有 14 个可被 RNA baseline 解释。去掉 3724：7 个同号，0 个 consistent。去掉 immediate-early genes 后 17 个中保留 14 个"
                "（失去 hypoxia、IL6-JAK-STAT3、allograft）。C1（同一 state 内的两个 clades）：只有 apical surface 超出 random splits。")
    wide(d, s, DF / "D8_hallmark_all_sets.png",
         [("怎么看图", ["列 = 全部 50 个 Hallmark sets；行 = 6 种 contrasts（上 raw，下 depth-thinned）。颜色 = 各 mice NES 的中位数；黑点 = ≥ 75% mice 方向一致。"])],
         [("主要发现", ["17 个 sets 方向一致（如 E2F、MYC、OXPHOS），但没有一个在 ≥ 50% contrasts 中超出 random splits；其中 14 个能被 RNA baseline 解释。"])],
         "方向在 mice 之间会重复，但都没有超过随机分组；目前不能下 pathway 层面的结论。", 3, fig_h=4.78, size=12)

    # 18 D9
    s = d.slide("例子：KP 3724 的一个 clade",
                "按结果前固定的规则选：focus mice 中细胞最多的 contrast，即 KP 3724 primary，clade c45（Mesenchymal-2），2,816 cells。"
                "每组 log-normalised expression 的平均 z，两个方向各 top 10 genes（raw counts）。RNA-OT-preferred：Vim、Lgals1、S100a4、"
                "Hmga1、Eno1。CRISPR-OT-preferred：Malat1、Neat1、Junb、Btg2、Nr4a1，且深度更低，更像细胞质量 / 应激，而不是某个 lineage "
                "program。3724 不满足 CRISPR resolution 规则。只是一个例子，不是统计结论。")
    side(d, s, DF / "D9_example_genes_3724_c45.png",
         [("怎么看图", ["三列 = 三组细胞（列名下是细胞数和 median depth）。",
                        "行 = 两组差异最大的 20 个 genes：上 10 个在 RNA-OT-preferred 更高，下 10 个在 CRISPR-OT-preferred 更高；颜色 = 平均 z-score。"]),
          ("主要发现", ["RNA-OT-preferred：Vim、Lgals1、S100a4、Hmga1、Eno1 更高。",
                        "CRISPR-OT-preferred：测序更浅，Malat1、Neat1、Junb、Btg2、Nr4a1 更高。"]),
          ("解释（未检验）", ["CRISPR-OT 多选的细胞更像低质量 / 应激的细胞，而不是某个 lineage program。"]),
          ("注意", ["按预先规则选出（focus mice 中细胞最多的 contrast）；3724 的 CRISPR 分辨率不足；只是一个例子。"])],
         "在这个 clade 里，CRISPR-OT 多选的细胞更浅、带应激信号；RNA-OT 多选的是典型的 mesenchymal 细胞。", 3, fig_w=8.0)

    # 19 controls
    s = d.slide("对照与局限", "对照都在读结果之前固定；之后加的内容在 plan 中以带日期的 post-hoc 章节记录。")
    d.table(s, 0.55, 0.8, [4.0, 8.35], [["对照", "说明了什么"],
                                         ["Within-site permutation null", "在 site 内随机打乱：state-lineage 关联是在 site 组成之外测得的"],
                                         ["Size-only coupling（只按 clade 大小）", "RNA-OT 在 clade 层面与它非常接近（0.98）"],
                                         ["Clade level vs cell level", "0.83 vs 0.04（3460 一个 unit 的 97 个 target clades）：clade 重叠 ≠ 细胞匹配"],
                                         ["去掉 KP 3724", "state-lineage 结果仍成立（0.120）；primary vs metastasis 的差别不成立"],
                                         ["合并 (Met) labels", "没有变化"],
                                         ["Sequencing depth", "unit 层面的分歧与深度无关（rho −0.01）；clade 内的偏好与深度有关（×1.40），所以 DEG 都另用 depth-thinned counts 跑"]],
            size=12.5, rowh=0.4)
    d.table(s, 0.55, 3.85, [4.0, 8.35], [["局限", "影响"],
                                          ["每个 site 只有一个 tumour", "没有 biological replicates；细胞层面的检验只能算探索性"],
                                          ["KP 3724 分辨率不足，且与 site 混杂", "单独报告；所有合并结果也给出去掉 3724 的版本"],
                                          ["CRISPR-OT 与 clade 用的是同一套 edits", "两者一致不是独立的 lineage validation"],
                                          ["单一时间点", "无法判断状态变化的方向和时间"]], size=12.5, rowh=0.4)
    takehome(s, "主要的 state-lineage 结果经得起这些检验；'primary vs metastasis 不同'的结论没有经住。"); part(s, 0)

    # 20 findings
    s = d.slide("主要发现与下一步", "")
    d.box(s, 0.55, 0.8, 6.3, 0.5, "观察到的（Observation）", hi=True, head_size=15)
    bullets(s, 0.6, 1.4, 6.2, 4.95, [("Part 1", ["两种 OT 匹配不同的细胞（top-15 重叠 0–2%），也 keep 不同的细胞（kappa 约 0）。",
                                                "CRISPR-OT 多出的匹配在同一 clade 内，RNA-OT 多出的跨 clade（45/45 units）。",
                                                "clade 层面的高重叠主要来自 clade 大小。"]),
                                     ("Part 2", ["有些 RNA states 集中在少数 clades（Early EMT-1、Mesenchymal-2），有些分布在很多 clades（Pre-EMT、AT2-like）。",
                                                 "primary vs metastasis 的关联因 mouse 而异，没有统一规律。"]),
                                     ("Part 3", ["clade 内，RNA-OT 偏好 RNA 更近、测序更深的细胞；没有 Hallmark pathway 通过所有检验。"])], size=12.5)
    d.box(s, 7.1, 0.8, 5.8, 0.5, "还不能说的 & 下一步", head_size=15)
    bullets(s, 7.15, 1.4, 5.7, 4.95, [("还不能说的", ["CRISPR-OT 更准确：它的 cost 和 clade 来自同一套 edits。",
                                                     "同一 clade 内的不同 states 代表某个特定方向的状态转变。",
                                                     "metastasis 的 state-lineage 关联普遍更强或更弱。",
                                                     "某条 Hallmark pathway 驱动了两种 OT 的分歧。"]),
                                      ("下一步", ["RNA-OT 能否找回真实的 fate？在 LARRY 上检验（clonal barcodes，day 2 / 4 / 6；正在 TACC 上运行）。",
                                                  "RNA-OT 偏好更深细胞这一点会不会影响 fate 预测？在 LARRY 上检查。",
                                                  "更多 resolved、带 metastases 的 mice，用来检验不含 3724 的 primary vs metastasis。"])], size=12.5)
    takehome(s, "State 和 lineage 在某些 states 上一致、在另一些上分开；两种 OT 看到的是这一结构的不同侧面。"); part(s, 0)

    set_ea(d.p)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    d.p.save(a.out); print(a.out, len(d.p.slides), "slides")


if __name__ == "__main__":
    main()
