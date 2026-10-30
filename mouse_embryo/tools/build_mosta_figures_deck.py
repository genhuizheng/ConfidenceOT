"""MOSTA figure deck: every panel of 30_mosta_figures.py on slides, main figures then supplementary.

    python mouse_embryo/tools/build_mosta_figures_deck.py PACKAGE_DIR [OUT.pptx]

PACKAGE_DIR is the unpacked mosta_figures_<date>.tar.gz (the folder holding figures/, tables/,
reports/).  Panels are placed as they were drawn, each cropped to its own content, never redrawn or
merged; every number on a slide is read from the package's tables when the deck is built.  The
deck and its cropped images are generated outputs and stay out of git.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PACKAGE = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else PACKAGE.parent.parent / "MOSTA_figures_deck.pptx"
ASSETS = OUT.parent / (OUT.stem + "_assets")
FIG = PACKAGE / "figures"
TAB = PACKAGE / "tables"

NAVY, INK, MUTED, RULE = "1F2A44", "2B2B2B", "6B7280", "C9CED6"
PANEL, ACCENT = "F4F6F9", "E4572E"
TITLE_FONT, BODY_FONT = "Cambria", "Calibri"
W, H = 13.333, 7.5
TOP, BOTTOM = 1.55, 7.0


def rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_)


def textbox(slide, x, y, w, h, paragraphs, size=14, colour=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            font=BODY_FONT, bold=False, after=4):
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
        p.space_after = Pt(opts.get("after", after))
        run = p.add_run()
        run.text = ("•  " + text) if opts.get("bullet") else text
        run.font.size = Pt(opts.get("size", size))
        run.font.name = opts.get("font", font)
        run.font.color.rgb = rgb(opts.get("colour", colour))
        run.font.bold = opts.get("bold", bold)
    return box


def panel_box(slide, x, y, w, h):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.05
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(PANEL)
    s.line.color.rgb = rgb(RULE)
    s.line.width = Pt(0.75)
    s.shadow.inherit = False
    return s


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.blank = self.prs.slide_layouts[6]
        self.count = 0
        self.used = []
        ASSETS.mkdir(parents=True, exist_ok=True)

    def slide(self, heading, kicker=None, section=""):
        slide = self.prs.slides.add_slide(self.blank)
        self.count += 1
        textbox(slide, 0.5, 0.28, 12.3, 0.7, [heading], size=24, colour=NAVY, font=TITLE_FONT, bold=True)
        if kicker:
            textbox(slide, 0.52, 0.9, 12.3, 0.5, [kicker], size=13, colour=MUTED)
        line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.52), Inches(1.4), Inches(12.8),
                                          Inches(1.4))
        line.line.color.rgb = rgb(RULE)
        line.line.width = Pt(0.75)
        tag = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(12.55), Inches(7.1), Inches(0.6), Inches(0.3))
        tag.fill.solid()
        tag.fill.fore_color.rgb = rgb(ACCENT)
        tag.line.fill.background()
        tag.shadow.inherit = False
        frame = tag.text_frame
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        p = frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = str(self.count)
        run.font.size, run.font.bold, run.font.name = Pt(11), True, BODY_FONT
        run.font.color.rgb = rgb("FFFFFF")
        if section:
            textbox(slide, 0.52, 7.12, 11.8, 0.3, [section], size=9, colour=MUTED)
        return slide

    def crop(self, path: Path) -> tuple[Path, float]:
        """The panel cropped to its non-white content (small margin), and its width / height."""
        image = Image.open(path).convert("RGB")
        array = np.asarray(image)
        ink = np.where((array < 245).any(axis=2))
        if len(ink[0]):
            pad = 8
            top, bottom = max(ink[0].min() - pad, 0), min(ink[0].max() + pad, array.shape[0])
            left, right = max(ink[1].min() - pad, 0), min(ink[1].max() + pad, array.shape[1])
            image = image.crop((left, top, right, bottom))
        name = "__".join(path.relative_to(FIG).with_suffix("").parts) + ".png"
        out = ASSETS / name
        image.save(out)
        self.used.append(path.relative_to(PACKAGE).as_posix())
        return out, image.width / image.height

    def image(self, slide, path: Path, x, y, w, h, caption=None, align="center"):
        """Fit the panel inside (x, y, w, h) keeping its aspect; optional caption under the box."""
        if not path.exists():
            raise FileNotFoundError(path)
        cropped, aspect = self.crop(path)
        ch = 0.28 if caption else 0.0
        bh = h - ch
        iw, ih = (w, w / aspect) if w / aspect <= bh else (bh * aspect, bh)
        ix = x + (w - iw) / 2 if align == "center" else x
        iy = y + (bh - ih) / 2
        slide.shapes.add_picture(str(cropped), Inches(ix), Inches(iy), Inches(iw), Inches(ih))
        if caption:
            textbox(slide, x, y + bh + 0.02, w, ch, [caption], size=10, colour=MUTED, align=PP_ALIGN.CENTER)

    def row(self, slide, paths, y, h, x0=0.5, x1=12.83, captions=None, gap=0.12):
        n = len(paths)
        w = (x1 - x0 - gap * (n - 1)) / n
        for i, path in enumerate(paths):
            self.image(slide, path, x0 + i * (w + gap), y, w, h, caption=None if captions is None else captions[i])


def arrow(group: str) -> str:
    return group.replace("->", "→")


def stem(group: str) -> str:
    return group.replace("->", "_to_")


def main() -> None:
    deck = Deck()
    main_dir, supp = FIG / "main", FIG / "supplementary"
    pairs = pd.read_csv(PACKAGE / "reports" / "PAIR_SELECTION.tsv", sep="\t")
    chosen = pairs[pairs["selected"]].set_index("group")
    slices = pd.read_csv(TAB / "representative_slices.tsv", sep="\t")
    composition = pd.read_csv(TAB / "Fig1B_composition_by_stage.tsv", sep="\t")
    rejection = pd.read_csv(TAB / "FigS2_rejection_per_pair.tsv", sep="\t")
    breakdown = pd.read_csv(TAB / "Fig5C_source_mass_breakdown.tsv", sep="\t").set_index("group")
    correspondence = pd.read_csv(TAB / "Fig4_correspondence_summary.tsv", sep="\t").set_index("group")
    shares = pd.read_csv(TAB / "FigS5_same_annotation_shares.tsv", sep="\t")
    spread = pd.read_csv(TAB / "FigS5_spread_and_coverage.tsv", sep="\t")
    asymmetry = pd.read_csv(TAB / "FigS3_asymmetry_correlations.tsv", sep="\t")
    agreement = pd.read_csv(TAB / "FigS7_m4e_m4r_agreement.tsv", sep="\t")
    warnings = pd.read_csv(TAB / "FigS6_m4e_inference_invalid_warnings.tsv", sep="\t")

    adjacent = rejection[rejection["kind"] == "adjacent"]
    same = rejection[rejection["kind"] == "same_stage"]
    every = rejection[rejection["source_final_rejection_rate"] >= 1.0]
    order = ["E9.5->E10.5", "E10.5->E11.5", "E11.5->E12.5", "E12.5->E13.5", "E13.5->E14.5", "E14.5->E15.5",
             "E15.5->E16.5"]
    med = adjacent.groupby("group")[["source_final_rejection_rate", "target_final_rejection_rate"]].median()
    share_med = shares.groupby("group")[["ConfidenceOT, retained bins", "balanced OT, same retained bins",
                                         "balanced OT, rejected bins"]].median()
    narrower = int((spread["median_spread_co"] < spread["median_spread_bot_same_bins"]).sum())

    # 1 title
    slide = deck.slide("MOSTA 小鼠胚胎 E9.5–E16.5：ConfidenceOT 的空间对应",
                       "所有图来自实际运行结果；每个 panel 单独成图，这里只做排版")
    textbox(slide, 0.6, 1.8, 12, 4.5, [
        ("数据：MOSTA Stereo-seq，53 张切片，8 个发育时期", {"bullet": True}),
        (f"比较：{len(adjacent)} 个相邻时期 pair + {len(same)} 个同期对照 pair，每个 pair 都跑 M4-E、M4-R 和 balanced OT",
         {"bullet": True}),
        ("表示：ranknm256_noscale_ds_cos（统一深度 → 每个 bin 前 256 个基因的秩 → 2,000 基因 → PCA 30 → 余弦代价）",
         {"bullet": True}),
        ("闸门：M4-E，within-side acceptance 0.99 校准", {"bullet": True}),
        ("运行：mosta_all_stages（10 月初 ds 正式结果）；图由 30_mosta_figures.py 生成", {"bullet": True}),
        ("读法：拒绝 = 这次比较、这个表示和校准下没有可支持的对应；不等于没有发育后代", {"bullet": True, "bold": True}),
    ], size=17, after=10)

    # 2 overview
    per_stage = composition.set_index("stage")["analysed"]
    slide = deck.slide(f"{int(composition['slices'].sum())} 张切片覆盖 E9.5–E16.5，每期分析 bin 数从 "
                       f"{int(per_stage.iloc[0]):,} 增到 {int(per_stage.iloc[-1]):,}",
                       "A：每个时期一张代表切片，同一比例尺；B：切片、bin 和比较数；C：分析流程", "Fig. 1")
    d = main_dir / "Fig1_embryogenesis_overview"
    deck.image(slide, d / "Fig1A_stage_series.png", 0.5, 1.5, 8.3, 2.9)
    deck.image(slide, d / "Fig1A_legend_annotations.png", 8.9, 1.55, 3.95, 2.8)
    deck.row(slide, [d / "Fig1B_sections_per_stage.png", d / "Fig1B_bins_per_stage.png", d / "Fig1B_comparisons.png"],
             4.45, 1.55, x1=8.8)
    deck.image(slide, d / "Fig1C_workflow.png", 0.5, 6.05, 8.3, 1.0)
    first, last = composition.iloc[0], composition.iloc[-1]
    textbox(slide, 9.0, 4.5, 3.85, 2.5, [
        f"{first['stage']}：{int(first['analysed']):,} 个分析 bin",
        f"{last['stage']}：{int(last['analysed']):,} 个分析 bin",
        f"合计 {int(composition['analysed'].sum()):,} 个分析 bin，Cavity {int(composition['cavity'].sum()):,} 个排除",
        "代表切片规则：注释种类最多，然后 bin 最多，然后按名字",
    ], size=12, colour=INK, after=6)

    # 3 atlas
    slide = deck.slide("代表切片的注释图谱", "每张切片用自己的比例尺；Cavity 为灰色；彼此是不同胚胎的切片，不是 3D 重建",
                       "Fig. 2")
    d = main_dir / "Fig2_spatial_atlas"
    atlas = sorted(d.glob("Fig2_E*.png"), key=lambda p: float(p.stem.split("_")[1][1:]))
    deck.row(slide, atlas, 1.5, 4.35)
    deck.image(slide, d / "Fig2_legend_annotations.png", 1.5, 5.9, 10.3, 1.15)

    # 4 selection
    slide = deck.slide("展示用 pair 的选择规则：组内 M4-E source 拒绝率的中位数",
                       "规则固定、事先写好；所有候选在 reports/PAIR_SELECTION.tsv 里，未被选中的照样参与汇总", "选择规则")
    rows = [("组", "角色", "pair", "source 拒绝", "target 拒绝", "名次")]
    for group in chosen.index:
        r = chosen.loc[group]
        rows.append((arrow(group), r["role"], r["pair_id"].replace("__", " → "), f"{r['m4e_source_rejected']:.3f}",
                     f"{r['m4e_target_rejected']:.3f}", f"{int(r['rank'])}/{int(r['pairs_in_group'])}"))
    table = slide.shapes.add_table(len(rows), 6, Inches(0.6), Inches(1.7), Inches(12.1), Inches(0.42 * len(rows))).table
    for j, width in enumerate((1.5, 1.8, 4.6, 1.4, 1.4, 1.4)):
        table.columns[j].width = Inches(width)
    for i, values in enumerate(rows):
        for j, value in enumerate(values):
            cell = table.cell(i, j)
            cell.text = str(value)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size, run.font.name = Pt(13), BODY_FONT
                    run.font.bold = i == 0
    textbox(slide, 0.6, 1.9 + 0.42 * len(rows), 12, 1.5, [
        "偶数个候选时取较低的中间值，并列按 pair id；注释数和 bin 数只报告，不参与选择",
        "相邻组分早、中、晚三个时期；同期对照取 E11.5 和 E14.5",
    ], size=13, colour=MUTED)

    # 5 rejection bars
    d = main_dir / "Fig3_spatial_retain_reject"
    bars = pd.read_csv(TAB / "Fig3_rejection_bar.tsv", sep="\t")
    src_adj = bars[(bars["kind"] == "adjacent") & (bars["side"] == "source")].set_index("group")["mean"]
    slide = deck.slide(f"每个 bin 被拒绝的比例：相邻时期 source 平均 {src_adj.min():.2f}–{src_adj.max():.2f}",
                       "柱 = 所有 pair 的平均拒绝率，误差线 = SD，点 = 每个 pair；左相邻时期，右同期对照；空心圆 = source 全部被拒绝",
                       "Fig. 3")
    deck.row(slide, [d / "Fig3_rejection_bar_adjacent.png", d / "Fig3_rejection_bar_same_stage.png"], 1.6, 5.3)

    # 6-10 retain / reject maps
    for group, r in chosen.iterrows():
        folder = d / (f"{stem(group)}_adjacent" if r["kind"] == "adjacent" else f"{group}_same_stage")
        src, tgt = r["source_sample"], r["target_sample"]
        label = f"{r['role']} {arrow(group)}" if r["kind"] == "adjacent" else f"同期对照 {group}"
        slide = deck.slide(f"{label}：source 拒绝 {r['m4e_source_rejected']:.3f}，target 拒绝 {r['m4e_target_rejected']:.3f}",
                           f"{src} → {tgt}；左两张是 source 的注释和闸门，右两张是 target 的", "Fig. 3")
        deck.row(slide, [folder / f"Fig3_source_annotation_{src}.png", folder / f"Fig3_source_m4e_gate_{src}.png",
                         folder / f"Fig3_target_annotation_{tgt}.png", folder / f"Fig3_target_m4e_gate_{tgt}.png"],
                 1.5, 4.6, captions=["source 注释", "source M4-E", "target 注释", "target M4-E"])
        deck.image(slide, d / "Fig3_legend_annotations.png", 0.5, 6.15, 8.6, 0.9)
        deck.image(slide, d / "Fig3_legend_status.png", 9.2, 6.3, 3.6, 0.6)

    # 10-12 correspondence
    d = main_dir / "Fig4_spatial_correspondence"
    for group in [g for g in chosen.index if chosen.loc[g, "kind"] == "adjacent"]:
        r, c = chosen.loc[group], correspondence.loc[group]
        folder = d / stem(group)
        slide = deck.slide(f"{arrow(group)}：保留 bin 在 target 上的对应位置",
                           f"ConfidenceOT 对 balanced OT（同一批 bin）；{r['pair_id'].replace('__', ' → ')}；保留 "
                           f"{c['source_retained']:.3f}，有支持质量的 "
                           f"{int(c['retained_with_supported_mass']):,} 个 bin；伙伴离散度中位数 "
                           f"{c['median_spread_confidenceot']:.1f} 对 {c['median_spread_balanced_same_bins']:.1f}", "Fig. 4")
        deck.image(slide, folder / "Fig4A_source_retained.png", 0.5, 1.5, 3.0, 4.6, caption="A source：保留的 bin")
        deck.image(slide, folder / "Fig4B_co_partner_centroids.png", 3.6, 1.5, 2.9, 2.3, caption="B ConfidenceOT 伙伴质心")
        deck.image(slide, folder / "Fig4C_pushforward_co.png", 6.6, 1.5, 2.9, 2.3, caption="C ConfidenceOT 推前密度")
        deck.image(slide, folder / "Fig4D_bot_partner_centroids.png", 3.6, 3.85, 2.9, 2.3, caption="D balanced OT 伙伴质心")
        deck.image(slide, folder / "Fig4E_pushforward_bot.png", 6.6, 3.85, 2.9, 2.3, caption="E balanced OT 推前密度")
        deck.image(slide, folder / "Fig4F_partner_spread.png", 9.65, 1.6, 3.2, 2.6, caption="F 伙伴离散度")
        deck.image(slide, d / "Fig4_legend_annotations.png", 0.5, 6.2, 8.9, 0.85)
        deck.image(slide, d / "Fig4_legend_context.png", 9.65, 4.5, 3.2, 0.5)
        textbox(slide, 9.65, 5.1, 3.2, 1.9, [
            "质心是 transport 加权的平均位置，不是追踪到的位置",
            "C、E 共用一个色标；两种方法推前的总质量相同",
        ], size=11, colour=MUTED, after=4)

    # 13 heatmaps
    d = main_dir / "Fig5_annotation_transitions"
    heat_group = next(p.stem.split("heatmap_co_")[1] for p in d.glob("Fig5A_heatmap_co_*.png"))
    slide = deck.slide(f"{heat_group.replace('_to_', '→')}：每个 source 注释的质量去了哪个 target 注释",
                       "左：balanced OT，全部质量；右：ConfidenceOT，保留的质量加上拒绝的一列。同一个量：source 注释质量的比例",
                       "Fig. 5A")
    deck.row(slide, [d / f"Fig5A_heatmap_bot_{heat_group}.png", d / f"Fig5A_heatmap_co_{heat_group}.png"], 1.5, 5.5,
             captions=["balanced OT", "ConfidenceOT (M4-E)"])

    # 14-16 alluvials
    for group in [g for g in chosen.index if chosen.loc[g, "kind"] == "adjacent"]:
        b = breakdown.loc[group]
        slide = deck.slide(f"{arrow(group)}：注释之间的流向，组内所有 pair 的平均",
                           f"ConfidenceOT 中 source 质量：同注释保留 {b['retained_same']:.3f}，其他注释保留 "
                           f"{b['retained_other']:.3f}，拒绝 {b['rejected']:.3f}", "Fig. 5B")
        deck.row(slide, [d / f"Fig5B_alluvial_bot_{stem(group)}.png", d / f"Fig5B_alluvial_co_{stem(group)}.png"],
                 1.5, 5.5, captions=["balanced OT", "ConfidenceOT (M4-E)，深灰为拒绝"])

    # 17 source / target summary
    early, late = breakdown.loc[order[:3], "rejected"], breakdown.loc[order[3:], "rejected"]
    slide = deck.slide(f"source 质量的拒绝比例：早期 {early.min():.2f}–{early.max():.2f}，E12.5 之后 "
                       f"{late.min():.2f}–{late.max():.2f}",
                       "source 和 target 分开画，不当作一个守恒的流；空心圆 = 所有 source bin 都被拒绝的 pair", "Fig. 5C–D")
    deck.row(slide, [d / "Fig5C_source_mass_breakdown.png", d / "Fig5D_target_rejection.png"], 1.5, 4.7)
    textbox(slide, 0.6, 6.25, 12.2, 0.8, [
        f"{len(every)} 个 pair 的 source 全部被拒绝（{', '.join(every['pair_id'].unique())}）。没有保留的 source，"
        "就没有任何被断言的对应，所以它们 target 的拒绝率 0 不是匹配率；照样画出，没有删除。",
    ], size=11, colour=INK)

    # 18 what the figures say
    slide = deck.slide("从这些图能读出的和不能读出的", None, "小结")
    rows = []
    for group in order:
        rows.append(f"{arrow(group)}：source 拒绝中位数 {med.loc[group, 'source_final_rejection_rate']:.3f}，target "
                    f"{med.loc[group, 'target_final_rejection_rate']:.3f}；同注释份额 ConfidenceOT "
                    f"{share_med.loc[group, 'ConfidenceOT, retained bins']:.2f}，balanced OT 同一批 bin "
                    f"{share_med.loc[group, 'balanced OT, same retained bins']:.2f}，balanced OT 在被拒 bin 上 "
                    f"{share_med.loc[group, 'balanced OT, rejected bins']:.2f}")
    same_med = same.groupby("group")["source_final_rejection_rate"].median()
    rho = asymmetry.set_index(["kind", "variable"])["spearman_rho"]
    textbox(slide, 0.6, 1.6, 12.2, 5.4, [("能读出的", {"bold": True, "colour": NAVY, "size": 15})] +
            [(t, {"bullet": True, "size": 11.5}) for t in rows] + [
        (f"在保留的 bin 上，ConfidenceOT 的伙伴离散度比 balanced OT 窄：{narrower}/{len(spread)} 个相邻 pair", {"bullet": True}),
        ("不能读出的 / 要注意的", {"bold": True, "colour": NAVY, "size": 15}),
        (f"同期对照也拒绝很多：source 拒绝中位数 {same_med.min():.2f}–{same_med.max():.2f}。拒绝反映的不只是发育变化，"
         "还有不同胚胎、不同切面之间的差异", {"bullet": True}),
        (f"source 拒绝率总是高于 target；和 bin 数之比几乎无关（相邻 ρ = {rho[('adjacent', 'log2_size_ratio')]:.2f}），"
         f"和注释种类之差弱相关（相邻 ρ = {rho[('adjacent', 'annotation_difference')]:.2f}）；只是相关，不是原因",
         {"bullet": True}),
        ("同注释份额衡量的是一致性，不是真值；拒绝不证明没有发育后代", {"bullet": True}),
    ], size=12.5, after=3)

    # supplementary: S1 atlas, two slides
    d = supp / "FigS1_all_stage_atlas"
    stages = sorted(d.glob("FigS1_E*.png"), key=lambda p: float(p.stem.split("_E")[1]))
    chunks, pending = [], []
    for path in stages:
        width, height = Image.open(path).size
        if width / height < 3:
            chunks += [pending] if pending else []
            chunks.append([path])
            pending = []
        else:
            pending.append(path)
            if len(pending) == 2:
                chunks.append(pending)
                pending = []
    chunks += [pending] if pending else []
    for part, chunk in enumerate(chunks):
        names = "、".join(p.stem.split("_")[1] for p in chunk)
        slide = deck.slide(f"全部切片：{names}", f"每个时期一张网格，切片按名字排列（{part + 1}/{len(chunks)}）", "Fig. S1")
        step = 4.45 / len(chunk)
        for i, path in enumerate(chunk):
            deck.image(slide, path, 0.5, 1.5 + i * step, 12.33, step - 0.05)
        deck.image(slide, d / "FigS1_legend_annotations.png", 1.5, 6.0, 10.3, 1.05)

    d = supp / "FigS2_rejection_summary"
    slide = deck.slide("每个 pair 的 M4-E 拒绝率", "上：相邻时期；下：同期对照；左 source，右 target；空心圆 = source 全部被拒绝",
                       "Fig. S2")
    deck.row(slide, [d / "FigS2_adjacent_source.png", d / "FigS2_adjacent_target.png"], 1.5, 2.75)
    deck.row(slide, [d / "FigS2_same_stage_source.png", d / "FigS2_same_stage_target.png"], 4.3, 2.75)

    d = supp / "FigS3_source_target_asymmetry"
    slide = deck.slide("source 拒绝率高于 target：每个 pair 都在对角线上方", "B、C 只是相关，不是原因", "Fig. S3")
    deck.row(slide, [d / "FigS3A_source_vs_target.png", d / "FigS3B_log2_size_ratio.png",
                     d / "FigS3C_annotation_difference.png"], 1.6, 4.6)
    textbox(slide, 0.6, 6.35, 12.2, 0.7, ["  ".join(
        f"{k.replace('_', '-')} / {v}: ρ = {x:.2f}, p = {p:.2g}" for k, v, x, p in
        asymmetry[["kind", "variable", "spearman_rho", "p_value"]].itertuples(index=False))], size=11, colour=MUTED)

    d = supp / "FigS4_annotation_rejection"
    slide = deck.slide("按注释的拒绝率，分开伙伴切片有没有这个注释", "上 source，下 target；左：伙伴切片没有这个注释，右：有；灰 = 没有这样的 pair",
                       "Fig. S4")
    deck.row(slide, [d / "FigS4_source_partner_absent.png", d / "FigS4_source_partner_present.png",
                     d / "FigS4_target_partner_absent.png", d / "FigS4_target_partner_present.png"], 1.5, 5.5,
             captions=["source，伙伴没有", "source，伙伴有", "target，伙伴没有", "target，伙伴有"])

    d = supp / "FigS5_balanced_comparison"
    slide = deck.slide("同一批保留的 bin 上，ConfidenceOT 和 balanced OT 的比较",
                       "A 同注释份额；B 伙伴离散度（每个 pair 的中位数）；C 支持质量覆盖", "Fig. S5")
    deck.image(slide, d / "FigS5A_same_annotation_share.png", 0.5, 1.6, 6.2, 4.0)
    deck.image(slide, d / "FigS5B_partner_spread.png", 6.85, 1.6, 2.9, 4.0)
    deck.image(slide, d / "FigS5C_supported_mass.png", 9.9, 1.6, 2.9, 4.0)
    textbox(slide, 0.6, 5.85, 12.2, 1.1, [
        f"B：{narrower}/{len(spread)} 个相邻 pair 中 ConfidenceOT 的伙伴离散度更窄（点在对角线下方）",
        f"A：balanced OT 在 M4-E 拒绝的 bin 上同注释份额中位数 {share_med['balanced OT, rejected bins'].min():.2f}–"
        f"{share_med['balanced OT, rejected bins'].max():.2f}，在保留的 bin 上 "
        f"{share_med['balanced OT, same retained bins'].min():.2f}–{share_med['balanced OT, same retained bins'].max():.2f}："
        "被拒的 bin 即使强行配对，注释也很少对得上",
    ], size=12)

    d = supp / "FigS6_calibration_diagnostics"
    invalid = int(warnings["pairs"].max()) if len(warnings) else 0
    slide = deck.slide("校准和收敛诊断", f"F：m4e_inference_valid 为 false 的 {invalid} 个 pair，记录的警告全部来自 M4-R 的验证",
                       "Fig. S6")
    deck.row(slide, [d / "FigS6A_rejection_cost.png", d / "FigS6B_within_side_acceptance.png",
                     d / "FigS6C_calibration_flags.png"], 1.5, 2.75)
    deck.row(slide, [d / "FigS6D_M4-E_convergence.png", d / "FigS6E_M4-R_convergence.png",
                     d / "FigS6F_inference_validity_warnings.png"], 4.3, 2.75)

    d = supp / "FigS7_m4e_vs_m4r_gates"
    gates = main_dir / "Fig3_spatial_retain_reject"
    for group, r in chosen.iterrows():
        src, tgt = r["source_sample"], r["target_sample"]
        folder = d / r["pair_id"]
        g3 = gates / (f"{stem(group)}_adjacent" if r["kind"] == "adjacent" else f"{group}_same_stage")
        a = agreement[agreement["pair_id"] == r["pair_id"]].set_index("side")
        slide = deck.slide(f"M4-E 对 M4-R：{arrow(group)}", f"{r['pair_id'].replace('__', ' → ')}；拒绝集合的 Jaccard："
                           f"source {a.loc['source', 'jaccard_rejected']:.2f}，target {a.loc['target', 'jaccard_rejected']:.2f}",
                           "Fig. S7")
        deck.row(slide, [g3 / f"Fig3_source_m4e_gate_{src}.png", folder / f"FigS7_source_m4r_gate_{src}.png",
                         g3 / f"Fig3_target_m4e_gate_{tgt}.png", folder / f"FigS7_target_m4r_gate_{tgt}.png"], 1.5, 5.0,
                 captions=["source M4-E", "source M4-R", "target M4-E", "target M4-R"])
        deck.image(slide, gates / "Fig3_legend_status.png", 3.0, 6.55, 7.3, 0.45)

    d = supp / "FigS8_annotation_heatmaps"
    for group in order:
        slide = deck.slide(f"{arrow(group)}：注释对应热图", "左 balanced OT，右 ConfidenceOT（含拒绝一列）；同一个量", "Fig. S8")
        deck.row(slide, [d / f"FigS8_heatmap_bot_{stem(group)}.png", d / f"FigS8_heatmap_co_{stem(group)}.png"], 1.5,
                 5.5, captions=["balanced OT", "ConfidenceOT (M4-E)"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    deck.prs.save(OUT)
    every_png = sorted(p.relative_to(PACKAGE).as_posix() for p in FIG.rglob("*.png"))
    missing = sorted(set(every_png) - set(deck.used))
    print(f"{OUT}: {deck.count} slides, {len(set(deck.used))}/{len(every_png)} panels placed")
    for path in missing:
        print("  not placed:", path)


if __name__ == "__main__":
    main()
