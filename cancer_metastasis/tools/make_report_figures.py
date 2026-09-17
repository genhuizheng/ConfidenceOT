"""Draw the figures for the PI report from the collected summary tables.

Input is the directory unpacked from ``collect_report_inputs.sh``. Every figure
is skipped with a note rather than crashing when its table is absent, because
which arms exist depends on how far the runs have got.

Design choices that are not taste. Colours come from a validated categorical
order, assigned in fixed slots and never cycled; scatter figures carry at most
three series because the full order does not clear colour-blind separation on
all pairs. Every series is directly labelled as well as legended, so identity
never rests on colour alone. There is one y axis per panel. Grid and axes are
recessive so the marks carry the eye.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical order, light surface. Slots are assigned by identity and
# never by rank, so a series keeps its colour across figures.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA = "#eda100", "#e87ba4"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
SURFACE = "#fcfcfb"

# Two androgen sets sit in the same figure and disagree, so each panel has to
# name the genes it scored rather than both reading "androgen".
SCORE_LABELS = {
    "androgen_signalling": "androgen targets\n(KLK3, NKX3-1, ...)",
    "androgen_response": "androgen response\n(HALLMARK)",
    "cell_division": "cell division",
    "epithelial_mesenchymal_transition": "EMT\n(HALLMARK)",
    "interferon_gamma_response": "interferon gamma\n(HALLMARK)",
}


def readable_limits(axis, values, *, separation=4.0, pad=0.12):
    """Rescale only when one point is far enough out to flatten the rest.

    One lymph node has lost its androgen targets almost completely, and on a
    full-range axis that single point flattens the four kept-cell differences
    the panel exists to show. Dropping it would be dishonest and keeping the
    axis wide makes the panel unreadable, so the axis is set from the remaining
    points and the excluded one is drawn at the boundary with its real value.

    With four patients a percentile rule would exclude the extremes by
    construction, so the test is relative instead: rescale only if the largest
    deviation from the median exceeds `separation` times the next largest. That
    fires on a point an order of magnitude out and leaves an ordinary spread
    alone.
    """
    values = np.asarray([v for v in values if np.isfinite(v)], float)
    if values.size < 4:
        return None
    deviation = np.abs(values - np.median(values))
    order = np.argsort(deviation)[::-1]
    largest, runner_up = deviation[order[0]], deviation[order[1]]
    if runner_up <= 0 or largest < separation * runner_up:
        return None
    inner = np.delete(values, order[0])
    low, high = float(inner.min()), float(inner.max())
    span = high - low or abs(high) or 1.0
    lower, upper = min(low - pad * span, 0.0), max(high + pad * span, 0.0)
    axis.set_ylim(lower, upper)
    return lower, upper


# The slide background, so a figure leaves no visible seam against it.
PAPER = "#ffffff"

# The embedding fills its panel now, so an in-panel label would otherwise land
# on the cells it is describing.
LABEL_BOX = dict(facecolor=PAPER, edgecolor="none", alpha=0.82, pad=1.6)

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "semibold",
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "grid.color": "#e6e6e2",
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "text.color": INK,
})


# The two schematics are the only figures with Chinese in them. Per-glyph
# fallback did not fire here, so those figures name a CJK face outright; it
# carries Latin too. The data figures keep DejaVu, which is what they were laid
# out against.
CJK = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
LATIN_FONTS = list(plt.rcParams["font.sans-serif"])


def finish(fig, path: Path, note: str | None = None) -> None:
    if note:
        # The note is placed in figure coordinates, so a two-line axis label
        # reaches down into it unless the axes are lifted first. Reserving the
        # room here fixes it once for every figure instead of per caller, and a
        # caller that already reserved more keeps its own value.
        reserved = max(fig.subplotpars.bottom, 0.2)
        if reserved > fig.subplotpars.bottom:
            fig.subplots_adjust(bottom=reserved)
        fig.text(0.01, 0.01, note, fontsize=10, color=MUTED, ha="left",
                 va="bottom")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    # A figure that asked for the CJK face gets it back to the Latin default
    # here, so one schematic cannot change the face of the figures after it
    # even if it raised on the way.
    plt.rcParams["font.sans-serif"] = LATIN_FONTS
    print(f"  wrote {path.name}")


def read(source: Path, name: str) -> pd.DataFrame | None:
    path = source / name
    if not path.exists():
        print(f"  skip: {name} absent")
        return None
    # "NA" is a category in cell_subtype, not a missing value.
    return pd.read_csv(path, keep_default_na=(name != "ov_gate_by_subtype.csv.gz"))


def gate_median(source: Path, name: str, statistic: str) -> float | None:
    table = read(source, name)
    if table is None or len(table.columns) < 2:
        return None
    key = table.columns[1]
    row = table.loc[table[key].eq(statistic), "median"]
    return float(row.iloc[0]) if len(row) else None


def figure_depth(source: Path, out: Path) -> None:
    """How much of the gate is sequencing depth, per dataset and setting."""
    wanted = [
        ("Ovarian\nstandard", "ov_gate_logcpm_summary.csv", "auc_predownsample_total_counts"),
        ("Ovarian\n+ rank", "ov_gate_rank256_summary.csv", "auc_predownsample_total_counts"),
        ("Prostate\nstandard", "pca_gate_logcpm_free_ds_summary.csv", "auc_total_counts"),
        ("Prostate\n+ rank", "pca_gate_rank_free_ds_summary.csv", "auc_total_counts"),
        ("Prostate\nno depth fix", "pca_gate_rank_free_raw_summary.csv", "auc_total_counts"),
    ]
    labels, values = [], []
    for label, name, statistic in wanted:
        value = gate_median(source, name, statistic)
        if value is not None:
            labels.append(label)
            values.append(value)
    if not values:
        print("  skip: no depth summaries")
        return

    # Plotted as the size of the depth effect, so zero is the goal and shorter
    # is better. The signed AUC needs the reader to hold "0.5 is perfect and
    # either direction is bad" in mind, which is one thought too many.
    sizes = [abs(v - 0.5) for v in values]
    fig, axis = plt.subplots(figsize=(9.5, 5.0))
    bars = axis.bar(labels, sizes, color=BLUE, width=0.6, zorder=3)
    for bar, size in zip(bars, sizes):
        axis.text(bar.get_x() + bar.get_width() / 2, size + 0.012, f"{size:.2f}",
                  ha="center", fontsize=14, color=INK, fontweight="semibold")
    axis.set_ylim(0, max(sizes) + 0.09)
    axis.set_ylabel("size of the depth effect\n(0 = none)")
    axis.set_title("How much of the gate was sequencing depth")
    axis.grid(axis="y", zorder=0)
    axis.set_axisbelow(True)
    finish(fig, out / "fig1_depth_in_gate.png", "Shorter is better. Zero is perfect.")


def figure_representation(source: Path, out: Path) -> None:
    """Simulated data with a known answer: which preprocessing removes depth."""
    arms = ["homogeneous_depth_cv0", "homogeneous_depth_cv_low",
            "homogeneous_depth_cv_mid", "homogeneous_depth_cv_high"]
    pretty = ["none", "low", "medium", "high"]
    # One thing that worked, in colour; three that did not, in grey. The eye
    # should land on the line that goes to zero without reading a legend first.
    series = [
        ("gene ranking", "sim_rank256_arms.csv", ORANGE, 2.6),
        ("standard", "sim_log_cpm_arms.csv", MUTED, 1.8),
        ("ranking, no gene median", "sim_rank_no_median_arms.csv", "#b9b9b3", 1.6),
        ("Pearson residuals", "sim_pearson_residuals_arms.csv", "#b9b9b3", 1.6),
    ]
    fig, axis = plt.subplots(figsize=(10.4, 5.4))
    placed, f1_note = [], ""
    for label, name, colour, width in series:
        table = read(source, name)
        if table is None or "auc_total_counts" not in table:
            continue
        table = table.set_index("arm")
        if not set(arms) <= set(table.index):
            continue
        # The size of the effect, so every line going down is an improvement.
        sizes = np.abs(table.loc[arms, "auc_total_counts"].to_numpy(float) - 0.5)
        axis.plot(pretty, sizes, marker="o", markersize=8, linewidth=width,
                  color=colour, zorder=3)
        placed.append((label, colour, float(sizes[-1])))
        if label == "gene ranking" and "perturbed_f1" in table:
            if "perturbed_depth_cv0" in table.index:
                f1_note = (f"Detection of a known 20% subpopulation barely moves: "
                           f"F1 {float(table.loc['perturbed_depth_cv0', 'perturbed_f1']):.2f}")
    if not placed:
        plt.close(fig)
        print("  skip: no simulation summaries")
        return

    # The three arms that fail all converge on the same value, so their end
    # labels land on top of one another and read as a smudge. Labels are spread
    # downwards to a minimum gap, and a leader is drawn when a label has been
    # moved far enough off its own line to be ambiguous.
    gap = 0.038
    placed.sort(key=lambda row: row[2], reverse=True)
    positions: list[float] = []
    for _, _, end in placed:
        target = end if not positions else min(end, positions[-1] - gap)
        positions.append(target)
    for (label, colour, end), y in zip(placed, positions):
        axis.annotate(label, (len(pretty) - 1, y), xytext=(11, 0),
                      textcoords="offset points", color=colour, fontsize=12,
                      va="center", fontweight="semibold")
        if abs(y - end) > gap / 2:
            axis.plot([len(pretty) - 1, len(pretty) - 0.88], [end, y],
                      color=colour, linewidth=0.9, zorder=2)
    axis.set_xlabel("how unequal the sequencing depth is")
    axis.set_ylabel("size of the depth effect\n(0 = none)")
    axis.set_title("Ranking genes within each cell is what removes it")
    axis.set_ylim(-0.03, 0.56)
    axis.set_xlim(-0.25, len(pretty) + 1.5)
    axis.grid(axis="y", zorder=0)
    axis.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.22)
    finish(fig, out / "fig2_representation_benchmark.png",
           f"Simulated data where the correct answer is known. {f1_note}")


def figure_similarity(source: Path, out: Path) -> None:
    """The retained fraction tracks how alike the two tissues are."""
    pairs = read(source, "ov_similarity_pairs.csv")
    if pairs is None:
        return
    needed = {"arm", "source_retained_fraction", "pseudobulk_pearson_hv"}
    if not needed <= set(pairs.columns):
        print("  skip: similarity columns absent")
        return
    report = source / "ov_similarity_report.json"
    rho = None
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        rho = data.get("results", {}).get("union", {}).get(
            "retained_vs_similarity_hv", {}).get("rho")

    fig, axis = plt.subplots(figsize=(9.2, 5.6))
    # Two series only; the validated order clears colour-blind separation on
    # all pairs for the first three slots.
    for arm, colour, label in (("matched", BLUE, "same patient"),
                               ("mismatched", ORANGE, "different patient")):
        subset = pairs[pairs["arm"].eq(arm)]
        if subset.empty:
            continue
        axis.scatter(subset["pseudobulk_pearson_hv"], subset["source_retained_fraction"],
                     s=42, color=colour, alpha=0.75, edgecolor=SURFACE,
                     linewidth=0.8, label=label, zorder=3)
    axis.set_xlabel("how alike the primary and the metastasis are\n(pseudobulk correlation)")
    axis.set_ylabel("fraction of primary cells retained")
    title = "What the retained fraction actually measures"
    if rho is not None:
        title += f"   (rho = {rho:.2f})"
    axis.set_title(title)
    axis.grid(zorder=0)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left")
    finish(fig, out / "fig3_similarity_readout.png",
           "188 tumour pairs. Each point is one primary sample against one metastasis.")


def figure_control(source: Path, out: Path) -> None:
    """Give a primary the wrong patient's metastasis and it retains nothing."""
    pairs = read(source, "ov_matched_vs_mismatched_pairs.csv")
    if pairs is None:
        return
    columns = {"retained_fraction_matched", "retained_fraction_mismatched"}
    if not columns <= set(pairs.columns):
        print("  skip: control columns absent")
        return
    matched = pairs["retained_fraction_matched"].to_numpy(float)
    mismatched = pairs["retained_fraction_mismatched"].to_numpy(float)

    fig, axis = plt.subplots(figsize=(9.2, 5.6))
    for a, b in zip(matched, mismatched):
        axis.plot([0, 1], [a, b], color=MUTED, linewidth=0.7, alpha=0.5, zorder=2)
    axis.scatter(np.zeros_like(matched), matched, s=46, color=BLUE,
                 edgecolor=SURFACE, linewidth=0.8, zorder=3)
    axis.scatter(np.ones_like(mismatched), mismatched, s=46, color=ORANGE,
                 edgecolor=SURFACE, linewidth=0.8, zorder=3)
    # Each median is labelled beside its own line rather than along the bottom,
    # where the two labels crowded the two-line category names underneath.
    for x, values, colour, side in ((0, matched, BLUE, -1), (1, mismatched, ORANGE, 1)):
        median = float(np.nanmedian(values))
        axis.plot([x - 0.16, x + 0.16], [median, median], color=INK,
                  linewidth=2.6, zorder=4)
        axis.annotate(f"median\n{median:.3f}", (x + side * 0.2, median),
                      ha="right" if side < 0 else "left", va="center",
                      fontsize=12, color=colour, fontweight="semibold")
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["own\nmetastasis", "another patient's\nmetastasis"])
    axis.set_xlim(-0.62, 1.62)
    axis.set_ylim(-0.04, 1.02)
    axis.set_ylabel("fraction of primary cells retained")
    axis.set_title("The method does use the metastasis it is given")
    axis.grid(axis="y", zorder=0)
    axis.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.2)
    finish(fig, out / "fig4_mismatched_control.png",
           "94 ovarian pairs, each line one primary sample. Paired test "
           "p = 8e-17.")


def figure_patients(source: Path, out: Path) -> None:
    """Every one of these patients already had a metastasis."""
    states = read(source, "ov_patient_states.csv")
    if states is None or "retained_fraction" not in states:
        return
    states = states.dropna(subset=["retained_fraction"]).sort_values("retained_fraction")
    lesion = states.get("lesion", pd.Series([""] * len(states))).astype(str)
    is_ascites = lesion.str.contains("ascites", case=False, na=False).to_numpy()
    colours = np.where(is_ascites, ORANGE, BLUE)

    fig, axis = plt.subplots(figsize=(12.2, 5.4))
    positions = np.arange(len(states))
    axis.bar(positions, states["retained_fraction"], color=colours, width=0.72, zorder=3)
    axis.set_xticks(positions)
    axis.set_xticklabels([str(p).replace("SPECTRUM-", "") for p in states["patient_id"]],
                         rotation=90, fontsize=9)
    axis.set_ylabel("fraction of primary cells retained")
    axis.set_title("The same measure spans 0% to 88% across patients who all had metastases")
    axis.grid(axis="y", zorder=0)
    axis.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    axis.legend(handles, ["solid metastasis", "ascites"], loc="upper left")
    finish(fig, out / "fig5_patient_spread.png",
           "29 ovarian patients, one designated metastasis each.")


def figure_gsea(source: Path, out: Path, faming: Path | None) -> None:
    """Our gate against an independent metastasis signature."""
    ours = read(source, "pca_gsea_rank256_cap085_ds.csv")
    if ours is None or "pathway" not in ours:
        return
    sets = ["ANDROGEN_RESPONSE", "EPITHELIAL_MESENCHYMAL_TRANSITION",
            "INTERFERON_GAMMA_RESPONSE", "G2M_CHECKPOINT", "E2F_TARGETS",
            "MITOTIC_SPINDLE"]
    ours = ours.assign(key=ours["pathway"].astype(str).str.replace("HALLMARK_", "", regex=False))
    ours = ours.set_index("key")
    # Our contrast is rejected-versus-retained, so flip the sign to express it
    # as "enriched in the retained, metastasis-like cells" and put it on the
    # same axis as a metastasis-versus-primary comparison.
    mine = {s: -float(ours.loc[s, "NES"]) for s in sets if s in ours.index}

    theirs: dict[str, float] = {}
    if faming and faming.exists():
        table = pd.read_excel(faming, sheet_name="GSEA_Hallmark_DESeq2", engine="openpyxl")
        table["key"] = (table["ID"].astype(str).str.replace("HALLMARK_", "", regex=False)
                        .str.replace("HALLMARK", "", regex=False).str.strip("_"))
        table = table.set_index("key")
        theirs = {s: float(table.loc[s, "NES"]) for s in sets if s in table.index}

    keys = [s for s in sets if s in mine]
    if not keys:
        print("  skip: no matching pathways")
        return
    labels = [k.replace("_", " ").replace("EPITHELIAL MESENCHYMAL TRANSITION", "EMT")
              .replace("INTERFERON GAMMA RESPONSE", "interferon")
              .replace("ANDROGEN RESPONSE", "androgen response")
              .replace("G2M CHECKPOINT", "G2M checkpoint")
              .replace("E2F TARGETS", "E2F targets")
              .replace("MITOTIC SPINDLE", "mitotic spindle") for k in keys]
    positions = np.arange(len(keys))
    height = 0.36

    fig, axis = plt.subplots(figsize=(10.6, 5.8))
    if theirs:
        axis.barh(positions + height / 2 + 0.01, [theirs.get(k, np.nan) for k in keys],
                  height=height, color=BLUE, label="tissue comparison (reference)", zorder=3)
    axis.barh(positions - height / 2 - 0.01, [mine[k] for k in keys],
              height=height, color=ORANGE, label="our retained cells", zorder=3)
    axis.axvline(0, color=INK2, linewidth=1.2, zorder=2)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("enrichment in metastasis / in retained cells  (NES)"
                    if theirs else "enrichment in the retained cells  (NES)")
    # Without the reference workbook there is only one set of bars, and a title
    # claiming agreement would not be supported by what the figure shows.
    axis.set_title("Androgen response agrees; proliferation is exactly reversed"
                   if theirs else
                   "Proliferation dominates what separates the retained cells")
    axis.grid(axis="x", zorder=0)
    axis.set_axisbelow(True)
    # A single series needs no legend box; the title already names it. With
    # both series the bars run left and right on every row, so there is no
    # free corner -- the legend goes under the axis instead of over a bar.
    if theirs:
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    # Androgen is the only agreement in the panel and also the weakest call on
    # our side, so the reader is given the FDR rather than left to weigh a bar
    # length against five others.
    fdr = {k: float(ours.loc[k, "fdr"]) for k in keys if "fdr" in ours.columns}
    detail = ",  ".join(f"{label} {fdr[k]:.3f}" for k, label in zip(keys, labels)
                        if k in fdr)
    fig.subplots_adjust(bottom=0.32 if theirs else 0.24)
    finish(fig, out / "fig6_gsea_agreement.png",
           ("Both bars point the same way when our retained cells resemble the "
            "metastasis.\n" if theirs else "")
           + f"Our FDR — {detail}.")


def figure_cellcycle(source: Path, out: Path) -> None:
    """The retained cells are the dividing cells."""
    table = read(source, "pca_deg_rank256_cap085_ds_discovery.csv")
    if table is None or "log2_fold_change" not in table:
        return
    significant = table[table["fdr"].lt(0.05)]
    top = significant.nsmallest(18, "log2_fold_change").iloc[::-1]
    cycle = top["gene"].astype(str).str.match(
        r"^(DIAPH3|TOP2A|NUSAP1|KIF11|ASPM|CENP[EF]|MKI67|STMN1|UBE2S|HELLS|BRIP1|"
        r"TPX2|CCNB\d|CDK1|AURK|PLK1|BUB1|TYMS|RRM2|NDC80|SMC\d)$")
    colours = np.where(cycle, ORANGE, MUTED)

    fig, axis = plt.subplots(figsize=(9.4, 6.4))
    magnitude = -top["log2_fold_change"].to_numpy(float)
    axis.barh(np.arange(len(top)), magnitude, color=colours, height=0.68, zorder=3)
    axis.set_yticks(np.arange(len(top)))
    axis.set_yticklabels(top["gene"].astype(str), fontsize=11)
    axis.set_xlabel("enrichment in the retained cells  (log2 fold change)")
    axis.set_title("The retained cells are the dividing cells")
    axis.grid(axis="x", zorder=0)
    axis.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE),
               plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    axis.legend(handles, ["cell-division gene", "other"], loc="lower right")
    finish(fig, out / "fig7_cell_cycle.png",
           # Signed the same way as the enrichment figure -- positive means
           # enriched in the retained cells -- so the two slides cannot appear
           # to disagree about the direction.
           "Prostate, depth-corrected arm. Gene-set test: E2F targets "
           "NES +3.4 in the retained cells, FDR < 0.001.")


def figure_methods_cartoon(out: Path) -> None:
    """What each preprocessing actually does to a deep and a shallow cell.

    Also a schematic. Two cells with identical biology are drawn over six
    genes, one sequenced 1.3x deeper than the other. The sixth gene is the
    point: it is lowly expressed, the deep cell detects it and the shallow cell
    reads zero. Dividing by the cell total fixes the scale but not that
    dropout, which is why log-CPM leaves a depth effect behind and why the two
    methods that survived do something else.
    """
    genes = np.arange(6)
    share = np.array([0.34, 0.24, 0.19, 0.12, 0.08, 0.03])
    deep = np.array([34.0, 24.0, 19.0, 12.0, 8.0, 3.0])
    shallow = np.array([26.0, 18.0, 14.0, 9.0, 6.0, 0.0])

    def cpm(counts):
        return np.log1p(counts / counts.sum() * 100)

    def equalised(counts, target):
        # Reads subsampled to a common total, so the shallow cell's dropout is
        # imposed on the deep one too.
        scaled = counts / counts.sum() * target
        return np.where(scaled < 0.6, 0.0, scaled)

    def ranked(counts):
        # Within the cell, by level relative to that gene, keeping the top four.
        scaled = counts / counts.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.where(share > 0, scaled / share, 0.0)
        order = np.argsort(np.argsort(-relative))
        return np.where(counts > 0, np.maximum(4 - order, 0), 0.0)

    def residual(counts):
        expected = counts.sum() * share
        return (counts - expected) / np.sqrt(np.maximum(expected, 1e-9))

    panels = [
        ("原始 counts", deep, shallow,
         "两个细胞生物学相同，\n"
         "只是测序深度差 1.3×。\n"
         "第 6 个基因表达低，\n"
         "浅的细胞读到 0。", None),
        ("log-CPM", cpm(deep), cpm(shallow),
         "标准做法：除以细胞\n"
         "总数再取 log。\n"
         "前五个基因对齐了，\n"
         "但第 6 个的 0 还在。", False),
        ("downsample", equalised(deep, shallow.sum()), shallow,
         "把深的细胞随机抽到\n"
         "和浅的一样的总 reads。\n"
         "深度相同后检出概率也\n"
         "相同，剩下的只是随机。",
         True),
        ("gene rank", ranked(deep), ranked(shallow),
         "每个细胞内按“相对该\n"
         "基因自身的异常程度”\n"
         "排序，只用排名。\n"
         "数值尺度完全消失。", True),
        ("Pearson residuals", residual(deep), residual(shallow),
         "用深度和基因均值算\n"
         "“应该测到多少”，\n"
         "再看实测差多少。\n"
         "掉到 0 的基因仍留大残差。",
         False),
    ]

    plt.rcParams["font.sans-serif"] = CJK
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 5.1))
    width = 0.38
    for axis, (title, a, b, blurb, works) in zip(axes, panels):
        axis.bar(genes - width / 2, a, width, color=ORANGE, zorder=3,
                 label="深")
        axis.bar(genes + width / 2, b, width, color=MUTED, zorder=3,
                 label="浅")
        axis.axhline(0, color=INK2, linewidth=1.0, zorder=2)
        axis.set_xticks(genes)
        axis.set_xticklabels([str(g + 1) for g in genes], fontsize=11)
        axis.set_yticks([])
        axis.set_xlabel("基因", fontsize=11.5, color=INK2)
        axis.grid(axis="y", zorder=0)
        axis.set_axisbelow(True)
        for spine in ("left", "right", "top"):
            axis.spines[spine].set_visible(False)
        # Not a tick or a cross: the CJK face has no glyph for either and both
        # came out as empty boxes.
        mark = "" if works is None else ("（有效）" if works
                                         else "（无效）")
        colour = INK if works is None else (AQUA if works else "#c2453a")
        axis.set_title(title + mark, fontsize=13, color=colour)
        axis.text(0.0, -0.22, blurb, transform=axis.transAxes, ha="left",
                  va="top", fontsize=11.5, color=INK2, linespacing=1.6)
    axes[0].legend(loc="upper right", fontsize=11.5)

    fig.suptitle("三种做法到底在做什"
                 "么：两个生物学相同、"
                 "深度不同的细胞",
                 fontsize=16, fontweight="semibold", y=1.02)
    fig.subplots_adjust(bottom=0.34, wspace=0.16)
    finish(fig, out / "fig_methods_cartoon.png",
           "示意图，非数据。有效 = "
           "在模拟和真实数据上都把 "
           "depth 效应降下来了。两个"
           "有效的必须一起用：rank "
           "自己不能消除 dropout，只做 "
           "rank 不做 downsample 时 AUC 是 0.06（"
           "0.5 才是无关）。")


def figure_depth_cartoon(out: Path) -> None:
    """A schematic of how a depth difference turns into a gate decision.

    The one figure here that is not data. The ovarian row -- a difference of
    0.0063 at p = 1.9e-8 -- is only readable once the mechanism is visible, and
    prose describing it did not land. Two primary populations with the same
    biology are pulled apart by sequencing depth alone, and the nearer one is
    kept. The depths shown are the real group medians in GSE180661.
    """
    plt.rcParams["font.sans-serif"] = CJK
    fig = plt.figure(figsize=(12.4, 5.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.78], wspace=0.14)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    rng = np.random.default_rng(11)

    def blob(axis, centre, colour, n=170, spread=0.42):
        points = rng.normal(centre, spread, size=(n, 2))
        axis.scatter(points[:, 0], points[:, 1], s=24, color=colour,
                     alpha=0.75, linewidth=0, zorder=3)

    def span(axis, start, end, colour, label, lift=0.3):
        axis.annotate("", xy=end, xytext=start,
                      arrowprops=dict(arrowstyle="<->", color=colour,
                                      linewidth=1.6, shrinkA=2, shrinkB=2))
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + lift,
                  label, ha="center", va="bottom", fontsize=12.5,
                  color=colour, fontweight="semibold")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim(-0.4, 10.4)
        axis.set_ylim(-1.0, 6.0)
        for spine in axis.spines.values():
            spine.set_visible(False)

    blob(axes[0], (2.1, 3.4), ORANGE)
    blob(axes[0], (2.1, 1.4), MUTED)
    blob(axes[0], (8.2, 2.4), BLUE)
    axes[0].text(2.1, 4.6, "A  primary", ha="center", fontsize=13,
                 color=ORANGE, fontweight="semibold")
    axes[0].text(2.1, 0.0, "B  primary", ha="center", fontsize=13,
                 color=MUTED, fontweight="semibold")
    axes[0].text(8.2, 3.7, "C  metastasis", ha="center", fontsize=13,
                 color=BLUE, fontweight="semibold")
    span(axes[0], (3.1, 3.3), (7.2, 2.6), INK, "距离相同")
    span(axes[0], (3.1, 1.5), (7.2, 2.2), INK, "", lift=0)
    axes[0].set_title("如果只有生物学差异\n"
                      "A 和 B 是同一种细胞",
                      fontsize=14)

    blob(axes[1], (5.4, 3.4), ORANGE)
    blob(axes[1], (1.2, 1.4), MUTED)
    blob(axes[1], (8.2, 2.4), BLUE)
    axes[1].text(5.4, 5.5, "A  深度高", ha="center", fontsize=13,
                 color=ORANGE, fontweight="semibold")
    axes[1].text(5.4, 4.95, "~13,800 counts", ha="center", fontsize=11.5,
                 color=ORANGE)
    axes[1].text(1.2, 0.0, "B  深度低", ha="center", fontsize=13,
                 color=MUTED, fontweight="semibold")
    axes[1].text(1.2, -0.55, "~10,400 counts", ha="center", fontsize=11.5,
                 color=MUTED)
    axes[1].text(8.2, 3.7, "C  metastasis", ha="center", fontsize=13,
                 color=BLUE, fontweight="semibold")
    span(axes[1], (6.3, 3.3), (7.4, 2.7), ORANGE, "近")
    span(axes[1], (2.1, 1.5), (7.2, 2.2), MUTED, "远", lift=-0.9)
    axes[1].set_title("加上测序深度的差异\n"
                      "生物学没变，A 却被"
                      "推向 C", fontsize=14)

    # What "depth" actually is, since the word carries the whole argument.
    legend = fig.add_subplot(grid[0, 2])
    legend.axis("off")
    # Broken by hand. `wrap=True` measures against the figure, not this column,
    # so a paragraph ran off the right edge.
    lines = [
        ("depth 是什么", True, 13.5, 0.06),
        ("一个细胞测到的 reads 总数\n"
         "（total counts），以及由此检出\n"
         "的基因数（nFeature）。",
         False, 12.5, 0.20),
        ("为什么细胞之间不一样",
         True, 13.5, 0.06),
        ("细胞被捕获的效率、建库\n"
         "和测序批次都会影响它。\n"
         "这是实验流程的差异，"
         "不是\n细胞本身的生物学。",
         False, 12.5, 0.26),
        ("为什么会影响距离", True, 13.5, 0.06),
        ("测得浅的细胞，很多本来\n"
         "表达的基因读数为 0。转录\n"
         "谱缺了一大块，计算距离\n"
         "时就显得“不像” C — 即"
         "使\n它的生物学与 A 一模一"
         "样。", False, 12.5, 0.30),
    ]
    y = 0.98
    for text, bold, size, drop in lines:
        legend.text(0.0, y, text, transform=legend.transAxes, va="top",
                    fontsize=size, color=INK if bold else INK2,
                    fontweight="semibold" if bold else "normal",
                    linespacing=1.55)
        y -= drop

    fig.suptitle("技术差异盖过生物学"
                 "差异：gate 读到的是深度",
                 fontsize=16, fontweight="semibold", y=1.02)
    fig.subplots_adjust(bottom=0.12)
    finish(fig, out / "fig_depth_cartoon.png",
           "示意图，非数据。深度"
           "数值是 GSE180661 里 retained 与 rejected "
           "两组的真实中位数（13,773 "
           "对 10,433，1.32×）。")


def figure_gate_vs_metastasis(source: Path, out: Path, dataset: str, name: str,
                              score: str, score_label: str, stem: str,
                              patient: str | None = None) -> None:
    """Both primary groups over the metastasis, in one panel.

    The earlier three-panel version put retained and rejected in a panel of
    their own, with the metastasis in a different panel, so where the retained
    cells sit *relative to the metastasis* could not be read off it. Here the
    metastasis is the grey backdrop and both primary groups are drawn over it,
    which is the comparison the figure exists to make.

    Colours: retained keeps the orange it has everywhere else in the deck, and
    rejected takes the next slot in the validated order rather than the grey
    now used for the backdrop.
    """
    cells = read(source, name)
    if cells is None or "umap_1" not in cells:
        return
    cells = cells.dropna(subset=["umap_1", "umap_2"])
    if patient is not None:
        if "patient_id" not in cells:
            print("  skip: no patient column")
            return
        cells = cells[cells["patient_id"].eq(patient)]
    if cells.empty:
        print("  skip: no placed cells")
        return

    primary = cells[cells["side"].eq("primary")]
    metastasis = cells[cells["side"].eq("metastasis")]
    kept = primary[primary["retained"].astype(bool)]
    dropped = primary[~primary["retained"].astype(bool)]
    if primary.empty:
        print("  skip: no primary cells")
        return

    trim = [2.0, 98.0] if patient is not None else [0.5, 99.5]
    span_x = np.percentile(cells["umap_1"].to_numpy(float), trim)
    span_y = np.percentile(cells["umap_2"].to_numpy(float), trim)
    pad_x, pad_y = 0.04 * np.ptp(span_x), 0.04 * np.ptp(span_y)
    aspect = float(np.ptp(span_x) / np.ptp(span_y))
    panel = max(4.6, 5.4 * aspect)

    fig, axes = plt.subplots(1, 2, figsize=(2 * panel + 1.1, 5.6))
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim(span_x[0] - pad_x, span_x[1] + pad_x)
        axis.set_ylim(span_y[0] - pad_y, span_y[1] + pad_y)
        axis.set_aspect("equal", adjustable="box")
        for spine in axis.spines.values():
            spine.set_visible(False)

    size = 2.6 if patient is not None else 1.3
    axes[0].scatter(metastasis["umap_1"], metastasis["umap_2"], s=size,
                    c="#d9d9d4", linewidth=0, rasterized=True, zorder=2)
    # Random order within the two primary groups, so neither buries the other.
    mixed = pd.concat([kept, dropped]).sample(frac=1.0, random_state=5)
    axes[0].scatter(mixed["umap_1"], mixed["umap_2"], s=size * 1.15,
                    c=np.where(mixed["retained"].astype(bool), ORANGE, AQUA),
                    linewidth=0, rasterized=True, zorder=3)
    axes[0].set_title("留下的细胞落在转"
                      "移灶的哪一侧", fontsize=14)
    # Bottom right, stacking upward: the cloud fills the panel, so a legend in
    # the top-left corner sat on the cells it was describing.
    for index, (text, colour, count) in enumerate((
            ("primary rejected", AQUA, len(dropped)),
            ("primary retained", ORANGE, len(kept)),
            ("metastasis", "#a8a8a2", len(metastasis)))):
        axes[0].text(0.98, 0.02 + index * 0.058,
                     f"{text}  n = {count:,}", color=colour, fontsize=12.5,
                     transform=axes[0].transAxes, va="bottom", ha="right",
                     fontweight="semibold", bbox=LABEL_BOX)

    if score in primary:
        values = primary[score].to_numpy(float)
        finite = np.isfinite(values)
        order = np.argsort(values[finite])
        low, high = np.nanpercentile(values[finite], [2, 98])
        dots = axes[1].scatter(primary["umap_1"].to_numpy()[finite][order],
                               primary["umap_2"].to_numpy()[finite][order],
                               s=size, c=values[finite][order], cmap="YlOrRd",
                               vmin=low, vmax=high, linewidth=0,
                               rasterized=True)
        bar = fig.colorbar(dots, ax=axes[1], fraction=0.045, pad=0.02)
        bar.set_label(score_label, fontsize=12)
        bar.set_ticks([low, high])
        bar.set_ticklabels(["low", "high"])
        bar.outline.set_visible(False)
        axes[1].set_title(score_label, fontsize=14)
    else:
        axes[1].set_visible(False)

    heading = dataset if patient is None else f"{dataset} — {patient}"
    plt.rcParams["font.sans-serif"] = CJK
    fig.suptitle(heading, fontsize=16, fontweight="semibold", y=1.01)
    note = (f"{len(primary):,} primary and {len(metastasis):,} metastatic "
            "malignant cells" + ("" if patient is None else ", one patient")
            + ", on the embedding published with the dataset.")
    finish(fig, out / f"fig_gatemap_{stem}.png", note)


def figure_gate_panel(source: Path, out: Path, name: str, stem: str,
                      title: str, note: str) -> None:
    """Just the kept/not-kept panel, for use as a small inset.

    The three-panel figure does not survive being shrunk into a corner of a
    slide. When the only job is to show that whole clusters are kept or
    dropped, one panel carries it and stays legible at a third of the width.
    """
    cells = read(source, name)
    if cells is None or "umap_1" not in cells:
        return
    cells = cells.dropna(subset=["umap_1", "umap_2"])
    primary = cells[cells["side"].eq("primary")]
    if primary.empty:
        print("  skip: no primary cells")
        return
    span_x = np.percentile(cells["umap_1"].to_numpy(float), [0.5, 99.5])
    span_y = np.percentile(cells["umap_2"].to_numpy(float), [0.5, 99.5])
    pad_x, pad_y = 0.04 * np.ptp(span_x), 0.04 * np.ptp(span_y)
    aspect = float(np.ptp(span_x) / np.ptp(span_y))

    fig, axis = plt.subplots(figsize=(5.2 * aspect, 5.2))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlim(span_x[0] - pad_x, span_x[1] + pad_x)
    axis.set_ylim(span_y[0] - pad_y, span_y[1] + pad_y)
    axis.set_aspect("equal", adjustable="box")
    for spine in axis.spines.values():
        spine.set_visible(False)
    kept = primary[primary["retained"].astype(bool)]
    dropped = primary[~primary["retained"].astype(bool)]
    axis.scatter(dropped["umap_1"], dropped["umap_2"], s=1.4, c="#d7d7d2",
                 linewidth=0, rasterized=True)
    axis.scatter(kept["umap_1"], kept["umap_2"], s=1.6, c=ORANGE,
                 linewidth=0, rasterized=True)
    share = float(primary["retained"].astype(bool).mean())
    axis.set_title(title, fontsize=15)
    axis.text(0.02, 0.98, f"kept  {share:.0%}", color=ORANGE, fontsize=14,
              transform=axis.transAxes, va="top", fontweight="semibold",
              bbox=LABEL_BOX)
    axis.text(0.02, 0.92, "not kept", color="#8a8a85", fontsize=14,
              transform=axis.transAxes, va="top", fontweight="semibold",
              bbox=LABEL_BOX)
    finish(fig, out / f"fig_gate_{stem}.png", note)


def figure_umap(source: Path, out: Path, dataset: str, name: str,
                score: str, score_label: str, patient: str | None = None,
                stem: str | None = None) -> None:
    """Three panels that read left to right: two tissues, the split, the reason.

    Axis numbers are omitted. UMAP coordinates have no units and no meaning
    beyond adjacency, so printing them invites a reader to compare values that
    are not comparable.

    With `patient` set, only that patient's cells are drawn. Clusters in this
    embedding are largely one patient each, so the pooled figure shows whole
    clusters kept or dropped and a reader cannot tell from it whether anything
    separates cells inside a patient. One patient answers that directly.
    """
    cells = read(source, name)
    if cells is None or "umap_1" not in cells:
        return
    cells = cells.dropna(subset=["umap_1", "umap_2"])
    if patient is not None:
        if "patient_id" not in cells:
            print("  skip: no patient column")
            return
        cells = cells[cells["patient_id"].eq(patient)]
        if cells.empty:
            print(f"  skip: {patient} not present")
            return
    if cells.empty:
        print("  skip: no placed cells")
        return
    primary = cells[cells["side"].eq("primary")]
    metastasis = cells[cells["side"].eq("metastasis")]

    # A handful of cells sit far outside the body of the embedding, and letting
    # them set the limits leaves the cloud filling barely half of each panel on
    # a slide. The limits come from a trimmed range instead, and equal aspect
    # keeps the embedding undistorted; `bbox_inches="tight"` then crops whatever
    # letterboxing that leaves.
    # One patient occupies a small part of the embedding and usually has a few
    # cells scattered far from its own cluster, so the subset is trimmed harder
    # than the pooled view: otherwise a dozen strays set the limits and the
    # cluster the panel is about shrinks into a corner.
    trim = [2.0, 98.0] if patient is not None else [0.5, 99.5]
    span_x = np.percentile(cells["umap_1"].to_numpy(float), trim)
    span_y = np.percentile(cells["umap_2"].to_numpy(float), trim)
    pad_x, pad_y = 0.04 * np.ptp(span_x), 0.04 * np.ptp(span_y)
    aspect = float(np.ptp(span_x) / np.ptp(span_y))

    # A tall, narrow cloud would give panels too narrow for their own titles,
    # which then overprint each other. Equal aspect is kept, so the floor adds
    # whitespace beside the cloud rather than distorting it.
    panel = max(3.7, 5.0 * aspect)
    fig, axes = plt.subplots(1, 3, figsize=(3 * panel + 1.2, 5.2))
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim(span_x[0] - pad_x, span_x[1] + pad_x)
        axis.set_ylim(span_y[0] - pad_y, span_y[1] + pad_y)
        axis.set_aspect("equal", adjustable="box")
        for spine in axis.spines.values():
            spine.set_visible(False)

    # Drawn in random order. Plotting one group after the other lets the larger
    # one bury the smaller, and whether the two tissues overlap is the whole
    # point of the panel: 103k metastatic cells painted over 84k primary ones
    # made the primary look absent.
    mixed = cells.sample(frac=1.0, random_state=3)
    axes[0].scatter(mixed["umap_1"], mixed["umap_2"], s=1.1, linewidth=0,
                    c=np.where(mixed["side"].eq("metastasis"), BLUE, "#c9c9c3"),
                    rasterized=True)
    axes[0].set_title("the two tissues")
    axes[0].text(0.02, 0.97, "primary", color="#8a8a85", fontsize=13,
                 transform=axes[0].transAxes, va="top", fontweight="semibold",
                 bbox=LABEL_BOX)
    axes[0].text(0.02, 0.91, "metastasis", color=BLUE, fontsize=13,
                 transform=axes[0].transAxes, va="top", fontweight="semibold",
                 bbox=LABEL_BOX)

    kept = primary[primary["retained"].astype(bool)]
    dropped = primary[~primary["retained"].astype(bool)]
    axes[1].scatter(dropped["umap_1"], dropped["umap_2"], s=1.1, c="#d7d7d2",
                    linewidth=0, rasterized=True)
    axes[1].scatter(kept["umap_1"], kept["umap_2"], s=1.3, c=ORANGE,
                    linewidth=0, rasterized=True)
    share = float(primary["retained"].astype(bool).mean())
    axes[1].set_title("which primary cells the method kept")
    axes[1].text(0.02, 0.97, f"kept  {share:.0%}", color=ORANGE, fontsize=13,
                 transform=axes[1].transAxes, va="top", fontweight="semibold",
                 bbox=LABEL_BOX)
    axes[1].text(0.02, 0.91, "not kept", color="#8a8a85", fontsize=13,
                 transform=axes[1].transAxes, va="top", fontweight="semibold",
                 bbox=LABEL_BOX)

    if score in primary:
        values = primary[score].to_numpy(float)
        finite = np.isfinite(values)
        order = np.argsort(values[finite])
        low, high = np.nanpercentile(values[finite], [2, 98])
        dots = axes[2].scatter(primary["umap_1"].to_numpy()[finite][order],
                               primary["umap_2"].to_numpy()[finite][order],
                               s=1.1, c=values[finite][order], cmap="YlOrRd",
                               vmin=low, vmax=high, linewidth=0, rasterized=True)
        bar = fig.colorbar(dots, ax=axes[2], fraction=0.04, pad=0.02)
        bar.set_label(score_label, fontsize=12)
        bar.set_ticks([low, high])
        bar.set_ticklabels(["low", "high"])
        bar.outline.set_visible(False)
        axes[2].set_title(score_label)
    else:
        axes[2].set_visible(False)

    heading = dataset if patient is None else f"{dataset} — {patient}"
    fig.suptitle(heading, fontsize=16, fontweight="semibold", y=1.02)
    note = "Every dot is one cell, on the embedding published with the dataset."
    if patient is not None:
        note = (f"One patient. {len(primary):,} primary and "
                f"{len(metastasis):,} metastatic malignant cells, on the same "
                "published embedding as the pooled figure.")
    finish(fig, out / f"fig_umap_{stem or dataset.split()[0].lower()}.png", note)


def figure_hallmark_groups(source: Path, out: Path, dataset: str, name: str,
                           scores: list[str]) -> None:
    """Per-patient differences, which is the unit the statistics were run in.

    Pooling cells across patients hides the effect. Scored per cell and pooled,
    cell division reads 0.50, 0.51 and 0.50 across rejected primary, retained
    primary and metastatic cells, while the paired differential expression on the
    same data puts TOP2A at 1.8-fold between the primary groups. Patients differ
    far more in their overall division rate than the two groups differ inside any
    one patient, so the pooled distribution is dominated by between-patient
    spread. Taking the difference within each patient first removes it, and
    matches the paired design the gene-level tests used.
    """
    cells = read(source, name)
    if cells is None or "group" not in cells or "patient_id" not in cells:
        return
    present = [s for s in scores if s in cells.columns]
    if not present:
        print("  skip: no requested scores present")
        return

    rows = []
    for patient, group in cells.groupby("patient_id"):
        kept = group[group["group"].eq("primary retained")]
        dropped = group[group["group"].eq("primary rejected")]
        distal = group[group["group"].eq("metastasis")]
        if len(kept) < 20 or len(dropped) < 20:
            continue
        for score in present:
            base = float(dropped[score].mean())
            rows.append({"patient": patient, "score": score,
                         "kept": float(kept[score].mean()) - base,
                         "metastasis": (float(distal[score].mean()) - base
                                        if len(distal) >= 20 else np.nan)})
    paired = pd.DataFrame(rows)
    if paired.empty:
        print("  skip: no patient has both groups")
        return
    patients = paired["patient"].nunique()

    fig, axes = plt.subplots(1, len(present), figsize=(3.5 * len(present), 5.4))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(7)
    for axis, score in zip(axes, present):
        subset = paired[paired["score"].eq(score)]
        columns = [("kept", 0, ORANGE), ("metastasis", 1, BLUE)]
        pooled = np.concatenate([subset[c].dropna().to_numpy(float)
                                 for c, _, _ in columns if subset[c].notna().any()])
        bounds = readable_limits(axis, pooled)
        for column, position, colour in columns:
            values = subset[column].dropna().to_numpy(float)
            if not values.size:
                continue
            jitter = rng.uniform(-0.1, 0.1, values.size)
            drawn, outside = values, np.zeros(values.size, bool)
            if bounds is not None:
                drawn = np.clip(values, *bounds)
                outside = drawn != values
            axis.scatter(jitter[~outside] + position, drawn[~outside], s=70,
                         color=colour, edgecolor=SURFACE, linewidth=1.0, zorder=3)
            axis.scatter(jitter[outside] + position, drawn[outside], s=70,
                         facecolor=SURFACE, edgecolor=colour, linewidth=1.8,
                         zorder=3)
            # The label sits inside the axis, away from the boundary it was
            # clipped to, so it cannot land on the title or the tick labels.
            for x, y, true in zip(jitter[outside] + position, drawn[outside],
                                  values[outside]):
                axis.annotate(f"{true:+.2f}", (x, y),
                              xytext=(0, 13 if true < np.median(values) else -15),
                              textcoords="offset points", ha="center", fontsize=10,
                              color=colour)
            # The median is of the real values, so a clipped point still counts.
            median = float(np.median(values))
            if bounds is None or bounds[0] <= median <= bounds[1]:
                axis.plot([position - 0.26, position + 0.26], [median, median],
                          color=INK, linewidth=2.4, zorder=4)
        axis.axhline(0, color=INK2, linewidth=1.2, linestyle="--", zorder=2)
        axis.set_xticks([0, 1])
        # Both columns name their baseline, so the contrast under test is on
        # the axis and cannot be read as kept-versus-metastasis.
        axis.set_xticklabels(["kept\nvs rejected", "metastasis\nvs rejected"],
                             fontsize=12)
        axis.set_xlim(-0.55, 1.55)
        axis.set_title(SCORE_LABELS.get(score, score.replace("_", " ")), fontsize=13)
        axis.grid(axis="y", zorder=0)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("difference from the rejected primary cells\n"
                       "(within each patient)")
    fig.suptitle(f"{dataset}: what separates the kept primary cells "
                 "from the rejected ones",
                 fontsize=16, fontweight="semibold", y=1.04)
    fig.subplots_adjust(wspace=0.46, bottom=0.22)
    finish(fig, out / f"fig_hallmark_{dataset.split()[0].lower()}.png",
           f"One dot per patient (n = {patients}). Orange is the contrast under "
           "test; blue is the metastasis on the same baseline, for reference. "
           "Hollow dots fall outside the axis and carry their real value.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Unpacked report_inputs directory")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--example-patient", default="SPECTRUM-OV-083",
                        help="the one patient drawn on its own")
    parser.add_argument("--faming-gsea", type=Path, default=None,
                        help="Step6b.GSEA_Hallmark.xlsx, for the reference bars")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, function in (
        ("depth", lambda: figure_depth(args.input_dir, args.output_dir)),
        ("representation", lambda: figure_representation(args.input_dir, args.output_dir)),
        ("similarity", lambda: figure_similarity(args.input_dir, args.output_dir)),
        ("control", lambda: figure_control(args.input_dir, args.output_dir)),
        ("patients", lambda: figure_patients(args.input_dir, args.output_dir)),
        ("gsea", lambda: figure_gsea(args.input_dir, args.output_dir, args.faming_gsea)),
        ("cell cycle", lambda: figure_cellcycle(args.input_dir, args.output_dir)),
        ("methods cartoon", lambda: figure_methods_cartoon(args.output_dir)),
        ("depth cartoon", lambda: figure_depth_cartoon(args.output_dir)),
        ("umap ovarian", lambda: figure_umap(
            args.input_dir, args.output_dir, "Ovarian cancer", "umap_ovarian.csv.gz",
            "cell_division", "cell division score")),
        ("gate vs metastasis, one ovarian patient",
         lambda: figure_gate_vs_metastasis(
             args.input_dir, args.output_dir, "Ovarian cancer",
             "umap_ovarian.csv.gz", "cell_division", "cell division score",
             "ovarian_patient", patient=args.example_patient)),
        ("gate vs metastasis, prostate", lambda: figure_gate_vs_metastasis(
            args.input_dir, args.output_dir, "Prostate cancer",
            "umap_prostate.csv.gz", "cell_division", "cell division score",
            "prostate")),
        ("pooled gate panel", lambda: figure_gate_panel(
            args.input_dir, args.output_dir, "umap_ovarian.csv.gz",
            "ovarian_all", "all 29 patients",
            "Whole clusters are kept or dropped, because a cluster here is "
            "largely one patient.")),
        # One patient, chosen for being legible rather than for its result:
        # 4,726 primary and 4,307 metastatic cells with the split near even at
        # 54% kept, so both groups are visible. Its cell-division gap, +0.126,
        # is the same sign as the cohort median of +0.089.
        ("umap ovarian, one patient", lambda: figure_umap(
            args.input_dir, args.output_dir, "Ovarian cancer",
            "umap_ovarian.csv.gz", "cell_division", "cell division score",
            patient=args.example_patient, stem="ovarian_patient")),
        ("umap prostate", lambda: figure_umap(
            args.input_dir, args.output_dir, "Prostate cancer", "umap_prostate.csv.gz",
            "cell_division", "cell division score")),
        ("hallmark ovarian", lambda: figure_hallmark_groups(
            args.input_dir, args.output_dir, "Ovarian cancer", "umap_ovarian.csv.gz",
            ["cell_division", "interferon", "mesenchymal"])),
        # The same three scores as the ovarian panel, so the two cancers can be
        # read side by side. Androgen is deliberately not scored per cell here:
        # the paired pseudobulk test is the instrument that call rests on, and a
        # per-cell mean over a focused set is the weaker measurement of the two.
        ("hallmark prostate", lambda: figure_hallmark_groups(
            args.input_dir, args.output_dir, "Prostate cancer", "umap_prostate.csv.gz",
            ["cell_division", "interferon", "mesenchymal"])),
    ):
        print(f"{name}:")
        try:
            function()
        except Exception as error:  # a missing column should not lose the rest
            print(f"  failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
