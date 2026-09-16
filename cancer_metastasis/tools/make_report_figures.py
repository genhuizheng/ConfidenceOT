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

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
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


def finish(fig, path: Path, note: str | None = None) -> None:
    if note:
        fig.text(0.01, 0.01, note, fontsize=10, color=MUTED, ha="left", va="bottom")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
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

    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    distance = [abs(v - 0.5) for v in values]
    # Sequential by magnitude of the problem: one hue, light to dark.
    shades = [plt.matplotlib.colors.to_hex(plt.cm.Blues(0.35 + 0.5 * d / max(distance + [0.01])))
              for d in distance]
    bars = axis.bar(labels, values, color=shades, width=0.62, zorder=3)
    axis.axhline(0.5, color=INK2, linewidth=1.4, linestyle="--", zorder=2)
    axis.text(len(labels) - 0.42, 0.512, "no depth effect", fontsize=11, color=INK2, ha="right")
    for bar, value in zip(bars, values):
        offset = 0.022 if value >= 0.5 else -0.05
        axis.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.2f}",
                  ha="center", fontsize=13, color=INK, fontweight="semibold")
    axis.set_ylim(0, max(1.0, max(values) + 0.12))
    axis.set_ylabel("how well depth predicts the gate\n(AUC; 0.5 = not at all)")
    axis.set_title("The gate was largely reading sequencing depth")
    axis.grid(axis="y", zorder=0)
    axis.set_axisbelow(True)
    finish(fig, out / "fig1_depth_in_gate.png",
           "Departure from 0.5 in either direction is a depth effect.")


def figure_representation(source: Path, out: Path) -> None:
    """Simulated data with a known answer: which preprocessing removes depth."""
    arms = ["homogeneous_depth_cv0", "homogeneous_depth_cv_low",
            "homogeneous_depth_cv_mid", "homogeneous_depth_cv_high"]
    pretty = ["none", "low", "medium", "high"]
    series = [
        ("standard (log-CPM)", "sim_log_cpm_arms.csv", BLUE),
        ("rank, top 256", "sim_rank256_arms.csv", ORANGE),
        ("rank without gene median", "sim_rank_no_median_arms.csv", AQUA),
        ("Pearson residuals", "sim_pearson_residuals_arms.csv", MAGENTA),
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                      gridspec_kw={"width_ratios": [1.55, 1]})
    drawn = 0
    for label, name, colour in series:
        table = read(source, name)
        if table is None or "auc_total_counts" not in table:
            continue
        table = table.set_index("arm")
        if not set(arms) <= set(table.index):
            continue
        values = table.loc[arms, "auc_total_counts"].to_numpy(float)
        left.plot(pretty, values, marker="o", markersize=8, linewidth=2,
                  color=colour, label=label, zorder=3)
        left.annotate(label, (len(pretty) - 1, values[-1]), xytext=(8, 0),
                      textcoords="offset points", color=colour, fontsize=11,
                      va="center", fontweight="semibold")
        drawn += 1
    if not drawn:
        plt.close(fig)
        print("  skip: no simulation summaries")
        return
    left.axhline(0.5, color=INK2, linewidth=1.4, linestyle="--", zorder=2)
    left.set_xlabel("how unequal the sequencing depth is")
    left.set_ylabel("depth effect on the gate\n(AUC; 0.5 = none)")
    left.set_title("Ranking genes within each cell removes the depth effect")
    left.set_ylim(0, 1.05)
    left.set_xlim(-0.25, len(pretty) + 0.9)
    left.grid(axis="y", zorder=0)
    left.set_axisbelow(True)
    left.legend(loc="lower left", ncol=1)

    names, f1s, colours = [], [], []
    for label, name, colour in series:
        table = read(source, name)
        if table is None or "perturbed_f1" not in table:
            continue
        table = table.set_index("arm")
        if "perturbed_depth_cv0" not in table.index:
            continue
        names.append(label.replace(" (log-CPM)", "").replace(", top 256", ""))
        f1s.append(float(table.loc["perturbed_depth_cv0", "perturbed_f1"]))
        colours.append(colour)
    if f1s:
        bars = right.barh(names, f1s, color=colours, height=0.6, zorder=3)
        for bar, value in zip(bars, f1s):
            right.text(value + 0.012, bar.get_y() + bar.get_height() / 2,
                       f"{value:.3f}", va="center", fontsize=12, color=INK)
        right.set_xlim(0, max(f1s) + 0.13)
        right.set_xlabel("detection of a known 20% subpopulation (F1)")
        right.set_title("At almost no cost in sensitivity")
        right.grid(axis="x", zorder=0)
        right.set_axisbelow(True)
        right.invert_yaxis()
    finish(fig, out / "fig2_representation_benchmark.png",
           "Simulated data where the correct answer is known.")


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

    fig, axis = plt.subplots(figsize=(7.4, 5.8))
    for a, b in zip(matched, mismatched):
        axis.plot([0, 1], [a, b], color=MUTED, linewidth=0.7, alpha=0.5, zorder=2)
    axis.scatter(np.zeros_like(matched), matched, s=46, color=BLUE,
                 edgecolor=SURFACE, linewidth=0.8, zorder=3)
    axis.scatter(np.ones_like(mismatched), mismatched, s=46, color=ORANGE,
                 edgecolor=SURFACE, linewidth=0.8, zorder=3)
    for x, values, colour in ((0, matched, BLUE), (1, mismatched, ORANGE)):
        median = float(np.nanmedian(values))
        axis.plot([x - 0.16, x + 0.16], [median, median], color=INK, linewidth=2.6, zorder=4)
        axis.text(x, -0.075, f"median {median:.3f}", ha="center", fontsize=12,
                  color=colour, fontweight="semibold")
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["own\nmetastasis", "another patient's\nmetastasis"])
    axis.set_xlim(-0.45, 1.45)
    axis.set_ylim(-0.13, 1.02)
    axis.set_ylabel("fraction of primary cells retained")
    axis.set_title("The method does use the metastasis it is given")
    axis.grid(axis="y", zorder=0)
    axis.set_axisbelow(True)
    finish(fig, out / "fig4_mismatched_control.png",
           "94 ovarian pairs, each line one primary sample. Paired test p = 8e-17.")


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
    axis.set_xlabel("enrichment in metastasis / in retained cells  (NES)")
    axis.set_title("Androgen response agrees; proliferation is exactly reversed")
    axis.grid(axis="x", zorder=0)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right")
    finish(fig, out / "fig6_gsea_agreement.png",
           "Both bars point the same way when our retained cells resemble the metastasis.")


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
           "Prostate, depth-corrected arm. Gene-set test: E2F targets NES -3.4, FDR < 0.001.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Unpacked report_inputs directory")
    parser.add_argument("output_dir", type=Path)
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
    ):
        print(f"{name}:")
        try:
            function()
        except Exception as error:  # a missing column should not lose the rest
            print(f"  failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
