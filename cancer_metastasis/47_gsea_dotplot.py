"""One signed GSEA dot plot per analysis: every arm, every hallmark pathway.

    python cancer_metastasis/47_gsea_dotplot.py OUT_DIR \\
        --analysis "Ovarian GSE180661::<down>/{rep}/arm_{cycle}/uniform/GSE180661/{label}" \\
        --analysis ...

This replaces the two one-sided figures of 43_retained_side_heatmap.py, which
split one GSEA into "higher in retained" and "higher in rejected" panels and
showed each side as an unsigned value. Here one figure holds the GSEA as it was
computed:

* **Colour** is NES of primary_rejected_vs_primary_retained, with retained as the
  reference: NES < 0 (blue) is higher in retained cells, NES > 0 (red) higher in
  rejected cells. Diverging, with a neutral grey midpoint, symmetric limits
  (default +/-3) shared by every figure of one run so colours compare across
  analyses; a point beyond them takes the end colour, the bar shows an arrow,
  and the exact NES is in the CSV.
* **Size** is -log10 FDR. GSEApy reports FDR = 0 below its permutation
  resolution; the runs used 2,000 permutations (tacc_factorial_downstream.slurm),
  so 0 is drawn at 1/2000 and the legend says so.
* **Outline** marks FDR < 0.05, the call 41_ made.

Rows are the arms of the cycling x representation design (43_'s cycling layout:
RNA, RNA+CNV and CNV; arms A-D; ranknm256_noscale_ds_cos and raw). An arm a
panel lacks -- prostate has no CNV arms -- is not drawn and the subtitle says so.

Columns are all 50 hallmark pathways, none dropped, grouped by the eight process
categories of the hallmark collection (Liberzon et al., Cell Systems 2015,
Fig. 2): cellular component, development, DNA damage, immune, metabolic,
pathway, proliferation, signalling. Within a category pathways keep the
category's order below, which is fixed and does not depend on any result.

Values are read from each arm's own all_gene_discovery_gsea_results.csv, not
from 41_'s matrices.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("heatmap43", HERE / "43_retained_side_heatmap.py")
heatmap43 = importlib.util.module_from_spec(_spec)
sys.modules["heatmap43"] = heatmap43  # dataclasses look the module up by name
_spec.loader.exec_module(heatmap43)

CONTRAST = heatmap43.CONTRAST
PERMUTATIONS = 2000
FDR_CALL = 0.05

# Liberzon et al. 2015, Fig. 2: the hallmark sets by process category.
CATEGORIES = [
    ("Cellular\ncomponent", ["APICAL_JUNCTION", "APICAL_SURFACE", "PEROXISOME"]),
    ("Development", ["ADIPOGENESIS", "ANGIOGENESIS", "EPITHELIAL_MESENCHYMAL_TRANSITION",
                     "MYOGENESIS", "SPERMATOGENESIS", "PANCREAS_BETA_CELLS"]),
    ("DNA\ndamage", ["DNA_REPAIR", "UV_RESPONSE_DN", "UV_RESPONSE_UP"]),
    ("Immune", ["ALLOGRAFT_REJECTION", "COAGULATION", "COMPLEMENT", "INTERFERON_ALPHA_RESPONSE",
                "INTERFERON_GAMMA_RESPONSE", "IL6_JAK_STAT3_SIGNALING", "INFLAMMATORY_RESPONSE"]),
    ("Metabolic", ["BILE_ACID_METABOLISM", "CHOLESTEROL_HOMEOSTASIS", "FATTY_ACID_METABOLISM",
                   "GLYCOLYSIS", "HEME_METABOLISM", "OXIDATIVE_PHOSPHORYLATION", "XENOBIOTIC_METABOLISM"]),
    ("Pathway", ["APOPTOSIS", "HYPOXIA", "PROTEIN_SECRETION", "UNFOLDED_PROTEIN_RESPONSE",
                 "REACTIVE_OXYGEN_SPECIES_PATHWAY"]),
    ("Proliferation", ["E2F_TARGETS", "G2M_CHECKPOINT", "MYC_TARGETS_V1", "MYC_TARGETS_V2",
                       "P53_PATHWAY", "MITOTIC_SPINDLE"]),
    ("Signaling", ["ANDROGEN_RESPONSE", "ESTROGEN_RESPONSE_EARLY", "ESTROGEN_RESPONSE_LATE",
                   "IL2_STAT5_SIGNALING", "KRAS_SIGNALING_UP", "KRAS_SIGNALING_DN", "MTORC1_SIGNALING",
                   "NOTCH_SIGNALING", "PI3K_AKT_MTOR_SIGNALING", "HEDGEHOG_SIGNALING",
                   "TGF_BETA_SIGNALING", "TNFA_SIGNALING_VIA_NFKB", "WNT_BETA_CATENIN_SIGNALING"]),
]
PATHWAYS = [p for _, members in CATEGORIES for p in members]
assert len(PATHWAYS) == 50 and len(set(PATHWAYS)) == 50

# The project's retained blue and rejected red, at their deepest matched step,
# through a neutral grey midpoint.
NEGATIVE = heatmap43.RAMPS["retained"][-2]
POSITIVE = heatmap43.RAMPS["rejected"][-2]
MIDPOINT = "#e9e9e9"
INK, MUTED, RULE = "#1a1a1a", "#5f5f5f", "#bdbdbd"

ARM_NOTE = ("A: nothing removed.  B: malignant cells outside G1 removed on both sides.  "
            "C: the 97 cell-cycle genes removed.  D: both.")


def arm_label(arm: str) -> str:
    rep, cycle, label = arm.split("/")
    rep_text = dict(heatmap43.CYCLING_REPS)[rep]
    preprocessing = "ds_cos" if label == "ranknm256_noscale_ds_cos" else "raw"
    if rep == "cnv":
        return f"{rep_text}  {cycle}"
    return f"{rep_text}  {preprocessing}  {cycle}"


def load(template: str) -> tuple[pd.DataFrame, list[str]]:
    layout = heatmap43.cycling_layout()
    rows, missing = [], []
    for arm in layout.arms:
        path = layout.arm_dir(template, arm) / "gsea" / "contrasts" / CONTRAST / "all_gene_discovery_gsea_results.csv"
        if not path.is_file():
            missing.append(arm)
            continue
        table = pd.read_csv(path)
        table = table[table["collection"].astype(str).eq("Hallmark")]
        table["pathway"] = table["pathway"].astype(str).str.replace("^HALLMARK_", "", regex=True)
        absent = sorted(set(PATHWAYS) - set(table["pathway"]))
        if absent:
            raise RuntimeError(f"{path}: hallmark pathways missing {absent}")
        for record in table.itertuples(index=False):
            if record.pathway in PATHWAYS:
                rows.append({"arm": arm, "pathway": record.pathway, "NES": float(record.NES),
                             "fdr": float(record.fdr), "source": str(path)})
    return pd.DataFrame(rows), missing


def draw(frame: pd.DataFrame, title: str, missing: list[str], limit: float, path: Path) -> None:
    arms = [arm for arm in heatmap43.cycling_layout().arms if arm in set(frame["arm"])]
    # A gap between representation blocks, as in the earlier figures.
    y_of, y, previous = {}, 0.0, None
    for arm in arms:
        block = arm.split("/")[0]
        if previous is not None and block != previous:
            y += 0.6
        y_of[arm] = y
        y += 1.0
        previous = block
    x_of, x = {}, 0.0
    category_spans = []
    for name, members in CATEGORIES:
        start = x
        for pathway in members:
            x_of[pathway] = x
            x += 1.0
        category_spans.append((name, start, x - 1.0))
        x += 0.6

    width = 2.6 + 0.205 * x
    height = 2.9 + 0.30 * y
    fig, ax = plt.subplots(figsize=(width, height))
    cmap = LinearSegmentedColormap.from_list("nes", [NEGATIVE, MIDPOINT, POSITIVE])
    norm = Normalize(vmin=-limit, vmax=limit)
    floor = 1.0 / PERMUTATIONS
    strength = -np.log10(frame["fdr"].clip(lower=floor))
    top = -np.log10(floor)
    sizes = 8 + 150 * (strength / top) ** 1.0
    called = frame["fdr"] < FDR_CALL
    xs = frame["pathway"].map(x_of).to_numpy()
    ys = frame["arm"].map(y_of).to_numpy()
    ax.scatter(xs[~called], ys[~called], s=sizes[~called], c=frame.loc[~called, "NES"],
               cmap=cmap, norm=norm, linewidths=0, zorder=2)
    ax.scatter(xs[called], ys[called], s=sizes[called], c=frame.loc[called, "NES"],
               cmap=cmap, norm=norm, edgecolors=INK, linewidths=0.9, zorder=3)

    ax.set_xlim(-0.8, x - 0.6 + 0.2)
    ax.set_ylim(y - 1.0 + 0.7, -0.7)
    ax.set_xticks([x_of[p] for p in PATHWAYS])
    ax.set_xticklabels([heatmap43.display_name("HALLMARK_" + p) for p in PATHWAYS], rotation=60,
                       ha="right", rotation_mode="anchor", fontsize=7.4, color=INK)
    ax.set_yticks([y_of[a] for a in arms])
    ax.set_yticklabels([arm_label(a) for a in arms], fontsize=7.8, color=INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(0.8)
    for name, start, end in category_spans:
        ax.annotate("", xy=(start - 0.3, -1.05), xytext=(end + 0.3, -1.05), xycoords="data",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), annotation_clip=False)
        ax.text((start + end) / 2, -1.25, name, ha="center", va="bottom", fontsize=8,
                color=INK, linespacing=1.0, clip_on=False)
    for name, start, end in category_spans[:-1]:
        ax.axvline(end + 0.8, color=RULE, lw=0.6, zorder=1)

    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.995)
    subtitle = ("GSEA of primary rejected vs primary retained (reference); hallmark, all 50 pathways.  "
                "Rows: arms.")
    if missing:
        names = dict(heatmap43.CYCLING_REPS)
        blocks = [names[rep] for rep, _ in heatmap43.CYCLING_REPS
                  if any(a.startswith(rep + "/") for a in missing)]
        subtitle += f"  {' and '.join(blocks)} arms were not run for this dataset."
    fig.text(0.01, 0.955, subtitle, ha="left", va="top", fontsize=8, color=MUTED)
    fig.text(0.01, 0.012, ARM_NOTE + "  ds_cos: ranknm256_noscale_ds_cos.", ha="left",
             va="bottom", fontsize=7.4, color=MUTED)

    # Legends: colour bar, size, outline.
    cax = fig.add_axes([0.905, 0.62, 0.012, 0.22])
    clipped = bool((frame["NES"].abs() > limit).any())
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       extend="both" if clipped else "neither")
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, length=0, colors=MUTED)
    cax.set_title("NES\nrejected vs\nretained", fontsize=7.4, color=INK, loc="left", pad=6)
    cax.text(2.6, 0.02, "higher in\nretained", transform=cax.transAxes, fontsize=6.6,
             color=MUTED, va="bottom")
    cax.text(2.6, 0.98, "higher in\nrejected", transform=cax.transAxes, fontsize=6.6,
             color=MUTED, va="top")
    handles = []
    for fdr in (0.1, 0.01, floor):
        value = -np.log10(fdr)
        label = f"{fdr:g}" if fdr > floor else f"<= {floor:g}*"
        handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="#9a9a9a",
                              markeredgecolor="none",
                              markersize=np.sqrt(8 + 150 * value / top), label=label))
    handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="#9a9a9a",
                          markeredgecolor=INK, markeredgewidth=0.9, markersize=7,
                          label=f"FDR < {FDR_CALL:g}"))
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.893, 0.52),
                        frameon=False, fontsize=7, title="FDR", title_fontsize=7.4,
                        labelcolor=INK, handletextpad=0.4, borderaxespad=0)
    legend.get_title().set_color(INK)
    fig.text(0.893, 0.20, f"* FDR 0 is below the\nresolution of {PERMUTATIONS:,}\npermutations",
             fontsize=6.4, color=MUTED, va="top")
    fig.subplots_adjust(left=1.9 / width, right=0.875, top=1 - 1.15 / height, bottom=2.05 / height)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--analysis", action="append", required=True,
                        help="TITLE::TEMPLATE, the template as 43_'s cycling layout takes it")
    parser.add_argument("--limit", type=float, default=3.0,
                        help="Colour limit |NES|, shared by every figure; points beyond it take the end "
                             "colour and the colour bar shows an arrow. The values are in the CSV")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    loaded = []
    for spec in args.analysis:
        title, template = spec.split("::", 1)
        frame, missing = load(template)
        if frame.empty:
            raise RuntimeError(f"{title}: no GSEA table found under {template}")
        loaded.append((title, frame, missing))
    limit = float(args.limit)
    record = {"limit": limit, "permutations": PERMUTATIONS, "figures": []}
    for title, frame, missing in loaded:
        stem = "gsea_dotplot_" + "".join(c if c.isalnum() else "_" for c in title).strip("_").lower()
        draw(frame, title, missing, limit, args.out / stem)
        frame.to_csv(args.out / f"{stem}.csv", index=False)
        record["figures"].append({"title": title, "stem": stem, "arms": int(frame["arm"].nunique()),
                                  "points_beyond_limit": int((frame["NES"].abs() > limit).sum()),
                                  "largest_abs_nes": float(frame["NES"].abs().max()),
                                  "missing_arms": missing, "points": int(len(frame))})
    (args.out / "gsea_dotplot.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
