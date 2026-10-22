"""Diagnostic figures of the rejection-coupling mass audit (reads only the audit tables).

    python lineage_validation/within_family_subclones/audit/plot_audit.py --audit lineage_validation/within_family_subclones/audit

  figures/A_rejection_<family>.png      eligible vs retained primary cells per source group, retention rate, by
                                        CRISPR lineage support; one panel per strategy, one row per run (M4-E)
  figures/B_mass_<family>.png           three compositions of the same source groups: eligible cells, retained
                                        cells, retained effective transport mass (M4-E)
  figures/C_ot_vs_independent_<family>.png  true metastatic subclone x primary subclone: CRISPR correspondence,
                                        raw within-family OT weight, independent-coupling expectation, enrichment
  figures/D_level1_vs_level2_<mouse>.png  family-level and subclone-level identification in the same run:
                                        retention rate and mass share per group, and the accuracy of raw-mass,
                                        independent-coupling, size and enrichment argmax per strategy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
STRATS = ["rna", "cnv", "rna_cnv", "cnv_then_rna", "rna_then_cnv"]
NAMES = {"rna": "RNA only", "cnv": "CNV only", "rna_cnv": "CNV + RNA", "cnv_then_rna": "CNV -> RNA", "rna_then_cnv": "RNA -> CNV"}
SUPPORT = {"lineage-supported (>= 5 metastatic cells)": "#1b7837", "lineage-supported": "#1b7837",
           "insufficient (1-4 metastatic cells)": "#fdb863", "no observed metastatic counterpart": "#762a83",
           "other family": "#999999", "ambiguous / unlabelled": "#d9d9d9", "": "#bdbdbd"}
GROUP_PALETTE = ["#1f78b4", "#ff7f00", "#6a3d9a", "#b15928", "#a6cee3", "#fdbf6f", "#cab2d6", "#e6ab02", "#8dd3c7", "#80b1d3",
                 "#bc80bd", "#666666", "#d9d9d9"]


def short(g):
    return str(g).replace("3508_Apc_", "").replace("3515_Lkb1_", "").replace("family unlabelled", "unlab.")


def run_label(pair):
    return pair.split("__")[1] + (" (" + pair.split("__")[2] + " source)" if "D_B" in pair and not pair.startswith("10k") else "")


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_rejection(rej, family, out):
    part = rej[(rej.family == family) & (rej.method == "M4-E") & rej.level.isin(["subclone L1", "family (primary site)", "family (clone)"])]
    pairs = sorted(part.pair.unique())
    fig, axes = plt.subplots(len(pairs), len(STRATS), figsize=(3.0 * len(STRATS), 0.6 + 2.6 * len(pairs)), squeeze=False)
    for r, pair in enumerate(pairs):
        for c, s in enumerate(STRATS):
            ax = axes[r, c]
            d = part[(part.pair == pair) & (part.strategy == s)]
            d = d.sort_values(["level", "group"])
            labels = [f"{'site ' if l.startswith('family') else ''}{short(g)}" for l, g in zip(d.level, d.group)]
            y = np.arange(len(d))
            ax.barh(y, d.eligible_cells, color="#f0f0f0", edgecolor="#999999", label="eligible")
            ax.barh(y, d.retained_cells, color=[SUPPORT.get(x, "#bdbdbd") for x in d.support], label="retained")
            for yi, (n, k, rr) in enumerate(zip(d.eligible_cells, d.retained_cells, d.retention_rate)):
                ax.text(n, yi, f" {k}/{n} = {rr:.2f}", va="center", fontsize=6)
            ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6); ax.invert_yaxis()
            ax.set_xlim(0, d.eligible_cells.max() * 1.6 if len(d) else 1)
            ax.set_title(f"{NAMES[s]}\n{run_label(pair)}", fontsize=7)
            ax.set_xlabel("primary cells (count)", fontsize=6)
    handles = [Patch(facecolor="#f0f0f0", edgecolor="#999999", label="eligible (analysed) cells")] + \
              [Patch(facecolor=v, label=f"retained, {k}") for k, v in SUPPORT.items() if k and k != "lineage-supported"]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{family}: primary cells retained by the M4-E gate, per source group (bar colour = CRISPR lineage support of the group)", fontsize=9)
    fig.tight_layout()
    save(fig, out, f"A_rejection_{family}")


def fig_mass(dec, family, out):
    part = dec[(dec.family == family) & (dec.method == "M4-E") & (dec.level == "subclone L1")]
    pairs = sorted(part.pair.unique())
    groups = sorted(part.group.unique(), key=lambda g: (g == "family unlabelled", g))
    colour = {g: GROUP_PALETTE[i] if g != "family unlabelled" else "#d9d9d9" for i, g in enumerate(groups)}
    comps = ["eligible cells", "retained cells", "effective transport mass"]
    fig, axes = plt.subplots(len(pairs), len(STRATS), figsize=(2.6 * len(STRATS), 0.8 + 2.8 * len(pairs)), squeeze=False)
    for r, pair in enumerate(pairs):
        for c, s in enumerate(STRATS):
            ax = axes[r, c]
            d = part[(part.pair == pair) & (part.strategy == s)]
            for k, comp in enumerate(comps):
                bottom = 0.0
                for g in groups:
                    v = d[(d.composition == comp) & (d.group == g)]["share"]
                    v = float(v.iloc[0]) if len(v) else 0.0
                    ax.bar(k, v, bottom=bottom, color=colour[g], edgecolor="white", width=0.7)
                    if v >= 0.06:
                        ax.text(k, bottom + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=6)
                    bottom += v
            ax.set_xticks(range(3)); ax.set_xticklabels(["eligible\ncells", "retained\ncells", "retained\ntransport mass"], fontsize=6)
            ax.set_ylim(0, 1); ax.set_title(f"{NAMES[s]}\n{run_label(pair)}", fontsize=7)
            if c == 0:
                ax.set_ylabel("share within the family", fontsize=7)
    fig.legend(handles=[Patch(facecolor=colour[g], label=short(g)) for g in groups], loc="lower center", ncol=len(groups), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{family}: composition of the family's primary source by CRISPR subclone (level 1), M4-E", fontsize=9)
    fig.tight_layout()
    save(fig, out, f"B_mass_{family}")


def fig_ot_vs_independent(vs, family, out):
    part = vs[(vs.family == family) & (vs.method == "M4-E") & (vs.level == "subclone L1") & vs.pair.str.contains("D_B")]
    if part.empty:
        return
    sources = sorted(part.source_group.unique(), key=lambda g: (g == "family unlabelled", g))
    trues = sorted(part.true_group.unique())
    fig, axes = plt.subplots(len(STRATS), 4, figsize=(12, 1.2 + 1.5 * len(STRATS) * max(1, len(trues) / 2)), squeeze=False)
    titles = ["CRISPR correspondence\n(1 = same subclone)", "raw OT weight\n(mean within-family S)", "independent coupling\n(retained mass share)",
              "enrichment S / share\n(blank: share < 0.01 or < 5 cells)"]
    for r, s in enumerate(STRATS):
        d = part[part.strategy == s]
        for c, (col, cmap, norm) in enumerate((("crispr_correspondence", "Greens", None), ("mean_S_within_family", "Purples", None),
                                               ("independent_expectation", "Purples", None), ("enrichment", "RdBu_r", TwoSlopeNorm(1.0, 0.0, 2.0)))):
            ax = axes[r, c]
            m = d.pivot_table(index="true_group", columns="source_group", values=col).reindex(index=trues, columns=sources)
            n = d.drop_duplicates("true_group").set_index("true_group")["retained_target_cells"].reindex(trues)
            im = ax.imshow(m.to_numpy(float), cmap=cmap, norm=norm, vmin=None if norm else 0, vmax=None if norm else 1, aspect="auto")
            for (i, j), v in np.ndenumerate(m.to_numpy(float)):
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
            ax.set_xticks(range(len(sources))); ax.set_xticklabels([short(x) for x in sources], fontsize=6, rotation=90)
            ax.set_yticks(range(len(trues))); ax.set_yticklabels([f"{t} (n={int(k)})" for t, k in n.items()], fontsize=6)
            if r == 0:
                ax.set_title(titles[c], fontsize=7)
            if c == 0:
                ax.set_ylabel(f"{NAMES[s]}\nmetastatic subclone", fontsize=7)
            if r == len(STRATS) - 1:
                ax.set_xlabel("primary subclone", fontsize=6)
            if r == len(STRATS) - 1:
                images = images if "images" in dir() else {}
                images[c] = im
    fig.subplots_adjust(wspace=0.55, hspace=0.45, bottom=0.12)
    for c, im in images.items():  # one colour bar per column, below the figure
        pos = axes[-1, c].get_position()
        fig.colorbar(im, cax=fig.add_axes([pos.x0, 0.025, pos.width, 0.01]), orientation="horizontal")
    fig.suptitle(f"{family}: OT matching weight vs the independent coupling of the retained block (D_B run, M4-E, retained metastatic cells)", fontsize=9)
    save(fig, out, f"C_ot_vs_independent_{family}")


def fig_levels(rej, cmp_, level1, subfamilies, pair, title, out, name):
    """Rows: level 1 (family / clone) and level 2 (CRISPR L1 subclones of each family), all in the same run `pair`."""
    rows = [("level 1 (" + level1[1].replace("family (", "").rstrip(")") + ")", level1[0], level1[1])] +            [(f"level 2: subclones of {short(f).split('_')[-1]}", f, "subclone L1") for f in subfamilies]
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 1.0 + 3.0 * len(rows)), squeeze=False, gridspec_kw={"width_ratios": [1.3, 1]})
    for r, (label, fam, level) in enumerate(rows):
        d = rej[(rej.family == fam) & (rej.level == level) & (rej.pair == pair) & (rej.method == "M4-E")]
        ax = axes[r, 0]
        groups = sorted(d.group.unique(), key=lambda g: (g == "family unlabelled", g))
        width = 0.8 / len(STRATS)
        for k, s in enumerate(STRATS):
            e = d[d.strategy == s].set_index("group").reindex(groups)
            x = np.arange(len(groups)) + k * width
            ax.bar(x, e.retention_rate, width=width, color=plt.get_cmap("tab10").colors[k], label=f"{NAMES[s]} retention")
            ax.scatter(x, e.pi_within_family if level != "family (clone)" else e.pi_whole_source, color="black", marker="_", s=60, zorder=3)
        ax.set_xticks(np.arange(len(groups)) + 0.4 - width / 2)
        sup = d.drop_duplicates("group").set_index("group")["support"].reindex(groups)
        ax.set_xticklabels([f"{short(g)}\n{'*' if 'supported' in str(sv) else ''}" for g, sv in zip(groups, sup)], fontsize=6, rotation=90 if len(groups) > 6 else 0)
        ax.set_ylim(0, 1.05); ax.set_ylabel("retention rate (bars)\nmass share (black ticks)", fontsize=7)
        ax.set_title(f"{label}: retention and retained mass share per source group (* = lineage-supported)", fontsize=7)
        if r == 0:
            ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper right")
        ax = axes[r, 1]
        c = cmp_[(cmp_.family == fam) & (cmp_.level == level) & (cmp_.pair == pair) & (cmp_.method == "M4-E")].set_index("strategy").reindex(STRATS)
        x = np.arange(len(STRATS))
        for k, (col, colour) in enumerate((("raw_mass_argmax", "#542788"), ("independent_argmax", "#bdbdbd"), ("size_argmax", "#878787"),
                                            ("enrichment_argmax", "#e08214"))):
            ax.bar(x + k * 0.2, c[col], width=0.2, color=colour, label=col.replace("_", " "))
        ax.set_xticks(x + 0.3); ax.set_xticklabels([NAMES[s] for s in STRATS], fontsize=6, rotation=30)
        ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy on the same cells", fontsize=7)
        n = c["scored_target_cells"].dropna()
        ax.set_title(f"{label}: identification accuracy ({int(n.max()) if len(n) else 0} metastatic cells)", fontsize=7)
        if r == 0:
            ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower right")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    save(fig, out, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    a = args.audit
    rej = pd.read_csv(a / "subclone_rejection.tsv", sep="\t")
    dec = pd.read_csv(a / "subclone_mass_decomposition.tsv", sep="\t")
    vs = pd.read_csv(a / "ot_vs_independent.tsv", sep="\t")
    cmp_ = pd.read_csv(a / "classifier_comparison.tsv", sep="\t")
    out = a / "figures"
    for fam in ["KPTracer_3508_3508_Apc_T2", "KPTracer_3515_3515_Lkb1_T1", "Quinn2021_10k_LG2", "Quinn2021_10k_LG3", "Quinn2021_10k_LG5"]:
        fig_rejection(rej, fam, out)
        fig_mass(dec, fam, out)
        fig_ot_vs_independent(vs, fam, out)
    fig_levels(rej, cmp_, ("KPTracer_3508_3508_Apc_T2", "family (primary site)"), ["KPTracer_3508_3508_Apc_T2"], "3508__D_C3__primaries__metastases",
               "KP-Tracer 3508, D_C3 run (pooled T1 + T2 -> pooled metastases, all T2 family), M4-E: level 1 primary site vs level 2 T2 subclones "
               "(T2 has one level-1 subclone, primary-only: level 2 not classifiable)", out, "D_level1_vs_level2_3508")
    fig_levels(rej, cmp_, ("KPTracer_3515_3515_Lkb1_T1", "family (primary site)"), ["KPTracer_3515_3515_Lkb1_T1"], "3515__D_C3__primaries__metastases",
               "KP-Tracer 3515, D_C3 run (pooled T1 + T2 + T3 -> pooled metastases, all T1 family), M4-E: level 1 primary site vs level 2 T1 subclones",
               out, "D_level1_vs_level2_3515")
    fig_levels(rej, cmp_, ("Quinn2021_10k (all clones)", "family (clone)"), ["Quinn2021_10k_LG2", "Quinn2021_10k_LG3", "Quinn2021_10k_LG5"],
               "10k__D_B__LL__metastases",
               "Quinn 2021 10k, D_B run (LL -> pooled tissues), M4-E: level 1 LL clone vs level 2 CRISPR subclones within LG2, LG3, LG5",
               out, "D_level1_vs_level2_10k")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
