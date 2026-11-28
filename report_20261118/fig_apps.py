"""Figure 5 - Biological applications with independent reference information: sciPlex2 (drug / dose) and Kang PBMC
(IFN-beta; no population gained or lost). Data: results/app_search_20261117/tacc_ot_20261123 (scored by
lineage_validation/app_search_20261117/09_score_apps.py; M4-E; calibration flags m4e_flags.tsv, shown, never removed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import glyphs as g  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results/app_search_20261117/tacc_ot_20261123"
OUT = fs.OUT / "Figure5"
DATA = OUT / "data"
CFG = {"l5best": ("best preprocessing", fs.METHOD["ConfidenceOT"]), "logcpm_pca": ("logCPM only", "#8FB0C6")}
DRUGS = ["Nutlin", "SAHA", "BMS", "Dex"]
DCOL = dict(zip(DRUGS, [fs.POP[0], fs.POP[3], fs.POP[2], fs.POP[1]]))
TYPES = ["CD14+ Monocytes", "FCGR3A+ Monocytes", "Dendritic cells", "Megakaryocytes", "B cells", "NK cells", "CD4 T cells", "CD8 T cells"]
SHORT = {"CD14+ Monocytes": "CD14+ mono", "FCGR3A+ Monocytes": "FCGR3A+ mono", "Dendritic cells": "DC", "Megakaryocytes": "Mk",
         "B cells": "B", "NK cells": "NK", "CD4 T cells": "CD4 T", "CD8 T cells": "CD8 T"}
MYELOID = {"CD14+ Monocytes", "FCGR3A+ Monocytes", "Dendritic cells"}


def load():
    DATA.mkdir(parents=True, exist_ok=True)
    fl = pd.read_csv(RES / "m4e_flags.tsv", sep="\t")
    fl["config"] = fl.config.map({"ranknm256-mi": "l5best", "logcpm__none": "logcpm_pca"})
    fl["valid"] = fl.calibration_m4e_inference_valid.astype(str).str.lower().eq("true")
    valid = fl.set_index(["unit", "config"]).valid
    dose = pd.read_csv(RES / "sciplex2/dose.tsv", sep="\t"); dose = dose[dose.method == "M4-E"].copy()
    dose["valid"] = [valid.get((u, c), True) for u, c in zip(dose.unit_id, dose.config)]
    cyc = pd.read_csv(RES / "sciplex2/cycle.tsv", sep="\t"); cyc = cyc[(cyc.method == "M4-E") & (cyc.side == "source")].copy()
    cyc["drug"] = cyc.unit_id.str.split("_").str[1]; cyc["diff"] = cyc.rejected_mean - cyc.retained_mean
    sp = pd.read_csv(RES / "sciplex2/dose_spearman.tsv", sep="\t"); sp = sp[sp.method == "M4-E"]
    kg = pd.read_csv(RES / "kang/gates.tsv", sep="\t")
    kg = kg[(kg.method == "M4-E") & (kg.side == "source") & (kg.label != "ALL") & (kg.unit_id != "Kang_ctrl_to_stim_pooled")]
    sl = pd.read_csv(RES / "kang/same_label.tsv", sep="\t"); sl = sl[sl.method == "M4-E"]
    ifn = pd.read_csv(RES / "kang/ifn_shift.tsv", sep="\t")
    ifn = ifn[(ifn.method == "M4-E") & (ifn.unit_id != "Kang_ctrl_to_stim_pooled")]
    dep = pd.concat([pd.read_csv(RES / f"{d}/depth.tsv", sep="\t").assign(dataset=d) for d in ("kang", "sciplex2")])
    dep = dep[(dep.method == "M4-E") & (dep.side == "source")]
    kflag = fl[(fl.dataset == "kang")]
    for n, t in (("b_sciplex2_dose", dose), ("b_sciplex2_spearman", sp), ("c_sciplex2_cycle", cyc), ("d_kang_rejection_by_type", kg),
                 ("e_kang_same_label", sl), ("e_kang_ifn_shift", ifn), ("f_depth_auroc", dep), ("flags", fl)):
        t.to_csv(DATA / f"{n}.tsv", sep="\t", index=False)
    return dose, cyc, sp, kg, sl, ifn, dep, kflag


def sax(sh, x, y, w, h):
    ax = sh.ax(x, y, w, h); ax.set_xlim(0, w); ax.set_ylim(0, h); fs.blank(ax)
    return ax


def panel_a(sh):
    sh.panel("a", 0.0, 3.62, 3.0, 5.40, "Two applications with independent references (schematic)")
    ax = sax(sh, 0.05, 3.64, 2.92, 1.46)
    # sciPlex2 row
    y = 1.08
    g.cell_cluster(ax, 0.18, y, fs.TGT, n=4, r=0.045, spread=0.09, seed=3)
    g.label(ax, 0.18, y - 0.22, "A549", size=fs.FS_ANNOT)
    g.arrow(ax, 0.38, y, 0.50, y)
    g.pill(ax, 0.52, y - 0.10, 0.78, 0.20, "vehicle -> dose", fs.TGT, fs_=fs.FS_SMALL)
    g.arrow(ax, 1.33, y, 1.45, y)
    g.bars_icon(ax, 1.68, y, 0.13, fs.TGT, vals=(0.95, 0.8, 0.6, 0.4, 0.25))
    g.label(ax, 1.92, y + 0.08, "sciPlex2: 4 drugs x 7 doses", ha="left", size=fs.FS_ANNOT)
    g.label(ax, 1.92, y - 0.08, "reference: dose, cells recovered", ha="left")
    g.rule(ax, 0.0, 2.92, 0.74)
    # Kang row
    y = 0.40
    g.cell_cluster(ax, 0.14, y + 0.05, fs.POP[0], n=3, r=0.04, spread=0.07, seed=1)
    g.cell_cluster(ax, 0.28, y - 0.05, fs.POP[1], n=3, r=0.04, spread=0.07, seed=2)
    g.label(ax, 0.30, y - 0.25, "PBMC, 8 donors", size=fs.FS_ANNOT)
    g.arrow(ax, 0.38, y, 0.50, y)
    g.pill(ax, 0.52, y - 0.10, 0.78, 0.20, "control -> IFN-beta", fs.SRC, fs_=fs.FS_SMALL)
    g.arrow(ax, 1.33, y, 1.45, y)
    g.cell_cluster(ax, 1.60, y + 0.05, fs.POP[0], n=3, r=0.04, spread=0.07, seed=4)
    g.cell_cluster(ax, 1.74, y - 0.05, fs.POP[1], n=3, r=0.04, spread=0.07, seed=5)
    g.label(ax, 1.92, y + 0.08, "Kang: same populations", ha="left", size=fs.FS_ANNOT)
    g.label(ax, 1.92, y - 0.08, "reference: no population lost", ha="left")


def panel_b(sh, dose, sp):
    sh.panel("b", 3.05, 3.62, 7.20, 5.40, "sciPlex2: rejection grows with dose (best preprocessing)")
    d = dose[dose.config == "l5best"]
    doses = sorted(d.dose.unique())
    ax = sh.ax(3.50, 4.02, 1.45, 1.02); fs.tidy(ax)
    for dr in DRUGS:
        t = d[d.drug == dr].sort_values("dose")
        ax.plot(t.dose, t.source_rejected, color=DCOL[dr], lw=1.0)
        for _, r in t.iterrows():
            ax.scatter(r.dose, r.source_rejected, s=10, color=DCOL[dr] if r.valid else "white", edgecolor=DCOL[dr], lw=0.7, zorder=3)
        rho = sp[(sp.config == "l5best") & (sp.drug == dr) & (sp.side == "source")].spearman_rejected_vs_dose.iloc[0]
        ax.text(1.02, {"Nutlin": 0.92, "SAHA": 0.77, "BMS": 0.62, "Dex": 0.47}[dr], f"{dr}  rho {rho:.2f}", transform=ax.transAxes,
                color=DCOL[dr], fontsize=fs.FS_SMALL, va="center")
    ax.set_xscale("log"); ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"]); ax.minorticks_off()
    ax.set_xlabel("Dose (uM)"); ax.set_ylabel("Vehicle cells rejected"); ax.set_ylim(-0.02, 1.02); ax.set_yticks([0, 0.5, 1])
    ax2 = sh.ax(6.30, 4.02, 0.85, 1.02); fs.tidy(ax2)
    for dr in DRUGS:
        t = d[d.drug == dr].sort_values("dose")
        ax2.plot(t.dose, t.target_cells, color=DCOL[dr], lw=0.9, marker="o", ms=2.2)
    ax2.set_xscale("log"); ax2.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"]); ax2.minorticks_off()
    ax2.set_xlabel("Dose (uM)"); ax2.set_ylabel("Treated cells"); ax2.set_ylim(0, 1400); ax2.set_yticks([0, 500, 1000])
    sh.text(3.50, 3.70, "open circle = calibration flag (kept); vehicle = dose 0 of the same drug", fontsize=fs.FS_SMALL, va="center")


def panel_c(sh, cyc):
    sh.panel("c", 0.0, 1.86, 3.0, 3.58, "Rejected vehicle cells are more proliferative")
    ax = sh.ax(0.62, 2.22, 2.28, 1.02); fs.tidy(ax)
    x = np.arange(len(DRUGS)); w = 0.18
    for k, (cfg, (lab, col)) in enumerate(CFG.items()):
        for j, score in enumerate(("G2M", "S")):
            t = cyc[(cyc.config == cfg) & (cyc.score == score)].groupby("drug")["diff"].median().reindex(DRUGS)
            off = (k * 2 + j - 1.5) * w
            ax.bar(x + off, t.to_numpy(), w * 0.92, color=col, alpha=1.0 if score == "G2M" else 0.55, lw=0,
                   label=f"{lab}, {score}")
    ax.axhline(0, color=fs.INK, lw=0.5)
    ax.set_xticks(x, DRUGS); ax.set_ylabel("Score, rejected - retained\n(median over doses)")
    ax.legend(loc="upper left", ncol=2, fontsize=fs.FS_SMALL, handlelength=0.9, columnspacing=0.6, borderaxespad=0, bbox_to_anchor=(0.0, 1.18))


def panel_d(sh, kg):
    sh.panel("d", 3.05, 1.86, 7.20, 3.58, "Kang: rejection concentrates in myeloid cells (8 donors)")
    ax = sh.ax(3.55, 2.36, 3.55, 0.88); fs.tidy(ax)
    rng = np.random.default_rng(0)
    for k, (cfg, (lab, col)) in enumerate(CFG.items()):
        t = kg[kg.config == cfg]
        for i, ty in enumerate(TYPES):
            v = t[t.label == ty].rejected_share.to_numpy()
            xx = i + (k - 0.5) * 0.32
            ax.scatter(xx + rng.uniform(-0.06, 0.06, len(v)), v, s=6, color=col, lw=0, alpha=0.8, zorder=3)
            ax.plot([xx - 0.12, xx + 0.12], [np.median(v)] * 2, color=fs.INK if k == 0 else col, lw=1.2, zorder=4)
        ax.scatter([], [], s=8, color=col, label=lab)
    ax.set_xticks(range(len(TYPES)), [SHORT[t] for t in TYPES], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Control cells rejected"); ax.set_ylim(-0.02, 1.02); ax.set_yticks([0, 0.5, 1])
    ax.legend(loc="upper right", fontsize=fs.FS_SMALL, handlelength=0.8, borderaxespad=0)
    for i, ty in enumerate(TYPES):
        if ty in MYELOID:
            ax.axvspan(i - 0.45, i + 0.45, color=fs.PALE, lw=0, zorder=0)


def panel_e(sh, sl, ifn):
    sh.panel("e", 0.0, 0.0, 4.25, 1.82, "Kang: identity kept, IFN response largest in myeloid cells")
    ax = sh.ax(0.50, 0.50, 0.85, 0.92); fs.tidy(ax)
    for k, (cfg, (lab, col)) in enumerate(CFG.items()):
        t = sl[sl.config == cfg]
        ax.scatter(np.full(len(t), k) + np.random.default_rng(1).uniform(-0.08, 0.08, len(t)), t.same_label_mass, s=7, color=col, lw=0, zorder=3)
        ax.scatter(np.full(len(t), k) + 0.25, t.independent_expectation, s=7, facecolor="white", edgecolor=col, lw=0.6, zorder=3)
    ax.set_xticks([0, 1], ["best", "logCPM"]); ax.set_xlim(-0.4, 1.5); ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Retained mass between\nthe same cell type")
    ax.text(0.02, 0.02, "filled: observed\nopen: expected", transform=ax.transAxes, fontsize=fs.FS_SMALL, va="bottom")
    ax2 = sh.ax(1.80, 0.50, 2.35, 0.92); fs.tidy(ax2)
    t = ifn[ifn.config == "l5best"]
    rng = np.random.default_rng(2)
    for i, ty in enumerate(TYPES):
        v = t[t.source_cell_type == ty].mean_shift.to_numpy()
        ax2.scatter(i + rng.uniform(-0.1, 0.1, len(v)), v, s=6, color=CFG["l5best"][1], lw=0, alpha=0.8, zorder=3)
        ax2.plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color=fs.INK, lw=1.2, zorder=4)
        if ty in MYELOID:
            ax2.axvspan(i - 0.45, i + 0.45, color=fs.PALE, lw=0, zorder=0)
    ax2.set_xticks(range(len(TYPES)), [SHORT[t] for t in TYPES], rotation=35, ha="right", rotation_mode="anchor")
    ax2.set_ylabel("IFN-a/g score,\nmatched - control"); ax2.set_ylim(0, 0.75); ax2.set_yticks([0, 0.25, 0.5, 0.75])


def panel_f(sh, dep):
    sh.panel("f", 4.30, 0.0, 7.20, 1.82, "Does rejection follow sequencing depth?")
    ax = sh.ax(4.85, 0.50, 2.25, 0.92); fs.tidy(ax)
    rng = np.random.default_rng(3)
    for j, ds in enumerate(("kang", "sciplex2")):
        for k, (cfg, (lab, col)) in enumerate(CFG.items()):
            v = dep[(dep.dataset == ds) & (dep.config == cfg)].depth_auroc_rejected_deeper.dropna().to_numpy()
            xx = j + (k - 0.5) * 0.36
            ax.scatter(xx + rng.uniform(-0.07, 0.07, len(v)), v, s=6, color=col, lw=0, alpha=0.8, zorder=3)
            ax.plot([xx - 0.13, xx + 0.13], [np.median(v)] * 2, color=fs.INK, lw=1.2, zorder=4)
            ax.text(xx, 1.04, f"{np.median(v):.2f}", ha="center", fontsize=fs.FS_SMALL, color=fs.INK)
    ax.axhline(0.5, **fs.REF_KW)
    ax.set_xticks([0, 1], ["Kang (9 units)", "sciPlex2 (28 units)"]); ax.set_ylim(0, 1.1); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Depth AUROC\n(> 0.5: rejected deeper)")


def write_text(dose, cyc, sp, kg, sl, ifn, dep, kflag):
    s = sp[(sp.config == "l5best") & (sp.side == "source")].set_index("drug").spearman_rejected_vs_dose
    kmed = kg[kg.config == "l5best"].groupby("label").rejected_share.median()
    imed = ifn[ifn.config == "l5best"].groupby("source_cell_type").mean_shift.median()
    sl_b = sl[sl.config == "l5best"]
    dmed = dep.groupby(["dataset", "config"]).depth_auroc_rejected_deeper.median()
    nflag = int((~dose[dose.config == "l5best"].valid).sum())
    cap = f"""# Figure 5 | Biological applications with independent reference information

**a**, Schematic. sciPlex2 (Srivatsan 2020): A549 cells, vehicle (dose 0 of each drug) -> each of 7 doses of Nutlin-3a,
SAHA, BMS-345541 and dexamethasone (28 units); the reference is the dose and the number of treated cells recovered,
independent of the RNA. Kang 2018: PBMC of 8 donors, control -> IFN-beta 6 h (8 donor units + pooled); the reference is
that no population is gained or lost (proportions unchanged), and cell types are never an OT input. Same frozen
pipeline as Figures 1-4 (best preprocessing = `ranknm256-midrank|none|scale|cos|pca50`, calibration v3, M4-E); logCPM
only = `logcpm|none|noscale|sqeuc|pca30` as comparison.

**b**, sciPlex2, best preprocessing. Left: fraction of vehicle (source) cells rejected against dose; Spearman over the 7
doses: Nutlin {s['Nutlin']:.2f}, SAHA {s['SAHA']:.2f}, BMS {s['BMS']:.2f}, Dex {s['Dex']:.2f}. Open circles: units whose
M4-E held-out calibration check is not valid ({nflag} of 28; shown, not removed). Right: treated cells recovered per dose.
At the highest doses few treated cells remain (Nutlin 100 uM: 86 cells) and rejection falls.

**c**, sciPlex2: G2M and S scores (mean log1p(CP10k) over the Tirosh 2016 lists) of rejected minus retained vehicle cells,
median over the 7 doses, for both configurations. With best preprocessing the rejected vehicle cells are more
proliferative (Nutlin, SAHA, BMS), consistent with drugs that reduce the proliferating fraction; logCPM only reverses
the sign for Nutlin, BMS and Dex.

**d**, Kang: fraction of control cells rejected per cell type, one point per donor (8 donor units, all with a valid
calibration check under best preprocessing), line = median. Shaded: myeloid types. Best preprocessing: CD14+
monocytes {kmed['CD14+ Monocytes']:.3f}, FCGR3A+ {kmed['FCGR3A+ Monocytes']:.3f}; T, B and NK at most
{max(kmed[t] for t in ['B cells', 'NK cells', 'CD4 T cells', 'CD8 T cells']):.3f}. No population is absent, so this is
rejection of large state change, not of loss.

**e**, Kang. Left: retained coupling mass between cells of the same type (filled) vs the independent expectation (open),
9 units; best preprocessing median {sl_b.same_label_mass.median():.2f} vs {sl_b.independent_expectation.median():.2f}. Right: per
retained control cell, the coupling-weighted IFN-alpha / IFN-gamma hallmark score of its matched stimulated cells minus
its own (best preprocessing, 8 donors, line = median): CD14+ {imed['CD14+ Monocytes']:.2f}, DC {imed['Dendritic cells']:.2f},
FCGR3A+ {imed['FCGR3A+ Monocytes']:.2f}, lymphoid {min(imed[t] for t in ['B cells', 'NK cells', 'CD4 T cells', 'CD8 T cells']):.2f}-{max(imed[t] for t in ['B cells', 'NK cells', 'CD4 T cells', 'CD8 T cells']):.2f}. The authors'
767-gene myeloid list is not in the files we hold; the hallmark sets stand in.

**f**, AUROC of per-cell total counts for rejected vs retained source cells (> 0.5: rejected cells are deeper), one point
per unit. Kang: best {dmed[('kang', 'l5best')]:.2f}, logCPM {dmed[('kang', 'logcpm_pca')]:.2f}; sciPlex2: best
{dmed[('sciplex2', 'l5best')]:.2f}, logCPM {dmed[('sciplex2', 'logcpm_pca')]:.2f}. logCPM only rejection in Kang follows depth.
"""
    (OUT / "caption.md").write_text(cap, encoding="utf-8")
    (OUT / "sources.md").write_text("""# Figure 5 sources

| Panel | Source | Derived table |
|---|---|---|
| a | schematic; PLAN_TACC.md sections 2 and 4 | — |
| b | `results/app_search_20261117/tacc_ot_20261123/sciplex2/{dose,dose_spearman}.tsv`, `m4e_flags.tsv` | `data/b_sciplex2_*.tsv` |
| c | `.../sciplex2/cycle.tsv` (side source) | `data/c_sciplex2_cycle.tsv` |
| d | `.../kang/gates.tsv` (side source, donor units) | `data/d_kang_rejection_by_type.tsv` |
| e | `.../kang/{same_label,ifn_shift}.tsv` | `data/e_kang_*.tsv` |
| f | `.../{kang,sciplex2}/depth.tsv` (side source) | `data/f_depth_auroc.tsv` |

Scored by `lineage_validation/app_search_20261117/09_score_apps.py` on TACC job 1164902 (OT jobs 1164877-1164882).
""", encoding="utf-8")
    notes = {
        "a": {"title": "Two applications with independent references", "compared": "Drug dose in A549 cells (sciPlex2) and IFN-beta stimulation of PBMC (Kang).",
              "observe": "Both have a reference the OT never sees: the dose and cells recovered, and the fact that no population is lost.",
              "meaning": "These test whether rejection tracks real biology, not labels derived from RNA."},
        "b": {"title": "sciPlex2: rejection grows with dose", "compared": "Fraction of vehicle cells rejected at each dose of four drugs.",
              "observe": f"Rejection rises with dose for Nutlin (rho {s['Nutlin']:.2f}) and SAHA (rho {s['SAHA']:.2f}); dexamethasone stays near zero.",
              "meaning": "The ordering matches the paper's response strength; at the top doses too few treated cells remain."},
        "c": {"title": "Rejected vehicle cells are more proliferative", "compared": "Cell-cycle scores of rejected vs retained vehicle cells.",
              "observe": "With best preprocessing rejected cells have higher G2M scores; logCPM only reverses the sign for three drugs.",
              "meaning": "Cycling vehicle cells lose their counterpart when the drug removes the proliferating fraction (consistent with the paper)."},
        "d": {"title": "Kang: rejection concentrates in myeloid cells", "compared": "Fraction of control cells rejected per cell type, 8 donors.",
              "observe": f"CD14+ monocytes {kmed['CD14+ Monocytes']:.2f} and other myeloid types are rejected most; lymphoid cells hardly at all.",
              "meaning": "No population is lost, so rejection marks the cells whose state changes most."},
        "e": {"title": "Identity kept, IFN response largest in myeloid cells", "compared": "Same-type matching and the IFN shift of matched cells.",
              "observe": f"Same-type mass {sl_b.same_label_mass.median():.2f} vs {sl_b.independent_expectation.median():.2f} expected; IFN shift highest in monocytes and DC.",
              "meaning": "Matching keeps cell identity and recovers the paper's myeloid-specific IFN programme."},
        "f": {"title": "Does rejection follow sequencing depth?", "compared": "Depth AUROC of rejected vs retained cells, both configurations.",
              "observe": f"Kang: best {dmed[('kang', 'l5best')]:.2f}, logCPM {dmed[('kang', 'logcpm_pca')]:.2f}.",
              "meaning": "With logCPM only, rejection largely follows depth; best preprocessing does not."},
    }
    (OUT / "panel_notes.json").write_text(json.dumps(notes, indent=1), encoding="utf-8")


def main():
    fs.apply()
    t = load()
    sh = fs.Sheet(7.2, 5.4, "Figure5")
    panel_a(sh); panel_b(sh, t[0], t[2]); panel_c(sh, t[1]); panel_d(sh, t[3]); panel_e(sh, t[4], t[5]); panel_f(sh, t[6])
    sh.save()
    write_text(*t)


if __name__ == "__main__":
    main()
