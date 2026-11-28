"""Future Work figure - existing biological applications from the collected data sets.

Status and unit counts: lineage_validation/app_search_20261117/PLAN_TACC.md (frozen 2026-11-18 + dated post-hoc DRAFT),
06_prepare_tacc.py / 07_verify_inputs.py outputs on TACC (68 units, 136 files, 0 problems; OT not submitted),
scratchpad audit apps.md. Pilot numbers: results/app_search_20261117/kang_pilot/score/gates.tsv (pilot, 3,000 cells per
side; not an evaluation against the original conclusions).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = fs.OUT / "FigureFW"
DATA = OUT / "data"
STAGES = ["Identified", "Downloaded,\naudited", "Prepared\nfor OT", "OT\ncompleted", "Biology\nevaluated"]

# dataset, paper, system, source -> target, units, independent reference labels, stage (1-5), note
ROWS = [
    ("LARRY in vitro", "Weinreb 2020", "Mouse haematopoiesis, in vitro", "day 2 -> day 4 / 6", "4", "DNA clone barcodes", 4, "TACC run; pilot scored"),
    ("Cook A549", "Cook 2020", "Human A549, TGF-beta1 on / off", "adjacent times; withdrawal", "8", "time (design) only", 4, "local run, all cells"),
    ("Kang PBMC", "Kang 2018", "Human PBMC, IFN-beta 6 h", "control -> stimulated", "9", "donor (SNP), condition", 5, "OT 2026-11-23"),
    ("sciPlex2", "Srivatsan 2020", "Human A549, 4 drugs x 7 doses", "dose 0 -> each dose", "28", "drug, dose, cells per dose", 5, ""),
    ("Schiebinger", "Schiebinger 2019", "Mouse MEF -> iPSC reprogramming", "adjacent days per condition", "14", "day, condition", 5, "68k-cell subset"),
    ("C. elegans embryo", "Packer 2019", "Whole embryo, 12 time bins", "adjacent time bins", "11", "invariant lineage (names inferred)", 5, ""),
    ("Mouse pancreas", "Bastidas-Ponce 2019", "Endocrinogenesis E12.5-E15.5", "stage -> next stage", "3", "stage only", 5, ""),
    ("LARRY in vivo", "Weinreb 2020", "Transplanted progenitors", "day 2 -> day 9 / 16", "3", "DNA clone barcodes", 5, ""),
    ("SPLINTR AML", "Fennell 2022", "Mouse AML, chemotherapy vs vehicle", "baseline -> each mouse", "(16)", "expressed clone barcodes", 2, "*proposed"),
    ("Zebrafish heart", "Hu 2022", "Cryoinjury, Ctrl -> 30 dpi", "adjacent times per treatment", "(9)", "scar trees, within heart", 2, "*proposed"),
]


def panel_a(sh, t):
    sh.panel("a", 0.0, 1.95, 7.20, 5.40, "Collected data sets: where each one stands (status 2026-11-27)")
    ax = sh.ax(0.05, 2.00, 7.10, 3.05); ax.set_xlim(0, 100); ax.set_ylim(len(t) + 1.95, -0.2); fs.blank(ax)
    cols = [(0.0, "Data set (paper)"), (17.5, "Biological system"), (37.0, "Source -> Target"), (52.6, "Units"),
            (57.0, "Independent reference")]
    sx = [75.5 + i * 5.6 for i in range(5)]
    for x, h in cols:
        ax.text(x, 0.35, h, fontsize=fs.FS_SMALL + 0.3, color=fs.INK, va="center")
    for i, s in enumerate(STAGES):
        ax.text(sx[i], 0.35, s, fontsize=fs.FS_SMALL - 0.3, color=fs.INK, va="center", ha="center", linespacing=0.95)
    ax.plot([0, 100], [1.0, 1.0], color=fs.INK, lw=0.5)
    stage_col = ["#B0B5B9", "#8FB0C6", fs.SRC, fs.RETAIN, "#2E6B4F"]
    for r, row in t.iterrows():
        y = r + 1.6
        if r % 2 == 0:
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((0, y - 0.5), 100, 1.0, color=fs.PALE, lw=0, zorder=0))
        ax.text(0.0, y, f"{row.dataset}", fontsize=fs.FS_ANNOT, va="center")
        ax.text(0.0, y + 0.33, row.paper, fontsize=fs.FS_SMALL - 0.4, va="center", color=fs.INK)
        ax.text(17.5, y, row.system, fontsize=fs.FS_SMALL, va="center")
        ax.text(37.0, y, row.source_to_target, fontsize=fs.FS_SMALL, va="center")
        ax.text(54.0, y, row.units, fontsize=fs.FS_SMALL, va="center", ha="center")
        ax.text(57.0, y, row.reference_labels, fontsize=fs.FS_SMALL, va="center")
        for i in range(5):
            on = i < row.stage
            ax.scatter(sx[i], y, s=22 if on else 14, color=stage_col[row.stage - 1] if on else "white",
                       edgecolor=stage_col[row.stage - 1] if on else fs.NEUTRAL, lw=0.6, zorder=3)
        ax.plot([sx[0], sx[4]], [y, y], color=fs.GUIDE, lw=0.6, zorder=1)


READINGS = [  # dataset, independent reference?, reading (from RESULTS_SUMMARY.md, 2026-11-27), category
    ("sciPlex2", "yes", "rejection rises with dose (Nutlin, SAHA); Dex near 0", "consistent"),
    ("Kang PBMC", "yes", "identity kept; myeloid rejection and IFN shift", "consistent"),
    ("C. elegans", "no (names from RNA)", "lineage-compatible mass 10-30x expected", "consistent"),
    ("Mouse pancreas", "no", "Fev+ -> alpha / beta; Ngn3 steps weak", "partly"),
    ("Schiebinger", "no (signatures from RNA)", "serum: pluripotent kept, MEF/SASP rejected; 2i mixed", "partly"),
    ("LARRY in vivo", "yes", "93-96% of day-2 cells rejected", "not evaluable"),
]
CAT = {"consistent": "#2E6B4F", "partly": "#C9A54B", "not evaluable": "#9A9A9A"}


def load():
    DATA.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(ROWS, columns=["dataset", "paper", "system", "source_to_target", "units", "reference_labels", "stage", "note"])
    t.to_csv(DATA / "a_dataset_status.tsv", sep="\t", index=False)
    r = pd.DataFrame(READINGS, columns=["dataset", "independent_reference", "reading", "category"])
    r.to_csv(DATA / "b_readings.tsv", sep="\t", index=False)
    return t, r


def panel_b(sh, r):
    sh.panel("b", 0.0, 0.0, 4.55, 1.90, "What the evaluated data sets show (best preprocessing, M4-E)")
    ax = sh.ax(0.05, 0.04, 4.45, 1.56); ax.set_xlim(0, 100); ax.set_ylim(len(r) + 0.6, -0.4); fs.blank(ax)
    for x, h in ((0, "Data set"), (22, "Independent reference"), (44, "Reading vs the original paper")):
        ax.text(x, 0.0, h, fontsize=fs.FS_SMALL + 0.3, va="center")
    ax.plot([0, 100], [0.45, 0.45], color=fs.INK, lw=0.5)
    for i, row in r.iterrows():
        y = i + 1.0
        ax.text(0, y, row.dataset, fontsize=fs.FS_ANNOT, va="center")
        ax.text(22, y, row.independent_reference, fontsize=fs.FS_SMALL, va="center")
        ax.scatter(45, y, s=18, color=CAT[row.category], zorder=3)
        ax.text(47, y, f"{row.category}: {row.reading}", fontsize=fs.FS_SMALL, va="center")


def panel_c(sh, g=None):
    import glyphs as gl
    sh.panel("c", 4.60, 0.0, 7.20, 1.90, "Next (schematic)")
    W, H = 2.55, 1.56
    ax = sh.ax(4.62, 0.04, W, H); ax.set_xlim(0, W); ax.set_ylim(0, H); fs.blank(ax)
    cards = [("SPLINTR AML", "clone barcodes; chemo vs vehicle;\nprepare after 3 open decisions"),
             ("Zebrafish heart", "scar trees within each heart;\nunit list to agree"),
             ("LARRY in vitro", "TACC run done; map scores\nto the original conclusions")]
    for i, (h, t_) in enumerate(cards):
        y = 1.18 - i * 0.50
        gl.pill(ax, 0.02, y - 0.09, 0.95, 0.18, h, fs.SRC, fs_=fs.FS_SMALL)
        gl.label(ax, 1.03, y, t_, ha="left")


def write_text(t, r):
    cap = """# Future Work | Collected data sets: status and what the evaluated ones show

**a**, The ten collected data sets with the biological system, the Source -> Target units (same frozen pipeline as
Figures 1-5), the reference information that does not come from the RNA the OT sees, and a five-stage status ladder
(filled circles; status 2026-11-27). The six prepared data sets (68 units) were run on TACC with best preprocessing and
logCPM only (jobs 1164877-1164882, 136 rows, all completed) and scored (job 1164902; Schiebinger signatures job 1176050).
LARRY in vitro and Cook A549 were run earlier (TACC / local) and are not yet mapped to their papers. SPLINTR and zebrafish
are downloaded and audited; their preparation waits on open decisions (unit counts in brackets are proposed).

**b**, Reading of each evaluated data set against the original paper (RESULTS_SUMMARY.md): consistent, partly
consistent, or not evaluable. The independent-reference column states whether the reference is independent of the RNA.
Schiebinger is read with the authors' gene signatures (WOT gene_sets.gmx); the WOT cell sets share no cell with the
68k-cell subset.

**c**, The applications closest to completion. Schematic.
"""
    (OUT / "caption.md").write_text(cap, encoding="utf-8")
    (OUT / "sources.md").write_text("""# Future Work sources

| Panel | Source |
|---|---|
| a | `lineage_validation/app_search_20261117/PLAN_TACC.md` (frozen + post-hoc 2026-11-18 / 2026-11-23); TACC job records; `data/a_dataset_status.tsv` |
| b | `results/app_search_20261117/tacc_ot_20261123/RESULTS_SUMMARY.md`; `data/b_readings.tsv` |
| c | PLAN_TACC post-hoc decisions |
""", encoding="utf-8")
    notes = {
        "a": {"title": "Where each data set stands", "compared": "Ten collected data sets on a five-stage status ladder.",
              "observe": "Six data sets (68 units) are now run and evaluated with the frozen pipeline; two earlier runs wait for mapping; two wait for preparation.",
              "meaning": "Evaluation has started; the remaining work is preparation and mapping, not new methods."},
        "b": {"title": "What the evaluated data sets show", "compared": "Each data set's result against the claims of its original paper.",
              "observe": "sciPlex2 and Kang are consistent with independent references; C. elegans is consistent but its names come from RNA; in Schiebinger serum the late steps follow the authors' signatures (2i mixed); LARRY in vivo is not evaluable.",
              "meaning": "The strongest applications are the ones with references the OT never sees (Figure 5)."},
        "c": {"title": "Next", "compared": "The data sets closest to completion.",
              "observe": "SPLINTR and zebrafish need their preparation decisions; LARRY in vitro needs mapping to the paper.",
              "meaning": "Next step: evaluate ConfidenceOT on the remaining applications with independent reference information."},
    }
    (OUT / "panel_notes.json").write_text(json.dumps(notes, indent=1), encoding="utf-8")


def main():
    fs.apply()
    t, g = load()
    sh = fs.Sheet(7.2, 5.4, "FigureFW")
    panel_a(sh, t); panel_b(sh, g); panel_c(sh)
    sh.save()
    write_text(t, g)


if __name__ == "__main__":
    main()
