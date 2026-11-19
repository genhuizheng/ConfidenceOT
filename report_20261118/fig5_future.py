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
    ("Kang PBMC", "Kang 2018", "Human PBMC, IFN-beta 6 h", "control -> stimulated", "9", "donor (SNP), condition", 3, "pilot run locally"),
    ("sciPlex2", "Srivatsan 2020", "Human A549, 4 drugs x 7 doses", "dose 0 -> each dose", "28", "drug, dose, cells per dose", 3, ""),
    ("Schiebinger", "Schiebinger 2019", "Mouse MEF -> iPSC reprogramming", "adjacent days per condition", "14", "day, condition", 3, "68k-cell subset"),
    ("C. elegans embryo", "Packer 2019", "Whole embryo, 12 time bins", "adjacent time bins", "11", "invariant lineage (names inferred)", 3, ""),
    ("Mouse pancreas", "Bastidas-Ponce 2019", "Endocrinogenesis E12.5-E15.5", "stage -> next stage", "3", "stage only", 3, ""),
    ("LARRY in vivo", "Weinreb 2020", "Transplanted progenitors", "day 2 -> day 9 / 16", "3", "DNA clone barcodes", 3, ""),
    ("SPLINTR AML", "Fennell 2022", "Mouse AML, chemotherapy vs vehicle", "baseline -> each mouse", "16*", "expressed clone barcodes", 2, "*proposed"),
    ("Zebrafish heart", "Hu 2022", "Cryoinjury, Ctrl -> 30 dpi", "adjacent times per treatment", "9*", "scar trees (within heart)", 2, "*proposed"),
]


def load():
    DATA.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(ROWS, columns=["dataset", "paper", "system", "source_to_target", "units", "reference_labels", "stage", "note"])
    t.to_csv(DATA / "a_dataset_status.tsv", sep="\t", index=False)
    g = pd.read_csv(REPO / "results/app_search_20261117/kang_pilot/score/gates.tsv", sep="\t")
    g = g[(g.unit == "K0_ctrl_to_stim") & (g.method == "M4-E")]
    g.to_csv(DATA / "c_kang_pilot_gates.tsv", sep="\t", index=False)
    return t, g


def panel_a(sh, t):
    sh.panel("a", 0.0, 1.95, 7.20, 5.40, "Collected data sets: where each one stands (status 2026-11-18)")
    ax = sh.ax(0.05, 2.00, 7.10, 3.05); ax.set_xlim(0, 100); ax.set_ylim(len(t) + 1.95, -0.2); fs.blank(ax)
    cols = [(0.0, "Data set (paper)"), (17.5, "Biological system"), (37.0, "Source -> Target"), (52.6, "Units"),
            (57.0, "Independent reference")]
    sx = [75.5 + i * 5.6 for i in range(5)]
    for x, h in cols:
        ax.text(x, 0.35, h, fontsize=fs.FS_SMALL + 0.3, color=fs.SUBTLE, va="center")
    for i, s in enumerate(STAGES):
        ax.text(sx[i], 0.35, s, fontsize=fs.FS_SMALL - 0.3, color=fs.SUBTLE, va="center", ha="center", linespacing=0.95)
    ax.plot([0, 100], [1.0, 1.0], color=fs.INK, lw=0.5)
    stage_col = ["#B0B5B9", "#8FB0C6", fs.SRC, fs.RETAIN, fs.REJECT]
    for r, row in t.iterrows():
        y = r + 1.6
        if r % 2 == 0:
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((0, y - 0.5), 100, 1.0, color=fs.PALE, lw=0, zorder=0))
        ax.text(0.0, y, f"{row.dataset}", fontsize=fs.FS_ANNOT, va="center")
        ax.text(0.0, y + 0.33, row.paper, fontsize=fs.FS_SMALL - 0.4, va="center", color=fs.SUBTLE)
        ax.text(17.5, y, row.system, fontsize=fs.FS_SMALL, va="center")
        ax.text(37.0, y, row.source_to_target, fontsize=fs.FS_SMALL, va="center")
        ax.text(54.0, y, row.units, fontsize=fs.FS_SMALL, va="center", ha="center")
        ax.text(57.0, y, row.reference_labels, fontsize=fs.FS_SMALL, va="center")
        for i in range(5):
            on = i < row.stage
            ax.scatter(sx[i], y, s=22 if on else 14, color=stage_col[row.stage - 1] if on else "white",
                       edgecolor=stage_col[row.stage - 1] if on else fs.NEUTRAL, lw=0.6, zorder=3)
        ax.plot([sx[0], sx[4]], [y, y], color=fs.GUIDE, lw=0.6, zorder=1)
    ax.text(0.0, len(t) + 1.55, "No data set has reached 'Biology evaluated'. 68 units are prepared and checked (Kang 9, sciPlex2 28, "
            "Schiebinger 14, C. elegans 11, pancreas 3, LARRY in vivo 3);\nOT has not been submitted for them. * proposed, preparation pending.",
            fontsize=fs.FS_SMALL, color=fs.SUBTLE, va="center", linespacing=1.15)


def panel_b(sh):
    import glyphs as gl
    sh.panel("b", 0.0, 0.0, 4.55, 1.90, "Most worth completing: independent reference information")
    W, H = 4.45, 1.58
    ax = sh.ax(0.05, 0.04, W, H); ax.set_xlim(0, W); ax.set_ylim(0, H); fs.blank(ax)
    fs.schematic_tag(ax, 1.0, 1.04)
    cards = [
        ("LARRY in vivo", "clone barcodes\n(DNA, not RNA)", "Do day-2 cells couple\nto later cells of\ntheir own clone?", "clone"),
        ("SPLINTR AML", "expressed barcodes;\nvehicle vs chemo", "Are clones lost after\nchemo rejected more\nthan after vehicle?", "barcode"),
        ("sciPlex2", "dose; cells\nrecovered per dose", "Does rejection grow\nwith dose where\ncells are lost?", "dose"),
        ("Kang PBMC", "no population absent\n(negative control)", "Does rejection stay\nlow? Is the shift\nlargest in myeloid?", "control"),
    ]
    w, gap = 1.04, 0.09
    for i, (h, ref, q, icon) in enumerate(cards):
        x = 0.02 + i * (w + gap)
        gl.frame(ax, x, 0.02, w, 1.40, color=fs.NEUTRAL, lw=0.6)
        gl.pill(ax, x + 0.06, 1.20, w - 0.12, 0.16, h, fs.SRC, fs_=fs.FS_ANNOT)
        cx, cy = x + w / 2, 0.96
        if icon == "clone":
            for k, c in enumerate((fs.POP[0], fs.POP[0], fs.POP[2])):
                gl.cell(ax, cx - 0.22 + k * 0.10, cy + 0.02, 0.035, c, nucleus=False)
            gl.arrow(ax, cx + 0.03, cy + 0.02, cx + 0.12, cy + 0.02)
            for k, c in enumerate((fs.POP[0], fs.POP[2])):
                gl.cell(ax, cx + 0.17 + k * 0.09, cy + 0.02, 0.035, c, nucleus=False)
        elif icon == "barcode":
            gl.edit_array(ax, cx - 0.25, cy - 0.015, 0.5, 0.07, [1, 3, 0, 5, 2, 0, 4, 1], fs.POP)
        elif icon == "dose":
            gl.bars_icon(ax, cx, cy + 0.01, 0.12, fs.TGT, vals=(0.95, 0.85, 0.7, 0.45, 0.25))
        else:
            gl.cell_cluster(ax, cx - 0.16, cy + 0.01, fs.POP[0], n=3, r=0.03, spread=0.05, seed=1)
            gl.arrow(ax, cx - 0.06, cy + 0.01, cx + 0.06, cy + 0.01, color=fs.RETAIN)
            gl.cell_cluster(ax, cx + 0.16, cy + 0.01, fs.POP[0], n=3, r=0.03, spread=0.05, seed=2)
        gl.label(ax, x + 0.07, 0.76, ref, ha="left", va="top", color=fs.SUBTLE)
        gl.label(ax, x + 0.07, 0.44, q, ha="left", va="top", color=fs.INK)


def panel_c(sh, g):
    sh.panel("c", 4.60, 0.0, 7.20, 1.90, "Pilot only: Kang control -> IFN-beta")
    s = g[(g.side == "source") & (g.label != "ALL")].sort_values("rejected_share")
    ax = sh.ax(5.62, 0.50, 1.25, 1.08); fs.tidy(ax)
    yy = np.arange(len(s))
    ax.barh(yy, s.rejected_share, color=fs.SRC, height=0.62)
    ax.set_yticks(yy, [f"{l.replace(' Monocytes', ' mono.').replace(' cells', '')} ({n:,})" for l, n in zip(s.label, s.cells)], fontsize=fs.FS_SMALL)
    ax.set_xlim(0, 0.36); ax.set_xticks([0, 0.1, 0.2, 0.3]); ax.set_xlabel("Control cells rejected", labelpad=1)
    for y, v in zip(yy, s.rejected_share):
        ax.text(v + 0.006, y, f"{v:.2f}", fontsize=fs.FS_SMALL - 0.4, va="center")
    allv = g[(g.side == "source") & (g.label == "ALL")].rejected_share.iloc[0]
    sh.text(4.70, 0.04, f"Pilot: 3,000 cells per side, 8 donors pooled, M4-E; all cells {allv:.2f}; (n) = cells.",
            fontsize=fs.FS_SMALL - 0.3, color=fs.SUBTLE, va="bottom")


def write_text(t, g):
    cap = """# Future Work | Existing biological applications from the collected data sets

**a**, The ten data sets collected and audited by the application search, with the biological system, the planned Source ->
Target units (same fixed pipeline as in Figures 1-4: main configuration, calibration v3, M4-E, at most 10,000 cells per
side), the reference information that does not come from the RNA the OT sees, and a five-stage status ladder (filled
circles). Stages: identified; downloaded and audited; prepared for OT (integer counts, manifest, checked by
`07_verify_inputs.py`: 136 input files, 0 problems); OT completed; biological evaluation completed. LARRY in vitro OT was
run on TACC (2026-11-13) and scored on a local 3,000-cell pilot; Cook A549 was run locally on all cells. No data set has
reached biological evaluation. SPLINTR and zebrafish unit counts are proposals (PLAN_TACC post-hoc draft, decisions
pending). Cell-type labels exist for most data sets but come from expression, so they are not listed as independent.

**b**, The applications with independent reference information that are closest to completion, and the question each one
would answer. Schematic.

**c**, Pilot only (local, 3,000 cells per side, control -> IFN-beta stimulated, 8 donors pooled, M4-E): fraction of control
cells rejected per cell type. Rejection is highest in CD14+ monocytes and partly follows sequencing depth (source depth AUROC
0.64); this is not yet an evaluation against the published result.
"""
    (OUT / "caption.md").write_text(cap, encoding="utf-8")
    (OUT / "sources.md").write_text("""# Future Work sources

| Panel | Source |
|---|---|
| a | `lineage_validation/app_search_20261117/PLAN_TACC.md` (frozen 2026-11-18 + post-hoc draft), TACC prepare/verify logs of 2026-11-18 (`$SCRATCH/OT_project/app_search_20261117/{<dataset>/prep/units.tsv, verify/inputs.tsv}`), scratchpad audit apps.md; table in `data/a_dataset_status.tsv` |
| b | PLAN_TACC scoring sections (what each data set can and cannot support) |
| c | `results/app_search_20261117/kang_pilot/score/gates.tsv` (unit K0_ctrl_to_stim, M4-E); `data/c_kang_pilot_gates.tsv` |
""", encoding="utf-8")
    notes = {
        "a": {"title": "Where can we apply ConfidenceOT next?", "compared": "Ten collected data sets on a five-stage status ladder.",
              "observe": "Six data sets (68 units) are prepared and checked; two are piloted; none has a biological evaluation yet.",
              "meaning": "The next step is evaluation, not more data collection or new method development."},
        "b": {"title": "Selected applications", "compared": "The data sets whose reference labels do not come from the RNA.",
              "observe": "Clone barcodes (LARRY in vivo, SPLINTR), dose and cell recovery (sciPlex2), and a no-loss control (Kang).",
              "meaning": "These can test rejection and matching against information the OT never sees."},
        "c": {"title": "Pilot: Kang IFN-beta", "compared": "Control vs IFN-beta-stimulated PBMC, 3,000 cells per side (pilot).",
              "observe": "Few cells are rejected overall; rejection is highest in CD14+ monocytes.",
              "meaning": "A first look only; the published comparison has not been made."},
    }
    (OUT / "panel_notes.json").write_text(json.dumps(notes, indent=1), encoding="utf-8")


def main():
    fs.apply()
    t, g = load()
    sh = fs.Sheet(7.2, 5.4, "FigureFW")
    panel_a(sh, t); panel_b(sh); panel_c(sh, g)
    sh.save()
    write_text(t, g)


if __name__ == "__main__":
    main()
