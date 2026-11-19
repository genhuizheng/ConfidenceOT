"""Collect the report documents from the built outputs: CAPTIONS.md, FIGURE_SOURCES.md, SLIDE_OUTLINE.md (read from
the deck itself, so the outline lists exactly the slides that exist) and a figure-panel <-> slide check.

    python -I assemble_docs.py --deck out/ConfidenceOT_20261118.pptx
"""
import argparse
import json
from pathlib import Path

from pptx import Presentation

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
STEMS = ["Figure1", "Figure2", "Figure3", "Figure4", "FigureFW"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deck", type=Path, required=True)
    a = p.parse_args()
    caps = ["# Figure captions (ConfidenceOT report, 2026-11-18)\n"]
    srcs = ["# Figure-to-source mapping (ConfidenceOT report, 2026-11-18)\n",
            "Every plotted number is in `out/<Figure>/data/*.tsv`, derived read-only from the tables below. Scripts: "
            "`report_20261118/fig1_framework.py`, `fig2_splatter.py`, `fig3_pdx.py`, `fig4_crispr.py`, `fig5_future.py`; "
            "Figure 1d input extracted on TACC by `tacc_fig1_example.py`.\n"]
    for s in STEMS:
        caps.append((OUT / s / "caption.md").read_text(encoding="utf-8").strip() + "\n")
        srcs.append((OUT / s / "sources.md").read_text(encoding="utf-8").strip().replace("# ", "## ", 1) + "\n")
    (OUT / "CAPTIONS.md").write_text("\n".join(caps), encoding="utf-8")
    (OUT / "FIGURE_SOURCES.md").write_text("\n".join(srcs), encoding="utf-8")
    prs = Presentation(str(a.deck))
    lines = ["# Slide-by-slide outline (read from the built deck)\n", "| Slide | Title | Shows | Notes (first sentence) |", "|---|---|---|---|"]
    seen = {}
    for i, sl in enumerate(prs.slides, 1):
        texts = [sh.text_frame.text for sh in sl.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        title = next((t for t in texts if "|" in t or t in ("Why do we need rejection?", "Next steps", "Summary")), texts[0] if texts else "")
        pics = [sh for sh in sl.shapes if sh.shape_type == 13 and sh.width > 914400 * 4]
        shows = "figure image" if pics else "text / native shapes"
        note = sl.notes_slide.notes_text_frame.text.split(". ")[0][:140] if sl.has_notes_slide else ""
        lines.append(f"| {i} | {title.splitlines()[0]} | {shows} | {note} |")
        seen[title.split(" |")[0]] = i
    # completeness check: every figure has an overview and every panel a slide
    check = ["\n## Overview / panel coverage\n", "| Figure | Overview slide | Panels | Panel slides |", "|---|---|---|---|"]
    for s, label in zip(STEMS, ["Figure 1", "Figure 2", "Figure 3", "Figure 4", "Future work"]):
        panels = list(json.loads((OUT / s / f"{s}_panels.json").read_text())["panels"])
        ov = seen.get(label, "MISSING")
        ps = [seen.get(f"{label} {k}" if label == "Future work" else f"{label}{k}", "MISSING") for k in panels]
        check.append(f"| {label} | {ov} | {', '.join(panels)} | {', '.join(map(str, ps))} |")
        assert ov != "MISSING" and "MISSING" not in ps, (label, ov, ps)
    (OUT / "SLIDE_OUTLINE.md").write_text("\n".join(lines + check) + "\n", encoding="utf-8")
    print("\n".join(check))


if __name__ == "__main__":
    main()
