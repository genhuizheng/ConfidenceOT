# ConfidenceOT research report, 2026-11-18: storyline, figure design and build rules

Main line: **Method → controlled validation → real biological application → independent lineage evaluation → broader
applications.** The levels of evidence differ and are never merged: Splatter has known truth; PDX has none (consistency
with the published biology only); CRISPR gives independent lineage information for RNA-only matching; the application
datasets are prepared, not evaluated.

## Slide-by-slide storyline

| # | Slide | One message | Why here / leads to |
|---|---|---|---|
| 1 | Title | ConfidenceOT: matching cells between samples, and saying when a cell has no match | — |
| 2 | Why do we need rejection? (introduction + story map) | Two samples rarely contain the same populations; forcing every cell into a match creates false correspondences | Sets the question Figure 1 answers |
| 3 | Figure 1 overview | ConfidenceOT = UOT matching + a per-cell keep/reject decision | — |
| 4 | Fig 1a Biological problem | Shared, source-only and target-only populations | Defines the three kinds of cells |
| 5 | Fig 1b Standard OT vs ConfidenceOT | Balanced OT forces every cell; UOT shrinks mass but decides nothing; ConfidenceOT keeps UOT and adds the decision | What is new |
| 6 | Fig 1c Workflow | The exact code path, counts → retained correspondence | How it is computed |
| 7 | Fig 1d Representative example (real Splatter result) | The lost population is rejected, shared cells are kept and matched to their own group | Leads to: how well does it work in general? |
| 8 | Figure 2 overview | What works, what fails, and under which conditions | — |
| 9 | Fig 2a Benchmark design | 5 populations, Cases 1–3, levels L1–L5 | — |
| 10 | Fig 2b Matching and rejection | High F1 at L1–L4, drop at L5 (emerged population) | — |
| 11 | Fig 2c Preprocessing sensitivity | The main configuration; cosine vs squared Euclidean matters most | — |
| 12 | Fig 2d Balanced OT vs UOT vs ConfidenceOT | Transported mass, conditional accuracy, correct mass and rejection are different quantities | — |
| 13 | Fig 2e Failure cases | log1p-PCA lets depth win over biology; a very large global depth gap makes ConfidenceOT reject shared cells | Leads to: on real data, with no truth, what does it show? |
| 14 | Figure 3 overview | A real cell-state change: LTL331 PDX, PRAD → NEPC after castration | — |
| 15 | Fig 3a Design | 8 time points, castration, relapse, states | — |
| 16 | Fig 3b Composition | The state composition changes with time (data, no OT) | — |
| 17 | Fig 3c OT matching | Rejection and retained state-to-state coupling per adjacent pair | — |
| 18 | Fig 3d Biological observations | Retained flows that agree with the published order; robustness over 296 configurations | — |
| 19 | Fig 3e Comparison with the paper | What agrees, what is not testable, what OT cannot claim | Leads to: is RNA matching supported by independent lineage? |
| 20 | Figure 4 overview | Independent CRISPR lineage information as a check on RNA-only matching | — |
| 21 | Fig 4a Design | RNA → RNA-only OT; CRISPR edits → evaluation only | — |
| 22 | Fig 4b Tumour-level | Metastases retain their own-family primary more than other primaries | — |
| 23 | Fig 4c Editing structure | 3457 T1: founder edits, clades 3457-A/B/C, RNA states | — |
| 24 | Fig 4d RNA-OT vs CRISPR-OT | Agreement at clade level, not at the level of exact cell partners | — |
| 25 | Fig 4e Limitations and controls | Resolution, missingness, clone size, site composition | Leads to: where next? |
| 26 | Future work: dataset overview | 10 collected datasets on a 5-stage status ladder; 68 units prepared, none evaluated | — |
| 27 | Selected applications | The 2–3 datasets with independent reference labels | — |
| 28 | Next steps | Evaluate ConfidenceOT on existing applications with independent reference information | — |
| 29 | Summary | One line per figure | — |

(Exact panel slides follow the final panels; one panel or two tightly linked panels per slide.)

## Figure build rules (all figures)

- Style: `report_20261118/figstyle.py` (`apply()`, `Sheet`, palette, font sizes). Never define new colours for an
  existing meaning. Side colours `SRC`/`TGT`; retained `RETAIN`; rejected `REJECT`; methods `METHOD[...]`; neutral greys.
- Size: every main figure is **7.2 in wide × 5.4 in tall** (183 × 137 mm), `Sheet(7.2, 5.4, "FigureN")`.
- Panels: register each panel with `sh.panel(letter, x0, y0, x1, y1, title=...)`. The rectangle is the crop used for
  the panel slide, so it must contain the panel's axes, labels, legend, n and statistics, and must not overlap another
  panel. Panel letters lower-case bold.
- One rendering: `sh.save()` writes `out/FigureN/FigureN.{png,pdf,svg}` and the panel crops
  `out/FigureN/FigureN_panel_<k>.{png,pdf,svg}` from the same figure object. Slides use only these files.
- Data: every plotted number comes from a table written by the figure script to `out/FigureN/data/<panel>_*.tsv`
  (derived read-only from the repository's result tables). No number typed by hand into a plot except design
  constants (e.g. level definitions).
- Real vs schematic: schematic panels/sub-panels carry `schematic_tag(ax)`; real-data panels state the data set,
  comparison unit and n in the panel.
- Each figure script also writes `out/FigureN/caption.md` (full caption: title, then one paragraph per panel with n,
  units and statistics), `out/FigureN/sources.md` (panel → source files, columns, filters, commit) and
  `out/FigureN/panel_notes.json`: `{"<k>": {"title": ..., "compared": ..., "observe": ..., "meaning": ...}}`
  (one or two short sentences each, plain English, for the panel slide).
- Language: simple and exact; no mechanistic claims without evidence; rejection is never called death, differentiation
  or ancestry; OT coupling is never called lineage.
- Never: invalidated / pre-fix / collided coupling files; superseded runs (see the audits); new OT runs; solver fits;
  writing into existing result folders.
