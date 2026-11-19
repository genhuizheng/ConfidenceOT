"""Supplementary figures for the 2026-11-18 deck: earlier figures of the CURRENT result sets, shown as they are (not part
of any main figure). Superseded / withdrawn figures are not listed (old-calibration LTL331 runs, rejected-vs-retained
GSEA, pre-fix OT-depth outputs, unpaired Splatter runs). Clade ids in the older CRISPR figures are mapped to the main
Figure 4 labels on each slide.
"""
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
V3 = R / "v3_grid_20261104/figures"
OTDB = R / "ot_depth_benchmark_20261117"
RLD = R / "rna_lineage_discordance_20261110/figures/representative_20261112"
LEGEND_SOLVERS = OTDB / "splatter/figures/legend_solvers.png"

# (group, file, title, caption, note)
SUPP = [
    ("Figure 2", V3 / "splatter/splatter_f1_top30.png", "Case 2 F1, top 30 of 296 configurations",
     "Splatter N = 5,000 grid. Rows ranked by Case 2 rejection F1 at L5 (outlined column); each cell = mean of 3 replicates "
     "over both Case 2 directions. Rank 1 = main configuration; red row = earlier production baseline (rank 171).", None),
    ("Figure 2", V3 / "splatter/splatter_all_cases_top30.png", "All three cases for the same 30 configurations",
     "Left: Case 1 false rejection rate (lower is better). Middle: Case 2 rejection F1. Right: Case 3 unmatched recall "
     "(effective readout). N = 5,000, mean of 3 replicates.", None),
    ("Figure 2", V3 / "splatter/splatter_f1_all296.png", "Case 2 F1 for all 296 configurations",
     "All configurations ranked by L5 Case 2 F1; left tracks give cost, PCA, gene scaling and encoding of each row. "
     "L1-L3 hardly separate configurations; L4-L5 do.", None),
    ("Figure 2", V3 / "splatter/levels_benchmark.png", "What each technical level does to the data",
     "Display of one local replicate (N = 200, replicate 1): target/source nCount and nFeature ratios, expression of 60 "
     "genes per level, and example gates against ground truth for each case. Ratios are from this replicate, not the "
     "N = 5,000 data.", None),
    ("Figure 2", OTDB / "splatter/figures/S1_gate_C1_rejected_fraction.png", "ConfidenceOT gate by level and cell class",
     "Fixed-cost benchmark, main cost (C1), N = 200 per side, mean of 8 replicates: rejected fraction of shared and "
     "unmatched cells, per level and case.", None),
    ("Figure 2", OTDB / "splatter/figures/S2_conflict_C2_wrong_similar_depth_share.png", "Depth lure with a log1p-PCA cost",
     "Conflict arm at L0 (target Groups 1-2 thinned to f): share of source Group 1-2 mass on wrong groups with "
     "source-like depth. ConfidenceOT (red) lies on top of entropic UOT; N = 200, mean +- s.e.m. of 8 replicates.", "legend"),
    ("Figure 2", OTDB / "splatter/figures/S2_global_C1_target_rejected_fraction.png", "Global depth gap at L0, main cost",
     "Whole target thinned to f, every population shared: rejected fraction of target cells (all false) by ConfidenceOT; "
     "N = 200, mean +- s.e.m. of 8 replicates.", "legend"),
    ("Figure 2", OTDB / "bot_uot/figures/conflict_f0.1_uot_minus_balanced_by_cost.png", "UOT minus balanced OT, by cost",
     "Conflict arm at f = 0.1: correct-group share of entropic UOT minus entropic balanced OT (epsilon 0.1), per level and "
     "cost. C1 main configuration; C2 log1p-PCA; C3 ranknm512-expected noscale cos PCA30; C4 logCPM noscale cos PCA30.", None),
    ("Figure 3", V3 / "pdx/pdx_slices_splatter_l5_best.png", "Matching drawn on the authors' UMAP",
     "Authors' UMAP, same coordinates in every slice. Lines: 250 retained source cells per pair, sampled by coupling "
     "mass, each to its retained target of largest mass (M4-E). Grey: rejected in every pair.", None),
    ("Figure 3", V3 / "pdx/pdx_transitions_splatter_l5_best.png", "Where each source state's retained mass goes",
     "Row-normalised retained coupling per adjacent pair (main configuration); shares >= 0.15 printed; blank = state "
     "absent or fully rejected.", None),
    ("Figure 3", V3 / "pdx/pdx_rejected_splatter_l5_best.png", "Rejected fraction by state and time point",
     "Main configuration. Left: as source of the next pair; right: as target of the previous pair. * fewer than 20 cells.", None),
    ("Figure 3", V3 / "pdx/pdx_rejected_median296.png", "Rejected fraction, median over 296 configurations",
     "Same layout as the previous slide, median over all 296 preprocessing configurations.", None),
    ("Figure 3", V3 / "pdx/pdx_flow_logcpm.png", "Comparison arm: logCPM",
     "Same display as Figure 3c for logcpm noscale sqeuc PCA30 (logCPM + PCA only, no cosine, no scaling). Rejection is higher at "
     "12->16 (95%) and 16->20 (98%) than with the main configuration.", None),
    ("Figure 4", RLD / "1b_cell_tree_3457_Apc_T1.png", "Family 3457 T1: cell-level CRISPR tree",
     "Raw CRISPR edits only (no OT): average-linkage tree on edit distance 1 - s, clade cut 0.5; tracks: clade, RNA "
     "state, site.", "Clade ids in this figure: c6 = 3457-A, c1 = 3457-B, c12 = 3457-C (Figure 4)."),
    ("Figure 4", RLD / "1a_cell_tree_3724_NT_T1.png", "Family 3724 T1: cell-level CRISPR tree",
     "1,244 of 20,992 cells drawn; median 7 informative edits per cell; not resolved by the frozen rule.",
     "Clade ids in this figure: c45 = 3724-A, c5 = 3724-B."),
    ("Figure 4", RLD / "4b_flow_3724_T1_all_metastases.png", "RNA-OT vs CRISPR-OT flow, 3724 T1 -> all metastases",
     "Clade-to-clade coupling on cells retained by both gates (common support), each coupling normalised to 1. Grey "
     "shared, blue RNA-enriched, orange CRISPR-enriched. Clade-level mass, not cell-to-cell matches.",
     "Clade ids in this figure: c45 = 3724-A, c5 = 3724-B."),
    ("Figure 4", RLD / "4a_flow_3460_T1_N6.png", "RNA-OT vs CRISPR-OT flow, 3460 T1 -> N6",
     "Same display for a resolved Lkb1 family: shared 57% of each coupling at clade level.",
     "c8 / c9 / c1 / c2 are clade ids of family 3460 Lkb1 T1 (no letter labels are used for this family)."),
    ("Figure 4", RLD / "4c_baseline_independent_coupling.png", "Is clade-level agreement more than clade sizes give?",
     "45 KP units: clade-level shared mass of RNA-OT vs CRISPR-OT against each coupling's shared mass with the "
     "independent (size-only) coupling. RNA-OT sits near the size-only line.", None),
    ("Figure 4", RLD / "4d_clade_vs_cell_level_3460_N6.png", "Same targets, two levels of agreement",
     "3460 T1 -> N6: clade-level shared share vs exact cell-level top-15 partner overlap per target clade (n cells).",
     "c8 / c9 / c1 / c2 are clade ids of family 3460 Lkb1 T1."),
    ("Figure 4", RLD / "2b_state_concentration_all_states.png", "Are RNA states tied to few clades?",
     "Per mouse and state (>= 50 cells, all eligible KP cells): largest clade share minus the within-site null (red = "
     "lineage-concentrated, blue = spread over clades); dot area = cells. Raw edits, no OT.", None),
]
