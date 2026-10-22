"""The two Level 2 OT methods (PLAN.md, "The two methods"). Exactly these two; no other OT run is made.

Imported by run_pairwise.py, score_pairwise.py and make_inputs.py (run manifest), so the three cannot drift.
"""

EPSILON = 0.1
# method: preprocessing label (Preprocessing.from_label), counts, solver, rejection setting
METHODS = {
    "A_standard_OT": {"preprocessing": "logcpm_noscale", "counts": "raw", "solver": "BOT",
                      "solver_call": "traditional_ot.balanced.balanced_ot(threshold=1e-9, max_iterations=10000)",
                      "rejection": "none (Balanced OT, uniform marginals)"},
    "B_our_OT": {"preprocessing": "ranknm256_noscale_ds_cos", "counts": "ds", "solver": "UOT",
                 "solver_call": "ConfidenceOT(backbone='uot', variant='exact', lambda_a=1, lambda_b=1, tolerance=1e-4)",
                 "rejection": "source/target rejection budget 0 -> bounds (0, 0); KL-relaxed marginals kept"},
}
