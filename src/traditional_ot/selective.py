"""Compatibility facade for the archived cost-substitution formulation.

The current selective balanced/UOT method is :func:`support_restricted_ot` in
``traditional_ot.support_restricted``. Imports from this module intentionally
continue to resolve so historical tests and benchmarks remain reproducible;
new code should not use this facade.
"""

from traditional_ot.archive.cost_substitution_selective import (
    FixedGateSelectiveOTResult,
    PosthocSelectiveOTResult,
    SelectiveCalibrationResult,
    SelectiveInnerSolverError,
    SelectiveOTError,
    SelectiveOTResult,
    SelectiveRefitResult,
    TwoStageSelectiveOTResult,
    UnsupportedSelectiveVariantError,
    budgeted_gate_update,
    calibrate_selective_rejection_cost,
    gate_coefficients,
    partner_restricted_statistics,
    posthoc_selective_ot,
    refit_selective_ot,
    selective_filtered_cost,
    selective_ot,
    solve_fixed_gate_selective_ot,
    two_stage_selective_ot,
)

__all__ = [
    "FixedGateSelectiveOTResult",
    "PosthocSelectiveOTResult",
    "SelectiveCalibrationResult",
    "SelectiveInnerSolverError",
    "SelectiveOTError",
    "SelectiveOTResult",
    "SelectiveRefitResult",
    "TwoStageSelectiveOTResult",
    "UnsupportedSelectiveVariantError",
    "budgeted_gate_update",
    "calibrate_selective_rejection_cost",
    "gate_coefficients",
    "partner_restricted_statistics",
    "posthoc_selective_ot",
    "refit_selective_ot",
    "selective_filtered_cost",
    "selective_ot",
    "solve_fixed_gate_selective_ot",
    "two_stage_selective_ot",
]
