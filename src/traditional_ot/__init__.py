"""Traditional optimal-transport baselines with cycle-safe lazy exports.

The M4 implementation now lives in :mod:`confidenceot`. Lazy compatibility
exports keep historical imports working without importing deprecated M4 shims
while ConfidenceOT itself is still being initialized.
"""

from __future__ import annotations
from importlib import import_module


def _exports(module: str, names: str) -> dict[str, str]:
    return {name: module for name in names.split()}


_EXPORTS: dict[str, str] = {}
_EXPORTS.update(_exports("traditional_ot.balanced", "BalancedOTResult CFBOTResult balanced_ot confidence_filtered_balanced_ot solve_fixed_gate_balanced_ot"))
_EXPORTS.update(_exports("traditional_ot.bidirectional", "BidirectionalCalibrationResult BidirectionalUOTResult CalibrationError GateUpdateDiagnostics InnerSolverError PostSelectionUOTResult PopulationTestResult calibrate_bidirectional_rejection_cost confidence_filtered_bidirectional_uot constrained_gate_update filtered_cost population_monte_carlo_test refit_post_selection_uot solve_fixed_bidirectional_uot"))
_EXPORTS.update(_exports("traditional_ot.bidirectional_balanced", "BalancedInnerSolverError BidirectionalBalancedOTResult bidirectional_balanced_filtered_cost confidence_filtered_bidirectional_balanced_ot solve_fixed_bidirectional_balanced_ot"))
_EXPORTS.update(_exports("traditional_ot.reservoir", "BirthDeathUOTResult birth_death_uot"))
_EXPORTS.update(_exports("traditional_ot.icpot", "ICPOTResult intent_controlled_partial_ot"))
_EXPORTS.update(_exports("traditional_ot.partial", "PartialOTResult partial_wasserstein_uniform"))
_EXPORTS.update(_exports("traditional_ot.entropic_partial", "ConfidenceFilteredEntropicPartialOTResult EntropicPartialRefitResult EntropicPartialOTResult PartialInnerSolverError TwoStageEntropicPartialOTResult confidence_filtered_entropic_partial_ot entropic_partial_ot exact_cardinality_gate_update partial_gate_coefficients refit_entropic_partial_ot solve_fixed_confidence_filtered_partial_ot two_stage_confidence_filtered_entropic_partial_ot"))
_EXPORTS.update(_exports("traditional_ot.outlier", "OutlierTrimmedOTResult cross_snapshot_outlier_scores outlier_trimmed_ot outlier_trimmed_uot"))
_EXPORTS.update(_exports("traditional_ot.sinkhorn", "traditional_method"))
_EXPORTS.update(_exports("traditional_ot.unbalanced", "CFUOTResult CalibrationResult UOTResult calibrate_rejection_cost confidence_filtered_uot solve_fixed_gate_uot unbalanced_ot"))
_EXPORTS.update(_exports("traditional_ot.selective", "FixedGateSelectiveOTResult SelectiveInnerSolverError SelectiveCalibrationResult SelectiveOTError SelectiveOTResult SelectiveRefitResult PosthocSelectiveOTResult TwoStageSelectiveOTResult UnsupportedSelectiveVariantError budgeted_gate_update calibrate_selective_rejection_cost gate_coefficients partner_restricted_statistics posthoc_selective_ot refit_selective_ot selective_filtered_cost selective_ot solve_fixed_gate_selective_ot two_stage_selective_ot"))
_EXPORTS.update(_exports("traditional_ot.support_restricted", "FixedSupportResult PrefixSelection SupportRestrictedOTResult prefix_select solve_fixed_support_ot support_restricted_ot"))
_EXPORTS.update(_exports("traditional_ot.soft_gated", "SoftGateInnerState SoftGatedMultiStartResult SoftGatedRun SoftGatedUOTResult multi_start_soft_gated_uot project_soft_coverage soft_gated_uot"))
_EXPORTS.update(_exports("traditional_ot.soft_gated_balanced", "BalancedSoftInnerState SoftGatedBalancedMultiStartResult SoftGatedBalancedOTResult SoftGatedBalancedRun balanced_soft_envelope_gradient multi_start_soft_gated_balanced_ot round_transport_plan soft_gated_balanced_ot"))

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'traditional_ot' has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
