"""CellOT public API.

The public namespace separates transport backbones from selective gate
mechanisms.  ``traditional_ot`` remains available as a compatibility layer,
but new benchmark and package code should import from ``cellot``.
"""

from cellot.baselines import (
    BalancedOTResult,
    PartialOTResult,
    UOTResult,
    balanced_ot,
    partial_wasserstein_uniform,
    unbalanced_ot,
)
from cellot.cost_matrix_gate import (
    BalancedCostMatrixGateResult,
    UOTCostMatrixGateResult,
    fit_balanced_cost_matrix_gate,
    fit_uot_cost_matrix_gate,
    solve_fixed_balanced_cost_matrix_gate,
    solve_fixed_uot_cost_matrix_gate,
)
from cellot.soft_gate import (
    SoftGatedBalancedMultiStartResult,
    SoftGatedBalancedOTResult,
    SoftGatedUOTResult,
    fit_soft_gate_balanced,
    fit_soft_gate_uot,
    multi_start_soft_gate_balanced,
    multi_start_soft_gate_uot,
)
from confidenceot import ConfidenceOT, ConfidenceOTResult, m4_exact, m4_reversible

__all__ = [
    "BalancedOTResult",
    "PartialOTResult",
    "UOTResult",
    "balanced_ot",
    "partial_wasserstein_uniform",
    "unbalanced_ot",
    "BalancedCostMatrixGateResult",
    "UOTCostMatrixGateResult",
    "fit_balanced_cost_matrix_gate",
    "fit_uot_cost_matrix_gate",
    "solve_fixed_balanced_cost_matrix_gate",
    "solve_fixed_uot_cost_matrix_gate",
    "SoftGatedBalancedMultiStartResult",
    "SoftGatedBalancedOTResult",
    "SoftGatedUOTResult",
    "fit_soft_gate_balanced",
    "fit_soft_gate_uot",
    "multi_start_soft_gate_balanced",
    "multi_start_soft_gate_uot",
    "ConfidenceOT",
    "ConfidenceOTResult",
    "m4_exact",
    "m4_reversible",
]
