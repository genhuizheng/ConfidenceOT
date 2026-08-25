"""Binary cost-matrix gate methods over balanced and UOT backbones.

These methods replace costs outside the accepted source/target mask.  They
are selective OT methods, not Traditional OT baselines.
"""

from confidenceot._cpu_uot import (
    BidirectionalUOTResult as UOTCostMatrixGateResult,
    confidence_filtered_bidirectional_uot as fit_uot_cost_matrix_gate,
    solve_fixed_bidirectional_uot as solve_fixed_uot_cost_matrix_gate,
)
from confidenceot._cpu_balanced import (
    BidirectionalBalancedOTResult as BalancedCostMatrixGateResult,
    confidence_filtered_bidirectional_balanced_ot as fit_balanced_cost_matrix_gate,
    solve_fixed_bidirectional_balanced_ot as solve_fixed_balanced_cost_matrix_gate,
)

__all__ = [
    "BalancedCostMatrixGateResult",
    "UOTCostMatrixGateResult",
    "fit_balanced_cost_matrix_gate",
    "fit_uot_cost_matrix_gate",
    "solve_fixed_balanced_cost_matrix_gate",
    "solve_fixed_uot_cost_matrix_gate",
]
