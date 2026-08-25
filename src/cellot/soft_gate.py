"""Continuous soft-gate methods over balanced and UOT backbones."""

from traditional_ot.soft_gated import (
    SoftGatedMultiStartResult as SoftGatedUOTMultiStartResult,
    SoftGatedUOTResult,
    multi_start_soft_gated_uot as multi_start_soft_gate_uot,
    soft_gated_uot as fit_soft_gate_uot,
)
from traditional_ot.soft_gated_balanced import (
    SoftGatedBalancedMultiStartResult,
    SoftGatedBalancedOTResult,
    multi_start_soft_gated_balanced_ot as multi_start_soft_gate_balanced,
    soft_gated_balanced_ot as fit_soft_gate_balanced,
)

__all__ = [
    "SoftGatedBalancedMultiStartResult",
    "SoftGatedBalancedOTResult",
    "SoftGatedUOTMultiStartResult",
    "SoftGatedUOTResult",
    "fit_soft_gate_balanced",
    "fit_soft_gate_uot",
    "multi_start_soft_gate_balanced",
    "multi_start_soft_gate_uot",
]
