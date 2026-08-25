"""Ungated OT backbones used as benchmark baselines."""

from traditional_ot.balanced import BalancedOTResult, balanced_ot
from traditional_ot.partial import PartialOTResult, partial_wasserstein_uniform
from traditional_ot.unbalanced import UOTResult, unbalanced_ot

__all__ = [
    "BalancedOTResult",
    "PartialOTResult",
    "UOTResult",
    "balanced_ot",
    "partial_wasserstein_uniform",
    "unbalanced_ot",
]
