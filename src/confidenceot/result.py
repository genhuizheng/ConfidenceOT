"""Public result types for ConfidenceOT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ConfidenceOTResult:
    """Backend-independent terminal result returned by :class:`ConfidenceOT`."""

    coupling: NDArray[np.float64]
    source_gate: NDArray[np.bool_]
    target_gate: NDArray[np.bool_]
    source_score: NDArray[np.float64]
    target_score: NDArray[np.float64]
    source_raw_gate: NDArray[np.bool_]
    target_raw_gate: NDArray[np.bool_]
    backbone: Literal["balanced", "uot"]
    variant: Literal["exact", "reversible"]
    rejection_cost: float
    device: str
    backend: str
    inner_converged: bool
    outer_converged: bool
    cycle_detected: bool
    cycle_length: int
    n_outer_iterations: int
    total_inner_iterations: int
    objective: float
    fit_seconds: float

    @property
    def source_rejection_rate(self) -> float:
        return float(np.mean(~self.source_gate))

    @property
    def target_rejection_rate(self) -> float:
        return float(np.mean(~self.target_gate))

    @property
    def source_raw_acceptance(self) -> float:
        return float(np.mean(self.source_raw_gate))

    @property
    def target_raw_acceptance(self) -> float:
        return float(np.mean(self.target_raw_gate))

