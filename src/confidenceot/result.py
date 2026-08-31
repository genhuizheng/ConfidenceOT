"""Public result types for ConfidenceOT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BinConfidence:
    """Per-bin diagnostics for a terminal ConfidenceOT gate decision.

    These values are readouts of an already completed fit.  They do not alter
    the transport solve or the binary gate.  For M4-R, ``decision_cost`` is the
    counterfactual conditional transport cost; for M4-E it is the
    support-restricted conditional transport cost.
    """

    decision_cost: NDArray[np.float64]
    rejection_cost: float
    signed_rejection_margin: NDArray[np.float64]
    relative_rejection_margin: NDArray[np.float64]
    gate_coefficient: NDArray[np.float64]
    raw_rejected: NDArray[np.bool_]
    final_rejected: NDArray[np.bool_]
    budget_overridden: NDArray[np.bool_]
    cost_kind: Literal["counterfactual", "support_restricted"]

    @property
    def counterfactual_cost(self) -> NDArray[np.float64]:
        """Return the M4-R counterfactual cost.

        The property is deliberately unavailable for M4-E because its gate is
        based on a different, support-restricted conditional cost.
        """
        if self.cost_kind != "counterfactual":
            raise AttributeError(
                "counterfactual_cost is defined only for the reversible M4-R variant."
            )
        return self.decision_cost

    def normalized_rejection_score(
        self, *, temperature: float | None = None
    ) -> NDArray[np.float64]:
        """Map the signed margin monotonically to ``[0, 1]`` for display.

        This is an algorithmic visualization score, not a calibrated
        probability.  The default temperature is the rejection cost.
        """
        scale = self.rejection_cost if temperature is None else float(temperature)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("temperature must be finite and positive.")
        values = np.clip(self.signed_rejection_margin / scale, -709.0, 709.0)
        return 1.0 / (1.0 + np.exp(-values))


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
    source_confidence: BinConfidence
    target_confidence: BinConfidence
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
