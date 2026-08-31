"""High-level public API for ConfidenceOT (M4-E and M4-R)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike

from confidenceot.cuda import CUDAUnavailableError, cuda_available, fit_cuda
from confidenceot.result import BinConfidence, ConfidenceOTResult


Backbone = Literal["balanced", "uot"]
Variant = Literal["exact", "reversible"]
Device = Literal["auto", "cpu", "cuda"]


class ConfidenceOT:
    """Bidirectional confidence-filtered OT with M4-E or M4-R gates.

    Parameters are deliberately shared across CPU and CUDA backends.  The CPU
    implementation remains the numerical reference; ``device='cuda'`` runs the
    PyTorch CUDA implementation.
    """

    def __init__(
        self,
        *,
        backbone: Backbone = "uot",
        variant: Variant = "reversible",
        rejection_cost: float = 0.5,
        epsilon: float = 0.1,
        lambda_a: float = 1.0,
        lambda_b: float = 1.0,
        source_rejection_budget: float = 0.15,
        target_rejection_budget: float = 0.15,
        tolerance: float = 1e-3,
        gate_tolerance: float = 0.0,
        max_iterations: int = 20_000,
        max_outer_iterations: int = 30,
        device: Device = "auto",
        cuda_dtype: Literal["float32", "float64"] = "float32",
        fallback_to_cpu: bool = False,
        warn_on_terminal: bool = True,
    ) -> None:
        if backbone not in ("balanced", "uot"):
            raise ValueError("backbone must be 'balanced' or 'uot'.")
        if variant not in ("exact", "reversible"):
            raise ValueError("variant must be 'exact' or 'reversible'.")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        if not 0 <= source_rejection_budget < 1 or not 0 <= target_rejection_budget < 1:
            raise ValueError("Rejection budgets must lie in [0, 1).")
        self.backbone = backbone
        self.variant = variant
        self.rejection_cost = float(rejection_cost)
        self.epsilon = float(epsilon)
        self.lambda_a = float(lambda_a)
        self.lambda_b = float(lambda_b)
        self.source_rejection_budget = float(source_rejection_budget)
        self.target_rejection_budget = float(target_rejection_budget)
        self.tolerance = float(tolerance)
        self.gate_tolerance = float(gate_tolerance)
        self.max_iterations = int(max_iterations)
        self.max_outer_iterations = int(max_outer_iterations)
        self.device = device
        self.cuda_dtype = cuda_dtype
        self.fallback_to_cpu = bool(fallback_to_cpu)
        self.warn_on_terminal = bool(warn_on_terminal)

    def fit(
        self,
        cost_matrix: ArrayLike,
        *,
        source_weights: ArrayLike | None = None,
        target_weights: ArrayLike | None = None,
        initial_source_gate: ArrayLike | None = None,
        initial_target_gate: ArrayLike | None = None,
    ) -> ConfidenceOTResult:
        cost = np.asarray(cost_matrix, dtype=np.float64)
        requested = self.device
        resolved = "cuda" if requested == "auto" and cuda_available() else ("cpu" if requested == "auto" else requested)
        kwargs = dict(
            backbone=self.backbone, variant=self.variant,
            rejection_cost=self.rejection_cost, epsilon=self.epsilon,
            lambda_a=self.lambda_a, lambda_b=self.lambda_b,
            source_weights=None if source_weights is None else np.asarray(source_weights),
            target_weights=None if target_weights is None else np.asarray(target_weights),
            initial_source_gate=None if initial_source_gate is None else np.asarray(initial_source_gate),
            initial_target_gate=None if initial_target_gate is None else np.asarray(initial_target_gate),
            source_rejection_budget=self.source_rejection_budget,
            target_rejection_budget=self.target_rejection_budget,
            tau=self.gate_tolerance, threshold=self.tolerance,
            max_iterations=self.max_iterations,
            max_outer_iterations=self.max_outer_iterations,
        )
        if resolved == "cuda":
            try:
                result = fit_cuda(cost, dtype=self.cuda_dtype, **kwargs)
                return self._warn_if_needed(result)
            except CUDAUnavailableError:
                if not self.fallback_to_cpu:
                    raise
                warnings.warn(
                    "CUDA was requested but is unavailable; ConfidenceOT is falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return self._warn_if_needed(self._fit_cpu(cost, **kwargs))

    def fit_many(
        self,
        cost_matrices: Sequence[ArrayLike],
        *,
        workers: int = 1,
    ) -> list[ConfidenceOTResult]:
        """Fit independent cost matrices concurrently in input order.

        Parallelism is deliberately outside each M4 solve, so Sinkhorn and
        Gauss--Seidel gate-update order are unchanged. CUDA fits receive
        independent streams in :func:`confidenceot.cuda.fit_cuda`; CPU fits
        use independent NumPy solver states. ``workers=1`` is the serial
        numerical reference.
        """
        if workers <= 0:
            raise ValueError("workers must be a positive integer.")
        matrices = tuple(cost_matrices)
        if workers == 1 or len(matrices) <= 1:
            return [self.fit(matrix) for matrix in matrices]
        with ThreadPoolExecutor(
            max_workers=min(int(workers), len(matrices)),
            thread_name_prefix="confidenceot-fit",
        ) as executor:
            return list(executor.map(self.fit, matrices))

    def _warn_if_needed(self, result: ConfidenceOTResult) -> ConfidenceOTResult:
        if not self.warn_on_terminal:
            return result
        messages: list[str] = []
        if not result.inner_converged:
            messages.append("the inner Sinkhorn solver reached its iteration cap")
        if not result.outer_converged:
            messages.append("the M4 outer loop reached its iteration cap")
        if result.cycle_detected:
            messages.append(f"a gate cycle of length {result.cycle_length} was detected")
        if messages:
            warnings.warn(
                "ConfidenceOT retained a finite terminal result: " + "; ".join(messages) + ".",
                RuntimeWarning,
                stacklevel=2,
            )
        return result

    @staticmethod
    def _fit_cpu(cost: np.ndarray, **kwargs: object) -> ConfidenceOTResult:
        from confidenceot._cpu_balanced import (
            confidence_filtered_bidirectional_balanced_ot as fit_balanced_cost_matrix_gate,
        )
        from confidenceot._cpu_uot import (
            confidence_filtered_bidirectional_uot as fit_uot_cost_matrix_gate,
        )

        backbone = str(kwargs["backbone"])
        started = time.perf_counter()
        common = dict(
            rejection_cost=kwargs["rejection_cost"], epsilon=kwargs["epsilon"],
            variant=kwargs["variant"], source_weights=kwargs["source_weights"],
            target_weights=kwargs["target_weights"],
            initial_source_gate=kwargs["initial_source_gate"],
            initial_target_gate=kwargs["initial_target_gate"],
            source_rejection_budget=kwargs["source_rejection_budget"],
            target_rejection_budget=kwargs["target_rejection_budget"],
            tau_s=kwargs["tau"], threshold=kwargs["threshold"],
            max_iterations=kwargs["max_iterations"],
            max_outer_iterations=kwargs["max_outer_iterations"],
        )
        if backbone == "balanced":
            fitted = fit_balanced_cost_matrix_gate(cost, **common)
        else:
            fitted = fit_uot_cost_matrix_gate(
                cost, lambda_a=kwargs["lambda_a"], lambda_b=kwargs["lambda_b"], **common
            )
        rejection_cost = float(kwargs["rejection_cost"])
        variant = str(kwargs["variant"])

        def confidence_readout(
            decision_cost: np.ndarray,
            coefficient: np.ndarray,
            raw_gate: np.ndarray,
            final_gate: np.ndarray,
        ) -> BinConfidence:
            decision = np.asarray(decision_cost, dtype=np.float64)
            raw_rejected = ~np.asarray(raw_gate, dtype=bool)
            final_rejected = ~np.asarray(final_gate, dtype=bool)
            margin = decision - rejection_cost
            return BinConfidence(
                decision_cost=decision,
                rejection_cost=rejection_cost,
                signed_rejection_margin=margin,
                relative_rejection_margin=margin / rejection_cost,
                gate_coefficient=np.asarray(coefficient, dtype=np.float64),
                raw_rejected=raw_rejected,
                final_rejected=final_rejected,
                budget_overridden=raw_rejected != final_rejected,
                cost_kind=(
                    "counterfactual" if variant == "reversible" else "support_restricted"
                ),
            )

        source_confidence = confidence_readout(
            fitted.source_gate_score,
            fitted.source_gate_coefficient,
            fitted.source_raw_gate,
            fitted.source_gate,
        )
        target_confidence = confidence_readout(
            fitted.target_gate_score,
            fitted.target_gate_coefficient,
            fitted.target_raw_gate,
            fitted.target_gate,
        )
        return ConfidenceOTResult(
            coupling=np.asarray(fitted.coupling),
            source_gate=np.asarray(fitted.source_gate),
            target_gate=np.asarray(fitted.target_gate),
            source_score=np.asarray(fitted.source_gate_coefficient),
            target_score=np.asarray(fitted.target_gate_coefficient),
            source_raw_gate=np.asarray(fitted.source_raw_gate),
            target_raw_gate=np.asarray(fitted.target_raw_gate),
            source_confidence=source_confidence,
            target_confidence=target_confidence,
            backbone=backbone,
            variant=variant,
            rejection_cost=rejection_cost,
            device="cpu",
            backend="numpy",
            inner_converged=bool(fitted.inner_converged),
            outer_converged=bool(fitted.outer_converged),
            cycle_detected=bool(fitted.cycle_detected),
            cycle_length=int(fitted.cycle_length),
            n_outer_iterations=int(fitted.n_outer_iterations),
            total_inner_iterations=int(fitted.total_inner_iterations),
            objective=float(fitted.objective),
            fit_seconds=time.perf_counter() - started,
        )


def m4_exact(cost_matrix: ArrayLike, **kwargs: object) -> ConfidenceOTResult:
    """Convenience function for M4-E."""
    return ConfidenceOT(variant="exact", **kwargs).fit(cost_matrix)


def m4_reversible(cost_matrix: ArrayLike, **kwargs: object) -> ConfidenceOTResult:
    """Convenience function for M4-R."""
    return ConfidenceOT(variant="reversible", **kwargs).fit(cost_matrix)
