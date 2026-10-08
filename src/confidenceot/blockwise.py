"""Memory-bounded twin of the dense torch solver in :mod:`confidenceot.cuda`.

``cuda._fit_cuda_impl`` keeps the N x M cost and several N x M working tensors
on the device at once, which caps it near 50,000 x 50,000 bins on a 96 GB GPU.
Every operation it applies to those tensors is a reduction along rows or along
columns -- a log-sum-exp, a weighted row or column sum.  A row reduction needs
one row of the cost; a column reduction can be accumulated over rows.  This
module carries out the same arithmetic on row blocks of the cost, rebuilt from
the coordinates each time a block is needed, and keeps only O(N + M) state
between blocks.

It is an implementation of that solver, not a variant of the method.  The
Sinkhorn updates and their order, the stopping rule, the gate coefficients,
the deterministic projection (:func:`confidenceot.cuda._project_gate`, called,
not copied), cycle detection, the final consistency solve, the decision costs
and the objective are the dense solver's, step for step.  Two things differ:
a reduction split across blocks sums in a different floating-point order, and
the dense coupling is never formed.  The result carries the two final dual
potentials instead, from which any block of the coupling is rebuilt exactly as
the dense solver forms it.  ``tests/test_blockwise.py`` holds the two solvers
to each other.

The cost is the production one, ``squared_euclidean(source, target) / scale``:
the expansion ``|a|^2 + |b|^2 - 2 a.b``, clipped at zero, then divided, as
``cancer_metastasis/02_run_pair.py`` builds it before calling the dense solver.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import time
from typing import Any, Iterator, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from confidenceot.cuda import (
    CUDAUnavailableError,
    _bounds_active,
    _generalized_kl,
    _project_gate,
    _weights,
)
from confidenceot.result import BinConfidence, ConfidenceOTResult

# 2**28 entries is 1 GiB per float32 working block.  A block holds a handful
# of such tensors at once, so the default stays far inside a 96 GB device.
DEFAULT_BLOCK_ELEMENTS = 1 << 28


@dataclass(frozen=True)
class CoordinateCost:
    """A cost matrix held as the coordinates it is computed from.

    ``cost[i, j] = max(|s_i|^2 + |t_j|^2 - 2 s_i.t_j, 0) / scale``, the value
    ``squared_euclidean(source, target) / scale`` gives, never stored whole.
    """

    source: NDArray[np.floating]
    target: NDArray[np.floating]
    scale: float

    def __post_init__(self) -> None:
        source = np.asarray(self.source)
        target = np.asarray(self.target)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source and target must be two-dimensional coordinates.")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source and target must share one feature dimension.")
        if min(source.shape[0], target.shape[0]) == 0:
            raise ValueError("source and target must be non-empty.")
        if not (np.all(np.isfinite(source)) and np.all(np.isfinite(target))):
            raise ValueError("coordinates must be finite.")
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ValueError("scale must be positive and finite.")

    @property
    def shape(self) -> tuple[int, int]:
        return int(np.asarray(self.source).shape[0]), int(np.asarray(self.target).shape[0])

    def dense(self) -> NDArray[np.floating]:
        """The whole matrix, built the production way.  For tests and checks."""
        from confidenceot.preprocessing import squared_euclidean

        return squared_euclidean(self.source, self.target) / self.scale


@dataclass(frozen=True)
class BlockwiseResult(ConfidenceOTResult):
    """A :class:`ConfidenceOTResult` whose coupling was never formed.

    ``coupling`` is ``None``.  ``log_u`` and ``log_v`` are the dual potentials
    of the final consistency solve; with the gates, ``rejection_cost``,
    ``epsilon`` and the weights they determine every entry of the coupling the
    dense solver would have returned.  :func:`transport_reductions` and
    :func:`materialize_coupling` rebuild it block by block.
    """

    log_u: NDArray[np.float64]
    log_v: NDArray[np.float64]
    source_weights: NDArray[np.float64]
    target_weights: NDArray[np.float64]
    epsilon: float
    block_rows: int
    dtype: str


@dataclass
class _Potentials:
    log_u: Any
    log_v: Any
    converged: bool
    iterations: int
    error: float


def _import_torch(device: str) -> Any:
    try:
        import torch
    except ImportError as error:
        raise CUDAUnavailableError(
            "Blockwise ConfidenceOT requires PyTorch. Install the `confidenceot[cuda]` extra."
        ) from error
    if device == "cuda" and not torch.cuda.is_available():
        raise CUDAUnavailableError(
            "PyTorch is installed without an available CUDA runtime. Install a CUDA-enabled "
            "PyTorch build and verify `torch.cuda.is_available()` first."
        )
    if device not in ("cpu", "cuda"):
        raise ValueError("Internal torch device must be 'cpu' or 'cuda'.")
    return torch


@contextmanager
def _full_precision_matmul(torch: Any) -> Iterator[None]:
    """Keep float32 matrix products in float32 for the duration of a fit.

    TF32 would round the cost's inner products to a 10-bit mantissa.  PyTorch
    leaves it off by default; a caller that switched it on globally must not
    change this solver's cost, so it is held off here and restored after.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def resolve_block_rows(block_rows: int | None, n_source: int, n_target: int) -> int:
    """Rows per block: explicit, or as many as fit DEFAULT_BLOCK_ELEMENTS."""
    if block_rows is None:
        return max(1, min(n_source, DEFAULT_BLOCK_ELEMENTS // max(n_target, 1)))
    if int(block_rows) < 1:
        raise ValueError("block_rows must be a positive integer.")
    return min(int(block_rows), n_source)


class _Geometry:
    """The coordinates on the device, and rows of the cost built from them."""

    def __init__(self, torch: Any, cost: CoordinateCost, device: Any, dtype: Any) -> None:
        self.torch = torch
        source = torch.as_tensor(np.asarray(cost.source), device=device, dtype=dtype)
        target = torch.as_tensor(np.asarray(cost.target), device=device, dtype=dtype)
        self.source = source
        self.target_t = target.T.contiguous()
        self.source_sq = torch.sum(source * source, dim=1)
        self.target_sq = torch.sum(target * target, dim=1)
        self.scale = float(cost.scale)
        self.n_source, self.n_target = cost.shape
        self.device = device
        self.dtype = dtype

    def block(self, start: int, stop: int) -> Any:
        """Rows ``start:stop`` of ``squared_euclidean(source, target) / scale``."""
        value = self.torch.addmm(
            self.source_sq[start:stop, None], self.source[start:stop], self.target_t,
            beta=1.0, alpha=-2.0,
        )
        value.add_(self.target_sq[None, :])
        value.clamp_min_(0.0)
        value.div_(self.scale)
        return value


class _GatedBlocks:
    """Row blocks of the optimisation cost under one pair of gates.

    The dense solver forms ``torch.where(active, cost, rejection_cost)`` with
    ``active = source_gate[:, None] & target_gate[None, :]`` and every quantity
    below from it.  Each method here is one of those quantities, accumulated
    over row blocks.
    """

    def __init__(
        self,
        torch: Any,
        geometry: _Geometry,
        source_gate: np.ndarray,
        target_gate: np.ndarray,
        *,
        rejection_cost: float,
        epsilon: float,
        block_rows: int,
        log_a: Any,
        log_b: Any,
    ) -> None:
        self.torch = torch
        self.geometry = geometry
        self.rejection_cost = float(rejection_cost)
        self.epsilon = float(epsilon)
        self.log_a = log_a
        self.log_b = log_b
        device = geometry.device
        self.source_gate = torch.as_tensor(np.asarray(source_gate, dtype=bool), device=device)
        self.target_gate = torch.as_tensor(np.asarray(target_gate, dtype=bool), device=device)
        self.target_inactive = ~self.target_gate
        self.any_target_inactive = bool(np.any(~np.asarray(target_gate, dtype=bool)))
        n = geometry.n_source
        self.ranges = [(start, min(start + block_rows, n)) for start in range(0, n, block_rows)]
        inactive = np.asarray(~np.asarray(source_gate, dtype=bool))
        self.inactive_rows: dict[int, Any] = {}
        for start, stop in self.ranges:
            rows = np.flatnonzero(inactive[start:stop])
            if rows.size:
                self.inactive_rows[start] = torch.as_tensor(rows, device=device, dtype=torch.long)

    def gate_(self, block: Any, start: int) -> Any:
        """In place: entries outside the active support become the rejection cost."""
        if self.any_target_inactive:
            block.masked_fill_(self.target_inactive[None, :], self.rejection_cost)
        rows = self.inactive_rows.get(start)
        if rows is not None:
            block.index_fill_(0, rows, self.rejection_cost)
        return block

    def log_kernel_(self, block: Any, start: int, stop: int) -> Any:
        """In place: optimisation cost -> ``log a_i + log b_j - cost / epsilon``."""
        block.div_(self.epsilon).neg_().add_(self.log_b[None, :]).add_(self.log_a[start:stop, None])
        return block

    def row_lse(self, log_v: Any) -> Any:
        """``logsumexp(log_kernel + log_v[None, :], dim=1)``."""
        torch = self.torch
        out = torch.empty(self.geometry.n_source, device=self.geometry.device, dtype=self.geometry.dtype)
        for start, stop in self.ranges:
            block = self.log_kernel_(self.gate_(self.geometry.block(start, stop), start), start, stop)
            out[start:stop] = torch.logsumexp(block.add_(log_v[None, :]), dim=1)
        return out

    def column_lse(self, log_u: Any) -> Any:
        """``logsumexp(log_kernel + log_u[:, None], dim=0)``, accumulated over blocks."""
        torch = self.torch
        out = torch.full(
            (self.geometry.n_target,), -math.inf, device=self.geometry.device, dtype=self.geometry.dtype
        )
        for start, stop in self.ranges:
            block = self.log_kernel_(self.gate_(self.geometry.block(start, stop), start), start, stop)
            out = torch.logaddexp(out, torch.logsumexp(block.add_(log_u[start:stop, None]), dim=0))
        return out

    def coupling_block(self, start: int, stop: int, log_u: Any, log_v: Any) -> tuple[Any, Any]:
        """Rows of the true cost and of ``exp(log_u + log_kernel + log_v)``."""
        cost = self.geometry.block(start, stop)
        block = self.log_kernel_(self.gate_(cost.clone(), start), start, stop)
        block.add_(log_u[start:stop, None]).add_(log_v[None, :])
        return cost, block.exp_()

    def source_reduction(self, potentials: _Potentials, *, variant: str) -> tuple[Any, Any]:
        """Source partner mass and gate coefficients against ``target_gate``."""
        torch = self.torch
        geometry = self.geometry
        n = geometry.n_source
        partner = torch.empty(n, device=geometry.device, dtype=geometry.dtype)
        coefficient = torch.empty(n, device=geometry.device, dtype=geometry.dtype)
        target_weight = self.target_gate.to(geometry.dtype)
        for start, stop in self.ranges:
            cost, coupling = self.coupling_block(start, stop, potentials.log_u, potentials.log_v)
            mass = coupling @ target_weight
            partner[start:stop] = mass
            if variant == "exact":
                coefficient[start:stop] = torch.sum(
                    coupling * self.target_gate[None, :] * (cost - self.rejection_cost), dim=1
                )
            else:
                counterfactual = self._source_counterfactual(cost, potentials)
                coefficient[start:stop] = mass * (counterfactual - self.rejection_cost)
        return partner, coefficient

    def target_reduction(
        self, potentials: _Potentials, source_weight_gate: np.ndarray, *, variant: str
    ) -> tuple[Any, Any]:
        """Target partner mass and gate coefficients against a given source gate.

        The coupling is the one these blocks' gates produced; the weights are
        ``source_weight_gate``, which in the outer loop is the source gate just
        projected, exactly as the dense solver uses it.
        """
        torch = self.torch
        geometry = self.geometry
        m = geometry.n_target
        weight_gate = torch.as_tensor(np.asarray(source_weight_gate, dtype=bool), device=geometry.device)
        weight = weight_gate.to(geometry.dtype)
        partner = torch.zeros(m, device=geometry.device, dtype=geometry.dtype)
        coefficient = torch.zeros(m, device=geometry.device, dtype=geometry.dtype)
        softmax = _ColumnSoftmax(torch, m, geometry.device, geometry.dtype) if variant != "exact" else None
        for start, stop in self.ranges:
            cost, coupling = self.coupling_block(start, stop, potentials.log_u, potentials.log_v)
            partner += coupling.T @ weight[start:stop]
            if variant == "exact":
                coefficient += torch.sum(
                    coupling * weight_gate[start:stop, None] * (cost - self.rejection_cost), dim=0
                )
            else:
                softmax.add(self._target_logits(cost, potentials, start, stop), cost)
        if softmax is not None:
            coefficient = partner * (softmax.value() - self.rejection_cost)
        return partner, coefficient

    def _source_counterfactual(self, cost: Any, potentials: _Potentials) -> Any:
        torch = self.torch
        logits = self.log_b[None, :] + potentials.log_v[None, :] - cost / self.epsilon
        return torch.sum(torch.softmax(logits, dim=1) * cost, dim=1)

    def _target_logits(self, cost: Any, potentials: _Potentials, start: int, stop: int) -> Any:
        return self.log_a[start:stop, None] + potentials.log_u[start:stop, None] - cost / self.epsilon


class _ColumnSoftmax:
    """``sum(softmax(logits, dim=0) * values, dim=0)`` over row blocks, stably."""

    def __init__(self, torch: Any, m: int, device: Any, dtype: Any) -> None:
        self.torch = torch
        self.maximum = torch.full((m,), -math.inf, device=device, dtype=dtype)
        self.total = torch.zeros(m, device=device, dtype=dtype)
        self.weighted = torch.zeros(m, device=device, dtype=dtype)

    def add(self, logits: Any, values: Any) -> None:
        torch = self.torch
        maximum = torch.maximum(self.maximum, torch.amax(logits, dim=0))
        rescale = torch.exp(self.maximum - maximum)
        weights = torch.exp(logits - maximum[None, :])
        self.total = self.total * rescale + weights.sum(dim=0)
        self.weighted = self.weighted * rescale + torch.sum(weights * values, dim=0)
        self.maximum = maximum

    def value(self) -> Any:
        return self.weighted / self.total


def _sinkhorn(
    torch: Any,
    blocks: _GatedBlocks,
    *,
    backbone: str,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    a: Any,
    b: Any,
    threshold: float,
    max_iterations: int,
    warm: tuple[Any, Any] | None,
) -> _Potentials:
    """``cuda._sinkhorn``, with the kernel applied block by block.

    The balanced error needs the row sums of the coupling just formed, which
    is the row log-sum-exp the next iteration's u-update computes anyway, so it
    is computed once and carried over: the update sequence, the error and the
    stopping iteration are the dense solver's.
    """
    log_a = blocks.log_a
    log_b = blocks.log_b
    if warm is None:
        log_u = torch.zeros_like(a)
        log_v = torch.zeros_like(b)
    else:
        log_u, log_v = warm[0].clone(), warm[1].clone()
    alpha = lambda_a / (lambda_a + epsilon)
    beta = lambda_b / (lambda_b + epsilon)
    converged = False
    error = math.inf
    pending = None
    iteration = -1
    for iteration in range(max_iterations):
        if backbone == "balanced":
            row = pending if pending is not None else blocks.row_lse(log_v)
            log_u = log_a - row
            column = blocks.column_lse(log_u)
            log_v = log_b - column
            pending = blocks.row_lse(log_v)
            source_mass = torch.exp(log_u + pending)
            target_mass = torch.exp(log_v + column)
            error_tensor = torch.maximum(
                torch.sum(torch.abs(source_mass - a)),
                torch.sum(torch.abs(target_mass - b)),
            )
        else:
            old_u, old_v = log_u, log_v
            log_u = alpha * (log_a - blocks.row_lse(log_v))
            log_v = beta * (log_b - blocks.column_lse(log_u))
            error_tensor = torch.maximum(
                torch.max(torch.abs(log_u - old_u)),
                torch.max(torch.abs(log_v - old_v)),
            )
        error = float(error_tensor.item())
        if not math.isfinite(error):
            break
        if error < threshold:
            converged = True
            break
    return _Potentials(log_u, log_v, converged, iteration + 1, error)


def fit_blockwise(cost: CoordinateCost, **kwargs: Any) -> BlockwiseResult:
    """Run one blockwise fit.  Same keyword arguments as ``cuda.fit_cuda``,
    plus ``block_rows``; ``_torch_device='cpu'`` runs the identical code on the
    CPU for tests."""
    return _fit_blockwise_impl(cost, **kwargs)


def _fit_blockwise_impl(
    cost: CoordinateCost,
    *,
    backbone: str,
    variant: str,
    rejection_cost: float,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    source_weights: np.ndarray | None,
    target_weights: np.ndarray | None,
    initial_source_gate: np.ndarray | None,
    initial_target_gate: np.ndarray | None,
    source_rejection_budget: float,
    target_rejection_budget: float,
    tau: float,
    threshold: float,
    max_iterations: int,
    max_outer_iterations: int,
    enforce_budget: bool,
    source_rejection_bounds: tuple[float, float] | None,
    target_rejection_bounds: tuple[float, float] | None,
    dtype: str,
    block_rows: int | None = None,
    _torch_device: str = "cuda",
) -> BlockwiseResult:
    if not isinstance(cost, CoordinateCost):
        raise TypeError("cost must be a CoordinateCost.")
    torch = _import_torch(_torch_device)
    device = torch.device(_torch_device)
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}.get(dtype)
    if torch_dtype is None:
        raise ValueError("dtype must be 'float32' or 'float64'.")
    n_source, n_target = cost.shape
    rows = resolve_block_rows(block_rows, n_source, n_target)
    # The bounds, the count limits and the initial gates exactly as the dense
    # solver resolves them, through the same CPU helpers.
    from confidenceot._cpu_uot import bounded_gate_counts, resolve_rejection_bounds

    source_gate_np = np.ones(n_source, dtype=bool) if initial_source_gate is None else np.asarray(initial_source_gate, dtype=bool).copy()
    target_gate_np = np.ones(n_target, dtype=bool) if initial_target_gate is None else np.asarray(initial_target_gate, dtype=bool).copy()
    if source_gate_np.shape != (n_source,) or target_gate_np.shape != (n_target,):
        raise ValueError("Initial gates have incompatible shapes.")
    source_interval = resolve_rejection_bounds(
        source_rejection_bounds, legacy_budget=source_rejection_budget,
        enforce_legacy_budget=enforce_budget, name="source_rejection_bounds",
    )
    target_interval = resolve_rejection_bounds(
        target_rejection_bounds, legacy_budget=target_rejection_budget,
        enforce_legacy_budget=enforce_budget, name="target_rejection_bounds",
    )
    source_min, source_max = bounded_gate_counts(n_source, source_interval)
    target_min, target_max = bounded_gate_counts(n_target, target_interval)
    if source_gate_np.sum() < source_min or target_gate_np.sum() < target_min:
        raise ValueError("Initial gates violate the rejection budget.")

    with _full_precision_matmul(torch):
        geometry = _Geometry(torch, cost, device, torch_dtype)
        a = _weights(torch, source_weights, n_source, device, torch_dtype)
        b = _weights(torch, target_weights, n_target, device, torch_dtype)
        log_a = torch.log(a)
        log_b = torch.log(b)

        def gated(source_gate: np.ndarray, target_gate: np.ndarray) -> _GatedBlocks:
            return _GatedBlocks(
                torch, geometry, source_gate, target_gate, rejection_cost=rejection_cost,
                epsilon=epsilon, block_rows=rows, log_a=log_a, log_b=log_b,
            )

        def solve(blocks: _GatedBlocks, warm: tuple[Any, Any] | None) -> _Potentials:
            return _sinkhorn(
                torch, blocks, backbone=backbone, epsilon=epsilon,
                lambda_a=lambda_a, lambda_b=lambda_b, a=a, b=b,
                threshold=threshold, max_iterations=max_iterations, warm=warm,
            )

        warm = None
        total_inner = 0
        inner_converged = True
        outer_converged = False
        cycle_detected = False
        cycle_length = 0
        seen = {(source_gate_np.tobytes(), target_gate_np.tobytes()): 0}
        outer = -1
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        started = time.perf_counter()
        for outer in range(max_outer_iterations):
            blocks = gated(source_gate_np, target_gate_np)
            result = solve(blocks, warm)
            total_inner += result.iterations
            inner_converged &= result.converged
            previous_source = source_gate_np.copy()
            previous_target = target_gate_np.copy()
            source_partner_t, source_coeff_t = blocks.source_reduction(result, variant=variant)
            source_coeff = source_coeff_t.detach().cpu().double().numpy()
            source_scale = source_partner_t.detach().cpu().double().numpy()
            source_gate_np = _project_gate(source_coeff, source_gate_np, minimum=source_min, tau=tau, scale=source_scale, maximum=source_max)

            target_partner_t, target_coeff_t = blocks.target_reduction(result, source_gate_np, variant=variant)
            target_coeff = target_coeff_t.detach().cpu().double().numpy()
            target_scale = target_partner_t.detach().cpu().double().numpy()
            target_gate_np = _project_gate(target_coeff, target_gate_np, minimum=target_min, tau=tau, scale=target_scale, maximum=target_max)

            if np.array_equal(source_gate_np, previous_source) and np.array_equal(target_gate_np, previous_target):
                outer_converged = True
                break
            key = (source_gate_np.tobytes(), target_gate_np.tobytes())
            if key in seen:
                cycle_detected = True
                length = outer + 1 - seen[key]
                cycle_length = length if cycle_length == 0 else min(cycle_length, length)
            else:
                seen[key] = outer + 1
            warm = (result.log_u, result.log_v)

        # The dense solver's final consistency solve, then its readouts in one pass.
        blocks = gated(source_gate_np, target_gate_np)
        result = solve(blocks, warm)
        total_inner += result.iterations
        inner_converged &= result.converged
        final = _final_readout(
            torch, blocks, result, source_gate_np, target_gate_np, variant=variant,
            backbone=backbone, epsilon=epsilon, lambda_a=lambda_a, lambda_b=lambda_b, a=a, b=b,
        )
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        elapsed = time.perf_counter() - started

    source_score = final["source_score"]
    target_score = final["target_score"]
    source_partner = final["source_partner"]
    target_partner = final["target_partner"]
    if variant == "reversible":
        source_decision_cost = final["source_cf"]
        target_decision_cost = final["target_cf"]
        cost_kind = "counterfactual"
    else:
        source_decision_cost = np.zeros_like(source_score)
        target_decision_cost = np.zeros_like(target_score)
        np.divide(source_score, source_partner, out=source_decision_cost, where=source_partner > 0.0)
        np.divide(target_score, target_partner, out=target_decision_cost, where=target_partner > 0.0)
        source_decision_cost[source_partner > 0.0] += rejection_cost
        target_decision_cost[target_partner > 0.0] += rejection_cost
        cost_kind = "support_restricted"
    source_raw_gate = source_score < 0.0
    target_raw_gate = target_score < 0.0

    def confidence_readout(
        decision_cost: np.ndarray, coefficient: np.ndarray, raw_gate: np.ndarray, final_gate: np.ndarray,
    ) -> BinConfidence:
        margin = decision_cost - rejection_cost
        raw_rejected = ~raw_gate
        final_rejected = ~final_gate
        return BinConfidence(
            decision_cost=decision_cost,
            rejection_cost=float(rejection_cost),
            signed_rejection_margin=margin,
            relative_rejection_margin=margin / rejection_cost,
            gate_coefficient=coefficient,
            raw_rejected=raw_rejected,
            final_rejected=final_rejected,
            budget_overridden=raw_rejected != final_rejected,
            cost_kind=cost_kind,
        )

    return BlockwiseResult(
        coupling=None,
        source_gate=source_gate_np,
        target_gate=target_gate_np,
        source_rejection_bounds=source_interval,
        target_rejection_bounds=target_interval,
        source_bounds_active=_bounds_active(float(np.mean(~source_gate_np)), source_interval),
        target_bounds_active=_bounds_active(float(np.mean(~target_gate_np)), target_interval),
        source_score=source_score,
        target_score=target_score,
        source_raw_gate=source_raw_gate,
        target_raw_gate=target_raw_gate,
        source_confidence=confidence_readout(source_decision_cost, source_score, source_raw_gate, source_gate_np),
        target_confidence=confidence_readout(target_decision_cost, target_score, target_raw_gate, target_gate_np),
        backbone=backbone,
        variant=variant,
        rejection_cost=float(rejection_cost),
        device=str(device),
        backend="torch-blockwise",
        inner_converged=inner_converged,
        outer_converged=outer_converged,
        cycle_detected=cycle_detected,
        cycle_length=cycle_length,
        n_outer_iterations=outer + 1,
        total_inner_iterations=total_inner,
        objective=final["objective"],
        fit_seconds=elapsed,
        log_u=result.log_u.detach().cpu().double().numpy(),
        log_v=result.log_v.detach().cpu().double().numpy(),
        source_weights=a.detach().cpu().double().numpy(),
        target_weights=b.detach().cpu().double().numpy(),
        epsilon=float(epsilon),
        block_rows=rows,
        dtype=dtype,
    )


def _final_readout(
    torch: Any,
    blocks: _GatedBlocks,
    result: _Potentials,
    source_gate: np.ndarray,
    target_gate: np.ndarray,
    *,
    variant: str,
    backbone: str,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    a: Any,
    b: Any,
) -> dict[str, Any]:
    """The dense solver's post-fit readouts, in one pass over the blocks.

    Partner masses, scores, counterfactual costs (M4-R only: M4-E's decision
    does not read them), the coupling's row and column sums and the objective.
    """
    geometry = blocks.geometry
    n, m = geometry.n_source, geometry.n_target
    dtype, device = geometry.dtype, geometry.device
    source_gate_t = torch.as_tensor(np.asarray(source_gate, dtype=bool), device=device)
    target_gate_t = torch.as_tensor(np.asarray(target_gate, dtype=bool), device=device)
    source_weight = source_gate_t.to(dtype)
    target_weight = target_gate_t.to(dtype)
    source_partner = torch.empty(n, device=device, dtype=dtype)
    source_score = torch.empty(n, device=device, dtype=dtype)
    source_mass = torch.empty(n, device=device, dtype=dtype)
    source_cf = torch.empty(n, device=device, dtype=dtype) if variant == "reversible" else None
    target_partner = torch.zeros(m, device=device, dtype=dtype)
    target_score = torch.zeros(m, device=device, dtype=dtype)
    target_mass = torch.zeros(m, device=device, dtype=dtype)
    softmax = _ColumnSoftmax(torch, m, device, dtype) if variant == "reversible" else None
    transport = 0.0
    coupling_kl = 0.0
    rejection_cost = blocks.rejection_cost
    for start, stop in blocks.ranges:
        cost, coupling = blocks.coupling_block(start, stop, result.log_u, result.log_v)
        source_partner[start:stop] = coupling @ target_weight
        target_partner += coupling.T @ source_weight[start:stop]
        source_mass[start:stop] = coupling.sum(dim=1)
        target_mass += coupling.sum(dim=0)
        if variant == "exact":
            source_score[start:stop] = torch.sum(coupling * target_gate_t[None, :] * (cost - rejection_cost), dim=1)
            target_score += torch.sum(coupling * source_gate_t[start:stop, None] * (cost - rejection_cost), dim=0)
        else:
            source_cf[start:stop] = blocks._source_counterfactual(cost, result)
            softmax.add(blocks._target_logits(cost, result, start, stop), cost)
        optimization = blocks.gate_(cost, start)
        transport += float(torch.sum(coupling * optimization).item())
        reference = a[start:stop, None] * b[None, :]
        coupling_kl += float(_generalized_kl(torch, coupling, reference).item())
    if variant == "reversible":
        target_cf = softmax.value()
        source_score = source_partner * (source_cf - rejection_cost)
        target_score = target_partner * (target_cf - rejection_cost)
    objective = transport + epsilon * coupling_kl
    if backbone == "uot":
        objective = (
            objective
            + lambda_a * float(_generalized_kl(torch, source_mass, a).item())
            + lambda_b * float(_generalized_kl(torch, target_mass, b).item())
        )
    as_numpy = lambda tensor: tensor.detach().cpu().double().numpy()  # noqa: E731
    readout = {
        "source_partner": as_numpy(source_partner),
        "target_partner": as_numpy(target_partner),
        "source_score": as_numpy(source_score),
        "target_score": as_numpy(target_score),
        "source_mass": as_numpy(source_mass),
        "target_mass": as_numpy(target_mass),
        "objective": float(objective),
    }
    if variant == "reversible":
        readout["source_cf"] = as_numpy(source_cf)
        readout["target_cf"] = as_numpy(target_cf)
    return readout


# --------------------------------------------------------------------------
# Rebuilding the coupling after a fit.
# --------------------------------------------------------------------------

def _rebuild(cost: CoordinateCost, result: BlockwiseResult, block_rows: int | None):
    """Device geometry, gated blocks and potentials for a finished fit."""
    device = "cuda" if str(result.device).startswith("cuda") else "cpu"
    torch = _import_torch(device)
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[result.dtype]
    geometry = _Geometry(torch, cost, torch.device(device), torch_dtype)
    a = torch.as_tensor(result.source_weights, device=geometry.device, dtype=torch_dtype)
    b = torch.as_tensor(result.target_weights, device=geometry.device, dtype=torch_dtype)
    n_source, n_target = cost.shape
    rows = resolve_block_rows(result.block_rows if block_rows is None else block_rows, n_source, n_target)
    blocks = _GatedBlocks(
        torch, geometry, result.source_gate, result.target_gate,
        rejection_cost=result.rejection_cost, epsilon=result.epsilon, block_rows=rows,
        log_a=torch.log(a), log_b=torch.log(b),
    )
    potentials = _Potentials(
        torch.as_tensor(result.log_u, device=geometry.device, dtype=torch_dtype),
        torch.as_tensor(result.log_v, device=geometry.device, dtype=torch_dtype),
        bool(result.inner_converged), 0, 0.0,
    )
    return torch, geometry, blocks, potentials, a, b


def materialize_coupling(cost: CoordinateCost, result: BlockwiseResult) -> NDArray[np.float64]:
    """The whole coupling the dense solver would have returned.  Small problems only."""
    with_torch = _rebuild(cost, result, None)
    torch, geometry, blocks, potentials = with_torch[:4]
    out = np.empty(cost.shape, dtype=np.float64)
    with _full_precision_matmul(torch):
        for start, stop in blocks.ranges:
            _, coupling = blocks.coupling_block(start, stop, potentials.log_u, potentials.log_v)
            out[start:stop] = coupling.detach().cpu().double().numpy()
    return out


def transport_reductions(
    cost: CoordinateCost,
    result: BlockwiseResult,
    *,
    source_groups: ArrayLike,
    n_source_groups: int,
    target_groups: ArrayLike,
    n_target_groups: int,
    source_points: ArrayLike,
    target_points: ArrayLike,
    support: Literal["active", "all"] = "active",
    source_strata: Mapping[str, ArrayLike] | None = None,
    target_strata: Mapping[str, ArrayLike] | None = None,
    block_rows: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Row and column summaries of a fitted coupling, without forming it.

    ``support='active'`` restricts the coupling to retained x retained bins,
    the support on which ConfidenceOT asserts a correspondence; ``'all'`` keeps
    every entry.  Writing ``P`` for the supported coupling, ``r_i`` and ``q_j``
    for its row and column sums, ``g`` for group codes and ``p`` for points:

    * ``source_mass`` r, ``source_total_mass`` (unrestricted row sums) and the
      target counterparts;
    * ``source_partner_groups[i, B] = sum_{j in B} P_ij`` and
      ``target_partner_groups[j, A] = sum_{i in A} P_ij``;
    * ``source_partner_points[i] = sum_j P_ij p_j`` with
      ``source_partner_square[i] = sum_j P_ij |p_j|^2``, and the target
      counterparts, from which partner centroids and spreads follow;
    * ``group_mass[A, B] = sum_{i in A, j in B} P_ij``;
    * for every named source stratum S, ``push_<S>[j, A] =
      sum_{i in S and A} a_i P_ij / r_i``: the nominal mass of S carried onto
      target bin j, by source group; for every target stratum T,
      ``pull_<T>[i, B] = sum_{j in T and B} b_j P_ij / q_j``;
    * ``cost_max``, the largest entry of the cost.

    Reductions are taken in float64 from float32 blocks of the coupling.
    """
    if support not in ("active", "all"):
        raise ValueError("support must be 'active' or 'all'.")
    torch, geometry, blocks, potentials, a, b = _rebuild(cost, result, block_rows)
    n, m = cost.shape
    device = geometry.device
    f64 = torch.float64
    source_codes = np.asarray(source_groups, dtype=np.int64)
    target_codes = np.asarray(target_groups, dtype=np.int64)
    if source_codes.shape != (n,) or target_codes.shape != (m,):
        raise ValueError("group codes must give one code per bin.")
    if source_codes.min(initial=0) < 0 or source_codes.max(initial=-1) >= n_source_groups:
        raise ValueError("source group codes out of range.")
    if target_codes.min(initial=0) < 0 or target_codes.max(initial=-1) >= n_target_groups:
        raise ValueError("target group codes out of range.")
    source_hot = torch.zeros((n, n_source_groups), device=device, dtype=f64)
    source_hot[torch.arange(n, device=device), torch.as_tensor(source_codes, device=device)] = 1.0
    target_hot = torch.zeros((m, n_target_groups), device=device, dtype=f64)
    target_hot[torch.arange(m, device=device), torch.as_tensor(target_codes, device=device)] = 1.0

    def with_square(points: ArrayLike, count: int) -> Any:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != count:
            raise ValueError("points must give one row per bin.")
        return torch.as_tensor(
            np.column_stack([values, np.sum(values * values, axis=1)]), device=device, dtype=f64
        )

    source_features = with_square(source_points, n)
    target_features = with_square(target_points, m)
    if support == "active":
        row_support = torch.as_tensor(np.asarray(result.source_gate, dtype=bool), device=device).to(f64)
        column_support = torch.as_tensor(np.asarray(result.target_gate, dtype=bool), device=device).to(f64)
    else:
        row_support = torch.ones(n, device=device, dtype=f64)
        column_support = torch.ones(m, device=device, dtype=f64)

    source_mass = torch.zeros(n, device=device, dtype=f64)
    source_total = torch.zeros(n, device=device, dtype=f64)
    target_mass = torch.zeros(m, device=device, dtype=f64)
    target_total = torch.zeros(m, device=device, dtype=f64)
    source_partner_groups = torch.zeros((n, n_target_groups), device=device, dtype=f64)
    target_partner_groups = torch.zeros((n_source_groups, m), device=device, dtype=f64)
    source_points_sum = torch.zeros((n, source_features.shape[1]), device=device, dtype=f64)
    target_points_sum = torch.zeros((target_features.shape[1], m), device=device, dtype=f64)
    group_mass = torch.zeros((n_source_groups, n_target_groups), device=device, dtype=f64)
    cost_max = 0.0

    def supported(start: int, stop: int) -> tuple[Any, Any, Any]:
        cost_block, coupling = blocks.coupling_block(start, stop, potentials.log_u, potentials.log_v)
        full = coupling.to(f64)
        restricted = full * row_support[start:stop, None] * column_support[None, :]
        return cost_block, full, restricted

    with _full_precision_matmul(torch):
        for start, stop in blocks.ranges:
            cost_block, full, restricted = supported(start, stop)
            cost_max = max(cost_max, float(cost_block.max().item()))
            source_total[start:stop] = full.sum(dim=1)
            target_total += full.sum(dim=0)
            source_mass[start:stop] = restricted.sum(dim=1)
            target_mass += restricted.sum(dim=0)
            by_target = restricted @ target_hot
            source_partner_groups[start:stop] = by_target
            target_partner_groups += source_hot[start:stop].T @ restricted
            source_points_sum[start:stop] = restricted @ target_features
            target_points_sum += source_features[start:stop].T @ restricted
            group_mass += source_hot[start:stop].T @ by_target
            del full, restricted, by_target

        pushes: dict[str, Any] = {}
        pulls: dict[str, Any] = {}
        if source_strata or target_strata:
            row_weight = torch.where(source_mass > 0, a.to(f64) / torch.where(source_mass > 0, source_mass, 1.0), 0.0)
            column_weight = torch.where(target_mass > 0, b.to(f64) / torch.where(target_mass > 0, target_mass, 1.0), 0.0)
            row_weights = {}
            for name, mask in (source_strata or {}).items():
                values = np.asarray(mask, dtype=bool)
                if values.shape != (n,):
                    raise ValueError(f"source stratum {name!r} must give one flag per source bin.")
                row_weights[name] = row_weight * torch.as_tensor(values, device=device).to(f64)
                pushes[name] = torch.zeros((n_source_groups, m), device=device, dtype=f64)
            column_weights = {}
            for name, mask in (target_strata or {}).items():
                values = np.asarray(mask, dtype=bool)
                if values.shape != (m,):
                    raise ValueError(f"target stratum {name!r} must give one flag per target bin.")
                column_weights[name] = target_hot * (column_weight * torch.as_tensor(values, device=device).to(f64))[:, None]
                pulls[name] = torch.zeros((n, n_target_groups), device=device, dtype=f64)
            for start, stop in blocks.ranges:
                _, full, restricted = supported(start, stop)
                for name, weight in row_weights.items():
                    pushes[name] += (source_hot[start:stop] * weight[start:stop, None]).T @ restricted
                for name, weight in column_weights.items():
                    pulls[name][start:stop] = restricted @ weight
                del full, restricted

    as_numpy = lambda tensor: tensor.detach().cpu().numpy()  # noqa: E731
    out: dict[str, Any] = {
        "source_mass": as_numpy(source_mass),
        "source_total_mass": as_numpy(source_total),
        "target_mass": as_numpy(target_mass),
        "target_total_mass": as_numpy(target_total),
        "source_partner_groups": as_numpy(source_partner_groups),
        "target_partner_groups": as_numpy(target_partner_groups.T),
        "source_partner_points": as_numpy(source_points_sum[:, :-1]),
        "source_partner_square": as_numpy(source_points_sum[:, -1]),
        "target_partner_points": as_numpy(target_points_sum[:-1].T),
        "target_partner_square": as_numpy(target_points_sum[-1]),
        "group_mass": as_numpy(group_mass),
        "cost_max": np.float64(cost_max),
    }
    for name, value in pushes.items():
        out[f"push_{name}"] = as_numpy(value.T)
    for name, value in pulls.items():
        out[f"pull_{name}"] = as_numpy(value)
    return out
