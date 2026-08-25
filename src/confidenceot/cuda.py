"""CUDA implementation of bidirectional ConfidenceOT using PyTorch.

PyTorch is imported lazily so the core CPU package has no mandatory GPU
dependency.  The complete Sinkhorn loop, coupling, counterfactual scores, and
M4 coefficients remain on the GPU.  Only the two O(N) gate coefficient vectors
are copied to the host for deterministic budget projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from confidenceot.result import ConfidenceOTResult


class CUDAUnavailableError(RuntimeError):
    """Raised when the CUDA backend was requested but is unavailable."""


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def cuda_device_name() -> str | None:
    if not cuda_available():
        return None
    import torch

    return str(torch.cuda.get_device_name(torch.cuda.current_device()))


@dataclass
class _InnerResult:
    coupling: Any
    log_u: Any
    log_v: Any
    a: Any
    b: Any
    converged: bool
    iterations: int
    error: float


def _project_gate(
    coefficients: np.ndarray,
    current: np.ndarray,
    *,
    minimum: int,
    tau: float,
    scale: np.ndarray,
) -> np.ndarray:
    """Deterministic budgeted sign projection matching the reference solver."""
    boundary = tau * scale
    negative = coefficients < -boundary
    positive = coefficients > boundary
    tie = ~(negative | positive)
    gate = negative.copy()
    gate[tie] = current[tie]
    missing = minimum - int(gate.sum())
    if missing > 0:
        candidates = np.flatnonzero(~gate)
        order = np.lexsort(
            (candidates, -current[candidates].astype(np.int8), coefficients[candidates])
        )
        gate[candidates[order[:missing]]] = True
    return gate


def _weights(torch: Any, values: np.ndarray | None, n: int, device: Any, dtype: Any) -> Any:
    if values is None:
        return torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"Weights must be positive and have shape ({n},).")
    tensor = torch.as_tensor(array, device=device, dtype=dtype)
    return tensor / tensor.sum()


def _sinkhorn(
    torch: Any,
    optimization_cost: Any,
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
) -> _InnerResult:
    log_a = torch.log(a)
    log_b = torch.log(b)
    log_kernel = log_a[:, None] + log_b[None, :] - optimization_cost / epsilon
    if warm is None:
        log_u = torch.zeros_like(a)
        log_v = torch.zeros_like(b)
    else:
        log_u, log_v = warm[0].clone(), warm[1].clone()
    alpha = lambda_a / (lambda_a + epsilon)
    beta = lambda_b / (lambda_b + epsilon)
    converged = False
    error = math.inf
    for iteration in range(max_iterations):
        if backbone == "balanced":
            log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
            log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
            log_coupling = log_u[:, None] + log_kernel + log_v[None, :]
            coupling = torch.exp(log_coupling)
            error_tensor = torch.maximum(
                torch.sum(torch.abs(coupling.sum(dim=1) - a)),
                torch.sum(torch.abs(coupling.sum(dim=0) - b)),
            )
        else:
            old_u, old_v = log_u, log_v
            log_u = alpha * (log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1))
            log_v = beta * (log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0))
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
    log_coupling = log_u[:, None] + log_kernel + log_v[None, :]
    coupling = torch.exp(log_coupling)
    return _InnerResult(coupling, log_u, log_v, a, b, converged, iteration + 1, error)


def _counterfactual_losses(torch: Any, result: _InnerResult, cost: Any, epsilon: float) -> tuple[Any, Any]:
    source_logits = torch.log(result.b)[None, :] + result.log_v[None, :] - cost / epsilon
    target_logits = torch.log(result.a)[:, None] + result.log_u[:, None] - cost / epsilon
    source_loss = torch.sum(torch.softmax(source_logits, dim=1) * cost, dim=1)
    target_loss = torch.sum(torch.softmax(target_logits, dim=0) * cost, dim=0)
    return source_loss, target_loss


def _generalized_kl(torch: Any, values: Any, reference: Any) -> Any:
    positive = values > 0
    safe_values = torch.where(positive, values, torch.ones_like(values))
    terms = reference - values + torch.where(
        positive, values * (torch.log(safe_values) - torch.log(reference)), torch.zeros_like(values)
    )
    return terms.sum()


def fit_cuda(
    cost_matrix: np.ndarray,
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
    dtype: str,
    _torch_device: str = "cuda",
) -> ConfidenceOTResult:
    try:
        import torch
    except ImportError as error:
        raise CUDAUnavailableError(
            "CUDA ConfidenceOT requires PyTorch. Install the `confidenceot[cuda]` extra."
        ) from error
    if _torch_device == "cuda" and not torch.cuda.is_available():
        raise CUDAUnavailableError(
            "PyTorch is installed without an available CUDA runtime. Install a CUDA-enabled "
            "PyTorch build and verify `torch.cuda.is_available()` first."
        )
    if _torch_device not in ("cpu", "cuda"):
        raise ValueError("Internal torch device must be 'cpu' or 'cuda'.")
    device = torch.device(_torch_device)
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}.get(dtype)
    if torch_dtype is None:
        raise ValueError("CUDA dtype must be 'float32' or 'float64'.")
    cost_np = np.asarray(cost_matrix, dtype=np.float64)
    if cost_np.ndim != 2 or min(cost_np.shape) == 0 or not np.all(np.isfinite(cost_np)) or np.any(cost_np < 0):
        raise ValueError("cost_matrix must be a finite non-negative 2D matrix.")
    n_source, n_target = cost_np.shape
    cost = torch.as_tensor(cost_np, device=device, dtype=torch_dtype)
    a = _weights(torch, source_weights, n_source, device, torch_dtype)
    b = _weights(torch, target_weights, n_target, device, torch_dtype)
    source_gate_np = np.ones(n_source, dtype=bool) if initial_source_gate is None else np.asarray(initial_source_gate, dtype=bool).copy()
    target_gate_np = np.ones(n_target, dtype=bool) if initial_target_gate is None else np.asarray(initial_target_gate, dtype=bool).copy()
    if source_gate_np.shape != (n_source,) or target_gate_np.shape != (n_target,):
        raise ValueError("Initial gates have incompatible shapes.")
    source_min = int(math.ceil((1.0 - source_rejection_budget) * n_source - 1e-12))
    target_min = int(math.ceil((1.0 - target_rejection_budget) * n_target - 1e-12))
    if source_gate_np.sum() < source_min or target_gate_np.sum() < target_min:
        raise ValueError("Initial gates violate the rejection budget.")

    warm = None
    total_inner = 0
    inner_converged = True
    outer_converged = False
    cycle_detected = False
    cycle_length = 0
    seen = {(source_gate_np.tobytes(), target_gate_np.tobytes()): 0}
    result = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for outer in range(max_outer_iterations):
        source_gate = torch.as_tensor(source_gate_np, device=device)
        target_gate = torch.as_tensor(target_gate_np, device=device)
        active = source_gate[:, None] & target_gate[None, :]
        optimization_cost = torch.where(active, cost, torch.as_tensor(rejection_cost, device=device, dtype=torch_dtype))
        result = _sinkhorn(
            torch, optimization_cost, backbone=backbone, epsilon=epsilon,
            lambda_a=lambda_a, lambda_b=lambda_b, a=a, b=b,
            threshold=threshold, max_iterations=max_iterations, warm=warm,
        )
        total_inner += result.iterations
        inner_converged &= result.converged
        previous_source = source_gate_np.copy()
        previous_target = target_gate_np.copy()
        source_partner_mass = result.coupling @ target_gate.to(torch_dtype)
        source_cf, target_cf = _counterfactual_losses(torch, result, cost, epsilon)
        if variant == "exact":
            source_coeff_t = torch.sum(result.coupling * target_gate[None, :] * (cost - rejection_cost), dim=1)
        else:
            source_coeff_t = source_partner_mass * (source_cf - rejection_cost)
        source_coeff = source_coeff_t.detach().cpu().double().numpy()
        source_scale = source_partner_mass.detach().cpu().double().numpy()
        source_gate_np = _project_gate(source_coeff, source_gate_np, minimum=source_min, tau=tau, scale=source_scale)

        new_source_gate = torch.as_tensor(source_gate_np, device=device)
        target_partner_mass = result.coupling.T @ new_source_gate.to(torch_dtype)
        if variant == "exact":
            target_coeff_t = torch.sum(result.coupling * new_source_gate[:, None] * (cost - rejection_cost), dim=0)
        else:
            target_coeff_t = target_partner_mass * (target_cf - rejection_cost)
        target_coeff = target_coeff_t.detach().cpu().double().numpy()
        target_scale = target_partner_mass.detach().cpu().double().numpy()
        target_gate_np = _project_gate(target_coeff, target_gate_np, minimum=target_min, tau=tau, scale=target_scale)

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

    # A final consistency solve ensures the returned coupling matches the gates.
    source_gate = torch.as_tensor(source_gate_np, device=device)
    target_gate = torch.as_tensor(target_gate_np, device=device)
    active = source_gate[:, None] & target_gate[None, :]
    optimization_cost = torch.where(active, cost, torch.as_tensor(rejection_cost, device=device, dtype=torch_dtype))
    result = _sinkhorn(
        torch, optimization_cost, backbone=backbone, epsilon=epsilon,
        lambda_a=lambda_a, lambda_b=lambda_b, a=a, b=b,
        threshold=threshold, max_iterations=max_iterations, warm=warm,
    )
    total_inner += result.iterations
    inner_converged &= result.converged
    source_partner_mass = result.coupling @ target_gate.to(torch_dtype)
    target_partner_mass = result.coupling.T @ source_gate.to(torch_dtype)
    source_cf, target_cf = _counterfactual_losses(torch, result, cost, epsilon)
    if variant == "exact":
        source_score_t = torch.sum(result.coupling * target_gate[None, :] * (cost - rejection_cost), dim=1)
        target_score_t = torch.sum(result.coupling * source_gate[:, None] * (cost - rejection_cost), dim=0)
    else:
        source_score_t = source_partner_mass * (source_cf - rejection_cost)
        target_score_t = target_partner_mass * (target_cf - rejection_cost)
    reference = a[:, None] * b[None, :]
    source_mass = result.coupling.sum(dim=1)
    target_mass = result.coupling.sum(dim=0)
    objective_t = torch.sum(result.coupling * optimization_cost) + epsilon * _generalized_kl(torch, result.coupling, reference)
    if backbone == "uot":
        objective_t = objective_t + lambda_a * _generalized_kl(torch, source_mass, a) + lambda_b * _generalized_kl(torch, target_mass, b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    source_score = source_score_t.detach().cpu().double().numpy()
    target_score = target_score_t.detach().cpu().double().numpy()
    return ConfidenceOTResult(
        coupling=result.coupling.detach().cpu().double().numpy(),
        source_gate=source_gate_np,
        target_gate=target_gate_np,
        source_score=source_score,
        target_score=target_score,
        source_raw_gate=source_score < 0.0,
        target_raw_gate=target_score < 0.0,
        backbone=backbone,
        variant=variant,
        rejection_cost=float(rejection_cost),
        device=str(device),
        backend="torch",
        inner_converged=inner_converged,
        outer_converged=outer_converged,
        cycle_detected=cycle_detected,
        cycle_length=cycle_length,
        n_outer_iterations=outer + 1,
        total_inner_iterations=total_inner,
        objective=float(objective_t.item()),
        fit_seconds=elapsed,
    )
