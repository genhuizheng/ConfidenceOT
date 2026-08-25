# ConfidenceOT

ConfidenceOT is a confidence-filtered optimal transport package for detecting
source and target observations that should not be forced into a transport
match. It exposes two related algorithms over balanced or KL-unbalanced OT
backbones:

- **M4-E (`exact`)** uses exact sequential gate updates and serves as the
  stable reference algorithm for estimating the rejection cost.
- **M4-R (`reversible`)** allows gate decisions to be reconsidered and serves
  as the deployment algorithm after the rejection cost is frozen.

The package provides a NumPy CPU reference implementation and an optional
PyTorch CUDA implementation through the same public API.

## Installation

ConfidenceOT requires Python 3.10 or newer.

```bash
git clone https://github.com/YOUR_USERNAME/ConfidenceOT.git
cd ConfidenceOT
python -m pip install -e .
python -m confidenceot
```

For CUDA, first install a PyTorch build compatible with the CUDA runtime on the
target workstation or cluster, then install the optional dependency:

```bash
python -m pip install -e ".[cuda]"
python -m confidenceot
```

The environment diagnostic must report `"cuda_available": true` before
`device="cuda"` can be used. ConfidenceOT intentionally does not pin a
platform-specific CUDA wheel.

## Basic usage

ConfidenceOT accepts a non-negative source-by-target cost matrix.

```python
import numpy as np
from confidenceot import ConfidenceOT

cost = np.array([
    [0.10, 1.20, 1.50],
    [1.10, 0.15, 1.30],
    [2.20, 2.00, 2.10],
])

model = ConfidenceOT(
    backbone="uot",              # "balanced" or "uot"
    variant="reversible",        # M4-E="exact", M4-R="reversible"
    rejection_cost=0.5,
    epsilon=0.1,
    lambda_a=1.0,
    lambda_b=1.0,
    source_rejection_budget=0.15,
    target_rejection_budget=0.15,
    tolerance=1e-3,
    device="auto",               # "auto", "cpu", or "cuda"
)

result = model.fit(cost)

print(result.coupling)
print(result.source_gate)
print(result.target_gate)
print(result.source_rejection_rate)
print(result.target_rejection_rate)
```

`True` in a gate means that the observation is retained. `False` means that it
is rejected by the confidence filter. Iteration caps and detected cycles are
reported as warnings while the finite terminal result is retained.

## M4-E and M4-R

Convenience functions are available when the rejection cost is already known:

```python
from confidenceot import m4_exact, m4_reversible

reference = m4_exact(cost, backbone="balanced", rejection_cost=0.5)
deployed = m4_reversible(cost, backbone="balanced", rejection_cost=0.5)
```

For label-free null calibration, use M4-E to estimate a candidate rejection
cost and M4-R to validate that frozen value on held-out null replicates:

```python
from confidenceot import calibrate_confidence_cost

calibration = calibrate_confidence_cost(
    calibration_nulls,
    validation_nulls,
    backbone="uot",
    source_raw_acceptance_target=0.10,
    target_raw_acceptance_target=0.10,
    source_rejection_budget=0.15,
    target_rejection_budget=0.15,
    tolerance=1e-3,
    device="auto",
)

frozen_cost = calibration.rejection_cost
final_model = ConfidenceOT(
    backbone="uot",
    variant="reversible",
    rejection_cost=frozen_cost,
)
```

Calibration uses raw gate signs for threshold selection. Rejection-budget
projection is recorded separately and does not redefine the calibration
criterion.

## CPU and CUDA

The backend is selected with `device`:

- `auto`: use CUDA when available, otherwise CPU;
- `cpu`: use the NumPy reference implementation;
- `cuda`: require the PyTorch CUDA implementation.

Use `cuda_dtype="float32"` for speed or `cuda_dtype="float64"` when closer
agreement with the CPU reference is required. Explicit CUDA requests fail
cleanly if CUDA is unavailable unless `fallback_to_cpu=True` is set.

Independent matrices can be fitted concurrently without changing the update
order inside any individual solve:

```python
results = model.fit_many(cost_matrices, workers=4)
```

CPU fits use independent NumPy solver states. Concurrent CUDA fits use private
CUDA streams and synchronize per stream rather than across the entire device.
Results are returned in input order, regardless of completion order.

## MOSTA rejected-bin DEG

The mouse-embryo discovery workflow tests each section independently. For a
candidate disappearance or emergence, its ConfidenceOT-rejected spatial bins
are compared with all retained bins on the same side and in the same section
using Scanpy's Wilcoxon test. Genes used in the OT representation are excluded
from DEG to avoid circular validation. Across-section output is a descriptive
consensus rank and does not pool section p-values.

```bash
python -m pip install -r mouse_embryo/requirements_deg.txt
python mouse_embryo/09_spatial_bin_deg_meta.py RUN_ROOT DATA_ROOT OUTPUT_ROOT
python mouse_embryo/10_run_spatial_bin_gseapy.py OUTPUT_ROOT MOUSE_PATHWAYS.gmt GSEA_ROOT
python mouse_embryo/11_visualize_spatial_bin_programs.py OUTPUT_ROOT FIGURE_ROOT --gsea-root GSEA_ROOT
```

## Returned diagnostics

`ConfidenceOTResult` contains:

- the transport coupling;
- source and target gates;
- raw, pre-budget gate decisions;
- source and target gate scores;
- rejection rates;
- convergence and cycle diagnostics;
- outer and inner iteration counts;
- objective value, backend, device, and fit time.

## Development

Run the test suite from the repository root:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The CPU implementation is the numerical reference. The torch kernel is tested
against it for balanced/UOT and exact/reversible configurations.

## Status

ConfidenceOT is research software under active development. Validate the
calibration protocol, numerical tolerances, and rejection budgets for the
intended dataset before scientific or production use.
