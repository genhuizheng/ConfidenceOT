# ConfidenceOT

Primary-to-metastasis application code is under `cancer_metastasis/`.  It audits exact
patient/sample pair eligibility, runs data-driven null calibration and M4-R inference,
exports per-cell confidence and population transitions, and supports a budget-cap pilot
before the full Vista SLURM array is launched.

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

ConfidenceOT accepts a non-negative source-by-target cost matrix. To go
from counts to that cost matrix, see [From counts to a cost
matrix](#from-counts-to-a-cost-matrix).

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

# Per-observation M4-R readouts; these do not change the fitted gate.
print(result.source_confidence.counterfactual_cost)
print(result.source_confidence.signed_rejection_margin)
print(result.source_confidence.normalized_rejection_score())
```

`True` in a gate means that the observation is retained. `False` means that it
is rejected by the confidence filter. Iteration caps and detected cycles are
reported as warnings while the finite terminal result is retained.

## From counts to a cost matrix

`ConfidenceOT.fit` takes a cost matrix, and on single-cell counts what decides
the answer is mostly upstream of it: the transform the representation is built
in, whether read depth was equalised first, and whether the cost is squared
Euclidean or cosine. `Preprocessing` carries that whole chain as one object you
can name, record and reuse.

```python
import numpy as np
from confidenceot import ConfidenceOT, Preprocessing

configuration = Preprocessing(
    normalisation="rank_value",   # cpm | log_cpm | log1p | precomputed
                                  # pearson_residuals | rank_value | rank_no_median
    rank_top_n=256,
    equalise_depth=True,          # subsample every cell to one shared depth
    equalise_quantile=0.10,
    n_hvg=2000,
    n_pcs=30,
    cost="cosine",                # squared_euclidean | cosine
)

print(configuration.label())      # "rank256_ds_cos"

model = ConfidenceOT(variant="exact", rejection_cost=0.5, device="cpu")
result, prepared = model.fit_counts(
    source_counts, target_counts,          # cells x genes, raw counts
    preprocessing=configuration, seed=0,
    source_genes=source_gene_names,        # optional; intersects the two sides
    target_genes=target_gene_names,
)

print(prepared.scale)             # the median sampled pair distance
print(prepared.provenance)        # the configuration plus what each step did
```

`fit_counts` returns both halves on purpose. `rejection_cost` is in units of
`prepared.scale`, and `prepared.provenance` is what a run record needs, so a
result on its own cannot be read back.

The steps are also available individually — `equalise_depth`, `unit_rows`,
`squared_euclidean`, `median_pair_scale`, `rank_value_encode` — and
`configuration.representation(...)` stops at the coordinates if you want the
representation without a cost.

`normalisation="precomputed"` is how a transform from another package enters:
compute it yourself, pass the matrix, and name it with `label_stem` so the run
is still identifiable.

```python
configuration = Preprocessing(
    normalisation="precomputed", label_stem="sctransform",
    equalise_depth=True, cost="cosine",
)
representation = configuration.representation(source_residuals,
                                               target_residuals, seed=0)
```

**Which options have evidence behind them.** On two independent simulators, the
choice that separated a gate's false-rejection rate was the cost: every squared
Euclidean configuration false-rejected 0.071–0.098 of cells that should not have
been rejected, every cosine one 0.000–0.057, with no overlap. `cpm` and `log1p`
exist because the chain needs them and were not measured. `equalise_depth` is
not attributed: on a simulation whose depth mechanism is multinomial, read
equalisation is that mechanism's exact inverse, so measuring it there is
circular. `cancer_metastasis/SEQUENCING_DEPTH_RESOLUTION.md` has the full table
and the caveats.

`Preprocessing` needs scikit-learn for the PCA step:
`pip install "confidenceot[preprocessing]"`. The solver itself does not, so
`fit(cost_matrix)` still runs on numpy and scipy alone.

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
python mouse_embryo/12_prepare_mouse_msigdb_gmt.py GENE_SET_ROOT
python mouse_embryo/10_run_spatial_bin_gseapy.py OUTPUT_ROOT MOUSE_PATHWAYS.gmt GSEA_ROOT
python mouse_embryo/11_visualize_spatial_bin_programs.py OUTPUT_ROOT FIGURE_ROOT --gsea-root GSEA_ROOT
```

## Returned diagnostics

`ConfidenceOTResult` contains:

- the transport coupling;
- source and target gates;
- raw, pre-budget gate decisions;
- source and target gate scores;
- structured per-observation confidence readouts under `source_confidence` and
  `target_confidence`, including the decision cost, signed and relative margin,
  raw and final rejection decisions, and a budget-override flag;
- rejection rates;
- convergence and cycle diagnostics;
- outer and inner iteration counts;
- objective value, backend, device, and fit time.

For M4-R, the signed rejection margin is the counterfactual conditional cost
minus the rejection cost. Positive values support rejection and negative
values support retention. `normalized_rejection_score()` is a monotone
visualization score, not a calibrated probability.

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
