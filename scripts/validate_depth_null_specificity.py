"""Standalone specificity check for the ConfidenceOT rejection decision.

This script is deliberately independent of the Splatter scaling benchmark.  It
builds its own counts so that the ground truth is a construction rather than a
simulator parameter, then runs the exact production path used by the cancer
workflow: ``raw counts -> normalisation -> joint HVG -> gene scaling -> joint
PCA -> squared Euclidean cost -> median scaling -> null calibration -> M4-E``.
``--normalization`` and ``--calibration-null`` select the two steps under test.

Two separate questions are measured, because they have different consequences.

1. Is the rejection *rate* informative?  The ``homogeneous`` arms contain one
   population with no incompatible cells at all, so a specific method should
   reject almost none of them.  Under ``--calibration-null
   cross_side_rotation`` the cost is the largest whose rotated-null acceptance
   stays at or below 10%, and for a homogeneous cloud the rotated null closely
   resembles the observed data, so the threshold lands below the median cost.
   If the homogeneous arms reject at the cap, the rate is fixed by the
   calibration target rather than by compatibility, and no rejection rate
   reported anywhere in the project carries biological information.  Under
   ``within_side_split`` the cost must instead accept two halves of one side,
   which contain no incompatible cells by construction.
2. Is the rejection *identity* driven by depth?  The homogeneous arms differ
   only in how widely per-cell sequencing depth is spread.  Every cell's
   underlying relative expression profile is drawn from the same distribution
   regardless of its depth, so any dependence of the gate on depth is a
   specificity failure.  ``auc_total_counts`` and
   ``spearman_decision_cost_total_counts`` use the same definitions as
   ``cancer_metastasis/25_diagnose_gate_covariates.py`` so simulated and real
   values can be read side by side.

The ``perturbed`` arm is a positive control at uniform depth: a genuine
subpopulation is present only on the source side, so a method with any power
should reject those cells and few others.  Comparing it against the
homogeneous arms separates sensitivity from specificity.

Passing ``--fixed-rejection-cost`` bypasses null calibration, which separates
"the calibration target forces this rejection rate" from "the cost geometry
forces it".  Running both modes is the point of the script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cancer_metastasis.common import prepare_joint_representation  # noqa: E402
from confidenceot import (  # noqa: E402
    ConfidenceOT,
    calibrate_confidence_cost,
    rotation_null_costs,
    within_side_null_costs,
)


ARMS = {
    # Depth spread only; no biological difference of any kind.
    "homogeneous_depth_cv0": {"depth_sigma": 0.0, "perturbed_fraction": 0.0},
    "homogeneous_depth_cv_low": {"depth_sigma": 0.3, "perturbed_fraction": 0.0},
    "homogeneous_depth_cv_mid": {"depth_sigma": 0.6, "perturbed_fraction": 0.0},
    "homogeneous_depth_cv_high": {"depth_sigma": 0.9, "perturbed_fraction": 0.0},
    # Positive control: a real source-only subpopulation at uniform depth.
    "perturbed_depth_cv0": {"depth_sigma": 0.0, "perturbed_fraction": 0.2},
}

# Enabled only by --depth-source: the same two questions asked at the depth
# distribution an actual dataset has, rather than at a chosen sigma.
OBSERVED_ARMS = {
    "homogeneous_depth_observed": {"depth_sigma": 0.0, "perturbed_fraction": 0.0},
    "perturbed_depth_observed": {"depth_sigma": 0.0, "perturbed_fraction": 0.2},
}


def squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    value = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(value, 0.0)


def equalise_reads(
    rng: np.random.Generator, counts: np.ndarray, target: int,
) -> np.ndarray:
    """Subsample every cell's reads to one shared total.

    The same operation as ``cancer_metastasis/27_downsample_counts.py`` and as
    ``scanpy.pp.downsample_counts``: exact multivariate hypergeometric sampling
    of reads without replacement. Cells already at or below the target are left
    untouched, exactly as the production script leaves them, so the residual
    depth gradient the simulation sees is the same one the real runs see.

    This step was missing from the simulation entirely, which is why its
    gene-ranking arm measured ranking *alone*. On real prostate data ranking
    alone leaves the gate at AUC 0.060 while ranking plus this step reaches
    0.493, so the arm the figure showed was never the production pipeline.
    """
    equalised = np.array(counts, dtype=np.int64, copy=True)
    totals = equalised.sum(axis=1)
    for index in np.flatnonzero(totals > target):
        equalised[index] = rng.multivariate_hypergeometric(
            equalised[index], int(target)
        )
    return equalised


SCTRANSFORM_R = r"""
suppressMessages({library(Matrix); library(Seurat)})
args <- commandArgs(trailingOnly = TRUE)
counts <- readMM(args[1])                       # genes x cells
rownames(counts) <- paste0("g", seq_len(nrow(counts)))
colnames(counts) <- paste0("c", seq_len(ncol(counts)))
object <- CreateSeuratObject(counts = as(counts, "CsparseMatrix"))
# return.only.var.genes = FALSE so the caller, not Seurat, picks the genes;
# the comparison is of the transform, not of two different gene selections.
object <- SCTransform(object, vst.flavor = "v2", verbose = FALSE,
                      return.only.var.genes = FALSE,
                      variable.features.n = nrow(counts))
# Seurat 5 takes `layer`, Seurat 4 takes `slot`, and which one is installed on
# a cluster is not knowable from here. Try both rather than pin a version.
residuals <- tryCatch(
  GetAssayData(object, assay = "SCT", layer = "scale.data"),
  error = function(e) GetAssayData(object, assay = "SCT", slot = "scale.data")
)
if (nrow(residuals) == 0 || ncol(residuals) == 0) {
  stop("SCTransform returned an empty scale.data")
}
write.table(as.matrix(residuals), file = args[2], sep = "\t",
            row.names = FALSE, col.names = FALSE)
"""


def external_normalised(
    counts: np.ndarray, method: str, *, seed: int,
) -> np.ndarray:
    """Cells x genes residuals from an external package.

    Kept in this script rather than in ``cancer_metastasis/common.py`` because
    the cancer pipeline must not acquire an R dependency; these exist to be
    compared against, not to be run in production.
    """
    if method == "scanpy_pearson":
        try:
            import anndata as ad
            import scanpy.experimental as se
        except Exception as error:  # noqa: BLE001 - message matters more
            raise RuntimeError(
                "scanpy_pearson needs a working scanpy; import failed with "
                f"{type(error).__name__}: {error}"
            ) from error
        adata = ad.AnnData(X=counts.astype(np.float32))
        se.pp.normalize_pearson_residuals(adata)
        return np.asarray(adata.X, dtype=np.float32)

    if method == "sctransform":
        import shutil
        import subprocess
        import tempfile
        from scipy import io as scipy_io
        from scipy import sparse as scipy_sparse

        rscript = shutil.which("Rscript")
        if rscript is None:
            raise RuntimeError(
                "sctransform needs Rscript on PATH with Seurat and Matrix "
                "installed; none found"
            )
        with tempfile.TemporaryDirectory() as workspace:
            work = Path(workspace)
            matrix_path, out_path = work / "counts.mtx", work / "residuals.tsv"
            script_path = work / "sctransform.R"
            # Seurat wants genes x cells.
            scipy_io.mmwrite(str(matrix_path),
                             scipy_sparse.csr_matrix(counts.T.astype(np.int32)))
            script_path.write_text(SCTRANSFORM_R, encoding="utf-8")
            finished = subprocess.run(
                [rscript, "--vanilla", str(script_path),
                 str(matrix_path), str(out_path)],
                capture_output=True, text=True, check=False,
            )
            if finished.returncode != 0 or not out_path.exists():
                raise RuntimeError(
                    "SCTransform failed:\n"
                    + (finished.stderr or finished.stdout)[-2000:]
                )
            residuals = np.loadtxt(out_path, dtype=np.float32)
        return residuals.T  # back to cells x genes

    raise ValueError(f"Unknown external representation {method!r}")


def external_joint_pca(
    source_counts: np.ndarray, target_counts: np.ndarray, method: str, *,
    n_hvg: int, n_pcs: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror `prepare_joint_representation`'s tail on external residuals.

    Same steps in the same order as the production path -- stack both sides,
    take the top-variance genes, centre, scale per gene, joint PCA -- so the
    only thing that differs between an external arm and ours is the transform.
    """
    from sklearn.decomposition import PCA

    joint = np.vstack([source_counts, target_counts])
    dense = external_normalised(joint, method, seed=seed)
    # Only the cell count has to match. SCTransform drops genes detected in too
    # few cells, so its residual matrix is narrower than the input, and the
    # next step selects genes by variance anyway.
    if dense.shape[0] != joint.shape[0]:
        raise RuntimeError(
            f"{method} returned {dense.shape[0]} cells, expected "
            f"{joint.shape[0]}"
        )
    if dense.shape[1] < 2:
        raise RuntimeError(f"{method} left {dense.shape[1]} genes")
    if dense.shape[1] != joint.shape[1]:
        print(f"    {method} kept {dense.shape[1]} of {joint.shape[1]} genes")
    variances = dense.var(axis=0)
    selected = np.argsort(-variances, kind="stable")[: min(n_hvg, dense.shape[1])]
    dense = np.array(dense[:, selected], dtype=np.float32)
    dense -= dense.mean(axis=0)
    std = dense.std(axis=0)
    dense /= np.where(std > 1e-8, std, 1.0)
    components = min(n_pcs, dense.shape[0] - 1, dense.shape[1])
    coordinates = PCA(n_components=components,
                      random_state=seed).fit_transform(dense)
    return coordinates[: len(source_counts)], coordinates[len(source_counts):]


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row, leaving any all-zero row alone.

    Cosine distance on these rows is the squared Euclidean distance on them:
    for unit vectors ``||a-b||^2 = 2 - 2 cos(a, b)``. Normalising here therefore
    turns the existing cost, scaling and calibration machinery into the cosine
    version without touching any of it, and makes explicit that "use cosine"
    means "discard each cell's magnitude" and nothing else.
    """
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norm > 0, norm, 1.0)


def rank_auc(values: np.ndarray, positive: np.ndarray) -> float:
    """Mann--Whitney AUC with ``positive`` as the positive class."""
    numeric = np.asarray(values, dtype=np.float64)
    label = np.asarray(positive, dtype=bool)
    finite = np.isfinite(numeric)
    numeric, label = numeric[finite], label[finite]
    n_positive = int(label.sum())
    n_negative = int(numeric.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(numeric)
    return float(
        (ranks[label].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 10 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def simulate_counts(
    rng: np.random.Generator,
    *,
    gene_mean: np.ndarray,
    n_cells: int,
    median_depth: float,
    depth_sigma: float,
    depth_pool: np.ndarray | None = None,
    dispersion: float,
    perturbed_fraction: float,
    perturbation_log2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return integer counts, realised depth, and the perturbed-cell mask.

    ``gene_mean`` is supplied by the caller and must be the *same* vector for
    the source and target side of one replicate.  Drawing it separately per
    side would give the two sides unrelated expression profiles, so the
    homogeneous arms would no longer be homogeneous and rejecting them would be
    correct rather than a specificity failure.

    Every cell's profile is then that shared vector with gamma overdispersion,
    so cells differ biologically only through the optional perturbed
    subpopulation.  Depth is applied afterwards by multinomial sampling, which
    is why depth carries no biological signal by construction.
    """
    n_genes = int(gene_mean.size)
    rates = rng.gamma(
        shape=dispersion, scale=gene_mean / dispersion, size=(n_cells, n_genes)
    )
    perturbed = np.zeros(n_cells, dtype=bool)
    if perturbed_fraction > 0.0:
        count = int(round(perturbed_fraction * n_cells))
        perturbed[rng.choice(n_cells, count, replace=False)] = True
        affected = rng.choice(n_genes, max(1, n_genes // 10), replace=False)
        rates[np.ix_(perturbed, affected)] *= 2.0 ** perturbation_log2
    rates /= rates.sum(axis=1, keepdims=True)
    if depth_pool is not None:
        # Bootstrap from an observed depth distribution. The synthetic sigmas
        # are a dose-response curve chosen for convenience; where the real data
        # sits on that curve is a separate question, and GSE180661's 1.32x
        # retained-to-rejected depth ratio is below even the smallest synthetic
        # arm, so it cannot be read off by interpolation.
        depth = rng.choice(depth_pool, size=n_cells, replace=True)
    elif depth_sigma <= 0.0:
        depth = np.full(n_cells, float(median_depth))
    else:
        depth = median_depth * rng.lognormal(0.0, depth_sigma, size=n_cells)
    depth = np.maximum(depth.round(), 100.0)
    counts = np.empty((n_cells, n_genes), dtype=np.int64)
    for index in range(n_cells):
        counts[index] = rng.multinomial(int(depth[index]), rates[index])
    return counts, np.asarray(counts.sum(axis=1), dtype=np.float64), perturbed


def as_anndata(counts: np.ndarray, prefix: str):
    import anndata as ad

    return ad.AnnData(
        X=counts.astype(np.float32),
        obs=pd.DataFrame(index=[f"{prefix}{i:05d}" for i in range(counts.shape[0])]),
        var=pd.DataFrame(index=[f"gene{j:05d}" for j in range(counts.shape[1])]),
    )


def load_depth_pool(path: Path | None, cap: int | None = None) -> np.ndarray | None:
    """Read observed per-cell depths from a stored cell_confidence.csv.

    ``cap`` applies the transform that read subsampling performs on the depth
    distribution: every cell above the shared target lands on it, every cell
    already below keeps its depth. Passing the target used by
    ``27_downsample_counts.py`` therefore predicts what the corrected data will
    behave like, without refitting a single pair.
    """
    if path is None:
        return None
    table = pd.read_csv(path)
    for column in ("predownsample_total_counts", "total_counts"):
        if column in table:
            values = pd.to_numeric(table[column], errors="coerce").to_numpy(np.float64)
            values = values[np.isfinite(values) & (values >= 100)]
            if values.size < 100:
                raise RuntimeError(f"{path}: only {values.size} usable depths")
            return np.minimum(values, float(cap)) if cap else values
    raise RuntimeError(f"{path} has neither total_counts nor a pre-downsample column")


def run_replicate(
    arm: str, settings: dict, replicate: int, args: argparse.Namespace,
    depth_pool: np.ndarray | None = None,
) -> dict[str, object]:
    seed = args.seed + 7919 * replicate + abs(hash(arm)) % 10_000
    rng = np.random.default_rng(seed)
    # One gene mean vector for both sides: the homogeneous arms are only
    # homogeneous if source and target share it.
    gene_mean = rng.lognormal(mean=0.0, sigma=1.6, size=args.n_genes)
    gene_mean /= gene_mean.sum()
    shared = dict(
        gene_mean=gene_mean, median_depth=args.median_depth,
        dispersion=args.dispersion,
        perturbation_log2=args.perturbation_log2,
        depth_pool=depth_pool if arm.endswith("_observed") else None,
    )
    source_counts, source_depth, perturbed = simulate_counts(
        rng, n_cells=args.n_cells, depth_sigma=settings["depth_sigma"],
        perturbed_fraction=settings["perturbed_fraction"], **shared,
    )
    # The target side never carries the perturbed subpopulation, so perturbed
    # source cells are the only genuinely incompatible cells in any arm.
    target_counts, _, _ = simulate_counts(
        rng, n_cells=args.n_cells, depth_sigma=settings["depth_sigma"],
        perturbed_fraction=0.0, **shared,
    )
    # Read equalisation, before anything else sees the counts. The depth the
    # diagnostics test against stays the pre-equalisation depth, because that
    # is the covariate the gate must not track -- the same reason
    # 27_downsample_counts.py emits predownsample_depth.csv.gz.
    if args.equalise_depth:
        pooled = np.concatenate([source_counts.sum(axis=1),
                                 target_counts.sum(axis=1)])
        target_depth = int(np.quantile(pooled, args.equalise_quantile))
        source_counts = equalise_reads(rng, source_counts, target_depth)
        target_counts = equalise_reads(rng, target_counts, target_depth)

    if args.external_representation:
        source_pca, target_pca = external_joint_pca(
            source_counts, target_counts, args.external_representation,
            n_hvg=args.n_hvg, n_pcs=args.n_pcs, seed=seed,
        )
        hvg = []
    else:
        source = as_anndata(source_counts, "s")
        target = as_anndata(target_counts, "t")
        source_pca, target_pca, hvg, _ = prepare_joint_representation(
            source, target, n_hvg=args.n_hvg, n_pcs=args.n_pcs, seed=seed,
            representation=args.representation, rank_top_n=args.rank_top_n,
            minimum_detection_rate=args.minimum_detection_rate,
        )
    if args.cost == "cosine":
        source_pca, target_pca = unit_rows(source_pca), unit_rows(target_pca)

    pairs = min(1_000_000, len(source_pca) * len(target_pca))
    sampled = np.sum(
        (
            source_pca[rng.integers(len(source_pca), size=pairs)]
            - target_pca[rng.integers(len(target_pca), size=pairs)]
        ) ** 2,
        axis=1,
    )
    positive = sampled[sampled > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    cost = squared_euclidean(source_pca, target_pca) / scale

    calibration_status = "fixed_user_supplied"
    calibration_valid = False
    m4e_inference_valid = False
    m4r_validation_clean = True
    if args.fixed_rejection_cost is not None:
        rejection_cost = float(args.fixed_rejection_cost)
    else:
        limit = min(args.calibration_max_cells, len(source_pca), len(target_pca))
        source_index = np.sort(rng.choice(len(source_pca), limit, replace=False))
        target_index = np.sort(rng.choice(len(target_pca), limit, replace=False))
        total = args.null_calibration_replicates + args.null_validation_replicates
        if args.calibration_null == "within_side_split":
            source_nulls = within_side_null_costs(
                source_pca[source_index], observed_scale=scale,
                seed=seed, n_replicates=total,
            )
            target_nulls = within_side_null_costs(
                target_pca[target_index], observed_scale=scale,
                seed=seed + 7, n_replicates=total,
            )
        else:
            source_nulls, target_nulls = rotation_null_costs(
                source_pca[source_index], target_pca[target_index],
                observed_scale=scale, seed=seed, n_replicates=total,
            )
        split = args.null_calibration_replicates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            calibration = calibrate_confidence_cost(
                source_nulls[:split] + target_nulls[:split],
                source_nulls[split:] + target_nulls[split:],
                backbone="uot", epsilon=args.epsilon,
                lambda_a=args.lambda_a, lambda_b=args.lambda_b,
                null_semantics=args.calibration_null,
                within_side_acceptance_minimum=args.within_side_acceptance_minimum,
                source_rejection_budget=args.source_rejection_budget,
                target_rejection_budget=args.target_rejection_budget,
                tolerance=args.tolerance, grid_size=args.calibration_grid_size,
                device="cpu", emit_warnings=False,
            )
        rejection_cost = float(calibration.rejection_cost)
        calibration_status = str(calibration.selection_status)
        calibration_valid = bool(calibration.calibration_valid)
        # calibration_valid is a strict conjunction that any M4-R terminal
        # warning zeroes, and M4-R exhausting its outer loop is expected rather
        # than disqualifying, so the component the inference rests on is
        # recorded separately.
        m4e_inference_valid = bool(calibration.m4e_inference_valid)
        m4r_validation_clean = bool(calibration.m4r_validation_clean)

    model = ConfidenceOT(
        backbone="uot", variant="exact", rejection_cost=rejection_cost,
        epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
        source_rejection_budget=args.source_rejection_budget,
        target_rejection_budget=args.target_rejection_budget,
        tolerance=args.tolerance, device="cpu", warn_on_terminal=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = model.fit(cost)
    retained = np.asarray(result.source_gate, dtype=bool)
    decision = np.asarray(result.source_confidence.decision_cost, dtype=np.float64)
    sign_rule = decision < rejection_cost

    record: dict[str, object] = {
        "arm": arm,
        "replicate": replicate,
        "depth_sigma": settings["depth_sigma"],
        "perturbed_fraction": settings["perturbed_fraction"],
        "rejection_cost_mode": (
            "fixed" if args.fixed_rejection_cost is not None else "null_calibrated"
        ),
        "calibration_null": (
            "none" if args.fixed_rejection_cost is not None else args.calibration_null
        ),
        "rejection_cost": rejection_cost,
        "calibration_selection_status": calibration_status,
        "calibration_valid": calibration_valid,
        "m4e_inference_valid": m4e_inference_valid,
        "m4r_validation_clean": m4r_validation_clean,
        "hvg_n": len(hvg),
        "source_n": int(retained.size),
        # Question 1: is the rate informative?
        "source_rejection_rate": float(np.mean(~retained)),
        "source_rejection_budget_cap": args.source_rejection_budget,
        "sign_rule_retained_fraction": float(np.mean(sign_rule)),
        "median_decision_cost": float(np.median(decision)),
        # Question 2: is the identity depth-driven?
        "auc_total_counts": rank_auc(source_depth, retained),
        "spearman_decision_cost_total_counts": rank_correlation(
            decision, source_depth
        ),
        "median_total_counts_retained": float(np.median(source_depth[retained]))
        if retained.any() else float("nan"),
        "median_total_counts_rejected": float(np.median(source_depth[~retained]))
        if (~retained).any() else float("nan"),
        "observed_depth_cv": float(source_depth.std() / source_depth.mean()),
    }
    if settings["perturbed_fraction"] > 0.0:
        # Sensitivity against the only ground-truth incompatible cells.
        true_positive = int(np.sum(perturbed & ~retained))
        predicted = int(np.sum(~retained))
        actual = int(np.sum(perturbed))
        precision = true_positive / predicted if predicted else float("nan")
        recall = true_positive / actual if actual else float("nan")
        record["perturbed_recall"] = recall
        record["perturbed_precision"] = precision
        record["perturbed_f1"] = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
            else float("nan")
        )
        record["auc_perturbed_rejected"] = rank_auc(decision, perturbed)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--n-cells", type=int, default=1500,
                        help="Cells per side; 1500 matches the real per-pair median")
    parser.add_argument("--n-genes", type=int, default=4000)
    parser.add_argument("--median-depth", type=float, default=10000.0)
    parser.add_argument("--dispersion", type=float, default=2.0)
    parser.add_argument("--perturbation-log2", type=float, default=1.0)
    parser.add_argument(
        "--minimum-detection-rate", type=float, default=0.0,
        help="Drop genes detected in fewer than this fraction of cells. "
             "Orthogonal to --representation and composes with any of them.",
    )
    parser.add_argument(
        "--representation",
        choices=("log_cpm", "rank_value", "rank_no_median", "pearson_residuals"),
        default="log_cpm",
        help="Cell representation. 'rank_value' is the Geneformer formulation: "
             "expression over each gene's nonzero median, ranked within the "
             "cell, top --rank-top-n kept. It is depth-invariant by "
             "construction, where subsampling to a common total left the gate "
             "tracking original depth at AUC 0.557 on GSE180661.",
    )
    parser.add_argument(
        "--rank-top-n", type=int, default=512,
        help="Genes kept per cell under rank_value. Must stay below the "
             "shallowest cell's detected-gene count or detection breadth, the "
             "surviving part of the depth effect, re-enters through the "
             "list length.",
    )
    parser.add_argument(
        "--external-representation",
        choices=("scanpy_pearson", "sctransform"), default=None,
        help="Normalise with an external package instead of --representation, "
             "as a reference point the reviewer can check: scanpy's "
             "experimental.pp.normalize_pearson_residuals, or Seurat's "
             "SCTransform v2 through Rscript. Composes with --equalise-depth "
             "and --cost. Needs a working scanpy, or Rscript with Seurat.",
    )
    parser.add_argument(
        "--equalise-depth", action="store_true",
        help="Subsample every cell's reads to one shared total before "
             "anything else, as 27_downsample_counts.py does. Composes with "
             "--representation; the production cancer path is this plus "
             "rank_value, a combination the simulation could not express.",
    )
    parser.add_argument(
        "--equalise-quantile", type=float, default=0.10,
        help="Pooled-depth quantile the shared target is taken from",
    )
    parser.add_argument(
        "--cost", choices=("squared_euclidean", "cosine"),
        default="squared_euclidean",
        help="Cost geometry. 'cosine' L2-normalises each cell's PCA "
             "coordinates first, which discards magnitude and nothing else.",
    )
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--source-rejection-budget", type=float, default=0.85)
    parser.add_argument("--target-rejection-budget", type=float, default=0.00)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--calibration-max-cells", type=int, default=2000)
    parser.add_argument("--calibration-grid-size", type=int, default=5)
    parser.add_argument("--null-calibration-replicates", type=int, default=5)
    parser.add_argument("--null-validation-replicates", type=int, default=5)
    parser.add_argument(
        "--calibration-null", default="within_side_split",
        choices=("within_side_split", "cross_side_rotation"),
        help="Null the rejection cost is calibrated against",
    )
    parser.add_argument(
        "--within-side-acceptance-minimum", type=float, default=0.90,
        help="Minimum within-side null acceptance the rejection cost must reach",
    )
    parser.add_argument(
        "--fixed-rejection-cost", type=float, default=None,
        help="Bypass null calibration to isolate the calibration target's effect",
    )
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--depth-source", type=Path, default=None,
        help="A stored cell_confidence.csv whose total_counts supply an "
             "observed depth distribution; enables the *_observed arms",
    )
    parser.add_argument(
        "--depth-cap", type=int, default=None,
        help="Clip the observed depths at this shared target, which is what "
             "read subsampling does to the distribution; use the target_depth "
             "from 27_downsample_counts.py to predict the corrected behaviour",
    )
    parser.add_argument(
        "--arm", action="append", choices=sorted({*ARMS, *OBSERVED_ARMS}),
        help="Restrict to these arms; default runs all available",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    depth_pool = load_depth_pool(args.depth_source, args.depth_cap)
    available = dict(ARMS)
    if depth_pool is not None:
        available.update(OBSERVED_ARMS)
    selected = args.arm or sorted(available)
    missing = [arm for arm in selected if arm not in available]
    if missing:
        raise RuntimeError(
            f"{missing} need --depth-source to supply an observed distribution"
        )
    records = []
    for arm in selected:
        for replicate in range(args.replicates):
            record = run_replicate(
                arm, available[arm], replicate, args, depth_pool
            )
            records.append(record)
            print(
                f"{arm} rep={replicate} "
                f"rejection={record['source_rejection_rate']:.3f} "
                f"auc_depth={record['auc_total_counts']:.3f} "
                f"rho_cost_depth={record['spearman_decision_cost_total_counts']:.3f}",
                flush=True,
            )

    replicate_table = pd.DataFrame(records)
    numeric = [
        column for column in replicate_table.columns
        if column not in {"arm", "rejection_cost_mode", "calibration_selection_status"}
        and pd.api.types.is_numeric_dtype(replicate_table[column])
    ]
    summary = (
        replicate_table.groupby("arm", sort=True)[numeric]
        .median()
        .reset_index()
    )
    replicate_table.to_csv(args.output_root / "depth_null_replicates.csv", index=False)
    summary.to_csv(args.output_root / "depth_null_arm_summary.csv", index=False)
    report = {
        "rejection_cost_mode": (
            "fixed" if args.fixed_rejection_cost is not None else "null_calibrated"
        ),
        "source_rejection_budget": args.source_rejection_budget,
        "target_rejection_budget": args.target_rejection_budget,
        "replicates_per_arm": args.replicates,
        "cells_per_side": args.n_cells,
        "depth_source": str(args.depth_source) if args.depth_source else None,
        "depth_cap": args.depth_cap,
        "arms": {arm: available[arm] for arm in selected},
        "expected_outcomes": {
            "homogeneous arms": (
                "No incompatible cell exists, so a specific method rejects "
                "almost none. A rate near the cap means the rate is set by the "
                "calibration target rather than by compatibility."
            ),
            "auc_total_counts": (
                "0.5 under specificity. Compare against the real data, where "
                "GSE180661 gives 0.67 and GSE225857 gives 0.79."
            ),
            "perturbed_depth_cv0": (
                "Positive control: recall should be high and precision should "
                "greatly exceed the perturbed fraction, otherwise the method "
                "has no power in this regime either."
            ),
        },
    }
    (args.output_root / "depth_null_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
