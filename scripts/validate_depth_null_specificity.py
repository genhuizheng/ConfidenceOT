"""Standalone specificity check for the ConfidenceOT rejection decision.

This script is deliberately independent of the Splatter scaling benchmark.  It
builds its own counts so that the ground truth is a construction rather than a
simulator parameter, then runs the production path through the same object the
cancer workflow uses, :class:`confidenceot.Preprocessing`: ``raw counts ->
optional read equalisation -> normalisation -> joint HVG -> gene scaling ->
joint PCA -> optional L2 normalisation -> squared Euclidean cost -> median
scaling -> null calibration -> M4-E``.

Running one object rather than a parallel copy of those steps is the point.
Until it was shared, this file and ``cancer_metastasis/02_run_pair.py`` each
wrote out the transform, the cosine step and the median scale, which meant this
screen measured configurations that the production runner only approximated.
``tests/test_preprocessing.py`` pins the move against values captured from the
old path.

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
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confidenceot import (  # noqa: E402
    ConfidenceOT,
    Preprocessing,
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
    # Detection breadth at a fixed total depth. Depth is constant in these
    # arms, so a method that only equalises totals has nothing left to do and
    # whatever effect remains is the low-gene-number effect on its own. They
    # are the arms the depth arms cannot substitute for: in every other arm
    # here, and in splatter, breadth is a function of depth, so "fixed the
    # depth" and "fixed the breadth" are one statement.
    "homogeneous_breadth_mild": {"depth_sigma": 0.0, "perturbed_fraction": 0.0,
                                 "breadth_fraction": 0.5,
                                 "breadth_ratio": 0.50},
    "homogeneous_breadth_strong": {"depth_sigma": 0.0,
                                   "perturbed_fraction": 0.0,
                                   "breadth_fraction": 0.5,
                                   "breadth_ratio": 0.20},
    # The same breadth split with the planted subpopulation still present, so
    # power can be read in the regime where the correction is working. A
    # correction that removes breadth by removing the cells' differences would
    # pass the two arms above and fail this one.
    "perturbed_breadth_strong": {"depth_sigma": 0.0, "perturbed_fraction": 0.2,
                                 "breadth_fraction": 0.5,
                                 "breadth_ratio": 0.20},
}

# Enabled only by --depth-source: the same two questions asked at the depth
# distribution an actual dataset has, rather than at a chosen sigma.
OBSERVED_ARMS = {
    "homogeneous_depth_observed": {"depth_sigma": 0.0, "perturbed_fraction": 0.0},
    "perturbed_depth_observed": {"depth_sigma": 0.0, "perturbed_fraction": 0.2},
}


def arm_offset(arm: str) -> int:
    """A per-arm seed offset that is the same in every process.

    ``abs(hash(arm)) % 10_000`` was used here, and Python randomises string
    hashing per process unless ``PYTHONHASHSEED`` is set, so every invocation
    of this script drew a different seed for the same arm.  Two runs of one
    configuration were therefore never the same run, which is not a small
    matter for a script whose output is a table of small differences between
    configurations.

    This does not recover the seeds of runs already made -- those are gone.
    What it fixes is that a rerun now reproduces, and that a difference between
    two arms is a difference between the arms.
    """
    return int(hashlib.sha256(arm.encode("utf-8")).hexdigest()[:8], 16) % 10_000


def preprocessing_for(args: argparse.Namespace):
    """Build the one configuration object every arm of this screen runs.

    The screen's whole output is a table indexed by configuration, so the
    configuration has to be the same object the production runner uses --
    otherwise the table describes a pipeline that only exists here.  That was
    the case until this was moved: the transform, the equalisation, the cosine
    step and the median scale were written out separately in this file and in
    ``cancer_metastasis/02_run_pair.py``.

    ``--equalise-depth`` matters more than its one line suggests.  Its arm was
    missing from the simulation entirely at first, so the gene-ranking arm
    measured ranking *alone*: on real prostate data ranking alone leaves the
    gate at AUC 0.060 while ranking plus equalisation reaches 0.493, and the
    figure was showing a pipeline nobody had run.
    """
    if args.external_representation:
        return Preprocessing(
            normalisation="precomputed",
            label_stem=args.external_representation,
            equalise_depth=bool(args.equalise_depth),
            equalise_quantile=args.equalise_quantile,
            n_hvg=args.n_hvg, n_pcs=args.n_pcs, cost=args.cost,
            regress_out=tuple(args.regress_out),
        )
    return Preprocessing(
        normalisation=args.representation,
        rank_top_n=args.rank_top_n,
        minimum_detection_rate=args.minimum_detection_rate,
        equalise_depth=bool(args.equalise_depth),
        equalise_quantile=args.equalise_quantile,
        n_hvg=args.n_hvg, n_pcs=args.n_pcs, cost=args.cost,
        regress_out=tuple(args.regress_out),
    )


SCTRANSFORM_R = r"""
# sctransform::vst, not Seurat::SCTransform. The algorithm lives in the
# standalone package and Seurat's function is a wrapper around this call, but
# its dependencies are only Matrix, Rcpp, RcppArmadillo, matrixStats, dplyr and
# ggplot2 -- none of which need zlib, libcurl or libssl. Installing Seurat into
# the rocker r-ver container fails on exactly those system headers and takes 19
# packages down with it, every one of them from the interactive and plotting
# chain this comparison never touches.
# --vanilla skips the Renviron files, so R_LIBS_USER is honoured only because
# it is in the process environment. Prepending it here makes that explicit and
# survives a caller that passes different R flags.
local_library <- Sys.getenv("R_LIBS_USER")
if (nzchar(local_library)) {
  .libPaths(c(local_library, .libPaths()))
}
suppressMessages({library(Matrix); library(sctransform)})
args <- commandArgs(trailingOnly = TRUE)
counts <- readMM(args[1])                       # genes x cells
rownames(counts) <- paste0("g", seq_len(nrow(counts)))
colnames(counts) <- paste0("c", seq_len(ncol(counts)))
counts <- as(counts, "CsparseMatrix")

# min_cells = 1 so the caller decides which genes survive; the comparison is of
# the transform, not of two different gene selections. v2 is the flavour Seurat
# itself defaults to.
fit <- vst(counts, vst.flavor = "v2", residual_type = "pearson",
           min_cells = 1, return_cell_attr = FALSE, verbosity = 0)
residuals <- fit$y
if (is.null(residuals) || nrow(residuals) == 0 || ncol(residuals) == 0) {
  stop("sctransform::vst returned no residuals")
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
        import os
        import shlex
        import shutil
        import subprocess
        import tempfile
        from scipy import io as scipy_io
        from scipy import sparse as scipy_sparse

        # R is often not on PATH at all -- here it is a conda environment
        # built for another project -- so the whole invocation is
        # configurable, container form included:
        #   CONFIDENCEOT_RSCRIPT="apptainer exec -B /scratch /path/r.sif Rscript"
        # CONFIDENCEOT_R_WORKDIR puts the handoff files somewhere the container
        # can actually see, since a container binds only some of the host.
        configured = os.environ.get("CONFIDENCEOT_RSCRIPT", "").strip()
        if configured:
            command = shlex.split(configured)
        elif shutil.which("Rscript"):
            command = ["Rscript"]
        else:
            raise RuntimeError(
                "sctransform needs R with the sctransform and Matrix "
                "packages. Either put Rscript on PATH, or set "
                "CONFIDENCEOT_RSCRIPT to the full invocation. On TACC, "
                "scripts/tacc/r_environment.sh finds one and sets it."
            )
        parent = os.environ.get("CONFIDENCEOT_R_WORKDIR") or None
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as workspace:
            work = Path(workspace)
            matrix_path, out_path = work / "counts.mtx", work / "residuals.tsv"
            script_path = work / "sctransform.R"
            # vst wants genes x cells.
            scipy_io.mmwrite(str(matrix_path),
                             scipy_sparse.csr_matrix(counts.T.astype(np.int32)))
            script_path.write_text(SCTRANSFORM_R, encoding="utf-8")
            finished = subprocess.run(
                [*command, "--vanilla", str(script_path),
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
    breadth_fraction: float = 0.0,
    breadth_ratio: float = 0.25,
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

    ``breadth_fraction`` is what makes the low-gene-number question answerable
    at all.  Without it, detection breadth is a function of total depth and
    nothing else -- more reads, more non-zero genes -- so in this simulation
    "removed the depth effect" and "removed the detection-breadth effect" are
    the same statement, and no arm can tell a method that does one from a
    method that does the other.  That is why the screen passed
    ``rank + equalise + cosine`` while the real data still shows the leading
    axes tracking detected genes at rho 0.645.

    Set it and that fraction of cells draws from a narrower slice of the gene
    panel -- ``breadth_ratio`` of it -- at the **same** total depth.  Detection
    breadth then varies with total counts held fixed, which is the arm a
    correction aimed at breadth has to clear and a correction aimed at depth
    cannot.

    Those cells are not marked perturbed.  Narrow and broad cells share one
    expression profile over the genes they both express, so there is no
    subpopulation to find: an arm built this way still has "reject nothing" as
    its correct answer, and any rejection it draws is a specificity failure in
    the same sense as the depth arms.
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
    if breadth_fraction > 0.0:
        if not 0.0 < breadth_ratio <= 1.0:
            raise ValueError("breadth_ratio must lie in (0, 1]")
        narrow = np.zeros(n_cells, dtype=bool)
        narrow[rng.choice(n_cells, int(round(breadth_fraction * n_cells)),
                          replace=False)] = True
        kept = max(2, int(round(breadth_ratio * n_genes)))
        for index in np.flatnonzero(narrow):
            # A different slice per cell, so "narrow" is not itself a
            # subpopulation sharing a profile -- which it would be if every
            # narrow cell were silenced on the same genes, and then rejecting
            # them would be correct.
            silenced = rng.choice(n_genes, n_genes - kept, replace=False)
            rates[index, silenced] = 0.0
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


def load_splatter_arm(
    root: Path, arm: str, replicate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts for one arm and replicate, as prepared by splatter.

    Written by ``scripts/generate_splatter_depth_arms.R``, which expresses the
    same depth ladder in splatter's own ``lib.loc`` and ``lib.scale``. Files
    are cells x genes, so no transpose here; the R side does it.

    Replicates are one-based on disk and zero-based in this script, which is
    the only place the two conventions meet.
    """
    from scipy.io import mmread

    directory = Path(root) / arm / f"rep_{replicate + 1:02d}"
    needed = ("source.mtx", "target.mtx", "source_perturbed.csv")
    missing = [name for name in needed if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{directory} is missing {', '.join(missing)}. Generate the arms "
            f"with scripts/generate_splatter_depth_arms.R first."
        )
    source = np.asarray(mmread(directory / "source.mtx").todense(),
                        dtype=np.int64)
    target = np.asarray(mmread(directory / "target.mtx").todense(),
                        dtype=np.int64)
    perturbed = (pd.read_csv(directory / "source_perturbed.csv")
                 .perturbed.to_numpy(dtype=bool))
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"{directory}: source has {source.shape[1]} genes, target "
            f"{target.shape[1]}"
        )
    if perturbed.size != source.shape[0]:
        raise ValueError(
            f"{directory}: {perturbed.size} perturbation flags for "
            f"{source.shape[0]} source cells"
        )
    return source, target, perturbed


def run_replicate(
    arm: str, settings: dict, replicate: int, args: argparse.Namespace,
    depth_pool: np.ndarray | None = None,
) -> dict[str, object]:
    seed = args.seed + 7919 * replicate + arm_offset(arm)
    rng = np.random.default_rng(seed)
    if args.splatter_arms:
        # Depth was applied by splatter's lib.scale rather than by the
        # multinomial construction below. Everything downstream -- equalisation,
        # the transform, the joint PCA, the gate, the diagnostics -- is
        # unchanged, so the two paths differ only in where the counts came from.
        source_counts, target_counts, perturbed = load_splatter_arm(
            args.splatter_arms, arm, replicate
        )
        if source_counts.shape[0] != args.n_cells:
            raise ValueError(
                f"{arm} rep {replicate}: the prepared arm has "
                f"{source_counts.shape[0]} source cells, --n-cells says "
                f"{args.n_cells}. Regenerate, or pass the matching --n-cells."
            )
        expected = int(round(settings["perturbed_fraction"] * args.n_cells))
        if int(perturbed.sum()) != expected:
            raise ValueError(
                f"{arm} rep {replicate}: the prepared arm plants "
                f"{int(perturbed.sum())} cells, this arm expects {expected}"
            )
        source_depth = source_counts.sum(axis=1).astype(np.float64)
    else:
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
        breadth = {
            "breadth_fraction": settings.get("breadth_fraction", 0.0),
            "breadth_ratio": settings.get("breadth_ratio", 0.25),
        }
        source_counts, source_depth, perturbed = simulate_counts(
            rng, n_cells=args.n_cells, depth_sigma=settings["depth_sigma"],
            perturbed_fraction=settings["perturbed_fraction"], **breadth,
            **shared,
        )
        # The target side never carries the perturbed subpopulation, so
        # perturbed source cells are the only genuinely incompatible cells in
        # any arm.
        target_counts, _, _ = simulate_counts(
            rng, n_cells=args.n_cells, depth_sigma=settings["depth_sigma"],
            perturbed_fraction=0.0, **breadth, **shared,
        )
    # Both covariates are read off the counts as simulated, before read
    # equalisation touches them, for the same reason 27_downsample_counts.py
    # emits predownsample_depth.csv.gz: these are what the gate must not
    # track, and after equalisation the total is near constant by
    # construction while the detected-gene count is not.
    source_detected = np.asarray((source_counts > 0).sum(axis=1),
                                 dtype=np.float64)
    configuration = preprocessing_for(args)
    if args.external_representation:
        # The transform runs in its own package, then re-enters the shared
        # chain as a precomputed representation. The alternative -- an R and a
        # scanpy dependency inside confidenceot -- would put two heavy optional
        # packages behind every import of the solver.
        #
        # Equalisation first, and drawn from the same generator in the same
        # order as every other arm, so an external arm and one of ours differ
        # in the transform and in nothing else.
        source_counts, target_counts, equalisation = configuration.equalise(
            source_counts, target_counts, rng=rng
        )
        joint = external_normalised(
            np.vstack([source_counts, target_counts]),
            args.external_representation, seed=seed,
        )
        expected_cells = len(source_counts) + len(target_counts)
        if joint.shape[0] != expected_cells:
            raise RuntimeError(
                f"{args.external_representation} returned {joint.shape[0]} "
                f"cells, expected {expected_cells}"
            )
        if joint.shape[1] < 2:
            raise RuntimeError(
                f"{args.external_representation} left {joint.shape[1]} genes"
            )
        if joint.shape[1] != source_counts.shape[1]:
            # SCTransform drops genes detected in too few cells, so a narrower
            # residual matrix is expected rather than wrong.
            print(f"    {args.external_representation} kept {joint.shape[1]} "
                  f"of {source_counts.shape[1]} genes")
        representation = configuration.representation(
            joint[: len(source_counts)], joint[len(source_counts):],
            seed=seed, extra_provenance=equalisation,
        )
        prepared = configuration.cost_from_representation(representation, rng=rng)
        hvg = []
    else:
        prepared = configuration.cost_matrix(
            source_counts, target_counts, seed=seed, rng=rng,
            source_genes=[f"gene{j:05d}" for j in range(source_counts.shape[1])],
            target_genes=[f"gene{j:05d}" for j in range(target_counts.shape[1])],
        )
        hvg = list(prepared.representation.selected_genes)
    source_pca = prepared.representation.source
    target_pca = prepared.representation.target
    scale = prepared.scale
    cost = prepared.cost

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
        # And the same question about detection breadth, which is a separate
        # quantity even though the depth arms cannot separate them. On the
        # breadth arms the depth is constant, so `auc_total_counts` there is
        # trivially 0.5 and these two columns carry the whole reading; on the
        # depth arms they move together and the pair is what shows it.
        "auc_detected_genes": rank_auc(source_detected, retained),
        "spearman_decision_cost_detected_genes": rank_correlation(
            decision, source_detected
        ),
        "median_detected_genes_retained": (
            float(np.median(source_detected[retained])) if retained.any()
            else float("nan")),
        "median_detected_genes_rejected": (
            float(np.median(source_detected[~retained])) if (~retained).any()
            else float("nan")),
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
        "--regress-out", action="append", default=[],
        choices=("detected_genes", "total_counts"),
        help="Remove this covariate's linear component from the principal "
             "components before the cost is built; may be repeated. Regressed "
             "on the covariate's rank, because the correlation being removed "
             "is measured as Spearman. It cannot tell technical variation "
             "from biological: at one depth a transcriptionally broader cell "
             "genuinely detects more genes. In *this* simulation it can, "
             "because breadth carries no biology here -- which is exactly why "
             "a good result on the depth arms would not transfer, and why the "
             "homogeneous_breadth arms exist.",
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
        "--splatter-arms", type=Path, default=None,
        help="Directory of arms prepared by "
             "scripts/generate_splatter_depth_arms.R. Takes the counts from "
             "splatter, with the depth ladder expressed as its lib.scale, "
             "instead of building them here. Everything after the counts is "
             "unchanged, so the two paths are directly comparable.",
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
                f"auc_genes={record['auc_detected_genes']:.3f} "
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
        # What was actually compared. Until now the only record of a run's
        # representation, cost and equalisation was its output directory's
        # name, so two runs side by side could not be told apart from their own
        # files -- and the whole point of the screen is that these differ.
        "counts_source": ("splatter" if args.splatter_arms else "internal"),
        "splatter_arms": (str(args.splatter_arms) if args.splatter_arms
                          else None),
        "representation": (args.external_representation
                           or args.representation),
        "representation_is_external": bool(args.external_representation),
        "rank_top_n": (args.rank_top_n
                       if args.representation == "rank_value" else None),
        "equalise_depth": bool(args.equalise_depth),
        "equalise_quantile": (args.equalise_quantile
                              if args.equalise_depth else None),
        "cost": args.cost,
        # The configuration as the object itself records it, so a run can be
        # identified from its own report. The hand-written fields above are
        # kept because existing collectors read them, but they were assembled
        # by listing arguments one at a time and `--regress-out` was already
        # missing from that list the first time it ran.
        "preprocessing": preprocessing_for(args).as_dict(),
        "preprocessing_label": preprocessing_for(args).label(),
        "n_genes": args.n_genes,
        "n_hvg": args.n_hvg,
        "n_pcs": args.n_pcs,
        "seed": args.seed,
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
