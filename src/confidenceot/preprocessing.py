"""Everything between counts and a cost matrix, as one named configuration.

ConfidenceOT itself takes a cost matrix and returns a gate.  What decides the
answer in practice is upstream of that: which transform the representation is
built in, whether read depth was equalised first, and whether the cost is
squared Euclidean or cosine.  The depth screen measured exactly those choices,
and its result is a table of configurations -- so the configuration has to be a
thing the code can name, record and hand around, not a set of flags reassembled
at each call site.

That is what :class:`Preprocessing` is.  One frozen object carries the whole
chain, ``cost_matrix`` runs it, ``as_dict`` records it and ``label`` prints the
short name the screen's tables use.

**Why this is a separate module and not part of the solver.**  Depth is a
property of the data, not of the transport problem, and this project's claim
that the correction is decoupled from the algorithm only means something if the
two can be read apart.  So :meth:`ConfidenceOT.fit` still takes a cost matrix,
and nothing here can reach inside a solve.  The convenience wrapper
:meth:`ConfidenceOT.fit_counts` composes the two in the obvious order and
returns both halves, so the composition is visible rather than hidden.

**What the evidence covers.**  The screen measured the depth effect and the
false-rejection rate for these combinations, N=1500 and N=5000, on two
independent simulators:

======================================  ======  ==========  =============
configuration                              AUC  effect/s.e.  retained:rejected
======================================  ======  ==========  =============
``sctransform + equalise + cosine``      0.507         0.1             1.08x
``rank + equalise + cosine``             0.526         0.5             1.10x
``Pearson + equalise + cosine``          0.527         0.5             1.11x
``log CPM + cosine``                     0.583         3.2             1.29x
``rank + equalise`` (Euclidean)          0.719         8.7             2.02x
``log CPM``, untreated                   0.990        19.5             4.92x
======================================  ======  ==========  =============

The ordering that survived both simulators is the cost: cosine separates and
Euclidean does not.  ``equalise_depth``'s own contribution is *not* attributed
-- on the hand-built simulation read equalisation is the exact inverse of that
simulation's depth mechanism, so its apparent benefit there was an artefact of
our own construction.  See ``cancer_metastasis/SEQUENCING_DEPTH_RESOLUTION.md``.

``cpm`` and ``log1p`` are provided because the chain needs them, not because
the screen measured them.  An option with no evidence behind it should not look
like one that has some.

``scikit-learn`` is required for the PCA step and is an optional dependency of
this package; it is imported where it is used, so ``fit(cost_matrix)`` on a
core install is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

Normalisation = Literal[
    "cpm", "log_cpm", "log1p", "precomputed",
    "pearson_residuals", "rank_value", "rank_no_median",
]
Cost = Literal["squared_euclidean", "cosine"]
Scale = Literal["median_sampled_pair", "none"]

NORMALISATIONS: tuple[str, ...] = (
    "cpm", "log_cpm", "log1p", "precomputed",
    "pearson_residuals", "rank_value", "rank_no_median",
)
COSTS: tuple[str, ...] = ("squared_euclidean", "cosine")
SCALES: tuple[str, ...] = ("median_sampled_pair", "none")

# Short names the screen's own tables use, so one object generates both the
# configuration and the row it is reported on.
_LABEL_STEM = {
    "cpm": "cpm",
    "log_cpm": "logcpm",
    "log1p": "log1p",
    "pearson_residuals": "pearson",
    "rank_value": "rank",
    "rank_no_median": "ranknm",
}


# --------------------------------------------------------------------------
# Primitives. Each is separately callable and separately testable, because
# every one of them has at some point been the thing under suspicion.
# --------------------------------------------------------------------------

def equalise_depth(
    counts: ArrayLike,
    *,
    rng: np.random.Generator,
    target: int | None = None,
    quantile: float = 0.10,
    return_untouched: bool = False,
):
    """Subsample every cell's reads to one shared total.

    Exact multivariate hypergeometric sampling of reads without replacement,
    the same operation as ``scanpy.pp.downsample_counts``.  Cells already at or
    below the target are left untouched; they are the residual depth gradient,
    so ``return_untouched`` hands back the mask and the pre-equalisation depth
    for a caller that has to report them.

    ``target`` fixes the shared depth.  Omit it and the ``quantile`` of the
    pooled depth is used, but a quantile makes the target a function of the cell
    set: two runs on different cells then differ in two ways at once, so pass
    ``target`` explicitly when reproducing one.

    The draw is taken over each cell's **nonzero** genes, not over the full gene
    vector.  The two are the same distribution -- a colour with zero balls
    contributes nothing -- but they consume the generator differently, and only
    the first is possible on a real matrix: densifying 150,000 cells by 30,000
    genes to subsample them is not an option.  Sparse and dense input therefore
    give the identical result here, which is the property that lets the
    simulation and the production stage be compared at all.

    A warning that is easy to lose: on data whose depth mechanism is
    ``Multinomial(depth, p)`` with ``p`` independent of depth, this operation is
    that mechanism's exact inverse, and any measurement of it on such data is
    circular.  That is what happened to this project's first depth simulation.
    """
    was_sparse = sparse.issparse(counts)
    matrix = sparse.csr_matrix(counts, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("counts must be a 2-D cells-by-genes matrix")
    original = np.asarray(matrix.sum(axis=1), dtype=np.int64).ravel()
    if target is None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must lie in (0, 1)")
        target = int(np.quantile(original, quantile))
    if target < 1:
        raise ValueError(f"target depth resolved to {target}; raise the quantile")
    untouched = original <= target
    rows = []
    for index in range(matrix.shape[0]):
        start, stop = matrix.indptr[index], matrix.indptr[index + 1]
        row = matrix.data[start:stop]
        if untouched[index] or row.size == 0:
            rows.append(row)
            continue
        rows.append(rng.multivariate_hypergeometric(row, int(target)))
    reduced = sparse.csr_matrix(
        (np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64),
         matrix.indices.copy(), matrix.indptr.copy()),
        shape=matrix.shape, dtype=np.int64,
    )
    # A gene that lost all of its reads would otherwise stay an explicit zero,
    # which makes the matrix disagree with itself about what was detected --
    # and detection breadth is exactly what the rank cut is measured against.
    reduced.eliminate_zeros()
    equalised = reduced if was_sparse else np.asarray(reduced.todense())
    if return_untouched:
        return equalised, original, untouched
    return equalised


def unit_rows(matrix: ArrayLike) -> NDArray[np.floating]:
    """L2-normalise each row, leaving any all-zero row alone.

    Cosine distance on these rows *is* the squared Euclidean distance on them:
    for unit vectors ``||a - b||^2 = 2 - 2 cos(a, b)``.  Normalising the
    representation therefore turns the existing cost, median scaling and null
    calibration into the cosine version without touching any of them, and makes
    explicit that "use cosine" means "discard each cell's magnitude and nothing
    else".

    Applied before the cost is built *and* before the median scale is
    estimated, so the scale is measured on the geometry the gate sees.

    The floating dtype is preserved rather than promoted to float64: the screen
    measured its conclusions on float32 PCA coordinates, and promoting here
    would make those runs non-reproducible for no gain, since the norm of a
    30-vector is nowhere near float32's limits.
    """
    values = np.asarray(matrix)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norm > 0.0, norm, 1.0)


def squared_euclidean(left: ArrayLike, right: ArrayLike) -> NDArray[np.float64]:
    """Pairwise squared Euclidean distance, clipped at zero.

    The clip matters: the expansion ``|a|^2 + |b|^2 - 2 a.b`` is not guaranteed
    non-negative in floating point, and a small negative entry in a cost matrix
    is not a small error -- it is a cell the gate can profit from keeping.
    """
    a = np.asarray(left)
    b = np.asarray(right)
    value = (
        np.sum(a * a, axis=1)[:, None]
        + np.sum(b * b, axis=1)[None, :]
        - 2.0 * a @ b.T
    )
    return np.maximum(value, 0.0)


def median_pair_scale(
    source: ArrayLike,
    target: ArrayLike,
    *,
    rng: np.random.Generator,
    pairs: int = 1_000_000,
) -> float:
    """Median squared distance over randomly sampled cross-side pairs.

    The cost is divided by this so that ``rejection_cost`` is comparable across
    datasets of different dimension and spread; without it the calibrated cost
    would carry the units of whatever representation happened to be used.

    Zero distances are excluded before the median.  They are duplicate or
    identical cells, and including them would drag the scale toward zero and
    inflate every cost.  If every sampled pair is at distance zero the scale is
    1.0, which leaves the costs untouched rather than producing NaN.

    ``rng`` is drawn from twice, source indices then target indices, in that
    order.  Callers that reuse the generator afterwards depend on that, so the
    order is part of the contract, not an implementation detail.
    """
    a = np.asarray(source)
    b = np.asarray(target)
    count = min(int(pairs), max(len(a) * len(b), 1))
    sampled = np.sum(
        (a[rng.integers(len(a), size=count)]
         - b[rng.integers(len(b), size=count)]) ** 2,
        axis=1,
    )
    positive = sampled[sampled > 0]
    return float(np.median(positive)) if positive.size else 1.0


def _nonzero_median_per_gene(matrix: Any) -> np.ndarray:
    """Median of each gene's nonzero values, as Geneformer defines its scale.

    Zeros are excluded because including them would make the median zero for
    almost every gene in single-cell data, leaving nothing to divide by.
    """
    columns = matrix.tocsc()
    medians = np.ones(columns.shape[1], dtype=np.float64)
    for gene in range(columns.shape[1]):
        start, stop = columns.indptr[gene], columns.indptr[gene + 1]
        if stop > start:
            medians[gene] = float(np.median(columns.data[start:stop]))
    return np.where(medians > 0, medians, 1.0)


def rank_value_encode(joint: Any, top_n: int, use_gene_median: bool = True) -> Any:
    """Encode each cell as its ranking of genes, in Geneformer's formulation.

    Each cell is divided by its own total, each gene by its nonzero median over
    the joint matrix, and the genes are then ranked within the cell.  The two
    divisions are what make the ordering informative: ranking raw expression
    puts the same housekeeping genes on top of every cell, whereas ranking
    ``expression / gene median`` puts whatever is unusually high in *this* cell
    on top.

    The median must be taken over both sides together.  Taken per side it would
    normalise away exactly the cross-side differences the gate exists to
    detect, and the method would be blind by construction.

    Only the top ``top_n`` genes are kept, and the value falls linearly from
    1.0 to 1/top_n across them.  A fixed cut is what makes the encoding
    depth-invariant: cells differ in how many genes they detect, so a variable
    cut would let detection breadth back in, which is the part of the depth
    effect that survives subsampling to a common total -- on GSE180661 the gate
    still tracked original depth at AUC 0.557 among cells all sitting at
    exactly 3,119 counts.

    ``top_n`` therefore has to stay below the shallowest cell's detected-gene
    count.  It is not enforced here, because the encoder cannot tell a
    deliberate choice from an accident; ``cancer_metastasis/tools/
    audit_rank_top_n.py`` measures it against real data and refuses.
    """
    if top_n < 2:
        raise ValueError("top_n must be at least 2")
    totals = np.asarray(joint.sum(axis=1)).ravel()
    scaled = sparse.diags(1.0 / np.maximum(totals, 1e-12)) @ sparse.csr_matrix(joint)
    if use_gene_median:
        medians = _nonzero_median_per_gene(scaled)
        scaled = (scaled @ sparse.diags(1.0 / medians)).tocsr()
    else:
        # The ablation: ranking library-normalised expression directly. Kept so
        # the claim that this is dominated by housekeeping genes, and therefore
        # nearly identical across cells, is measured rather than asserted.
        scaled = scaled.tocsr()

    rows, columns, values = [], [], []
    for cell in range(scaled.shape[0]):
        start, stop = scaled.indptr[cell], scaled.indptr[cell + 1]
        if stop == start:
            continue
        data = scaled.data[start:stop]
        genes = scaled.indices[start:stop]
        keep = min(top_n, data.size)
        # Stable sort so ties break on gene order, making the encoding
        # reproducible rather than dependent on the sort implementation.
        order = np.argsort(-data, kind="stable")[:keep]
        rows.append(np.full(keep, cell, dtype=np.int64))
        columns.append(genes[order])
        values.append(1.0 - np.arange(keep, dtype=np.float64) / float(top_n))
    if not rows:
        raise ValueError("Every cell is empty; nothing to rank")
    return sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=scaled.shape,
    )


def pearson_residual_gene_variance(
    joint: Any, theta: float, chunk: int = 512
) -> np.ndarray:
    """Per-gene variance of analytic Pearson residuals.

    The residual of a count against the depth-only expectation
    ``mu = cell_total * gene_share`` under a negative binomial with dispersion
    ``theta``.  Its point is that a zero in a deep cell and a zero in a shallow
    cell receive different residuals, which is the part of the depth effect
    that library-size division cannot reach and that subsampling to a common
    total does not remove.

    Computed in gene blocks because the residual matrix is dense -- every entry,
    including the zeros, has a nonzero residual -- and materialising all of it
    for tens of thousands of genes is unnecessary when only the most variable
    are kept.
    """
    totals = np.asarray(joint.sum(axis=1), dtype=np.float64).ravel()
    gene_totals = np.asarray(joint.sum(axis=0), dtype=np.float64).ravel()
    grand = float(gene_totals.sum())
    if grand <= 0:
        raise ValueError("Joint matrix carries no counts")
    share = gene_totals / grand
    limit = np.sqrt(joint.shape[0])
    columns = joint.tocsc()
    variances = np.zeros(joint.shape[1], dtype=np.float64)
    for start in range(0, joint.shape[1], chunk):
        stop = min(start + chunk, joint.shape[1])
        expected = np.outer(totals, share[start:stop])
        scale = np.sqrt(expected + expected * expected / theta)
        block = columns[:, start:stop].toarray()
        residual = (block - expected) / np.maximum(scale, 1e-12)
        np.clip(residual, -limit, limit, out=residual)
        variances[start:stop] = residual.var(axis=0)
    return variances


def pearson_residual_block(joint: Any, genes: np.ndarray, theta: float) -> np.ndarray:
    """Dense analytic Pearson residuals for the chosen genes."""
    totals = np.asarray(joint.sum(axis=1), dtype=np.float64).ravel()
    gene_totals = np.asarray(joint.sum(axis=0), dtype=np.float64).ravel()
    share = gene_totals / float(gene_totals.sum())
    limit = np.sqrt(joint.shape[0])
    block = joint.tocsc()[:, genes].toarray()
    expected = np.outer(totals, share[genes])
    scale = np.sqrt(expected + expected * expected / theta)
    residual = (block - expected) / np.maximum(scale, 1e-12)
    np.clip(residual, -limit, limit, out=residual)
    return residual.astype(np.float32)


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Representation:
    """Joint coordinates plus a record of how they were produced."""

    source: NDArray[np.floating]
    target: NDArray[np.floating]
    selected_genes: list[Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class CostMatrix:
    """A cost matrix, the scale it was divided by, and its provenance.

    Both halves are returned because a cost matrix on its own is unreadable:
    the same numbers mean different things at different scales, and every
    downstream calibration is stated in units of this ``scale``.
    """

    cost: NDArray[np.float64]
    scale: float
    representation: Representation

    @property
    def provenance(self) -> dict[str, Any]:
        return self.representation.provenance


@dataclass(frozen=True)
class Preprocessing:
    """The whole path from counts to a cost matrix, as one recordable object.

    Parameters
    ----------
    equalise_depth:
        Subsample every cell to one shared read depth before anything else sees
        the counts.  Carried in the selected configuration because it is cheap,
        decoupled and preserves integer counts -- not because the screen
        attributed an effect to it.
    equalise_quantile, equalise_target:
        The shared depth, as a quantile of the pooled depth or as an explicit
        count.  Prefer the explicit count when reproducing a run.
    normalisation:
        One of :data:`NORMALISATIONS`.  ``precomputed`` passes the matrices
        through untouched, which is how an externally computed representation
        (scanpy's Pearson residuals, R's SCTransform) enters without this
        package acquiring those dependencies.
    rank_top_n:
        Genes kept per cell under a rank normalisation.  See
        :func:`rank_value_encode` for why this has a hard upper bound set by the
        data.
    residual_theta:
        Negative-binomial dispersion for ``pearson_residuals``.
    minimum_detection_rate:
        Drop genes detected in fewer than this fraction of cells.  Orthogonal
        to the transform, and composes with any of them: a gene detected in
        very few cells carries mostly its own zero pattern, which is what depth
        writes into the data.
    n_hvg, n_pcs:
        Genes kept by joint variance, and principal components retained.
    cost:
        ``squared_euclidean`` or ``cosine``.  This is the choice the screen
        separated on; see the module docstring.
    scale:
        Divide the cost by the median sampled cross-side pair distance, or not
        at all.  ``none`` leaves the cost in the representation's own units,
        which makes a calibrated ``rejection_cost`` incomparable across runs.
    label_stem:
        Overrides the stem of :meth:`label`.  Required for ``precomputed``,
        where only the caller knows what produced the matrix.
    """

    equalise_depth: bool = False
    equalise_quantile: float = 0.10
    equalise_target: int | None = None
    normalisation: Normalisation = "log_cpm"
    rank_top_n: int = 256
    residual_theta: float = 100.0
    minimum_detection_rate: float = 0.0
    n_hvg: int = 2000
    n_pcs: int = 30
    cost: Cost = "squared_euclidean"
    scale: Scale = "median_sampled_pair"
    scale_sample_pairs: int = 1_000_000
    label_stem: str | None = None

    def __post_init__(self) -> None:
        if self.normalisation not in NORMALISATIONS:
            raise ValueError(
                f"normalisation must be one of {NORMALISATIONS}; "
                f"got {self.normalisation!r}"
            )
        if self.cost not in COSTS:
            raise ValueError(f"cost must be one of {COSTS}; got {self.cost!r}")
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}; got {self.scale!r}")
        if self.rank_top_n < 2:
            raise ValueError("rank_top_n must be at least 2")
        if not 0.0 <= self.minimum_detection_rate <= 1.0:
            raise ValueError("minimum_detection_rate must lie in [0, 1]")
        if self.n_hvg < 2 or self.n_pcs < 1:
            raise ValueError("n_hvg must be >= 2 and n_pcs >= 1")
        if self.equalise_target is not None and self.equalise_target < 1:
            raise ValueError("equalise_target must be a positive read count")
        if not 0.0 < self.equalise_quantile < 1.0:
            raise ValueError("equalise_quantile must lie in (0, 1)")
        if self.normalisation == "precomputed" and not (self.label_stem or ""):
            raise ValueError(
                "normalisation='precomputed' needs a non-empty label_stem: "
                "only the caller knows what produced the matrix, and an "
                "unnamed configuration cannot be reported"
            )

    # ---- naming and recording -------------------------------------------

    def label(self) -> str:
        """The short name this configuration is reported under.

        ``rank_value`` at 256 with equalisation and cosine gives
        ``rank256_ds_cos``, which is the name in the screen's tables -- so a
        result and the object that produced it cannot drift apart.
        """
        stem = self.label_stem or _LABEL_STEM[self.normalisation]
        if self.label_stem is None and self.normalisation.startswith("rank"):
            stem = f"{stem}{self.rank_top_n}"
        parts = [stem]
        if self.equalise_depth:
            parts.append("ds")
        if self.cost == "cosine":
            parts.append("cos")
        return "_".join(parts)

    @classmethod
    def from_label(cls, label: str, **overrides: Any) -> "Preprocessing":
        """Rebuild a configuration from the name it is reported under.

        ``rank256_ds_cos`` gives back rank encoding at 256 with equalisation and
        the cosine cost.  This exists so that a pipeline names the
        configuration **once** -- in one job script, as one string -- instead of
        setting three variables that can disagree.  The failure it removes is
        real and silent: a rank cut set while the transform was left at its
        default produces a run with an ignored cut and no name.

        The label carries the three axes the depth screen varied and nothing
        else.  ``n_hvg``, ``n_pcs``, ``residual_theta``,
        ``minimum_detection_rate`` and ``equalise_quantile`` come from the
        defaults unless passed in ``overrides``, so a label identifies a
        configuration only together with those -- which is why
        :meth:`as_dict` records all of them and not just the label.

        A stem this class does not own is treated as an external transform, so
        ``sct_ds_cos`` round-trips to ``precomputed`` named ``sct``. Anything
        the caller then supplies is its own business, which is the point.
        """
        if not label or not label.strip():
            raise ValueError("label must be a non-empty string")
        parts = label.strip().split("_")
        cost = "squared_euclidean"
        if parts[-1] == "cos":
            cost, parts = "cosine", parts[:-1]
        equalise = False
        if parts and parts[-1] == "ds":
            equalise, parts = True, parts[:-1]
        stem = "_".join(parts)
        if not stem:
            raise ValueError(f"label {label!r} names no transform")
        settings: dict[str, Any] = {"cost": cost, "equalise_depth": equalise}
        inverse = {value: key for key, value in _LABEL_STEM.items()}
        rank_stems = tuple(
            value for key, value in _LABEL_STEM.items() if key.startswith("rank")
        )
        for rank_stem in sorted(rank_stems, key=len, reverse=True):
            digits = stem[len(rank_stem):]
            if stem.startswith(rank_stem) and digits.isdigit():
                settings["normalisation"] = inverse[rank_stem]
                settings["rank_top_n"] = int(digits)
                break
        else:
            if stem in inverse:
                settings["normalisation"] = inverse[stem]
            else:
                settings["normalisation"] = "precomputed"
                settings["label_stem"] = stem
        settings.update(overrides)
        configuration = cls(**settings)
        if configuration.label() != label.strip():
            # A label that does not survive the round trip would let a job
            # script ask for one configuration and record another.
            raise ValueError(
                f"label {label!r} does not round-trip; it rebuilt as "
                f"{configuration.label()!r}"
            )
        return configuration

    def as_dict(self) -> dict[str, Any]:
        """The configuration as plain JSON-safe values, for the run record.

        Fields that do not apply are recorded as ``None`` rather than omitted,
        so a table built from many runs has one column set.
        """
        rank = self.normalisation.startswith("rank")
        return {
            "label": self.label(),
            "equalise_depth": bool(self.equalise_depth),
            "equalise_quantile": (
                self.equalise_quantile
                if self.equalise_depth and self.equalise_target is None else None
            ),
            "equalise_target": (
                self.equalise_target if self.equalise_depth else None
            ),
            "normalisation": self.normalisation,
            "rank_top_n": self.rank_top_n if rank else None,
            "residual_theta": (
                self.residual_theta
                if self.normalisation == "pearson_residuals" else None
            ),
            "minimum_detection_rate": self.minimum_detection_rate,
            "n_hvg": self.n_hvg,
            "n_pcs": self.n_pcs,
            "cost": self.cost,
            "scale": self.scale,
        }

    # ---- the steps ------------------------------------------------------

    def equalise(
        self,
        source_counts: ArrayLike,
        target_counts: ArrayLike,
        *,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], dict[str, Any]]:
        """Apply read equalisation to both sides against one shared target.

        One target across both sides, not one per side: a per-side target would
        leave the two sides of a pair at different depths, which is the
        difference the correction exists to remove.
        """
        source = np.asarray(source_counts)
        target = np.asarray(target_counts)
        if not self.equalise_depth:
            return source, target, {"equalise_depth": False}
        pooled = np.concatenate([np.asarray(source).sum(axis=1),
                                 np.asarray(target).sum(axis=1)])
        shared = (
            int(self.equalise_target) if self.equalise_target is not None
            else int(np.quantile(pooled, self.equalise_quantile))
        )
        equalised_source = equalise_depth(source, rng=rng, target=shared)
        equalised_target = equalise_depth(target, rng=rng, target=shared)
        untouched = int((pooled <= shared).sum())
        return equalised_source, equalised_target, {
            "equalise_depth": True,
            "equalise_target_depth": shared,
            "equalise_target_was_explicit": self.equalise_target is not None,
            "cells_already_at_or_below_target": untouched,
        }

    def representation(
        self,
        source_counts: ArrayLike,
        target_counts: ArrayLike,
        *,
        seed: int,
        source_genes: Sequence[Any] | None = None,
        target_genes: Sequence[Any] | None = None,
        extra_provenance: dict[str, Any] | None = None,
    ) -> Representation:
        """Joint representation of both sides, in one shared coordinate system.

        Every step that could see one side alone is deliberately taken on the
        two sides stacked: the gene-median division, the variance ranking, the
        centring and scaling, and the PCA.  Taken per side, each of them would
        normalise away the cross-side difference the gate exists to detect.

        ``source_genes``/``target_genes`` name the columns.  Supplied, the two
        matrices are intersected on them, first occurrence winning, in sorted
        order.  Omitted, the matrices must already share a column space.

        ``precomputed`` input is kept dense throughout.  An externally computed
        representation -- Pearson residuals, SCTransform -- has a nonzero entry
        almost everywhere, so a sparse container would cost memory and change
        the gene variances: ``var(axis=0)`` on a dense float32 block and
        ``E[x^2] - E[x]^2`` over a float64 sparse one agree in exact arithmetic
        and not in floating point, and the second one decides which genes are
        kept.  Keeping the container the caller chose is what makes an external
        arm comparable with the arm it is being compared against.
        """
        dense_path = self.normalisation == "precomputed" and not sparse.issparse(
            source_counts
        )
        if dense_path:
            source_x = np.asarray(source_counts)
            target_x = np.asarray(target_counts)
        else:
            source_x = sparse.csr_matrix(source_counts, dtype=np.float64)
            target_x = sparse.csr_matrix(target_counts, dtype=np.float64)
        if (source_genes is None) != (target_genes is None):
            raise ValueError("supply both gene name sequences, or neither")
        if source_genes is None:
            if source_x.shape[1] != target_x.shape[1]:
                raise ValueError(
                    f"without gene names the two sides must share a column "
                    f"space; got {source_x.shape[1]} and {target_x.shape[1]}"
                )
            common: list[Any] = list(range(source_x.shape[1]))
        else:
            source_first: dict[str, int] = {}
            target_first: dict[str, int] = {}
            for index, key in enumerate(source_genes):
                source_first.setdefault(str(key), index)
            for index, key in enumerate(target_genes):
                target_first.setdefault(str(key), index)
            common = sorted(set(source_first) & set(target_first))
            if len(common) < 2:
                raise ValueError("Fewer than two common genes")
            source_x = source_x[:, [source_first[key] for key in common]]
            target_x = target_x[:, [target_first[key] for key in common]]

        detection_note = "no detection-rate filter"
        if self.minimum_detection_rate > 0.0:
            counted = _stack(source_x, target_x)
            detected = (np.asarray((counted > 0).sum(axis=0)).ravel()
                        / counted.shape[0])
            keep = detected >= self.minimum_detection_rate
            if int(keep.sum()) < 2:
                raise ValueError(
                    f"Detection rate >= {self.minimum_detection_rate} leaves "
                    f"{int(keep.sum())} genes"
                )
            source_x = source_x[:, keep]
            target_x = target_x[:, keep]
            common = [gene for gene, flag in zip(common, keep) if flag]
            detection_note = (
                f"genes detected in >= {self.minimum_detection_rate:.3f} of "
                f"cells; {int(keep.sum())} of {len(keep)} kept"
            )

        source_n = source_x.shape[0]
        limit = min(self.n_hvg, len(common))
        if self.normalisation == "pearson_residuals":
            counts = sparse.vstack([source_x, target_x], format="csr")
            variances = pearson_residual_gene_variance(counts, self.residual_theta)
            selected = np.argsort(-variances, kind="stable")[:limit]
            dense = pearson_residual_block(counts, selected, self.residual_theta)
            transform = (
                f"joint analytic Pearson residuals, theta="
                f"{self.residual_theta:g}, genes ranked by residual variance"
            )
            hvg_note = "top residual variance"
        elif self.normalisation in ("rank_value", "rank_no_median"):
            joint = rank_value_encode(
                sparse.vstack([source_x, target_x], format="csr"),
                self.rank_top_n,
                use_gene_median=self.normalisation == "rank_value",
            )
            selected = _top_variance(joint, limit)
            dense = joint[:, selected].toarray().astype(np.float32)
            transform = (
                f"joint rank encoding, top {self.rank_top_n} genes per cell, "
                + ("expression divided by each gene's nonzero median over both "
                   "sides" if self.normalisation == "rank_value" else
                   "expression ranked directly, without the gene-median "
                   "division")
            )
            hvg_note = "top variance of the rank encoding"
        elif dense_path:
            joint = _stack(source_x, target_x)
            selected = _top_variance(joint, limit)
            dense = np.array(joint[:, selected], dtype=np.float32)
            transform = "values used as supplied, without retransformation"
            hvg_note = "top variance of the supplied values"
        else:
            source_x, transform = _scale_rows(source_x, self.normalisation)
            target_x, _ = _scale_rows(target_x, self.normalisation)
            joint = sparse.vstack([source_x, target_x], format="csr")
            selected = _top_variance(joint, limit)
            dense = joint[:, selected].toarray().astype(np.float32)
            hvg_note = "top variance after the row transformation"

        dense -= dense.mean(axis=0)
        std = dense.std(axis=0)
        dense /= np.where(std > 1e-8, std, 1.0)
        from sklearn.decomposition import PCA  # optional dependency

        components = min(self.n_pcs, dense.shape[0] - 1, dense.shape[1])
        coordinates = PCA(n_components=components,
                          random_state=seed).fit_transform(dense)
        source_pca = coordinates[:source_n]
        target_pca = coordinates[source_n:]
        if self.cost == "cosine":
            source_pca, target_pca = unit_rows(source_pca), unit_rows(target_pca)

        provenance = {
            **self.as_dict(),
            "transform": transform,
            "detection_filter": detection_note,
            "joint_hvg": hvg_note,
            "joint_pca": "centered and gene-scaled PCA",
            "common_gene_n": len(common),
            "hvg_n": int(len(selected)),
            "pca_components": int(components),
            "cosine_applied_before_scale": self.cost == "cosine",
            "seed": int(seed),
            **(extra_provenance or {}),
        }
        return Representation(
            source=source_pca,
            target=target_pca,
            selected_genes=[common[index] for index in selected],
            provenance=provenance,
        )

    def cost_matrix(
        self,
        source_counts: ArrayLike,
        target_counts: ArrayLike,
        *,
        seed: int,
        rng: np.random.Generator | None = None,
        source_genes: Sequence[Any] | None = None,
        target_genes: Sequence[Any] | None = None,
        equalise: bool = True,
    ) -> CostMatrix:
        """Counts in, cost matrix out: the whole configuration in one call.

        ``rng`` drives read equalisation and the scale estimate.  It defaults to
        ``default_rng(seed)``; pass one explicitly to keep a caller's existing
        stream, since the scale estimate draws from it twice and anything drawn
        afterwards depends on that.

        ``equalise=False`` skips the equalisation step while keeping it in the
        record, for the case where the counts arrive already equalised -- which
        is how the production pipeline works, since it equalises at the file
        level so that every later stage sees the same matrix.
        """
        generator = rng if rng is not None else np.random.default_rng(seed)
        if equalise:
            source_counts, target_counts, record = self.equalise(
                source_counts, target_counts, rng=generator
            )
        else:
            record = {
                "equalise_depth": bool(self.equalise_depth),
                "equalise_applied_here": False,
                "equalise_note": (
                    "counts were supplied already equalised; this step was "
                    "skipped, not omitted from the configuration"
                ),
            }
        representation = self.representation(
            source_counts, target_counts, seed=seed,
            source_genes=source_genes, target_genes=target_genes,
            extra_provenance=record,
        )
        return self.cost_from_representation(representation, rng=generator)

    def cost_from_representation(
        self,
        representation: Representation,
        *,
        rng: np.random.Generator,
    ) -> CostMatrix:
        """The cost and its scale, from coordinates that already exist.

        Separate from :meth:`cost_matrix` so a caller holding its own
        representation -- a UMAP overlay, a stored PCA, an external transform --
        reaches the identical cost construction instead of writing a third copy
        of it.
        """
        if self.scale == "median_sampled_pair":
            scale = median_pair_scale(
                representation.source, representation.target,
                rng=rng, pairs=self.scale_sample_pairs,
            )
        else:
            scale = 1.0
        cost = squared_euclidean(representation.source,
                                 representation.target) / scale
        return CostMatrix(cost=cost, scale=scale, representation=representation)


def _stack(source: Any, target: Any) -> Any:
    """Stack the two sides, keeping whichever container they arrived in."""
    if sparse.issparse(source) or sparse.issparse(target):
        return sparse.vstack([source, target], format="csr")
    return np.vstack([source, target])


def _top_variance(joint: Any, limit: int) -> NDArray[np.integer]:
    """Indices of the highest-variance columns.

    Sparse input uses ``E[x^2] - E[x]^2``, which avoids densifying; dense input
    uses ``var`` directly.  The split is deliberate rather than incidental: it
    is what each of the two existing code paths already did, and the two are
    not bit-identical, so unifying them would have moved the gene selection in
    exactly the arms that exist to be compared.
    """
    if sparse.issparse(joint):
        mean = np.asarray(joint.mean(axis=0)).ravel()
        mean2 = np.asarray(joint.multiply(joint).mean(axis=0)).ravel()
        variances = mean2 - mean * mean
    else:
        variances = np.asarray(joint).var(axis=0)
    return np.argsort(-variances, kind="stable")[:limit]


def _scale_rows(matrix: Any, normalisation: str) -> tuple[Any, str]:
    """Row-wise library scaling for the non-rank, non-residual transforms."""
    if normalisation == "precomputed":
        return matrix, "values used as supplied, without retransformation"
    if normalisation == "log1p":
        scaled = sparse.csr_matrix(matrix, copy=True)
        if scaled.data.size and np.min(scaled.data) < 0:
            return matrix, "values contain negatives; used as provided"
        scaled.data = np.log1p(scaled.data)
        return scaled, "values -> log1p"
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scaled = sparse.diags(1e4 / np.maximum(totals, 1.0)) @ sparse.csr_matrix(matrix)
    if normalisation == "cpm":
        return scaled, "raw counts -> library size 1e4"
    scaled = sparse.csr_matrix(scaled)
    scaled.data = np.log1p(scaled.data)
    return scaled, "raw counts -> library size 1e4 -> log1p"


__all__ = [
    "COSTS",
    "CostMatrix",
    "NORMALISATIONS",
    "Preprocessing",
    "Representation",
    "SCALES",
    "equalise_depth",
    "median_pair_scale",
    "pearson_residual_block",
    "pearson_residual_gene_variance",
    "rank_value_encode",
    "squared_euclidean",
    "unit_rows",
]
