"""The four factorial cells the production API cannot name.

The benchmark varies four binary choices, which is sixteen cells. Twelve are
expressible as a ``Preprocessing`` label. The other four put library-size
normalisation and rank encoding on together, and those are one field in the
production code -- ``normalisation`` takes one value -- so they have no name.

They are also not a separate condition. Ranking is done within a cell, and
library-size normalisation to 1e4 followed by ``log1p`` is a strictly
increasing per-cell map, so it cannot reorder the genes of a cell. The rank
encoder then divides by the row sum itself, which is another per-cell positive
scale, and assigns values by rank position alone. The encoded matrix is
therefore identical, bit for bit, and so is everything downstream of it.

Asserting that here is what lets the benchmark run twelve configurations and
still report a sixteen-cell design honestly. It costs a second; running the
four as full conditions would cost a quarter of the grid to rediscover an
identity.
"""

import numpy as np
import pytest

from confidenceot import Preprocessing


def library_normalise(counts: np.ndarray) -> np.ndarray:
    """What the production code calls ``log_cpm``: to 1e4, then log1p."""
    totals = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(1e4 * counts / totals)


@pytest.mark.parametrize("cost", ["squared_euclidean", "cosine"])
@pytest.mark.parametrize("scale_genes", [True, False])
def test_rank_absorbs_library_normalisation(cost, scale_genes):
    rng = np.random.default_rng(4)
    source = rng.poisson(0.5, size=(80, 300)).astype(np.float64)
    target = rng.poisson(0.5, size=(90, 300)).astype(np.float64)
    configuration = Preprocessing(
        normalisation="rank_no_median", rank_top_n=64, n_hvg=150, n_pcs=6,
        scale_genes=scale_genes, cost=cost)

    plain = configuration.representation(source, target, seed=1)
    normalised = configuration.representation(
        library_normalise(source), library_normalise(target), seed=1)

    assert np.array_equal(np.asarray(plain.source),
                          np.asarray(normalised.source))
    assert np.array_equal(np.asarray(plain.target),
                          np.asarray(normalised.target))


def test_the_twelve_labels_round_trip():
    """A label that does not rebuild itself would silently run something else."""
    labels = (
        "raw raw_noscale raw_cos raw_noscale_cos "
        "logcpm logcpm_noscale logcpm_cos logcpm_noscale_cos "
        "ranknm256 ranknm256_noscale ranknm256_cos ranknm256_noscale_cos"
    ).split()
    assert len(labels) == 12
    for label in labels:
        assert Preprocessing.from_label(label).label() == label
