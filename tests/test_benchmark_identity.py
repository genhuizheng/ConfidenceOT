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

Runs under pytest, and also as a plain script:

    python tests/test_benchmark_identity.py

The second form exists because this file has to run on TACC, where the
benchmark's conda environment has no pytest. A claim about the design that
can only be checked where the grid does not run is not much of a check, so
the parametrisation is an explicit tuple and the assertions name their cell.
"""

import numpy as np

from confidenceot import Preprocessing

# The four unnameable cells: both geometries, gene scaling on and off.
CELLS = tuple((cost, scale_genes)
              for cost in ("squared_euclidean", "cosine")
              for scale_genes in (True, False))

LABELS = (
    "raw raw_noscale raw_cos raw_noscale_cos "
    "logcpm logcpm_noscale logcpm_cos logcpm_noscale_cos "
    "ranknm256 ranknm256_noscale ranknm256_cos ranknm256_noscale_cos"
).split()


def library_normalise(counts: np.ndarray) -> np.ndarray:
    """What the production code calls ``log_cpm``: to 1e4, then log1p."""
    totals = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(1e4 * counts / totals)


def check_one_cell(cost: str, scale_genes: bool) -> None:
    rng = np.random.default_rng(4)
    source = rng.poisson(0.5, size=(80, 300)).astype(np.float64)
    target = rng.poisson(0.5, size=(90, 300)).astype(np.float64)
    configuration = Preprocessing(
        normalisation="rank_no_median", rank_top_n=64, n_hvg=150, n_pcs=6,
        scale_genes=scale_genes, cost=cost)

    plain = configuration.representation(source, target, seed=1)
    normalised = configuration.representation(
        library_normalise(source), library_normalise(target), seed=1)

    where = f"cost={cost} scale_genes={scale_genes}"
    for side in ("source", "target"):
        before = np.asarray(getattr(plain, side))
        after = np.asarray(getattr(normalised, side))
        assert np.array_equal(before, after), (
            f"{where}: the {side} representation changed under library-size "
            f"normalisation, so rank encoding is not absorbing it; largest "
            f"difference {np.abs(before - after).max():.3e}")


def test_rank_absorbs_library_normalisation():
    for cost, scale_genes in CELLS:
        check_one_cell(cost, scale_genes)


def test_the_twelve_labels_round_trip():
    """A label that does not rebuild itself would silently run something else."""
    assert len(LABELS) == 12
    for label in LABELS:
        rebuilt = Preprocessing.from_label(label).label()
        assert rebuilt == label, f"{label} rebuilt itself as {rebuilt}"


if __name__ == "__main__":
    import sys
    import traceback

    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            print(f"ok    {name}")
    print(f"\n{failures} failed")
    sys.exit(1 if failures else 0)
