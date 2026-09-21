"""The preprocessing configuration, and proof the refactor changed no number.

The values in :data:`REFERENCE` were captured from the pre-refactor code path --
``cancer_metastasis.common.prepare_joint_representation`` followed by the median
scale estimate and cost construction written out by hand at each call site --
before any of it moved into :mod:`confidenceot.preprocessing`.

That matters more than a usual regression test. Every configuration claim in
``cancer_metastasis/SEQUENCING_DEPTH_RESOLUTION.md`` is a measurement made
through the old path. If the new path computes anything different, those
measurements no longer describe the shipped code, and the table has to be
re-measured rather than re-read. So this pins the cost matrix itself, the scale
it was divided by, *and* the state the random generator is left in -- the
calibration draws its null subsamples from that same generator immediately
afterwards, so a changed state silently changes every calibrated cost.
"""

from __future__ import annotations

import hashlib
import unittest

import numpy as np
from scipy import sparse

from confidenceot import Preprocessing
from confidenceot.preprocessing import (
    equalise_depth,
    median_pair_scale,
    squared_euclidean,
    unit_rows,
)

# (normalisation, cost) -> scale, cost digest, cost sum, generator's next draw.
REFERENCE = {
    ("log_cpm", "squared_euclidean"):
        (32.7376251221, "231ce82a4194c19a", 1446.41503906),
    ("log_cpm", "cosine"):
        (2.7441825867, "9898d4148e4e58f1", 1361.22229004),
    ("rank_value", "squared_euclidean"):
        (23.9733467102, "7416bc54fe4c3227", 1525.6472168),
    ("rank_value", "cosine"):
        (2.5678529739, "0cb1b0943e0ce33c", 1303.99829102),
    ("rank_no_median", "squared_euclidean"):
        (26.8493061066, "76dec1f5ded63a7e", 1485.16552734),
    ("rank_no_median", "cosine"):
        (2.5853693485, "b94301bfc873deb8", 1357.75769043),
    ("pearson_residuals", "squared_euclidean"):
        (31.7141265869, "75977276d289ada1", 1686.28637695),
    ("pearson_residuals", "cosine"):
        (3.0601861477, "b450f68ff363b7d2", 1359.49536133),
}
REFERENCE_NEXT_DRAW = 378915
SELECTED_GENE_HEAD = {
    "log_cpm": ["G051", "G052", "G009", "G040", "G049"],
    "rank_value": ["G054", "G011", "G019", "G028", "G016"],
    "rank_no_median": ["G042", "G043", "G058", "G059", "G056"],
    "pearson_residuals": ["G051", "G039", "G046", "G049", "G052"],
}

N_GENES = 60
GENES = [f"G{index:03d}" for index in range(N_GENES)]


def fixture(seed: int = 7, n_source: int = 40, n_target: int = 35):
    """Counts with a deliberate depth gradient, identical on every run.

    The lognormal multiplier is the point: without depth spread the transforms
    are nearly indistinguishable and the fixture would pass whatever the cost
    did.
    """
    rng = np.random.default_rng(seed)

    def side(n: int) -> np.ndarray:
        counts = rng.poisson(rng.uniform(0.2, 3.0, size=N_GENES),
                             size=(n, N_GENES))
        depth = rng.lognormal(0.0, 0.6, size=n)[:, None]
        return np.rint(counts * depth).astype(np.int64).astype(np.float64)

    return side(n_source), side(n_target)


def digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(array, dtype=np.float64).round(10)).tobytes()
    ).hexdigest()[:16]


class PreprocessingEquivalenceTest(unittest.TestCase):
    def test_reproduces_the_pre_refactor_cost_matrix(self):
        for (normalisation, cost), expected in REFERENCE.items():
            with self.subTest(normalisation=normalisation, cost=cost):
                scale, cost_digest, cost_sum = expected
                source, target = fixture()
                configuration = Preprocessing(
                    normalisation=normalisation, cost=cost,
                    n_hvg=30, n_pcs=5, rank_top_n=8,
                )
                rng = np.random.default_rng(99)
                prepared = configuration.cost_matrix(
                    source, target, seed=11, rng=rng,
                    source_genes=GENES, target_genes=GENES,
                )
                self.assertAlmostEqual(prepared.scale, scale, places=7)
                self.assertEqual(digest(prepared.cost), cost_digest)
                self.assertAlmostEqual(float(prepared.cost.sum()), cost_sum,
                                       places=5)
                # The calibration subsamples from this same generator next.
                self.assertEqual(int(rng.integers(1_000_000)),
                                 REFERENCE_NEXT_DRAW)

    def test_reproduces_the_pre_refactor_gene_selection(self):
        for normalisation, head in SELECTED_GENE_HEAD.items():
            with self.subTest(normalisation=normalisation):
                source, target = fixture()
                configuration = Preprocessing(
                    normalisation=normalisation, n_hvg=30, n_pcs=5,
                    rank_top_n=8,
                )
                representation = configuration.representation(
                    source, target, seed=11,
                    source_genes=GENES, target_genes=GENES,
                )
                self.assertEqual(representation.selected_genes[:5], head)


class PreprocessingContractTest(unittest.TestCase):
    def test_label_matches_the_screen_table(self):
        self.assertEqual(Preprocessing().label(), "logcpm")
        self.assertEqual(Preprocessing(cost="cosine").label(), "logcpm_cos")
        self.assertEqual(
            Preprocessing(normalisation="rank_value", rank_top_n=256,
                          equalise_depth=True, cost="cosine").label(),
            "rank256_ds_cos",
        )
        self.assertEqual(
            Preprocessing(normalisation="pearson_residuals",
                          equalise_depth=True, cost="cosine").label(),
            "pearson_ds_cos",
        )
        # An externally computed representation is named by its caller, because
        # nothing else knows what produced it.
        self.assertEqual(
            Preprocessing(normalisation="precomputed", label_stem="sct",
                          equalise_depth=True, cost="cosine").label(),
            "sct_ds_cos",
        )

    def test_every_label_rebuilds_the_configuration_that_printed_it(self):
        for label in ("cpm", "logcpm", "log1p", "pearson", "rank256",
                      "ranknm256", "logcpm_cos", "pearson_ds_cos",
                      "rank256_ds_cos", "rank512", "sct_ds_cos",
                      "scanpy_pearson_ds"):
            with self.subTest(label=label):
                self.assertEqual(Preprocessing.from_label(label).label(), label)

    def test_from_label_recovers_the_three_screened_axes(self):
        configuration = Preprocessing.from_label("rank256_ds_cos")
        self.assertEqual(configuration.normalisation, "rank_value")
        self.assertEqual(configuration.rank_top_n, 256)
        self.assertTrue(configuration.equalise_depth)
        self.assertEqual(configuration.cost, "cosine")
        # An unknown stem is an external transform, named by the caller.
        external = Preprocessing.from_label("sct_ds_cos")
        self.assertEqual(external.normalisation, "precomputed")
        self.assertEqual(external.label_stem, "sct")

    def test_from_label_takes_the_axes_it_does_not_carry_as_overrides(self):
        configuration = Preprocessing.from_label("rank256_ds_cos", n_pcs=50,
                                                 equalise_target=3119)
        self.assertEqual(configuration.n_pcs, 50)
        self.assertEqual(configuration.equalise_target, 3119)
        self.assertEqual(configuration.label(), "rank256_ds_cos")

    def test_a_label_that_would_not_round_trip_is_refused(self):
        # 'rank_ds' omits the cut, so it would rebuild as 'rank256_ds' and a
        # job script would record a configuration it did not ask for.
        for label in ("", "   ", "_ds_cos", "cos", "rank_ds"):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    Preprocessing.from_label(label)

    def test_precomputed_without_a_name_is_refused(self):
        with self.assertRaises(ValueError):
            Preprocessing(normalisation="precomputed")

    def test_unknown_options_are_refused(self):
        for kwargs in ({"normalisation": "sctransform"}, {"cost": "euclidean"},
                        {"scale": "mean"}, {"rank_top_n": 1},
                        {"minimum_detection_rate": 1.5},
                        {"equalise_quantile": 0.0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    Preprocessing(**kwargs)

    def test_record_omits_inapplicable_fields_without_dropping_columns(self):
        record = Preprocessing(normalisation="rank_value",
                               equalise_depth=True).as_dict()
        self.assertEqual(record["rank_top_n"], 256)
        self.assertIsNone(record["residual_theta"])
        self.assertEqual(record["equalise_quantile"], 0.10)
        explicit = Preprocessing(equalise_depth=True,
                                 equalise_target=3119).as_dict()
        # With an explicit target the quantile did not decide anything, so
        # recording it would misreport the run.
        self.assertIsNone(explicit["equalise_quantile"])
        self.assertEqual(explicit["equalise_target"], 3119)
        self.assertEqual(set(record), set(explicit))


class PrimitiveTest(unittest.TestCase):
    def test_unit_rows_normalises_and_leaves_zero_rows_alone(self):
        rows = unit_rows(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(rows[0], [0.6, 0.8])
        np.testing.assert_array_equal(rows[1], [0.0, 0.0])
        self.assertEqual(rows.dtype, np.float32)
        self.assertEqual(unit_rows(np.array([[1, 2]])).dtype, np.float64)

    def test_cosine_cost_is_the_euclidean_cost_on_unit_rows(self):
        rng = np.random.default_rng(3)
        left = rng.normal(size=(6, 4))
        right = rng.normal(size=(5, 4))
        cosine = 1.0 - (unit_rows(left) @ unit_rows(right).T)
        np.testing.assert_allclose(
            squared_euclidean(unit_rows(left), unit_rows(right)),
            2.0 * cosine, atol=1e-12,
        )

    def test_equalise_depth_hits_the_target_and_spares_shallow_cells(self):
        counts = np.array([[10, 10, 10], [1, 1, 0], [50, 0, 50]])
        equalised = equalise_depth(counts, rng=np.random.default_rng(0),
                                   target=6)
        np.testing.assert_array_equal(equalised.sum(axis=1), [6, 2, 6])
        # Reads are moved, never invented.
        self.assertTrue(np.all(equalised <= counts))

    def test_equalise_depth_matches_the_production_stage_bit_for_bit(self):
        """The equalised objects already on disk were produced by this.

        ``cancer_metastasis/27_downsample_counts.py`` wrote the depth-equalised
        h5ads that the ovarian and prostate re-runs reuse rather than
        regenerate. Its ``downsample_matrix`` now delegates here, so this is
        the check that delegating did not silently invalidate those files.
        """
        def reference(matrix, target, rng):
            # downsample_matrix as it stood before it delegated.
            matrix = sparse.csr_matrix(matrix, dtype=np.int64)
            original = np.asarray(matrix.sum(axis=1), dtype=np.int64).ravel()
            untouched = original <= target
            rows = []
            for index in range(matrix.shape[0]):
                start, end = matrix.indptr[index], matrix.indptr[index + 1]
                counts = matrix.data[start:end]
                if untouched[index] or counts.size == 0:
                    rows.append(counts)
                    continue
                rows.append(rng.multivariate_hypergeometric(counts,
                                                            int(target)))
            reduced = sparse.csr_matrix(
                (np.concatenate(rows) if rows
                 else np.zeros(0, dtype=np.int64),
                 matrix.indices.copy(), matrix.indptr.copy()),
                shape=matrix.shape, dtype=np.int64)
            reduced.eliminate_zeros()
            return reduced, original, untouched

        rng = np.random.default_rng(11)
        counts = (rng.poisson(0.4, size=(50, 120))
                  * rng.integers(1, 40, size=(50, 1)))
        want, want_original, want_untouched = reference(
            counts, 80, np.random.default_rng(5))
        got, got_original, got_untouched = equalise_depth(
            sparse.csr_matrix(counts), rng=np.random.default_rng(5),
            target=80, return_untouched=True,
        )
        self.assertEqual((want != got).nnz, 0)
        np.testing.assert_array_equal(want_original, got_original)
        np.testing.assert_array_equal(want_untouched, got_untouched)

    def test_dense_and_sparse_input_equalise_identically(self):
        """Otherwise the simulation and the production stage are two operations.

        The draw is over each cell's nonzero genes. Drawing over the full gene
        vector instead is the same distribution -- a colour with zero balls
        contributes nothing -- but consumes the generator differently, so a
        dense path that did that would give the simulation different counts
        from the ones the real pipeline produces, and the screen's conclusion
        would not transfer.
        """
        rng = np.random.default_rng(21)
        counts = (rng.poisson(0.3, size=(30, 80))
                  * rng.integers(1, 30, size=(30, 1)))
        dense = equalise_depth(counts, rng=np.random.default_rng(7), target=50)
        from_sparse = equalise_depth(sparse.csr_matrix(counts),
                                     rng=np.random.default_rng(7), target=50)
        self.assertFalse(sparse.issparse(dense))
        self.assertTrue(sparse.issparse(from_sparse))
        np.testing.assert_array_equal(dense,
                                      np.asarray(from_sparse.todense()))

    def test_equalise_depth_leaves_no_explicit_zeros(self):
        # A gene that lost all its reads must not stay a stored zero: the rank
        # cut is audited against detected-gene counts, which read the stored
        # pattern.
        rng = np.random.default_rng(31)
        counts = rng.poisson(1.0, size=(20, 40)) * 20
        reduced = equalise_depth(sparse.csr_matrix(counts),
                                 rng=np.random.default_rng(2), target=30)
        self.assertEqual(reduced.nnz, int((reduced.toarray() != 0).sum()))

    def test_equalise_depth_refuses_a_target_of_zero(self):
        with self.assertRaises(ValueError):
            equalise_depth(np.zeros((3, 3), dtype=int),
                           rng=np.random.default_rng(0), quantile=0.5)

    def test_median_pair_scale_ignores_coincident_pairs(self):
        # Two identical points contribute distance zero; including them would
        # drag the scale toward zero and inflate every cost.
        points = np.array([[0.0], [0.0], [3.0]])
        scale = median_pair_scale(points, points,
                                  rng=np.random.default_rng(1), pairs=2000)
        self.assertGreater(scale, 0.0)

    def test_median_pair_scale_survives_all_coincident_points(self):
        points = np.zeros((4, 2))
        self.assertEqual(
            median_pair_scale(points, points, rng=np.random.default_rng(1),
                              pairs=10),
            1.0,
        )

    def test_median_pair_scale_draws_source_then_target(self):
        # The draw order is contractual: callers reuse the generator.
        left = np.arange(6.0).reshape(3, 2)
        right = np.arange(8.0).reshape(4, 2)
        rng = np.random.default_rng(5)
        median_pair_scale(left, right, rng=rng, pairs=7)
        reference = np.random.default_rng(5)
        reference.integers(3, size=7)
        reference.integers(4, size=7)
        self.assertEqual(int(rng.integers(10_000)),
                         int(reference.integers(10_000)))


class EqualisationInTheChainTest(unittest.TestCase):
    def test_both_sides_reach_one_shared_depth(self):
        source, target = fixture()
        configuration = Preprocessing(equalise_depth=True,
                                      equalise_quantile=0.10)
        rng = np.random.default_rng(4)
        equal_source, equal_target, record = configuration.equalise(
            source, target, rng=rng
        )
        shared = record["equalise_target_depth"]
        for side in (equal_source, equal_target):
            totals = side.sum(axis=1)
            self.assertTrue(np.all(totals <= shared))
        self.assertTrue(record["equalise_depth"])
        self.assertFalse(record["equalise_target_was_explicit"])

    def test_skipping_equalisation_still_records_the_configuration(self):
        source, target = fixture()
        configuration = Preprocessing(equalise_depth=True, n_hvg=20, n_pcs=4)
        prepared = configuration.cost_matrix(
            source, target, seed=2, rng=np.random.default_rng(2),
            source_genes=GENES, target_genes=GENES, equalise=False,
        )
        # The run equalised upstream, at the file level. The record has to say
        # the configuration includes it and that this call did not apply it,
        # or a reader cannot tell a skipped step from an omitted one.
        self.assertTrue(prepared.provenance["equalise_depth"])
        self.assertFalse(prepared.provenance["equalise_applied_here"])


class GeneAlignmentTest(unittest.TestCase):
    def test_sides_are_intersected_on_gene_names(self):
        rng = np.random.default_rng(8)
        source = rng.poisson(1.0, size=(20, 6)).astype(np.float64)
        target = rng.poisson(1.0, size=(18, 5)).astype(np.float64)
        configuration = Preprocessing(n_hvg=4, n_pcs=3)
        representation = configuration.representation(
            source, target, seed=1,
            source_genes=["A", "B", "C", "D", "E", "F"],
            target_genes=["F", "E", "D", "C", "Z"],
        )
        self.assertEqual(representation.provenance["common_gene_n"], 4)
        self.assertTrue(set(representation.selected_genes)
                        <= {"C", "D", "E", "F"})

    def test_mismatched_widths_without_names_are_refused(self):
        configuration = Preprocessing()
        with self.assertRaises(ValueError):
            configuration.representation(np.ones((4, 5)), np.ones((4, 6)),
                                         seed=1)

    def test_one_sided_gene_names_are_refused(self):
        configuration = Preprocessing()
        with self.assertRaises(ValueError):
            configuration.representation(
                np.ones((4, 2)), np.ones((4, 2)), seed=1,
                source_genes=["A", "B"],
            )


class PrecomputedRepresentationTest(unittest.TestCase):
    """An external transform must reach the identical tail, not a similar one."""

    def test_reproduces_the_hand_written_external_tail(self):
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(12)
        n_source, n_target, n_genes = 30, 25, 40
        supplied = rng.normal(size=(n_source + n_target, n_genes)).astype(
            np.float32)
        n_hvg, n_pcs, seed = 15, 4, 5

        # scripts/validate_depth_null_specificity.py's external_joint_pca, as
        # it stood before the move: dense variance, top-n, centre, gene-scale,
        # joint PCA.
        variances = supplied.var(axis=0)
        selected = np.argsort(-variances, kind="stable")[
            : min(n_hvg, supplied.shape[1])]
        dense = np.array(supplied[:, selected], dtype=np.float32)
        dense -= dense.mean(axis=0)
        std = dense.std(axis=0)
        dense /= np.where(std > 1e-8, std, 1.0)
        components = min(n_pcs, dense.shape[0] - 1, dense.shape[1])
        expected = PCA(n_components=components,
                       random_state=seed).fit_transform(dense)

        prepared = Preprocessing(
            normalisation="precomputed", label_stem="sct",
            n_hvg=n_hvg, n_pcs=n_pcs,
        ).representation(supplied[:n_source], supplied[n_source:], seed=seed)
        np.testing.assert_array_equal(prepared.source, expected[:n_source])
        np.testing.assert_array_equal(prepared.target, expected[n_source:])

    def test_dense_input_is_not_routed_through_a_sparse_container(self):
        # The two variance formulas are not bit-identical, and the gene
        # selection depends on which one ran. This pins the dispatch.
        rng = np.random.default_rng(13)
        supplied = rng.normal(size=(20, 30)).astype(np.float32)
        configuration = Preprocessing(normalisation="precomputed",
                                      label_stem="external", n_hvg=8, n_pcs=3)
        prepared = configuration.representation(
            supplied[:10], supplied[10:], seed=1,
        )
        self.assertEqual(prepared.provenance["joint_hvg"],
                         "top variance of the supplied values")


class FitCountsTest(unittest.TestCase):
    def test_fit_counts_returns_the_gate_and_the_cost_it_used(self):
        from confidenceot import ConfidenceOT

        source, target = fixture()
        configuration = Preprocessing(normalisation="rank_value",
                                      rank_top_n=8, cost="cosine",
                                      n_hvg=20, n_pcs=4)
        model = ConfidenceOT(variant="exact", rejection_cost=0.5,
                             device="cpu", warn_on_terminal=False)
        result, prepared = model.fit_counts(
            source, target, preprocessing=configuration, seed=3,
            source_genes=GENES, target_genes=GENES,
        )
        self.assertEqual(prepared.cost.shape, (len(source), len(target)))
        self.assertEqual(result.source_gate.shape, (len(source),))
        self.assertEqual(result.target_gate.shape, (len(target),))
        self.assertEqual(prepared.provenance["label"], "rank8_cos")
        # Same configuration, same seed, same cost matrix.
        again = configuration.cost_matrix(
            source, target, seed=3, source_genes=GENES, target_genes=GENES,
        )
        np.testing.assert_array_equal(prepared.cost, again.cost)

    def test_a_plain_dict_is_not_accepted_as_a_configuration(self):
        from confidenceot import ConfidenceOT

        source, target = fixture()
        with self.assertRaises(TypeError):
            ConfidenceOT(device="cpu").fit_counts(
                source, target, preprocessing={"cost": "cosine"}, seed=1,
            )


if __name__ == "__main__":
    unittest.main()
