"""The saved MOSTA couplings rebuild the fitted couplings, entry for entry.

mouse_embryo/mosta_coupling.py saves the dual potentials, gates, weights and
settings of each fit instead of the dense coupling.  These tests fit M4-E,
M4-R and balanced OT blockwise on the CPU on clouds where part of the target
is rejected, save them as 23_ does, and rebuild from the files alone: whole
couplings against ``materialize_coupling`` and the dense solver, subsets by
position, bin id and mask, multi-block rebuilds, conditional probabilities and
masses.

Runs with ``python -m unittest discover -s tests -p test_mosta_coupling.py``.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mouse_embryo"))

from confidenceot import ConfidenceOT, median_pair_scale  # noqa: E402
from confidenceot.preprocessing import squared_euclidean, unit_rows  # noqa: E402


def clouds(seed: int, n: int, m: int, dimension: int = 6):
    """test_blockwise's clouds: a quarter of the target far from every source point."""
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(n, dimension))
    target = rng.normal(size=(m, dimension))
    source[:, 0] += 3.0
    target[:, 0] += 3.0
    target[: m // 4, 0] -= 6.0
    return unit_rows(source).astype(np.float32), unit_rows(target).astype(np.float32)


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class SavedCouplingsRebuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mosta_coupling
        from confidenceot.blockwise import CoordinateCost

        cls.mosta_coupling = mosta_coupling
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        source, target = clouds(11, 90, 110)
        cls.source, cls.target = source, target
        scale = median_pair_scale(source, target, rng=np.random.default_rng(3))
        cls.cost = CoordinateCost(source, target, scale)
        common = dict(rejection_cost=0.6, epsilon=0.1, tolerance=1e-6, device="cpu", cuda_dtype="float64")
        cls.models = {
            "M4-E": ConfidenceOT(backbone="uot", variant="exact", **common),
            "M4-R": ConfidenceOT(backbone="uot", variant="reversible", **common),
            "balanced": ConfidenceOT(backbone="balanced", variant="exact", source_rejection_budget=0.0,
                                     target_rejection_budget=0.0, **common),
        }
        cls.fits = {method: model.fit_blockwise(cls.cost, block_rows=17) for method, model in cls.models.items()}
        representation = root / "representations" / "pair" / "representation.npz"
        representation.parent.mkdir(parents=True)
        np.savez(representation, source=source, target=target)
        cls.pair_dir = root / "pairs" / "pair"
        cls.pair_dir.mkdir(parents=True)
        cls.source_ids = np.array([f"s{i}" for i in range(len(source))])
        cls.target_ids = np.array([f"t{j}" for j in range(len(target))])
        mosta_coupling.save_coupling(
            cls.pair_dir, cls.fits, cls.models, pair_id="pair",
            representation_path="representations/pair/representation.npz",
            representation_sha256=mosta_coupling.sha256(representation),
            representation_shapes=(source.shape, target.shape),
            source_bin_ids=cls.source_ids, target_bin_ids=cls.target_ids, cost_scale=scale, block_rows=17,
        )
        cls.coupling = mosta_coupling.PairCoupling(cls.pair_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_case_rejects_part_of_the_target(self) -> None:
        gate = self.fits["M4-E"].target_gate
        self.assertGreater(int((~gate).sum()), 0)
        self.assertGreater(int(gate.sum()), 0)

    def test_whole_coupling_equals_the_fit(self) -> None:
        from confidenceot.blockwise import materialize_coupling

        for method, fit in self.fits.items():
            np.testing.assert_allclose(self.coupling.mass(method), materialize_coupling(self.cost, fit),
                                       rtol=1e-12, atol=0, err_msg=method)

    def test_whole_coupling_equals_the_dense_solver(self) -> None:
        from confidenceot.cuda import fit_cuda

        dense_cost = squared_euclidean(self.source.astype(np.float64), self.target.astype(np.float64)) / self.cost.scale
        for method, model in self.models.items():
            dense = fit_cuda(dense_cost, dtype="float64", _torch_device="cpu", **model._solver_kwargs())
            np.testing.assert_array_equal(dense.target_gate, self.fits[method].target_gate)
            np.testing.assert_allclose(self.coupling.mass(method), dense.coupling, rtol=1e-9, atol=0, err_msg=method)

    def test_subsets_by_position_id_and_mask(self) -> None:
        rows = np.array([3, 0, 41, 89, 7])
        columns = np.array([5, 109, 2, 60])
        for method in self.fits:
            whole = self.coupling.mass(method)
            by_position = self.coupling.mass(method, rows, columns)
            np.testing.assert_allclose(by_position, whole[np.ix_(rows, columns)], rtol=1e-12, atol=0)
            np.testing.assert_array_equal(self.coupling.mass(method, self.source_ids[rows], self.target_ids[columns]),
                                          by_position)
            mask_rows = np.zeros(len(self.source_ids), dtype=bool)
            mask_rows[rows] = True
            mask_columns = np.zeros(len(self.target_ids), dtype=bool)
            mask_columns[columns] = True
            np.testing.assert_allclose(self.coupling.mass(method, mask_rows, mask_columns),
                                       whole[np.ix_(np.sort(rows), np.sort(columns))], rtol=1e-12, atol=0)
            np.testing.assert_allclose(self.coupling.mass(method, block_rows=7), whole, rtol=1e-12, atol=0)

    def test_conditional_probabilities_use_every_partner(self) -> None:
        rows = np.array([1, 2, 50])
        columns = np.array([0, 30, 99])
        fit = self.fits["M4-E"]
        for support in ("all", "active"):
            whole = self.coupling.mass("M4-E")
            if support == "active":
                whole = whole * (fit.source_gate[:, None] & fit.target_gate[None, :])
            with np.errstate(divide="ignore", invalid="ignore"):
                given_source = whole / whole.sum(axis=1, keepdims=True)
                given_target = whole / whole.sum(axis=0, keepdims=True)
            np.testing.assert_allclose(
                self.coupling.mass("M4-E", rows, columns, support=support, normalise="source"),
                given_source[np.ix_(rows, columns)], rtol=1e-12, atol=0)
            np.testing.assert_allclose(
                self.coupling.mass("M4-E", rows, columns, support=support, normalise="target"),
                given_target[np.ix_(rows, columns)], rtol=1e-12, atol=0)
        retained = np.flatnonzero(fit.source_gate)
        rows_sum = self.coupling.mass("M4-E", retained, None, support="active", normalise="source").sum(axis=1)
        np.testing.assert_allclose(rows_sum[np.isfinite(rows_sum)], 1.0, rtol=0, atol=1e-12)

    def test_active_support_is_zero_off_the_gates(self) -> None:
        fit = self.fits["M4-E"]
        active = self.coupling.mass("M4-E", support="active")
        self.assertTrue(np.all(active[~fit.source_gate] == 0))
        self.assertTrue(np.all(active[:, ~fit.target_gate] == 0))
        self.assertGreater(float(active.sum()), 0.0)

    def test_marginals_match_the_reductions(self) -> None:
        from confidenceot.blockwise import transport_reductions

        fit = self.fits["M4-E"]
        n, m = self.cost.shape
        summary = transport_reductions(
            self.cost, fit, source_groups=np.zeros(n, dtype=int), n_source_groups=1,
            target_groups=np.zeros(m, dtype=int), n_target_groups=1,
            source_points=np.zeros((n, 2)), target_points=np.zeros((m, 2)), support="active")
        rows, columns = self.coupling.marginals("M4-E", support="active")
        np.testing.assert_allclose(rows, summary["source_mass"], rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(columns, summary["target_mass"], rtol=1e-12, atol=1e-18)

    def test_a_changed_representation_is_refused(self) -> None:
        representation = Path(self.temporary.name) / "other.npz"
        np.savez(representation, source=self.source + 1, target=self.target)
        with self.assertRaises(ValueError):
            self.mosta_coupling.PairCoupling(self.pair_dir, representation=representation)

    def test_unknown_bin_id_is_refused(self) -> None:
        with self.assertRaises(KeyError):
            self.coupling.mass("M4-E", np.array(["nope"]), None)


if __name__ == "__main__":
    unittest.main()
