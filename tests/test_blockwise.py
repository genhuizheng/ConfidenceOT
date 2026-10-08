"""The blockwise solver against the dense torch solver it replaces.

``confidenceot.blockwise`` claims to be the dense solver in
``confidenceot.cuda`` carried out on row blocks, not a different method.  These
tests hold it to that on the CPU, where both run the same torch code: the
gates, decision costs, scores, iteration counts, convergence flags and
objective must agree, and so must every summary rebuilt from the potentials
against the same summary taken from the dense coupling.

Every case uses several blocks, so the column reductions really are
accumulated across blocks.  float64 isolates the algorithm from rounding and
is held to a tight tolerance; float32 is what production runs and is held to
the tolerance its rounding allows.

Runs with ``python -m unittest tests.test_blockwise``.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # pragma: no cover - the CPU package has no torch dependency
    HAVE_TORCH = False

from confidenceot import ConfidenceOT
from confidenceot.preprocessing import squared_euclidean, unit_rows


def clouds(seed: int, n: int, m: int, dimension: int = 6, shift: float = 0.0, dtype=np.float64):
    """Two point clouds on the unit sphere.

    Both sit around one pole; with ``shift`` a quarter of the target is moved
    to the opposite pole, far from every source point, so a calibrated-looking
    rejection cost rejects it and keeps the rest.
    """
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(n, dimension))
    target = rng.normal(size=(m, dimension))
    source[:, 0] += 3.0
    target[:, 0] += 3.0
    target[: m // 4, 0] -= 2.0 * shift
    return unit_rows(source).astype(dtype), unit_rows(target).astype(dtype)


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class BlockwiseMatchesDense(unittest.TestCase):
    def fit_both(self, model: ConfidenceOT, source, target, *, block_rows: int, dtype: str):
        from confidenceot.blockwise import CoordinateCost, fit_blockwise
        from confidenceot.cuda import fit_cuda

        rng = np.random.default_rng(3)
        from confidenceot import median_pair_scale

        scale = median_pair_scale(source, target, rng=rng)
        cost = CoordinateCost(source, target, scale)
        dense_cost = squared_euclidean(source, target) / scale
        kwargs = model._solver_kwargs()
        dense = fit_cuda(np.asarray(dense_cost, dtype=np.float64), dtype=dtype, _torch_device="cpu", **kwargs)
        blockwise = fit_blockwise(cost, dtype=dtype, block_rows=block_rows, _torch_device="cpu", **kwargs)
        return cost, dense, blockwise

    def assert_same_fit(self, dense, blockwise, *, rtol: float, atol: float) -> None:
        np.testing.assert_array_equal(blockwise.source_gate, dense.source_gate)
        np.testing.assert_array_equal(blockwise.target_gate, dense.target_gate)
        np.testing.assert_array_equal(blockwise.source_raw_gate, dense.source_raw_gate)
        np.testing.assert_array_equal(blockwise.target_raw_gate, dense.target_raw_gate)
        for side in ("source", "target"):
            dense_value = getattr(dense, f"{side}_confidence")
            block_value = getattr(blockwise, f"{side}_confidence")
            np.testing.assert_allclose(block_value.decision_cost, dense_value.decision_cost, rtol=rtol, atol=atol)
            np.testing.assert_allclose(getattr(blockwise, f"{side}_score"), getattr(dense, f"{side}_score"), rtol=rtol, atol=atol * 1e-3)
            self.assertEqual(block_value.cost_kind, dense_value.cost_kind)
        self.assertEqual(blockwise.inner_converged, dense.inner_converged)
        self.assertEqual(blockwise.outer_converged, dense.outer_converged)
        self.assertEqual(blockwise.cycle_detected, dense.cycle_detected)
        self.assertEqual(blockwise.cycle_length, dense.cycle_length)
        self.assertEqual(blockwise.n_outer_iterations, dense.n_outer_iterations)
        self.assertEqual(blockwise.total_inner_iterations, dense.total_inner_iterations)
        self.assertAlmostEqual(blockwise.objective, dense.objective, delta=max(abs(dense.objective), 1.0) * rtol)
        self.assertEqual(blockwise.source_bounds_active, dense.source_bounds_active)
        self.assertEqual(blockwise.target_bounds_active, dense.target_bounds_active)

    def test_uot_exact_float64(self) -> None:
        source, target = clouds(1, 90, 110, shift=3.0)
        model = ConfidenceOT(backbone="uot", variant="exact", rejection_cost=0.6, tolerance=1e-6)
        cost, dense, blockwise = self.fit_both(model, source, target, block_rows=17, dtype="float64")
        self.assertGreater(int((~dense.target_gate).sum()), 0, "the case must reject something")
        self.assertGreater(dense.n_outer_iterations, 1, "the gate must move at least once")
        self.assert_same_fit(dense, blockwise, rtol=1e-9, atol=1e-12)
        from confidenceot.blockwise import materialize_coupling

        np.testing.assert_allclose(materialize_coupling(cost, blockwise), dense.coupling, rtol=1e-9, atol=1e-16)

    def test_uot_reversible_float64(self) -> None:
        source, target = clouds(2, 80, 70, shift=3.0)
        model = ConfidenceOT(backbone="uot", variant="reversible", rejection_cost=0.7, tolerance=1e-6)
        cost, dense, blockwise = self.fit_both(model, source, target, block_rows=9, dtype="float64")
        self.assert_same_fit(dense, blockwise, rtol=1e-9, atol=1e-12)

    def test_balanced_all_retained_float64(self) -> None:
        """The balanced comparator: budget 0 means bounds (0, 0), nothing rejected."""
        source, target = clouds(3, 75, 95, shift=3.0)
        model = ConfidenceOT(backbone="balanced", variant="exact", rejection_cost=0.5,
                             source_rejection_budget=0.0, target_rejection_budget=0.0, tolerance=1e-6)
        cost, dense, blockwise = self.fit_both(model, source, target, block_rows=11, dtype="float64")
        self.assertTrue(dense.source_gate.all() and dense.target_gate.all())
        self.assert_same_fit(dense, blockwise, rtol=1e-9, atol=1e-12)
        from confidenceot.blockwise import materialize_coupling

        np.testing.assert_allclose(materialize_coupling(cost, blockwise), dense.coupling, rtol=1e-9, atol=1e-16)

    def test_uot_exact_float32_production_settings(self) -> None:
        """float32 and the production tolerance: rounding is the only difference."""
        source, target = clouds(4, 150, 130, shift=3.0, dtype=np.float32)
        model = ConfidenceOT(backbone="uot", variant="exact", rejection_cost=0.6, tolerance=1e-4)
        cost, dense, blockwise = self.fit_both(model, source, target, block_rows=23, dtype="float32")
        self.assert_same_fit(dense, blockwise, rtol=1e-4, atol=1e-5)

    def test_uot_reversible_float32_production_settings(self) -> None:
        source, target = clouds(5, 120, 140, shift=3.0, dtype=np.float32)
        model = ConfidenceOT(backbone="uot", variant="reversible", rejection_cost=0.7, tolerance=1e-4)
        cost, dense, blockwise = self.fit_both(model, source, target, block_rows=31, dtype="float32")
        self.assert_same_fit(dense, blockwise, rtol=1e-4, atol=1e-5)

    def test_one_block_and_many_blocks_agree(self) -> None:
        source, target = clouds(6, 60, 50, shift=3.0)
        model = ConfidenceOT(backbone="uot", variant="exact", rejection_cost=0.6, tolerance=1e-6)
        from confidenceot.blockwise import CoordinateCost, fit_blockwise
        from confidenceot import median_pair_scale

        cost = CoordinateCost(source, target, median_pair_scale(source, target, rng=np.random.default_rng(0)))
        one = fit_blockwise(cost, dtype="float64", block_rows=60, _torch_device="cpu", **model._solver_kwargs())
        many = fit_blockwise(cost, dtype="float64", block_rows=7, _torch_device="cpu", **model._solver_kwargs())
        self.assert_same_fit(one, many, rtol=1e-10, atol=1e-13)

    def test_api_method_matches_function(self) -> None:
        source, target = clouds(7, 40, 45, shift=3.0)
        from confidenceot.blockwise import CoordinateCost, fit_blockwise

        cost = CoordinateCost(source, target, 1.3)
        model = ConfidenceOT(backbone="uot", variant="exact", rejection_cost=0.6, device="cpu", cuda_dtype="float64")
        via_api = model.fit_blockwise(cost, block_rows=6)
        direct = fit_blockwise(cost, dtype="float64", block_rows=6, _torch_device="cpu", **model._solver_kwargs())
        np.testing.assert_array_equal(via_api.source_gate, direct.source_gate)
        np.testing.assert_allclose(via_api.log_u, direct.log_u)
        self.assertIsNone(via_api.coupling)

    def test_transport_reductions_match_the_dense_coupling(self) -> None:
        from confidenceot.blockwise import CoordinateCost, fit_blockwise, transport_reductions
        from confidenceot.cuda import fit_cuda

        source, target = clouds(8, 70, 85, shift=3.0)
        cost = CoordinateCost(source, target, 1.1)
        model = ConfidenceOT(backbone="uot", variant="exact", rejection_cost=0.6, tolerance=1e-6)
        kwargs = model._solver_kwargs()
        dense = fit_cuda(np.asarray(cost.dense(), dtype=np.float64), dtype="float64", _torch_device="cpu", **kwargs)
        blockwise = fit_blockwise(cost, dtype="float64", block_rows=13, _torch_device="cpu", **kwargs)
        rng = np.random.default_rng(9)
        source_groups = rng.integers(0, 4, size=70)
        target_groups = rng.integers(0, 5, size=85)
        source_points = rng.uniform(0, 300, size=(70, 2))
        target_points = rng.uniform(0, 300, size=(85, 2))
        strata_source = {"retained": dense.source_gate, "rejected": ~dense.source_gate}
        strata_target = {"retained": dense.target_gate}
        summary = transport_reductions(
            cost, blockwise, source_groups=source_groups, n_source_groups=4,
            target_groups=target_groups, n_target_groups=5,
            source_points=source_points, target_points=target_points, support="active",
            source_strata=strata_source, target_strata=strata_target, block_rows=8,
        )
        # The same summaries from the dense coupling, written out plainly.
        coupling = dense.coupling
        supported = coupling * dense.source_gate[:, None] * dense.target_gate[None, :]
        source_hot = np.eye(4)[source_groups]
        target_hot = np.eye(5)[target_groups]
        r = supported.sum(axis=1)
        q = supported.sum(axis=0)
        tol = dict(rtol=1e-8, atol=1e-14)
        np.testing.assert_allclose(summary["source_mass"], r, **tol)
        np.testing.assert_allclose(summary["target_mass"], q, **tol)
        np.testing.assert_allclose(summary["source_total_mass"], coupling.sum(axis=1), **tol)
        np.testing.assert_allclose(summary["target_total_mass"], coupling.sum(axis=0), **tol)
        np.testing.assert_allclose(summary["source_partner_groups"], supported @ target_hot, **tol)
        np.testing.assert_allclose(summary["target_partner_groups"], supported.T @ source_hot, **tol)
        np.testing.assert_allclose(summary["group_mass"], source_hot.T @ supported @ target_hot, **tol)
        np.testing.assert_allclose(summary["source_partner_points"], supported @ target_points, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(summary["source_partner_square"], supported @ np.sum(target_points ** 2, axis=1), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(summary["target_partner_points"], supported.T @ source_points, rtol=1e-8, atol=1e-10)
        a = np.full(70, 1 / 70)
        b = np.full(85, 1 / 85)
        row_weight = np.divide(a, r, out=np.zeros_like(r), where=r > 0)
        column_weight = np.divide(b, q, out=np.zeros_like(q), where=q > 0)
        for name, mask in strata_source.items():
            expected = supported.T @ (source_hot * (row_weight * mask)[:, None])
            np.testing.assert_allclose(summary[f"push_{name}"], expected, **tol)
        expected = supported @ (target_hot * (column_weight * strata_target["retained"])[:, None])
        np.testing.assert_allclose(summary["pull_retained"], expected, **tol)
        # Rejected rows carry no supported mass, so their push-forward is empty.
        np.testing.assert_allclose(summary["push_rejected"], 0.0, atol=1e-18)
        self.assertAlmostEqual(float(summary["cost_max"]), float(cost.dense().max()), places=10)

    def test_coordinate_cost_is_the_production_cost(self) -> None:
        source, target = clouds(10, 33, 21)
        from confidenceot.blockwise import CoordinateCost, _Geometry

        cost = CoordinateCost(source, target, 0.9)
        geometry = _Geometry(torch, cost, torch.device("cpu"), torch.float64)
        built = np.vstack([geometry.block(start, min(start + 5, 33)).numpy() for start in range(0, 33, 5)])
        expected = squared_euclidean(source.astype(np.float64), target.astype(np.float64)) / 0.9
        np.testing.assert_allclose(built, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
