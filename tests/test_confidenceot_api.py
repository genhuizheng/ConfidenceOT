import unittest

import numpy as np

from confidenceot import (
    CUDAUnavailableError,
    ConfidenceOT,
    cuda_available,
    m4_exact,
    m4_reversible,
)
from confidenceot.cuda import fit_cuda
from cellot import fit_balanced_cost_matrix_gate, fit_uot_cost_matrix_gate


class ConfidenceOTAPITest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(10, 3))
        y = rng.normal(size=(11, 3))
        self.cost = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)

    def test_cpu_balanced_matches_reference_gates(self) -> None:
        expected = fit_balanced_cost_matrix_gate(
            self.cost,
            rejection_cost=0.5,
            epsilon=0.1,
            variant="exact",
            source_rejection_budget=0.15,
            target_rejection_budget=0.15,
            threshold=1e-3,
            max_iterations=2_000,
            max_outer_iterations=10,
        )
        actual = ConfidenceOT(
            backbone="balanced",
            variant="exact",
            device="cpu",
            max_iterations=2_000,
            max_outer_iterations=10,
        ).fit(self.cost)
        np.testing.assert_array_equal(actual.source_gate, expected.source_gate)
        np.testing.assert_array_equal(actual.target_gate, expected.target_gate)
        np.testing.assert_allclose(actual.coupling, expected.coupling)

    def test_cpu_uot_matches_reference_gates(self) -> None:
        expected = fit_uot_cost_matrix_gate(
            self.cost,
            rejection_cost=0.5,
            epsilon=0.1,
            lambda_a=1.0,
            lambda_b=1.0,
            variant="reversible",
            source_rejection_budget=0.15,
            target_rejection_budget=0.15,
            threshold=1e-3,
            max_iterations=2_000,
            max_outer_iterations=10,
        )
        actual = ConfidenceOT(
            backbone="uot",
            variant="reversible",
            device="cpu",
            max_iterations=2_000,
            max_outer_iterations=10,
        ).fit(self.cost)
        np.testing.assert_array_equal(actual.source_gate, expected.source_gate)
        np.testing.assert_array_equal(actual.target_gate, expected.target_gate)
        np.testing.assert_allclose(actual.coupling, expected.coupling)

    def test_convenience_functions(self) -> None:
        self.assertEqual(m4_exact(self.cost, device="cpu").variant, "exact")
        self.assertEqual(m4_reversible(self.cost, device="cpu").variant, "reversible")

    def test_explicit_cuda_fails_cleanly_when_unavailable(self) -> None:
        if cuda_available():
            self.skipTest("CUDA is available on this test host.")
        with self.assertRaises(CUDAUnavailableError):
            ConfidenceOT(device="cuda").fit(self.cost)

    def test_torch_kernel_matches_numpy_reference_on_cpu(self) -> None:
        common = dict(
            rejection_cost=0.5,
            epsilon=0.1,
            lambda_a=1.0,
            lambda_b=1.0,
            source_weights=None,
            target_weights=None,
            initial_source_gate=None,
            initial_target_gate=None,
            source_rejection_budget=0.15,
            target_rejection_budget=0.15,
            tau=0.0,
            threshold=1e-3,
            max_iterations=2_000,
            max_outer_iterations=10,
            dtype="float64",
            _torch_device="cpu",
        )
        for backbone in ("balanced", "uot"):
            for variant in ("exact", "reversible"):
                expected = ConfidenceOT(
                    backbone=backbone,
                    variant=variant,
                    device="cpu",
                    max_iterations=2_000,
                    max_outer_iterations=10,
                    warn_on_terminal=False,
                ).fit(self.cost)
                actual = fit_cuda(self.cost, backbone=backbone, variant=variant, **common)
                np.testing.assert_array_equal(actual.source_gate, expected.source_gate)
                np.testing.assert_array_equal(actual.target_gate, expected.target_gate)
                np.testing.assert_allclose(actual.coupling, expected.coupling, rtol=2e-9, atol=2e-11)


if __name__ == "__main__":
    unittest.main()
