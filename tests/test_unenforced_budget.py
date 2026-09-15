import unittest

import numpy as np

from confidenceot import ConfidenceOT
from confidenceot._cpu_uot import bounded_gate_counts, resolve_rejection_bounds


class UnenforcedBudgetTest(unittest.TestCase):
    """The budget is reported rather than imposed.

    Enforcing the cardinality floor makes the rejection rate equal the budget
    whenever the rejection cost wants to reject more, so the parameter becomes
    the answer instead of a bound on it.
    """

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        source = rng.normal(size=(40, 4))
        target = rng.normal(size=(40, 4))
        observed = np.sum((source[:, None] - target[None, :]) ** 2, axis=2)
        self.cost = observed / float(np.median(observed[observed > 0]))

    def fit(self, *, enforce: bool, rejection_cost: float = 0.05):
        return ConfidenceOT(
            backbone="uot", variant="exact", rejection_cost=rejection_cost,
            source_rejection_budget=0.85, target_rejection_budget=0.85,
            enforce_rejection_budget=enforce, device="cpu",
            warn_on_terminal=False,
        ).fit(self.cost)

    def test_an_all_rejected_gate_is_reachable(self) -> None:
        # A cost far below every pairwise cost means "reject everything", and
        # that verdict must be expressible rather than clipped to the floor.
        result = self.fit(enforce=False)
        self.assertEqual(result.source_rejection_rate, 1.0)
        self.assertFalse(result.source_gate.any())
        self.assertTrue(np.all(np.isfinite(result.coupling)))

    def test_the_floor_clips_the_same_fit_when_enforced(self) -> None:
        enforced = self.fit(enforce=True)
        self.assertLessEqual(enforced.source_rejection_rate, 0.85 + 1e-12)
        self.assertGreater(self.fit(enforce=False).source_rejection_rate,
                           enforced.source_rejection_rate)

    def test_bounds_report_which_end_bound(self) -> None:
        # Unenforced, the interval is (0, 1) and nothing can bind, so a gate
        # that rejects everything is reported as the data's verdict.
        unenforced = self.fit(enforce=False)
        self.assertEqual(unenforced.source_rejection_bounds, (0.0, 1.0))
        self.assertEqual(unenforced.source_bounds_active, (False, False))
        self.assertEqual(unenforced.source_rejection_rate, 1.0)
        # Enforced, the upper end binds and says so: the parameter, not the
        # data, set the answer at that end.
        enforced = self.fit(enforce=True)
        self.assertEqual(enforced.source_rejection_bounds, (0.0, 0.85))
        self.assertEqual(enforced.source_bounds_active, (False, True))
        # A cost above every pairwise cost retains everything, so neither end
        # binds and both flags stay quiet.
        quiet = self.fit(enforce=False, rejection_cost=50.0)
        self.assertEqual(quiet.source_bounds_active, (False, False))
        self.assertEqual(quiet.source_rejection_rate, 0.0)

    def test_legacy_budget_maps_onto_an_interval(self) -> None:
        def bounds(**kwargs):
            return resolve_rejection_bounds(None, **kwargs)

        # Unenforced, the budget states a reference rather than a constraint.
        self.assertEqual(bounds(legacy_budget=0.85), (0.0, 1.0))
        self.assertEqual(
            bounds(legacy_budget=0.85, enforce_legacy_budget=True), (0.0, 0.85)
        )
        # A budget of zero is an intent, not a cap: the source-only design sets
        # the target budget to zero to hold every target cell in the reference
        # distribution, and that must hold either way.
        self.assertEqual(bounds(legacy_budget=0.0), (0.0, 0.0))
        self.assertEqual(
            bounded_gate_counts(64, bounds(legacy_budget=0.0)), (64, 64)
        )
        self.assertEqual(
            bounded_gate_counts(64, bounds(legacy_budget=0.85))[0], 0
        )

    def test_counts_are_not_inflated_by_floating_point(self) -> None:
        # (1 - 0.85) * 1000 evaluates to 150.00000000000003, so a bare ceiling
        # returns 151 and retains one cell more than asked for. The CUDA path
        # shares this helper, so both devices agree.
        for n, expected in ((1000, 150), (200, 30), (2000, 300)):
            self.assertEqual(bounded_gate_counts(n, (0.0, 0.85))[0], expected)
        # Genuinely fractional counts still round outward.
        self.assertEqual(bounded_gate_counts(1001, (0.0, 0.85))[0], 151)
        self.assertEqual(bounded_gate_counts(7, (0.0, 0.5))[0], 4)

    def test_a_lower_bound_caps_retention(self) -> None:
        # The mirror of the budget: "reject at least 80%" is "retain at most
        # 20%", which turns the gate into a ranked selection of that size.
        self.assertEqual(bounded_gate_counts(1000, (0.80, 1.0)), (0, 200))
        self.assertEqual(bounded_gate_counts(1000, (0.80, 0.85)), (150, 200))
        self.assertEqual(bounded_gate_counts(1000, (0.0, 1.0)), (0, 1000))
        with self.assertRaisesRegex(ValueError, "empty"):
            resolve_rejection_bounds((0.9, 0.5))


if __name__ == "__main__":
    unittest.main()
