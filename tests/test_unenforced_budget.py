import unittest

import numpy as np

from confidenceot import ConfidenceOT
from confidenceot._cpu_uot import _coverage_floor


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

    def test_budget_diagnostic_reports_the_breach(self) -> None:
        unenforced = self.fit(enforce=False)
        self.assertFalse(unenforced.budget_enforced)
        self.assertTrue(unenforced.source_budget_exceeded)
        self.assertEqual(unenforced.source_rejection_budget, 0.85)
        # A cost above every pairwise cost retains everything, so nothing is
        # breached and the diagnostic must stay quiet.
        quiet = self.fit(enforce=False, rejection_cost=50.0)
        self.assertFalse(quiet.source_budget_exceeded)
        self.assertEqual(quiet.source_rejection_rate, 0.0)

    def test_zero_budget_stays_binding_without_enforcement(self) -> None:
        # The source-only cancer design sets the target budget to zero to hold
        # every target cell in the reference distribution, so that must survive
        # the default change.
        self.assertEqual(_coverage_floor(64, 0.0, enforce=False), 64)
        self.assertEqual(_coverage_floor(64, 0.85, enforce=False), 0)

    def test_floor_is_not_inflated_by_floating_point(self) -> None:
        # (1 - 0.85) * 1000 evaluates to 150.00000000000003, so a bare ceiling
        # returns 151 and retains one cell more than asked for. The CUDA path
        # shares this helper, so both devices agree.
        self.assertEqual(_coverage_floor(1000, 0.85), 150)
        self.assertEqual(_coverage_floor(200, 0.85), 30)
        self.assertEqual(_coverage_floor(2000, 0.85), 300)
        # Genuinely fractional floors still round up.
        self.assertEqual(_coverage_floor(1001, 0.85), 151)
        self.assertEqual(_coverage_floor(7, 0.5), 4)


if __name__ == "__main__":
    unittest.main()
