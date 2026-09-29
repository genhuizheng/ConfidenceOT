"""The comparator rejection rule, on the cases that define it.

Runs under pytest and as a plain script, because the machine that runs the
grid has no pytest:

    python tests/test_comparator_rejection.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

from comparator_rejection import (  # noqa: E402
    FIXED_CUTOFF,
    areas,
    candidate_cutoffs,
    operating_point,
    oracle_cutoffs,
    score_side,
    traditional_ot_score,
    unmatchedness,
)


def test_unmatchedness_is_the_normalised_mass_deficit():
    # Three source cells, marginal 1/3 each. The first is fully transported,
    # the second half, the third not at all.
    third = 1.0 / 3.0
    coupling = np.array([[third, 0.0, 0.0],
                         [0.0, third / 2.0, 0.0],
                         [0.0, 0.0, 0.0]])
    marginal = np.full(3, third)
    u = unmatchedness(coupling, marginal, side="source")
    assert np.allclose(u, [0.0, 0.5, 1.0]), u
    # The target side reads the other margin of the same plan.
    v = unmatchedness(coupling, marginal, side="target")
    assert np.allclose(v, [0.0, 0.5, 1.0]), v


def test_excess_mass_floors_at_zero_rather_than_going_negative():
    coupling = np.array([[1.0, 0.0], [0.0, 0.0]])
    marginal = np.full(2, 0.5)
    u = unmatchedness(coupling, marginal, side="source")
    # The first cell received twice its own marginal: excess, not evidence of
    # rejection, so 0 rather than -1.
    assert np.allclose(u, [0.0, 1.0]), u


def test_partial_ot_score_is_binary_so_the_cutoff_adds_no_parameter():
    """An exact cardinality matching gives a cell all of its mass or none."""
    unit = 0.25
    coupling = np.zeros((4, 4))
    coupling[0, 1] = coupling[2, 3] = unit          # two of four matched
    marginal = np.full(4, unit)
    u = unmatchedness(coupling, marginal, side="source")
    assert set(np.unique(u)) == {0.0, 1.0}, u
    truth = np.array([False, True, False, True])
    scored = score_side(method="Partial OT", side="source", truth=truth,
                        score=u)
    assert scored.summary["score_is_binary"] is True
    # Every cutoff in (0,1) is the same rule, so the oracle cannot beat the
    # fixed one.
    assert scored.summary["oracle_f1_f1"] <= scored.summary["fixed_f1"] + 1e-12
    for cutoff in (0.1, 0.5, 0.9):
        assert np.array_equal(u > cutoff, u > FIXED_CUTOFF)


def test_traditional_ot_rejects_nothing_and_has_no_area():
    truth = np.array([True, False, False, False])
    scored = score_side(method="Traditional OT", side="source", truth=truth,
                        score=traditional_ot_score(4))
    assert scored.summary["score_kind"] == "mass_deficit"
    assert not scored.fixed_prediction.any()
    assert scored.summary["fixed_rejected_n"] == 0
    assert scored.summary["fixed_recall"] == 0.0
    # Nothing was predicted positive, so precision has no denominator.
    assert np.isnan(scored.summary["fixed_precision"])
    # A constant score orders nothing, which is exactly chance: 0.5 is the
    # measurement, and the flag beside it says why.
    assert scored.summary["roc_auc"] == 0.5
    assert scored.summary["score_is_constant"] is True
    # And it has no oracle: maximising F1 over its only two rules would
    # report recall 1.0 for rejecting every cell.
    assert scored.summary["oracle_f1_defined"] is False
    assert scored.summary["oracle_roc_defined"] is False
    assert scored.summary["oracle_undefined_because"] == "the score is constant"


def test_no_positives_leaves_the_rate_defined_and_the_rest_not():
    """The all_shared case: nothing should be rejected on either side."""
    truth = np.zeros(6, dtype=bool)
    u = np.array([0.0, 0.1, 0.6, 0.9, 0.2, 0.0])
    scored = score_side(method="Vanilla UOT", side="source", truth=truth,
                        score=u)
    assert scored.summary["fixed_rejected_n"] == 2          # 0.6 and 0.9
    assert np.isclose(scored.summary["fixed_false_rejection_rate"], 2 / 6)
    # Recall has no denominator: there was nothing to find.
    assert np.isnan(scored.summary["fixed_recall"])
    # Precision does have one -- two cells were called and both were wrong --
    # so it is 0, which is a result rather than a missing value.
    assert scored.summary["fixed_precision"] == 0.0
    # One class only, so an optimum would be a tie-break artefact.
    assert scored.summary["oracle_roc_defined"] is False
    assert scored.summary["oracle_f1_defined"] is False


def test_oracle_beats_the_fixed_cutoff_on_a_badly_scaled_score():
    """A score with the right ordering and the wrong scale."""
    truth = np.array([True, True, True, False, False, False])
    # Every value is below 0.5, so the fixed cutoff rejects nothing at all
    # while the ordering is perfect.
    u = np.array([0.40, 0.35, 0.30, 0.20, 0.10, 0.05])
    scored = score_side(method="Vanilla UOT", side="source", truth=truth,
                        score=u)
    assert scored.summary["fixed_rejected_n"] == 0
    assert np.isnan(scored.summary["fixed_precision"])
    assert scored.summary["oracle_f1_f1"] == 1.0
    assert scored.summary["oracle_roc_youden_j"] == 1.0
    # And the threshold-free area sees the ordering the fixed cutoff missed.
    assert scored.summary["roc_auc"] == 1.0
    assert scored.summary["pr_auc"] == 1.0


def test_confidenceot_keeps_its_native_gate_for_the_binary_decision():
    truth = np.array([True, True, False, False])
    # The gate names the cells it KEEPS. It is right about the first and
    # wrong about the second.
    gate = np.array([False, True, True, True])
    cost = np.array([0.9, 0.2, 0.3, 0.1])       # higher means more rejectable
    scored = score_side(method="ConfidenceOT", side="source", truth=truth,
                        score=cost, native_gate=gate,
                        score_kind="decision_cost")
    assert scored.summary["binary_decision_from"] == "native_gate"
    assert np.array_equal(scored.fixed_prediction, [True, False, False, False])
    assert scored.summary["fixed_precision"] == 1.0
    assert np.isclose(scored.summary["fixed_recall"], 0.5)
    # The score is its decision cost, not a mass deficit, and it is not
    # converted into one.
    assert scored.summary["score_kind"] == "decision_cost"


def test_the_areas_match_a_reference_implementation():
    rng = np.random.default_rng(11)
    for _ in range(12):
        truth = rng.random(40) < 0.3
        if truth.all() or not truth.any():
            continue
        score = rng.random(40)
        # Ties on purpose, which is where a hand-rolled AUC usually breaks.
        score[rng.random(40) < 0.3] = 0.5
        mine = areas(score, truth)
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError:
            continue
        assert np.isclose(mine["roc_auc"], roc_auc_score(truth, score)), mine
        assert np.isclose(mine["pr_auc"],
                          average_precision_score(truth, score)), mine


def test_candidate_cutoffs_reach_both_extremes():
    score = np.array([0.0, 0.5, 1.0])
    cutoffs = candidate_cutoffs(score)
    assert (score > cutoffs[0]).all(), "rejecting every cell must be reachable"
    assert not (score > cutoffs[-1]).any(), "rejecting nothing must be reachable"


def test_ties_break_towards_the_more_conservative_cutoff():
    truth = np.array([True, False])
    # Two cutoffs give the same F1; the larger one rejects fewer cells.
    u = np.array([1.0, 0.0])
    out = oracle_cutoffs(u, truth)
    assert out["oracle_f1_cutoff"] == 0.0, out
    assert operating_point(u > out["oracle_f1_cutoff"], truth)["rejected_n"] == 1


def test_operating_point_refuses_a_shape_mismatch():
    try:
        operating_point(np.zeros(3, dtype=bool), np.zeros(4, dtype=bool))
    except ValueError:
        return
    raise AssertionError("a shape mismatch must not be scored silently")


if __name__ == "__main__":
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
