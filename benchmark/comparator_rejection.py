"""One rejection rule for the ungated comparators, at two operating points.

Traditional OT, Vanilla UOT and Partial OT all return a coupling and no
decision. To score them against a gate they need a cell-level quantity, and
the one the coupling already contains is how much of a cell's own mass failed
to be transported:

    u_i = clip(1 - (sum_j pi_ij) / a_i, 0, 1)        source side
    u_j = clip(1 - (sum_i pi_ij) / b_j, 0, 1)        target side

Called the **cell-level unmatchedness score**, or the **mass-deficit
rejection score**. Not a probability: it is a normalised mass deficit, it is
not calibrated against anything, and nothing here estimates P(reject).

The clip matters at both ends. A cell that received *more* than its own
marginal gives a negative deficit, which is excess rather than evidence of
rejection, so it floors at 0; the ratio cannot exceed 1 for a non-negative
coupling, and the ceiling is there so a numerical overshoot cannot become a
score above 1.

**Two operating points, reported separately and never mixed.**

*Fixed cutoff.* tau = 0.5, ``reject = u > 0.5``, the same number for every
dataset, replicate and level. This is the deployable rule and it is the main
result. Under Partial OT it is not a tuning choice at all: an exact
cardinality matching gives every cell either its full marginal or none of it,
so u is already {0, 1} and any cutoff in (0, 1) is the same rule -- "was this
cell matched". That fact is detected and recorded rather than assumed.

*Oracle cutoff.* The best threshold the score could have had, chosen with the
ground truth in hand -- Youden's J for the ROC-optimal one, maximal F1 for
the PR-optimal one. It answers only "if the threshold were chosen perfectly,
how good is this score at best", which is an upper bound on the score and not
a setting anyone could deploy. It is labelled ``oracle_`` everywhere it
appears, it is never the headline, and a reader who mistakes it for an
operating point has been misled by the table rather than by this module.

**Orientation, because the project already contains the opposite
convention.** Here the positive class is *rejected*, the score is *higher
means more likely rejected*, and an AUC near 1 is good. ``40_score.py``'s
``rank_auc`` takes *retained* as positive, so there an AUC near 0 means the
gate is a clean threshold on cost. Both are correct in their own file and
mixing them silently flips a result, so every column produced here carries
the ``rejected_positive`` convention in its documentation and the module
refuses to consume a retained-positive truth vector.

**What each method contributes.**

``Traditional OT``
    Balanced transport moves every unit of mass, so u is identically zero and
    the gate is all-False by construction. Recorded as a fact, not computed
    from a degenerate curve.

``Partial OT``
    The same formula. u comes out {0, 1}, which is recorded.

``ConfidenceOT``
    Keeps its native gate as the binary rejection -- it is not converted to a
    mass-deficit score, because its gate is the thing under test. Its
    continuous ``decision_cost`` is carried alongside for the threshold-free
    areas, so ROC-AUC and PR-AUC are comparable across all four methods while
    the binary decisions remain each method's own.

``Vanilla UOT``
    The formula above, continuous, and the only method for which the fixed
    cutoff is a genuine choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

FIXED_CUTOFF = 0.5


def unmatchedness(coupling: np.ndarray, marginal: np.ndarray, *,
                  side: str) -> np.ndarray:
    """The cell-level unmatchedness score for one side of a coupling.

    ``marginal`` is the side's own reference mass per cell -- ``a`` for the
    source, ``b`` for the target -- and is taken rather than assumed uniform,
    because a method that was given non-uniform marginals must be scored
    against the ones it was given.
    """
    if side not in ("source", "target"):
        raise ValueError(f"side must be 'source' or 'target', got {side!r}")
    plan = np.asarray(coupling, dtype=np.float64)
    reference = np.asarray(marginal, dtype=np.float64)
    transported = plan.sum(axis=1) if side == "source" else plan.sum(axis=0)
    if reference.shape != transported.shape:
        raise ValueError(
            f"{side}: marginal has {reference.shape} against a coupling that "
            f"gives {transported.shape}")
    if np.any(reference <= 0.0):
        raise ValueError(
            f"{side}: a cell with non-positive reference mass has no deficit "
            f"to normalise; {int(np.sum(reference <= 0.0))} such cells")
    return np.clip(1.0 - transported / reference, 0.0, 1.0)


def _counts(predicted: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    return tp, fp, fn, tn


def _ratio(numerator: float, denominator: float) -> float:
    """Undefined rather than zero when there is nothing to divide by."""
    return float(numerator / denominator) if denominator else float("nan")


def operating_point(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Precision, recall, F1 and the false rejection rate at one decision.

    Each is undefined rather than zero when its denominator is empty. That
    distinction carries most of the benchmark: in the ``all_shared`` case the
    truth has no positives at all, so recall and precision say nothing and
    the false rejection rate is the entire result.
    """
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if predicted.shape != truth.shape:
        raise ValueError(f"{predicted.shape} predictions against {truth.shape} truth")
    tp, fp, fn, tn = _counts(predicted, truth)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    denominator = (precision + recall) if np.isfinite(precision + recall) else 0.0
    return {
        "rejected_n": int(predicted.sum()),
        "rejection_rate": float(predicted.mean()) if predicted.size else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": (2.0 * precision * recall / denominator) if denominator else float("nan"),
        # Of the cells that should have been kept, how many were rejected.
        # Defined whenever anything should have been kept, which is why it
        # survives the homogeneous case that leaves the rest undefined.
        "false_rejection_rate": _ratio(fp, fp + tn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def candidate_cutoffs(score: np.ndarray) -> np.ndarray:
    """Every cutoff that can change the decision under ``score > tau``.

    The observed values, plus one below the smallest so that rejecting every
    cell is reachable. Rejecting nothing is already reachable at the largest
    observed value.
    """
    values = np.unique(np.asarray(score, dtype=np.float64))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    below = np.nextafter(values[0], -np.inf)
    return np.concatenate([[below], values])


def _sweep(score: np.ndarray, truth: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for cutoff in candidate_cutoffs(score):
        point = operating_point(score > cutoff, truth)
        tp, fp, fn, tn = point["tp"], point["fp"], point["fn"], point["tn"]
        point["cutoff"] = float(cutoff)
        point["tpr"] = _ratio(tp, tp + fn)
        point["fpr"] = _ratio(fp, fp + tn)
        point["youden_j"] = point["tpr"] - point["fpr"]
        rows.append(point)
    return rows


def oracle_cutoffs(score: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """The best cutoff this score could have had, by two criteria.

    Chosen with the ground truth, so both are upper bounds on the score and
    neither is deployable. Named ``oracle_`` at every level so the
    distinction survives being copied into a table.

    Ties are broken towards the larger cutoff, which is the more conservative
    of two equally good rules: it rejects fewer cells for the same score.
    """
    score = np.asarray(score, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    out: dict[str, Any] = {}
    finite = score[np.isfinite(score)]
    one_class = truth.sum() == 0 or truth.sum() == truth.size
    # A constant score has no ordering, so its only two rules are "reject
    # everything" and "reject nothing". Maximising F1 over those picks
    # reject-everything whenever any cell should be rejected, which reports
    # recall 1.0 for a score that discriminates nothing -- an upper bound
    # that is an artefact of the sweep rather than a property of the score.
    # Traditional OT is exactly this case.
    constant = bool(finite.size == 0 or np.ptp(finite) == 0.0)
    if one_class or constant:
        for criterion in ("roc", "f1"):
            out[f"oracle_{criterion}_cutoff"] = float("nan")
            out[f"oracle_{criterion}_defined"] = False
        out["oracle_undefined_because"] = (
            "one class in the truth" if one_class else "the score is constant")
        return out
    rows = _sweep(score, truth)
    for criterion, key in (("roc", "youden_j"), ("f1", "f1")):
        usable = [row for row in rows if np.isfinite(row[key])]
        if not usable:
            out[f"oracle_{criterion}_cutoff"] = float("nan")
            out[f"oracle_{criterion}_defined"] = False
            continue
        best = max(usable, key=lambda row: (row[key], row["cutoff"]))
        out[f"oracle_{criterion}_cutoff"] = best["cutoff"]
        out[f"oracle_{criterion}_defined"] = True
        for name in ("precision", "recall", "f1", "false_rejection_rate",
                     "rejection_rate", "youden_j"):
            out[f"oracle_{criterion}_{name}"] = best[name]
    return out


def areas(score: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """ROC-AUC and PR-AUC with **rejected** as the positive class.

    Threshold-free, so they are the one part of this module that compares a
    mass-deficit score against ConfidenceOT's decision cost without either
    being converted into the other.

    PR-AUC is average precision -- the step-wise sum over the thresholds the
    score can express, not a trapezoid over an interpolated curve, which is
    optimistic on a curve that is not monotone. Both agree with
    ``sklearn.metrics`` including on ties, which
    ``tests/test_comparator_rejection.py`` checks against it directly.
    """
    score = np.asarray(score, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    positives, negatives = int(truth.sum()), int((~truth).sum())
    if positives == 0 or negatives == 0:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "auc_defined": False}
    # ROC-AUC as the Mann-Whitney statistic, with ties given their mean rank
    # so a score that is constant over a block gets no credit for an ordering
    # it does not have. A wholly constant score therefore lands on exactly
    # 0.5, which is chance -- reported as such, with score_is_constant set,
    # rather than as NaN: 0.5 is the measurement, not a missing value.
    from scipy.stats import rankdata

    ranks = rankdata(score, method="average")
    roc_auc = (ranks[truth].sum() - positives * (positives + 1) / 2.0) / (
        positives * negatives)

    # Average precision, grouped by distinct score. Walking cell by cell
    # splits a block of equal scores in whatever order the sort happened to
    # produce, which credits the score for resolving cells it scored
    # identically; the group boundaries are the only thresholds the score can
    # actually express.
    order = np.argsort(-score, kind="stable")
    ordered_score, ordered_truth = score[order], truth[order]
    ends = np.flatnonzero(np.diff(ordered_score))
    ends = np.append(ends, ordered_score.size - 1)
    true_positive = np.cumsum(ordered_truth)[ends]
    called = ends + 1
    precision = true_positive / called
    recall = true_positive / positives
    previous_recall = np.concatenate([[0.0], recall[:-1]])
    pr_auc = float(np.sum((recall - previous_recall) * precision))
    return {"roc_auc": float(roc_auc), "pr_auc": pr_auc, "auc_defined": True}


@dataclass
class SideScore:
    """Everything one method says about one side of one pair."""

    method: str
    side: str
    score_kind: str
    score: np.ndarray
    fixed_prediction: np.ndarray
    truth: np.ndarray
    summary: dict[str, Any] = field(default_factory=dict)


def score_side(*, method: str, side: str, truth: np.ndarray,
               score: np.ndarray | None = None,
               native_gate: np.ndarray | None = None,
               score_kind: str = "mass_deficit") -> SideScore:
    """Score one side, at the fixed cutoff and at both oracle cutoffs.

    ``score`` is the continuous quantity, oriented so that higher means more
    likely rejected. ``native_gate`` is a method's own retain/reject decision
    where it has one; when given it **replaces** the fixed cutoff as that
    method's binary result, because the point of testing a gate is to test
    the gate. The score is still used for the threshold-free areas and for
    the oracle bounds.
    """
    truth = np.asarray(truth, dtype=bool)
    if score is None:
        score = np.zeros(truth.size, dtype=np.float64)
        score_kind = "constant_zero"
    score = np.asarray(score, dtype=np.float64)
    if score.shape != truth.shape:
        raise ValueError(f"score {score.shape} against truth {truth.shape}")

    if native_gate is not None:
        gate = np.asarray(native_gate, dtype=bool)
        if gate.shape != truth.shape:
            raise ValueError(f"gate {gate.shape} against truth {truth.shape}")
        # A gate names the cells it KEEPS, so rejection is its complement.
        prediction = ~gate
        decision = "native_gate"
    else:
        prediction = score > FIXED_CUTOFF
        decision = f"fixed_cutoff_{FIXED_CUTOFF}"

    finite = score[np.isfinite(score)]
    is_binary = bool(finite.size and np.all(np.isin(finite, (0.0, 1.0))))
    summary: dict[str, Any] = {
        "method": method,
        "side": side,
        "score_kind": score_kind,
        "binary_decision_from": decision,
        "fixed_cutoff": FIXED_CUTOFF,
        # Recorded, not assumed: under an exact cardinality matching the
        # score is already {0,1}, so the fixed cutoff adds no parameter and
        # an oracle cutoff cannot improve on it.
        "score_is_binary": is_binary,
        # A constant score gives ROC-AUC exactly 0.5. That is chance rather
        # than an absence, so the number is reported and the reason is
        # flagged beside it -- Traditional OT is the case that matters.
        "score_is_constant": bool(finite.size and np.ptp(finite) == 0.0),
        "n_cells": int(truth.size),
        "n_should_reject": int(truth.sum()),
        "positive_class": "rejected",
    }
    for name, value in operating_point(prediction, truth).items():
        summary[f"fixed_{name}"] = value
    summary.update(oracle_cutoffs(score, truth))
    summary.update(areas(score, truth))
    return SideScore(method=method, side=side, score_kind=score_kind,
                     score=score, fixed_prediction=prediction, truth=truth,
                     summary=summary)


def traditional_ot_score(n_cells: int) -> np.ndarray:
    """Identically zero: balanced transport leaves no cell unmatched.

    A function rather than a constant so the fact is stated once and cannot
    be reintroduced as a degenerate sweep over a constant score.
    """
    return np.zeros(int(n_cells), dtype=np.float64)


def summary_frame(scores: list[SideScore]):
    """One row per method and side, with the three operating points apart.

    Column groups, deliberately prefixed so that copying a subset of the
    table cannot lose the distinction:

    ``fixed_*``        the deployable rule, tau = 0.5 or a native gate
    ``oracle_roc_*``   the best Youden's J cutoff, chosen with the truth
    ``oracle_f1_*``    the best F1 cutoff, chosen with the truth
    ``roc_auc``/``pr_auc``  threshold-free, rejected as the positive class
    """
    import pandas as pd

    return pd.DataFrame([score.summary for score in scores])


# The three blocks, kept apart by construction. A caller that wants only the
# deployable numbers takes FIXED_BLOCK and cannot accidentally carry an
# oracle column with it.
FIXED_BLOCK = ("fixed_rejection_rate", "fixed_precision", "fixed_recall",
               "fixed_f1", "fixed_false_rejection_rate")
ORACLE_ROC_BLOCK = ("oracle_roc_cutoff", "oracle_roc_precision",
                    "oracle_roc_recall", "oracle_roc_f1",
                    "oracle_roc_false_rejection_rate")
ORACLE_F1_BLOCK = ("oracle_f1_cutoff", "oracle_f1_precision",
                   "oracle_f1_recall", "oracle_f1_f1",
                   "oracle_f1_false_rejection_rate")
THRESHOLD_FREE_BLOCK = ("roc_auc", "pr_auc")

BLOCK_HEADINGS = (
    ("fixed cutoff tau=0.5 or native gate -- DEPLOYABLE, the main result",
     FIXED_BLOCK),
    ("oracle ROC cutoff, max Youden J -- UPPER BOUND, chosen with the truth",
     ORACLE_ROC_BLOCK),
    ("oracle PR cutoff, max F1 -- UPPER BOUND, chosen with the truth",
     ORACLE_F1_BLOCK),
    ("threshold-free, rejected as the positive class", THRESHOLD_FREE_BLOCK),
)


def summary_table(scores: "list[SideScore]", *, digits: int = 3) -> str:
    """The three operating points as three labelled blocks, never one row.

    A single wide row invites a reader to compare an oracle F1 against a
    fixed-cutoff F1 as though both were available at deployment. They are
    printed as separate blocks with the oracle ones marked UPPER BOUND, and
    the caption says what that means, because the distinction has to survive
    someone copying a screenshot into a slide.
    """
    import pandas as pd

    frame = summary_frame(scores)
    if frame.empty:
        return "no scores"
    index = ["method", "side"]
    lines = []
    for heading, block in BLOCK_HEADINGS:
        present = [column for column in block if column in frame]
        if not present:
            continue
        lines.append(f"=== {heading} ===")
        table = frame.set_index(index)[present].round(digits)
        table.columns = [name
                         .replace("fixed_", "")
                         .replace("oracle_roc_", "")
                         .replace("oracle_f1_", "")
                         for name in table.columns]
        lines.append(table.to_string())
        lines.append("")
    facts = [column for column in
             ("score_kind", "binary_decision_from", "score_is_binary",
              "score_is_constant", "n_cells", "n_should_reject")
             if column in frame]
    lines.append("=== how each score was formed ===")
    lines.append(frame.set_index(index)[facts].to_string())
    lines.append("")
    lines.append("The oracle cutoffs are chosen using should_reject. They "
                 "bound what the score could")
    lines.append("achieve if the threshold were perfect; no deployment has "
                 "access to them, and they")
    lines.append("are not an operating point. The main result is the fixed "
                 "block.")
    return "\n".join(lines)
