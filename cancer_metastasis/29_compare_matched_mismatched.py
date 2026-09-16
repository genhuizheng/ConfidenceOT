"""Compare the gate on matched pairs against the cross-patient control.

``28_build_mismatched_manifest.py`` re-runs every pair with a different
patient's metastasis on the target side and the *same* primary cells on the
source side.  This script reads the two runs together and asks the question the
matched analysis cannot answer on its own: does it matter whose metastasis is
on the other side?

The per-pair statistics, in decreasing order of force:

``gate_jaccard``
    Overlap of the two retained sets over the primary cells both runs saw.  The
    two runs gate the same cells, so this is a direct measure of how much the
    partner's identity changes the answer.  Read it against
    ``chance_gate_jaccard``, the overlap two independent subsets of the same
    two sizes would reach, which is the floor.  The ceiling is 1.0, meaning the
    partner is irrelevant and the gate is a property of the primary side alone.

``spearman_decision_cost``
    Rank correlation of the transport cost of the same cell under the two
    partners.  ``gate_jaccard`` can be high merely because both runs retain a
    similar *number* of cells; this asks whether the underlying cost ordering
    is the same object.  Near 1.0 means the cost is computed from the primary
    cell and not from its relationship to any particular metastasis.

``retained_fraction_delta``
    How much the retained fraction moves.  Weakest of the three, because two
    unrelated gates can coincidentally retain similar fractions, but it is the
    number the biological claim is stated in.

``rejection_cost_matched`` / ``rejection_cost_mismatched``
    An internal check rather than a result.  The within-side null is built by
    splitting one side against itself, so the source threshold is calibrated
    without reference to the partner and should barely move.  A large shift
    would mean the calibration depends on the partner through some path the
    design does not intend, and would have to be resolved before the rest of
    the comparison could be read.

Interpretation is fixed here, before the numbers exist.  ``gate_jaccard`` near
its chance floor means the split is patient-specific and the matched result
survives.  ``gate_jaccard`` near 1.0 with ``spearman_decision_cost`` near 1.0
means the gate never used the metastasis at all, and the retained set cannot be
evidence about metastatic compatibility whatever its differential expression
turns out to say.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REQUIRED = ("method", "side", "observation_id", "retained", "decision_cost")

SUMMARY_ORDER = (
    "shared_cell_n",
    "retained_fraction_matched",
    "retained_fraction_mismatched",
    "retained_fraction_delta",
    "gate_jaccard",
    "chance_gate_jaccard",
    "label_concordance",
    "spearman_decision_cost",
    "rejection_cost_matched",
    "rejection_cost_mismatched",
)


def confidence_path(root: Path, pair_id: str, scope: str, budget_tag: str | None) -> Path | None:
    matches = sorted(root.glob(f"{pair_id}/{scope}/{budget_tag or '*'}/cell_confidence.csv"))
    if not matches:
        return None
    if len(matches) > 1:
        tags = sorted({item.parent.name for item in matches})
        raise RuntimeError(
            f"{pair_id}: several budget tags completed ({tags}); pass --budget-tag"
        )
    return matches[0]


def load_side(path: Path, method: str, side: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = [name for name in REQUIRED if name not in table.columns]
    if missing:
        raise RuntimeError(f"{path}: missing columns {missing}")
    table = table.loc[table["method"].eq(method) & table["side"].eq(side)].copy()
    table["observation_id"] = table["observation_id"].astype(str)
    if table["observation_id"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate observation_id for {method}/{side}")
    return table.set_index("observation_id")


def compare_pair(matched: pd.DataFrame, mismatched: pd.DataFrame) -> dict[str, float]:
    shared = matched.index.intersection(mismatched.index)
    if not len(shared):
        return {"shared_cell_n": 0}
    left = matched.loc[shared]
    right = mismatched.loc[shared]
    kept_matched = left["retained"].to_numpy(bool)
    kept_mismatched = right["retained"].to_numpy(bool)

    union = int((kept_matched | kept_mismatched).sum())
    fraction_matched = float(kept_matched.mean())
    fraction_mismatched = float(kept_mismatched.mean())
    # Overlap two independent subsets of these two sizes would reach. With
    # equal sizes this reduces to the f / (2 - f) baseline used elsewhere.
    denominator = fraction_matched + fraction_mismatched - fraction_matched * fraction_mismatched
    record = {
        "shared_cell_n": int(len(shared)),
        "matched_cell_n": int(len(matched)),
        "mismatched_cell_n": int(len(mismatched)),
        "retained_fraction_matched": fraction_matched,
        "retained_fraction_mismatched": fraction_mismatched,
        "retained_fraction_delta": fraction_mismatched - fraction_matched,
        "gate_jaccard": (
            float((kept_matched & kept_mismatched).sum() / union) if union else float("nan")
        ),
        "chance_gate_jaccard": (
            float(fraction_matched * fraction_mismatched / denominator)
            if denominator > 0 else float("nan")
        ),
        "label_concordance": float((kept_matched == kept_mismatched).mean()),
    }
    cost_matched = pd.to_numeric(left["decision_cost"], errors="coerce").to_numpy(float)
    cost_mismatched = pd.to_numeric(right["decision_cost"], errors="coerce").to_numpy(float)
    finite = np.isfinite(cost_matched) & np.isfinite(cost_mismatched)
    if finite.sum() >= 3 and np.ptp(cost_matched[finite]) > 0 and np.ptp(cost_mismatched[finite]) > 0:
        record["spearman_decision_cost"] = float(
            stats.spearmanr(cost_matched[finite], cost_mismatched[finite]).statistic
        )
    else:
        record["spearman_decision_cost"] = float("nan")
    for name, frame in (("matched", left), ("mismatched", right)):
        if "rejection_cost" in frame:
            values = pd.to_numeric(frame["rejection_cost"], errors="coerce").dropna()
            record[f"rejection_cost_{name}"] = (
                float(values.iloc[0]) if len(values) else float("nan")
            )
    return record


def summarize(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in SUMMARY_ORDER:
        if name not in pairs:
            continue
        values = pd.to_numeric(pairs[name], errors="coerce").dropna()
        if not len(values):
            continue
        rows.append({
            "statistic": name,
            "pair_n": int(len(values)),
            "median": float(values.median()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mismatched_manifest", type=Path)
    parser.add_argument("matched_root", type=Path)
    parser.add_argument("mismatched_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--scope", default="scope_malignant")
    parser.add_argument("--matched-budget-tag", default=None)
    parser.add_argument("--mismatched-budget-tag", default=None)
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--side", default="source", choices=("source", "target"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.mismatched_manifest)
    for name in ("pair_id", "matched_pair_id"):
        if name not in manifest.columns:
            raise RuntimeError(f"{args.mismatched_manifest}: missing column {name!r}")

    records = []
    skipped: list[dict[str, str]] = []
    for row in manifest.to_dict("records"):
        matched_path = confidence_path(
            args.matched_root, str(row["matched_pair_id"]), args.scope,
            args.matched_budget_tag,
        )
        mismatched_path = confidence_path(
            args.mismatched_root, str(row["pair_id"]), args.scope,
            args.mismatched_budget_tag,
        )
        if matched_path is None or mismatched_path is None:
            skipped.append({
                "matched_pair_id": str(row["matched_pair_id"]),
                "pair_id": str(row["pair_id"]),
                "reason": "matched missing" if matched_path is None else "mismatched missing",
            })
            continue
        record = compare_pair(
            load_side(matched_path, args.method, args.side),
            load_side(mismatched_path, args.method, args.side),
        )
        if not record.get("shared_cell_n"):
            skipped.append({
                "matched_pair_id": str(row["matched_pair_id"]),
                "pair_id": str(row["pair_id"]),
                "reason": "no shared primary cells",
            })
            continue
        record["matched_pair_id"] = str(row["matched_pair_id"])
        record["pair_id"] = str(row["pair_id"])
        record["source_patient_id"] = str(row.get("source_patient_id", ""))
        record["target_patient_id"] = str(row.get("target_patient_id", ""))
        records.append(record)

    if not records:
        raise RuntimeError("No pair completed in both runs")
    pairs = pd.DataFrame(records)
    args.output_root.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_root / "matched_vs_mismatched_pairs.csv", index=False)
    summary = summarize(pairs)
    summary.to_csv(args.output_root / "matched_vs_mismatched_summary.csv", index=False)

    # The paired test that matters: is the overlap between the two runs larger
    # than two independent gates of the same sizes would reach? A gate that
    # used the partner would sit near the chance floor.
    excess = pairs["gate_jaccard"] - pairs["chance_gate_jaccard"]
    usable = excess.dropna()
    jaccard_vs_chance_p = (
        float(stats.wilcoxon(usable).pvalue)
        if len(usable) >= 6 and float(np.ptp(usable.to_numpy())) > 0 else float("nan")
    )
    delta = pairs["retained_fraction_delta"].dropna()
    retained_shift_p = (
        float(stats.wilcoxon(delta).pvalue)
        if len(delta) >= 6 and float(np.ptp(delta.to_numpy())) > 0 else float("nan")
    )

    report = {
        "mismatched_manifest": str(args.mismatched_manifest),
        "matched_root": str(args.matched_root),
        "mismatched_root": str(args.mismatched_root),
        "method": args.method,
        "side": args.side,
        "pairs_compared": int(len(pairs)),
        "pairs_skipped": len(skipped),
        "gate_jaccard_median": float(pairs["gate_jaccard"].median()),
        "chance_gate_jaccard_median": float(pairs["chance_gate_jaccard"].median()),
        "gate_jaccard_excess_over_chance_median": float(usable.median()) if len(usable) else None,
        "gate_jaccard_vs_chance_signed_rank_p": jaccard_vs_chance_p,
        "spearman_decision_cost_median": float(pairs["spearman_decision_cost"].median()),
        "retained_fraction_matched_median": float(pairs["retained_fraction_matched"].median()),
        "retained_fraction_mismatched_median": float(pairs["retained_fraction_mismatched"].median()),
        "retained_fraction_delta_signed_rank_p": retained_shift_p,
        "reading": (
            "gate_jaccard near chance_gate_jaccard means the retained set is "
            "patient-specific. gate_jaccard near 1.0 with "
            "spearman_decision_cost near 1.0 means the gate did not use the "
            "metastatic side, so the retained set cannot be evidence about "
            "metastatic compatibility."
        ),
    }
    (args.output_root / "matched_vs_mismatched_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if skipped:
        pd.DataFrame(skipped).to_csv(
            args.output_root / "matched_vs_mismatched_skipped.csv", index=False
        )
    print(json.dumps(report, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
