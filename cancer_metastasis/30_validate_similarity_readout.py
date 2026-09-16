"""Test whether the retained fraction is a primary--metastasis similarity readout.

The gate was built to say which primary malignant cells are compatible with a
matched metastasis, and the retained fraction was read as an estimate of how
many such cells there are.  Four independent observations say it is measuring
something else:

* swapping in another patient's metastasis moves the retained fraction from
  0.351 to 0.000 across 94 pairs (signed-rank p = 8e-17), and preserves only
  rho = 0.378 of the cost ordering;
* 64% of one patient's primary cells carry different labels against different
  lesions of that same patient;
* naming a different lesion for SPECTRUM-OV-003 moved its retained fraction
  from 0.420 to 0.032 over the same 1,332 cells;
* across patients, with one lesion each, it ranges from 0.000 to 0.879.

The mechanism predicts exactly this.  The threshold is calibrated by splitting
one side against itself and dividing by the median cross-side distance, so

    c ~ within-side heterogeneity / cross-side separation

and a cell is retained when its distance to the metastasis is below the spread
of the primary against itself.  The denominator is a similarity, so the
retained fraction should track how alike the two sides are.

**This script tests that against a measure the algorithm never sees.**
Similarity is computed between pseudobulk profiles: sum the counts of each
side, take log-CPM, correlate.  No PCA, no transport, no gene selection by the
solver.  Using anything internal -- the calibrated cost, the cross-side scale --
would be circular, since those are the very quantities the mechanism is stated
in.

Matched and mismatched arms are read together on purpose.  The matched pairs
alone span a narrow band of similarity; the mismatched pairs extend it
downward, and a relationship that holds across the union is tested over a much
wider range than either arm supports on its own.

**Prespecified reading.**

*Supported* if ``spearman(retained_fraction, pseudobulk_spearman)`` is strongly
positive across the union of arms, and the mismatched pairs sit at the low end
of both axes continuously with the matched ones rather than as a separate
cluster.  The retained fraction is then a similarity measure -- a real
quantity, but not the count of metastasis-competent cells it was read as.

*Not supported* if the correlation is near zero.  The retained fraction is then
reading something else again, and the residual depth signal that survived
downsampling (AUC 0.587) is the first thing to suspect.

*Confounded* if ``spearman(pseudobulk_spearman, min_cell_n)`` is comparable in
size to the result.  A pseudobulk built from more cells is less noisy and
correlates better with anything, so cell count could produce the relationship
on its own.  That column is reported next to the result for this reason, and a
conclusion cannot be drawn without reading it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

from common import expression_matrix, gene_keys, load_exact_side


def sample_pseudobulk(paths: list[str], sample: str) -> pd.Series:
    """Total counts per gene over every cell of one sample."""
    data = load_exact_side(paths, sample)
    matrix = sparse.csr_matrix(expression_matrix(data))
    totals = np.asarray(matrix.sum(axis=0), dtype=np.float64).ravel()
    keys = [str(value) for value in gene_keys(data)]
    # A key can repeat when several rows of var map to one gene; summing keeps
    # the profile a count vector.
    return pd.Series(totals, index=keys).groupby(level=0).sum()


def log_cpm(counts: pd.Series) -> pd.Series:
    total = float(counts.sum())
    if total <= 0:
        raise RuntimeError("Pseudobulk carries no counts")
    return np.log1p(counts / total * 1e6)


def similarity(left: pd.Series, right: pd.Series, basis: pd.Index) -> dict[str, float]:
    shared = left.index.intersection(right.index)
    both = shared[(left.reindex(shared) > 0) | (right.reindex(shared) > 0)]
    record: dict[str, float] = {"shared_gene_n": int(len(both))}
    if len(both) < 50:
        return {**record, "pseudobulk_spearman": float("nan"),
                "pseudobulk_pearson_hv": float("nan")}
    first, second = log_cpm(left.reindex(both, fill_value=0.0)), log_cpm(
        right.reindex(both, fill_value=0.0)
    )
    record["pseudobulk_spearman"] = float(
        stats.spearmanr(first.to_numpy(), second.to_numpy()).statistic
    )
    # A gene basis fixed across every sample, so each pair is scored on the
    # same axes rather than on whatever happens to be shared between its two
    # sides.
    fixed = both.intersection(basis)
    record["hv_gene_n"] = int(len(fixed))
    record["pseudobulk_pearson_hv"] = (
        float(np.corrcoef(
            log_cpm(left.reindex(fixed, fill_value=0.0)).to_numpy(),
            log_cpm(right.reindex(fixed, fill_value=0.0)).to_numpy(),
        )[0, 1])
        if len(fixed) >= 50 else float("nan")
    )
    return record


def gate_fraction(root: Path, pair_id: str, scope: str, method: str) -> dict[str, float]:
    matches = sorted(root.glob(f"{pair_id}/{scope}/*/cell_confidence.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one cell_confidence.csv for {pair_id}; found {len(matches)}"
        )
    table = pd.read_csv(matches[0], usecols=["method", "side", "retained"])
    table = table.loc[table["method"].eq(method)]
    record = {}
    for side in ("source", "target"):
        values = table.loc[table["side"].eq(side), "retained"]
        record[f"{side}_retained_fraction"] = (
            float(values.astype(bool).mean()) if len(values) else float("nan")
        )
        record[f"{side}_gated_cell_n"] = int(len(values))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--arm", action="append", nargs=3, required=True,
        metavar=("LABEL", "MANIFEST", "GATE_ROOT"),
        help="Named arm to include; repeat for matched and mismatched",
    )
    parser.add_argument("--scope", default="scope_malignant")
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--hv-gene-n", type=int, default=2000)
    args = parser.parse_args()

    # One pseudobulk per distinct file, reused by every pair that cites it.
    cache: dict[str, pd.Series] = {}

    def profile(paths_json: str, sample: str) -> pd.Series:
        paths = json.loads(str(paths_json))
        key = "|".join(sorted(str(value) for value in paths)) + "::" + str(sample)
        if key not in cache:
            cache[key] = sample_pseudobulk([str(value) for value in paths], str(sample))
        return cache[key]

    arms = []
    for label, manifest_csv, gate_root in args.arm:
        manifest = pd.read_csv(manifest_csv)
        for row in manifest.to_dict("records"):
            arms.append((label, Path(gate_root), row))

    # Every profile is built before any pair is scored, so the high-variance
    # basis is the same set of genes for every arm and every pair.
    for _, _, row in arms:
        profile(row["source_h5ads_json"], row["source_sample"])
        profile(row["target_h5ads_json"], row["target_sample"])
    matrix = pd.DataFrame({key: log_cpm(value) for key, value in cache.items()})
    basis = (
        matrix.var(axis=1).sort_values(ascending=False)
        .head(args.hv_gene_n).index
    )

    records = []
    for label, gate_root, row in arms:
        source = profile(row["source_h5ads_json"], row["source_sample"])
        target = profile(row["target_h5ads_json"], row["target_sample"])
        record = {
            "arm": label,
            "pair_id": str(row["pair_id"]),
            "source_patient_id": str(row.get("source_patient_id", row["patient_id"])),
            "target_patient_id": str(row.get("target_patient_id", row["patient_id"])),
            "source_sample": str(row["source_sample"]),
            "target_sample": str(row["target_sample"]),
        }
        record.update(similarity(source, target, basis))
        record.update(gate_fraction(gate_root, record["pair_id"], args.scope, args.method))
        record["min_cell_n"] = min(
            record["source_gated_cell_n"], record["target_gated_cell_n"]
        )
        records.append(record)

    pairs = pd.DataFrame(records)
    args.output_root.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_root / "similarity_readout_pairs.csv", index=False)

    def rank_correlation(frame: pd.DataFrame, left: str, right: str) -> dict:
        usable = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(usable) < 6 or usable[left].nunique() < 3 or usable[right].nunique() < 3:
            return {"rho": None, "p": None, "n": int(len(usable))}
        result = stats.spearmanr(usable[left].to_numpy(), usable[right].to_numpy())
        return {
            "rho": float(result.statistic),
            "p": float(result.pvalue),
            "n": int(len(usable)),
        }

    report = {"arms": sorted({label for label, _, _ in arms}), "results": {}}
    for scope_label, frame in [("union", pairs)] + [
        (label, pairs[pairs["arm"].eq(label)]) for label in report["arms"]
    ]:
        report["results"][scope_label] = {
            "pair_n": int(len(frame)),
            "retained_fraction_median": float(
                pd.to_numeric(frame["source_retained_fraction"], errors="coerce").median()
            ),
            "pseudobulk_spearman_median": float(
                pd.to_numeric(frame["pseudobulk_spearman"], errors="coerce").median()
            ),
            "retained_vs_similarity": rank_correlation(
                frame, "source_retained_fraction", "pseudobulk_spearman"
            ),
            "retained_vs_similarity_hv": rank_correlation(
                frame, "source_retained_fraction", "pseudobulk_pearson_hv"
            ),
            # Read this before the result above. A pseudobulk built from more
            # cells is less noisy and correlates better with anything.
            "similarity_vs_cell_n": rank_correlation(
                frame, "pseudobulk_spearman", "min_cell_n"
            ),
            "retained_vs_cell_n": rank_correlation(
                frame, "source_retained_fraction", "min_cell_n"
            ),
        }
    (args.output_root / "similarity_readout_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
