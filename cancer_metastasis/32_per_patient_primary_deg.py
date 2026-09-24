"""Per-patient primary retained-vs-rejected differential expression, method B.

Method A sums each patient's primary cells into two pseudobulk vectors and asks
whether there is a **common** effect across patients. This asks a different
question: within each patient separately, which genes separate the retained
cells from the rejected ones, and how **consistent** is that across patients.
The combination step (``33_combine_per_patient_primary_deg.py``) then reports
gene overlap, direction agreement and a patient-level meta-analysis, so the
number of patients supporting a gene is a column of the result rather than
something recovered afterwards.

**The cells and labels come from method A's own output, not from a second
resolution of the same rule.** ``21_prepare_four_state_malignant_pseudobulk.py``
writes ``four_state_cell_classification.csv.gz`` per patient, and this script
reads it. That matters more than it looks: which lesion a patient is paired
against changes the labels of its primary cells -- 64% of one patient's cells
carry different labels against different lesions of that same patient -- so two
methods that resolved the lesion independently could disagree about which cells
are retained while appearing to be the same analysis. Reading the classification
makes the two methods share cell sets by construction.

**Counts come from the ORIGINAL matrices, labels from the equalised gate.**
The same split method A uses. Read equalisation is justified for the
representation and the gate and was never priced for what it costs the
expression, so the expression side reads the untouched counts. See section 2e
of ``EXPERIMENT_2026-09-23_PREPROCESSING_ARMS.md``.

**A patient below the floor is skipped, not failed.** With retention running
from 0.37 to 0.95, a high-retention patient can have very few rejected cells,
and an array task that exits non-zero would block the combination step through
``afterok``. The skip is recorded in ``diagnostics.json`` with its reason and
counted by the combination step, because a silently absent patient and a
deliberately skipped one must not look alike.

Usage, one patient per invocation so it arrays cleanly:

    python cancer_metastasis/32_per_patient_primary_deg.py \\
        FOUR_STATE_ROOT ORIGINAL_MANIFEST_CSV OUTPUT_ROOT --index 0

    python cancer_metastasis/32_per_patient_primary_deg.py \\
        FOUR_STATE_ROOT ORIGINAL_MANIFEST_CSV OUTPUT_ROOT --print-patient-count
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from common import (
    expression_kind,
    expression_matrix,
    gene_keys,
    load_exact_side,
    normalize_expression,
)

# The two statuses the contrast is between. consensus_classification also emits
# `incomplete_pair_coverage` and `site_or_cap_discordant`; both are excluded and
# counted, because a cell whose label depends on which lesion it was scored
# against has no single answer to contribute.
STATUSES = ("retained", "rejected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("four_state_root", type=Path,
                        help="Output root of 21_prepare_four_state_malignant_pseudobulk.py")
    parser.add_argument("manifest_csv", type=Path,
                        help="The ORIGINAL pair manifest, not the downsampled one")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--index", type=int,
                        help="Zero-based patient index within the sorted patient list")
    parser.add_argument("--print-patient-count", action="store_true",
                        help="Print the patient count and exit, for array sizing")
    parser.add_argument("--minimum-cells-per-status", type=int, default=10,
                        help="Floor per status; matches the DEG prespecification's "
                             "2026-09-24 amendment")
    parser.add_argument("--target-sum", type=float, default=10_000.0,
                        help="Library size each cell is scaled to before log1p")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    if not args.print_patient_count and args.index is None:
        parser.error("--index is required unless --print-patient-count is given")
    return args


def patient_directories(four_state_root: Path) -> list[Path]:
    """Method A's per-patient outputs, in a stable order.

    Sorted by directory name, which 21_ prefixes with the zero-padded index it
    used, so this order is method A's order and the two arrays line up.
    """
    root = four_state_root / "patients"
    if not root.is_dir():
        raise FileNotFoundError(
            f"No patients/ directory under {four_state_root}. Run "
            f"21_prepare_four_state_malignant_pseudobulk.py first."
        )
    return sorted(path for path in root.iterdir() if path.is_dir())


def source_paths(manifest: pd.DataFrame, pair_id: str) -> list[str]:
    """The h5ad paths for one pair's source side, from the original manifest."""
    rows = manifest[manifest["pair_id"].astype(str) == str(pair_id)]
    if len(rows) != 1:
        raise KeyError(
            f"pair_id {pair_id!r} matched {len(rows)} rows of the manifest; "
            f"expected exactly one. The four-state root and the manifest must "
            f"describe the same pairs."
        )
    row = rows.iloc[0]
    column = "source_h5ads_json"
    if column in row and pd.notna(row[column]):
        return [str(value) for value in json.loads(str(row[column]))]
    return [str(row["source_h5ad"])]


def load_primary_cells(
    manifest: pd.DataFrame, selected: pd.DataFrame, wanted: pd.DataFrame
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, str]:
    """Counts for the classified primary cells, in classification order.

    Returns the matrix, the gene keys, the per-cell status, and the declared
    expression kind. Samples are intersected on gene key rather than assumed to
    agree, because a patient can contribute more than one source library.
    """
    per_sample: list[tuple[sparse.csr_matrix, np.ndarray, np.ndarray]] = []
    kinds: set[str] = set()
    for sample, block in wanted.groupby("sample", sort=True):
        pairs = selected[selected["source_sample"].astype(str) == str(sample)]
        if pairs.empty:
            raise KeyError(
                f"sample {sample!r} appears in the classification but in no "
                f"selected pair; selected_pairs.csv and the classification "
                f"disagree."
            )
        data = load_exact_side(source_paths(manifest, pairs.iloc[0]["pair_id"]), str(sample))
        kinds.add(expression_kind(data))
        lookup = {str(value): index for index, value in enumerate(data.obs_names)}
        ids = block["observation_id"].astype(str).tolist()
        missing = [value for value in ids if value not in lookup]
        if missing:
            # The gate was computed on the equalised matrices and the counts are
            # read from the original ones. Equalisation subsets cells, so every
            # gate id must still be present here; if one is not, the manifest
            # and the gate root are describing different cells and nothing
            # downstream would notice.
            raise KeyError(
                f"{len(missing)} classified primary cells absent from "
                f"{sample!r} in the original matrices, e.g. {missing[:3]}"
            )
        rows = np.asarray([lookup[value] for value in ids], dtype=np.int64)
        matrix = sparse.csr_matrix(expression_matrix(data))[rows]
        per_sample.append((matrix, np.asarray([str(k) for k in gene_keys(data)]),
                           block["consensus_status"].to_numpy()))
    if len(kinds) != 1:
        raise ValueError(f"Source samples declare different expression kinds: {sorted(kinds)}")

    shared = per_sample[0][1]
    for _, genes, _ in per_sample[1:]:
        shared = np.intersect1d(shared, genes)
    if shared.size == 0:
        raise ValueError("No gene key is shared by every source sample of this patient")

    blocks, statuses = [], []
    for matrix, genes, status in per_sample:
        order = pd.Index(genes).get_indexer(shared)
        blocks.append(matrix[:, order])
        statuses.append(status)
    return (sparse.vstack(blocks).tocsr(), shared,
            np.concatenate(statuses), sorted(kinds)[0])


def wilcoxon_table(
    expression: sparse.csr_matrix, genes: np.ndarray, status: np.ndarray
) -> pd.DataFrame:
    """Rejected against retained, one patient, Scanpy's Wilcoxon.

    Matches the call in ``11_run_robust_target_deg.py`` -- BH correction, tie
    correction, expression fractions reported -- so a per-patient effect from
    the primary side is comparable with the target-side analysis rather than
    being a second convention.
    """
    try:
        import anndata as ad
        import scanpy as sc
    except (ImportError, AttributeError) as error:
        raise RuntimeError("scanpy>=1.10,<2 is required for per-patient DEG") from error

    analysis = ad.AnnData(
        X=expression,
        obs=pd.DataFrame({"confidence_status": pd.Categorical(status)},
                         index=[f"cell_{index}" for index in range(len(status))]),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    sc.tl.rank_genes_groups(
        analysis,
        groupby="confidence_status",
        groups=["rejected"],
        reference="retained",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        tie_correct=True,
        pts=True,
        n_genes=analysis.n_vars,
        use_raw=False,
    )
    return sc.get.rank_genes_groups_df(analysis, group="rejected").rename(columns={
        "names": "gene", "scores": "wilcoxon_score",
        "logfoldchanges": "log2_fold_change", "pvals": "p_value",
        "pvals_adj": "fdr", "pct_nz_group": "fraction_rejected",
        "pct_nz_reference": "fraction_retained",
    })


def main() -> None:
    args = parse_args()
    directories = patient_directories(args.four_state_root)
    if args.print_patient_count:
        print(len(directories))
        return
    if not 0 <= args.index < len(directories):
        raise IndexError(
            f"--index {args.index} outside 0..{len(directories) - 1}"
        )
    source = directories[args.index]
    patient = source.name.split("_", 1)[-1]
    output = args.output_root / source.name
    if args.skip_completed and (output / "PER_PATIENT_DEG_READY").is_file():
        print(f"{patient}: already complete", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)

    classification_path = source / "four_state_cell_classification.csv.gz"
    selected_path = source / "selected_pairs.csv"
    for path in (classification_path, selected_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is absent. Method B reads method A's classification "
                f"rather than re-resolving the lesion, so method A must have "
                f"completed for this patient."
            )
    classification = pd.read_csv(classification_path)
    selected = pd.read_csv(selected_path)
    manifest = pd.read_csv(args.manifest_csv)

    primary = classification[classification["side"].astype(str) == "primary"]
    excluded = primary[~primary["consensus_status"].isin(STATUSES)]
    wanted = primary[primary["consensus_status"].isin(STATUSES)].copy()
    counts = {status: int((wanted["consensus_status"] == status).sum())
              for status in STATUSES}

    report: dict[str, object] = {
        "patient_id": patient,
        "patient_index": args.index,
        "four_state_root": str(args.four_state_root),
        "manifest_csv": str(args.manifest_csv),
        "counts_source": "original matrices, not the equalised ones",
        "selected_pair_n": int(len(selected)),
        "selected_pairs": selected["pair_id"].astype(str).tolist(),
        "primary_classified_n": int(len(primary)),
        "status_n": counts,
        "excluded_status_n": excluded["consensus_status"].value_counts().to_dict(),
        "minimum_cells_per_status": args.minimum_cells_per_status,
    }

    short = [status for status in STATUSES
             if counts[status] < args.minimum_cells_per_status]
    if short:
        report["skipped"] = True
        report["skip_reason"] = (
            f"below the floor of {args.minimum_cells_per_status} in: "
            + ", ".join(f"{status} ({counts[status]})" for status in short)
        )
        (output / "diagnostics.json").write_text(json.dumps(report, indent=2),
                                                 encoding="utf-8")
        (output / "PER_PATIENT_DEG_SKIPPED").write_text(
            str(report["skip_reason"]) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return

    expression, genes, status, kind = load_primary_cells(manifest, selected, wanted)
    normalised, provenance = normalize_expression(expression, kind, args.target_sum)
    table = wilcoxon_table(normalised, genes, status)
    table.insert(0, "patient_id", patient)
    table["retained_n"] = counts["retained"]
    table["rejected_n"] = counts["rejected"]
    table.to_csv(output / "per_patient_primary_deg.csv.gz", index=False,
                 compression="gzip")

    report["skipped"] = False
    report["expression_kind"] = kind
    report["normalisation"] = provenance
    report["gene_n"] = int(len(genes))
    report["cell_n"] = int(expression.shape[0])
    # Reported rather than applied: the combination step decides what counts as
    # significant, because the thresholds belong to the prespecification and not
    # to one patient's fit.
    report["fdr_below_005_n"] = int((table["fdr"] < 0.05).sum())
    report["fdr_below_005_abs_log2fc_above_1_n"] = int(
        ((table["fdr"] < 0.05) & (table["log2_fold_change"].abs() >= 1.0)).sum()
    )
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
    (output / "PER_PATIENT_DEG_READY").write_text("ready\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
