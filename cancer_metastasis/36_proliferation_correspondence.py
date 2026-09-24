"""Is the gate matching proliferation to proliferation, or just picking cyclers?

The paired differential expression on GSE180661 returned 90 genes at two-fold
and every one of them higher in the retained cells, with the top twelve a pure
mitotic signature -- KIF20A, PLK1, PIF1, DLGAP5, CDCA3, GAS2L3, FAM83D, KIF2C,
DEPDC1, GTSE1, CCNA2, CKAP2L. Retained primary cells are cycling and rejected
ones are not, in 26 of 27 patients.

Two readings survive that, and they differ in what the gate is:

**Correspondence.** The metastasis is proliferative, so the primary cells
closest to it are the proliferative ones. The gate is doing its job and
proliferation is the biology that job surfaces.

**A cycle detector.** Cycling cells carry more RNA and detect more genes, the
leading axis tracks the detected-gene count at rho 0.645 on this data, and the
gate retains whatever sits at that end of the axis regardless of the partner.
The differential expression then recovers cell-cycle genes because the gate
selected cycling cells, and the metastasis had nothing to do with it.

**What separates them, and it is a single number.** Under correspondence the
retained-minus-rejected proliferation gap should track the metastasis's own
proliferation: a pair whose metastasis cycles hard should show a large gap, and
a pair whose metastasis is quiescent should show a small one or none. Under a
cycle detector the gap is a property of the primary alone and is flat against
the partner.

The gap is near-universally positive, which is suggestive but not decisive on
its own: ovarian metastases may simply all be proliferative. The correlation is
the test.

The score is the mean of log1p CPM over a fixed G2M set, computed on the
**original** count matrices, the same ones the differential expression read.
It is deliberately not a rank-based or regressed quantity: the question is
whether two crude proliferation levels move together, and a score that shared
machinery with the representation would be answering with the representation.

Usage:

    python cancer_metastasis/36_proliferation_correspondence.py \\
        FOUR_STATE_ROOT TRIMMED_MANIFEST_CSV OUTPUT_ROOT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

from common import expression_matrix, gene_keys, load_exact_side

# Tirosh et al. G2M, the standard set, trimmed to symbols these objects carry.
# Fixed rather than derived from the result: scoring with the genes the
# contrast just selected would be circular.
G2M = (
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "SMC4",
    "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "ANP32E", "TUBB4B", "GTSE1",
    "KIF11", "HJURP", "CDCA3", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA",
    "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3",
    "CBX5", "CENPA",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("four_state_root", type=Path)
    parser.add_argument("manifest_csv", type=Path,
                        help="The manifest the pseudobulk was built from")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--minimum-cells", type=int, default=10,
                        help="A state below this is scored as missing rather "
                             "than as a number from a handful of cells")
    return parser.parse_args()


def side_paths(manifest: pd.DataFrame, pair_id: str, side: str) -> list[str]:
    rows = manifest[manifest["pair_id"].astype(str) == str(pair_id)]
    if len(rows) != 1:
        raise KeyError(f"pair_id {pair_id!r} matched {len(rows)} manifest rows")
    row = rows.iloc[0]
    column = f"{side}_h5ads_json"
    if column in row and pd.notna(row[column]):
        return [str(value) for value in json.loads(str(row[column]))]
    return [str(row[f"{side}_h5ad"])]


def score(paths: list[str], sample: str, ids: list[str]) -> tuple[float, int]:
    """Mean log1p CPM over the G2M set for the named cells."""
    data = load_exact_side(paths, sample)
    lookup = {str(value): index for index, value in enumerate(data.obs_names)}
    rows = np.asarray([lookup[value] for value in ids if value in lookup],
                      dtype=np.int64)
    if rows.size == 0:
        return float("nan"), 0
    matrix = sparse.csr_matrix(expression_matrix(data))[rows]
    keys = np.asarray([str(key) for key in gene_keys(data)])
    columns = np.flatnonzero(np.isin(keys, G2M))
    if columns.size == 0:
        return float("nan"), int(rows.size)
    library = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
    library[library <= 0] = np.nan
    cpm = np.asarray(matrix[:, columns].todense(), dtype=np.float64)
    cpm = cpm / library[:, None] * 1e6
    return float(np.nanmean(np.log1p(cpm))), int(rows.size)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for directory in sorted(
            path for path in (args.four_state_root / "patients").iterdir()
            if path.is_dir()):
        classification_path = directory / "four_state_cell_classification.csv.gz"
        selected_path = directory / "selected_pairs.csv"
        if not (classification_path.is_file() and selected_path.is_file()):
            continue
        patient = directory.name.split("_", 1)[-1]
        classification = pd.read_csv(classification_path)
        selected = pd.read_csv(selected_path)
        record: dict[str, object] = {"patient_id": patient}
        for side, statuses, label in (
                ("primary", ("retained",), "primary_retained"),
                ("primary", ("rejected",), "primary_rejected"),
                # Every malignant metastatic cell, not only the matched ones:
                # the question is how proliferative that lesion is, which is a
                # property of the lesion rather than of the matching.
                ("metastasis", ("retained", "rejected"), "metastasis_all")):
            block = classification[
                classification["side"].astype(str).eq(side)
                & classification["consensus_status"].isin(statuses)]
            values, cells = [], 0
            for sample, part in block.groupby("sample", sort=True):
                column = "source" if side == "primary" else "target"
                pairs = selected[selected[f"{column}_sample"].astype(str)
                                 == str(sample)]
                if pairs.empty:
                    continue
                value, n = score(
                    side_paths(manifest, pairs.iloc[0]["pair_id"], column),
                    str(sample), part["observation_id"].astype(str).tolist())
                if n:
                    values.append(value * n)
                    cells += n
            record[label] = (sum(values) / cells
                             if cells >= args.minimum_cells else float("nan"))
            record[f"{label}_n"] = cells
        rows.append(record)
        print(f"{patient}: " + "  ".join(
            f"{key}={record[key]:.3f}" for key in
            ("primary_retained", "primary_rejected", "metastasis_all")
            if isinstance(record.get(key), float) and record[key] == record[key]),
            flush=True)

    table = pd.DataFrame(rows)
    table["gap"] = table["primary_retained"] - table["primary_rejected"]
    table.to_csv(args.output_root / "proliferation_correspondence.csv",
                 index=False)

    usable = table.dropna(subset=["gap", "metastasis_all"])
    report: dict[str, object] = {
        "four_state_root": str(args.four_state_root),
        "score": "mean log1p CPM over the Tirosh G2M set, original counts",
        "patient_n": int(len(table)),
        "patients_with_metastasis_score": int(len(usable)),
        "gap_median": float(table["gap"].median()),
        "gap_positive_n": int((table["gap"] > 0).sum()),
        "gap_negative_n": int((table["gap"] < 0).sum()),
    }
    if len(usable) >= 6:
        rho, p_value = stats.spearmanr(usable["gap"], usable["metastasis_all"])
        report["spearman_gap_vs_metastasis"] = float(rho)
        report["spearman_p_value"] = float(p_value)
        # Stated before the number is read, so the reading is not chosen by it.
        report["reading"] = {
            "correspondence": (
                "a clearly positive correlation: the gap follows how "
                "proliferative the partner lesion is, so the gate is matching "
                "proliferation to proliferation"),
            "cycle_detector": (
                "a correlation near zero with the gap still positive nearly "
                "everywhere: the gap is a property of the primary alone, the "
                "partner does not enter it, and the differential expression "
                "recovers cell-cycle genes because the gate selected cycling "
                "cells"),
            "caveat": (
                "neither reading makes the gate a test of metastatic "
                "competence; the first makes proliferation the biology the "
                "filter surfaces, the second makes it an artefact of the "
                "detected-gene axis the preprocessing does not remove"),
        }
    (args.output_root / "diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
