"""Validate primary-cell OT gates with metastasis-derived pyUCell signatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon

from cancer_metastasis.common import cell_qc_table, load_exact_side
from cancer_metastasis.gse180661.primary_pseudobulk import (
    collapsed_raw_counts,
    malignant_annotation_mask,
    one_result_directory,
    paths_for,
)


DATASET_LABELS = {
    "GSE180661": ["Ovarian.cancer.cell"],
    "GSE181919": ["Malignant.cells"],
    "GSE225857": [
        "Tu01_AREG", "Tu02_DEFA5", "Tu03_SRRM2", "Tu04_RGMB",
        "Tu05_PCNA", "Tu06_NKD1", "Tu07_MKI67", "Tu08_GNG13",
        "Tu09_MUC2", "Tu10_COL3A1", "Tu11_PLA2G2A",
    ],
}


def confidence_path(ot_root: Path, pair_id: str, budget_tag: str) -> Path:
    return one_result_directory(ot_root, pair_id, budget_tag) / "cell_confidence.csv"


def analyzed_ids(
    manifest: pd.DataFrame, ot_root: Path, budget_tag: str, side: str
) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = {}
    sample_column = f"{side}_sample"
    for row in manifest.to_dict("records"):
        table = pd.read_csv(confidence_path(ot_root, str(row["pair_id"]), budget_tag))
        table = table.loc[table["method"].eq("M4-E") & table["side"].eq(side)]
        key = (str(row["patient_id"]), str(row[sample_column]))
        result.setdefault(key, set()).update(table["observation_id"].astype(str))
    return result


def load_analyzed_side(
    manifest: pd.DataFrame,
    ot_root: Path,
    budget_tag: str,
    side: str,
    labels: list[str],
) -> ad.AnnData:
    selected_ids = analyzed_ids(manifest, ot_root, budget_tag, side)
    sample_column = f"{side}_sample"
    records = manifest.drop_duplicates(["patient_id", sample_column])
    pieces = []
    for _, row in records.iterrows():
        patient = str(row["patient_id"])
        sample = str(row[sample_column])
        data = load_exact_side(paths_for(row, side), sample)
        data.obs_names = data.obs_names.astype(str)
        keep = malignant_annotation_mask(data, labels)
        data = data[keep].copy()
        wanted = selected_ids[(patient, sample)]
        keep = data.obs_names.astype(str).isin(wanted)
        data = data[keep].copy()
        if data.n_obs != len(wanted):
            missing = len(wanted) - data.n_obs
            raise RuntimeError(f"{patient}/{sample}/{side}: {missing} analyzed cells missing")
        data.obs["observation_id"] = data.obs_names.astype(str)
        data.obs["patient_id"] = patient
        data.obs["sample_id_exact"] = sample
        data.obs["side"] = "primary" if side == "source" else "metastasis"
        data.obs_names = pd.Index(
            [f"{patient}::{sample}::{value}" for value in data.obs_names], dtype=str
        )
        pieces.append(data)
    if not pieces:
        raise RuntimeError(f"No analyzed malignant cells for side={side}")
    return ad.concat(pieces, join="inner", merge="same", index_unique=None)


def collapsed_anndata(source: ad.AnnData, target: ad.AnnData) -> ad.AnnData:
    combined = ad.concat([source, target], join="inner", merge="same", index_unique=None)
    matrix, genes, _ = collapsed_raw_counts(combined, set())
    return ad.AnnData(
        X=sparse.csr_matrix(matrix),
        obs=combined.obs.copy(),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )


def patient_pseudobulk(data: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metadata = []
    for (patient, side), positions in data.obs.groupby(
        ["patient_id", "side"], sort=True, observed=True
    ).indices.items():
        values = np.asarray(data.X[positions].sum(axis=0)).ravel().astype(np.int64)
        sample_id = f"{patient}__{side}"
        rows.append(pd.Series(values, index=data.var_names, name=sample_id))
        metadata.append({
            "sample_id": sample_id,
            "patient_id": str(patient),
            "side": str(side),
            "cell_n": int(len(positions)),
            "library_size": int(values.sum()),
        })
    counts = pd.DataFrame(rows).fillna(0).astype(np.int64)
    meta = pd.DataFrame(metadata).set_index("sample_id")
    complete = meta.groupby("patient_id")["side"].agg(
        lambda values: set(values) == {"primary", "metastasis"}
    )
    keep_patients = set(complete.index[complete])
    keep = meta["patient_id"].isin(keep_patients)
    return counts.loc[keep], meta.loc[keep]


def run_paired_deg(
    counts: pd.DataFrame, metadata: pd.DataFrame, n_cpus: int
) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    keep = counts.sum(axis=0).ge(10)
    counts = counts.loc[:, keep]
    design = metadata.copy()
    design["patient_id"] = design["patient_id"].astype("category")
    design["side"] = pd.Categorical(
        design["side"], categories=["primary", "metastasis"]
    )
    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=counts,
        metadata=design,
        design="~patient_id + side",
        refit_cooks=True,
        inference=inference,
        quiet=False,
    )
    dds.deseq2()
    statistics = DeseqStats(
        dds,
        contrast=["side", "metastasis", "primary"],
        alpha=0.05,
        cooks_filter=True,
        independent_filter=True,
        inference=inference,
        quiet=False,
    )
    statistics.summary()
    result = statistics.results_df.reset_index().rename(columns={
        "index": "gene", "baseMean": "base_mean",
        "log2FoldChange": "log2_fold_change",
        "lfcSE": "lfc_standard_error", "stat": "wald_statistic",
        "pvalue": "p_value", "padj": "fdr",
    })
    if "gene" not in result:
        result = result.rename(columns={result.columns[0]: "gene"})
    return result.sort_values(
        ["wald_statistic", "gene"], ascending=[False, True], kind="stable"
    )


def make_signatures(deg: pd.DataFrame, sizes: list[int]) -> dict[str, list[str]]:
    ranked = deg.loc[
        deg["log2_fold_change"].gt(0)
        & deg["wald_statistic"].replace([np.inf, -np.inf], np.nan).notna()
    ].sort_values(["wald_statistic", "gene"], ascending=[False, True])
    if len(ranked) < max(sizes):
        raise RuntimeError(f"Only {len(ranked)} metastasis-enriched genes available")
    return {f"MetastasisTop{size}": ranked.head(size)["gene"].tolist() for size in sizes}


def gate_occurrences(
    manifest: pd.DataFrame,
    ot_root: Path,
    budget_tag: str,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    lookup_columns = ["patient_id", "sample_id_exact", "observation_id"]
    rows = []
    for row in manifest.to_dict("records"):
        confidence = pd.read_csv(
            confidence_path(ot_root, str(row["pair_id"]), budget_tag)
        )
        confidence = confidence.loc[
            confidence["method"].eq("M4-E") & confidence["side"].eq("source")
        ].copy()
        confidence["patient_id"] = str(row["patient_id"])
        confidence["sample_id_exact"] = str(row["source_sample"])
        confidence["pair_id"] = str(row["pair_id"])
        confidence["metastatic_sample"] = str(row["target_sample"])
        confidence["gate"] = np.where(
            confidence["retained"].astype(bool), "retained", "rejected"
        )
        rows.append(confidence)
    long = pd.concat(rows, ignore_index=True)
    return long.merge(scores, on=lookup_columns, how="left", validate="many_to_one")


def summarize_scores(long: pd.DataFrame, score_columns: list[str]):
    pair = long.groupby(
        ["patient_id", "pair_id", "gate"], observed=True
    )[score_columns].median().reset_index()
    patient = pair.groupby(
        ["patient_id", "gate"], observed=True
    )[score_columns].median().reset_index()
    tests = []
    for level, table, key in (
        ("pair", pair, "pair_id"), ("patient", patient, "patient_id")
    ):
        for score in score_columns:
            wide = table.pivot(index=key, columns="gate", values=score).dropna()
            delta = wide["retained"] - wide["rejected"]
            try:
                p_value = float(wilcoxon(delta).pvalue) if len(delta) else np.nan
            except ValueError:
                p_value = 1.0
            tests.append({
                "level": level,
                "signature": score.removesuffix("_UCell"),
                "unit_n": int(len(delta)),
                "retained_median": float(wide["retained"].median()),
                "rejected_median": float(wide["rejected"].median()),
                "median_retained_minus_rejected": float(delta.median()),
                "paired_wilcoxon_p": p_value,
            })
    return pair, patient, pd.DataFrame(tests)


def violin_figure(
    table: pd.DataFrame,
    score_columns: list[str],
    dataset: str,
    unit: str,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, len(score_columns), figsize=(5 * len(score_columns), 5))
    axes = np.atleast_1d(axes)
    colors = ["#2b8cbe", "#d7301f"]
    for axis, score in zip(axes, score_columns):
        values = [
            table.loc[table["gate"].eq(gate), score].dropna().to_numpy()
            for gate in ("retained", "rejected")
        ]
        parts = axis.violinplot(values, positions=[1, 2], showmedians=True, widths=0.82)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.72)
        axis.set_xticks([1, 2], ["Retained", "Rejected"])
        axis.set_ylabel("UCell score")
        axis.set_title(score.removesuffix("_UCell").replace("MetastasisTop", "Top "))
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        f"{dataset}: metastasis-derived signatures in primary OT gate states\n{unit}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_only_root", type=Path)
    parser.add_argument("gse180661_manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--dataset", choices=sorted(DATASET_LABELS), required=True)
    parser.add_argument("--budget-tag", default="budget_source_0.85_target_0.00")
    parser.add_argument("--signature-sizes", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--n-cpus", type=int, default=16)
    parser.add_argument("--ucell-chunk-size", type=int, default=1000)
    args = parser.parse_args()

    try:
        import pyucell as uc
    except ImportError as error:
        raise RuntimeError("pyucell>=0.7,<0.8 is required") from error

    dataset = args.dataset
    destination = args.output_root / dataset
    destination.mkdir(parents=True, exist_ok=False)
    manifest_path = (
        args.gse180661_manifest if dataset == "GSE180661"
        else args.source_only_root / dataset / "manifest" / "pair_manifest_malignant_eligible.csv"
    )
    manifest = pd.read_csv(manifest_path)
    ot_root = args.source_only_root / dataset / "ot"
    labels = DATASET_LABELS[dataset]
    source = load_analyzed_side(manifest, ot_root, args.budget_tag, "source", labels)
    target = load_analyzed_side(manifest, ot_root, args.budget_tag, "target", labels)
    data = collapsed_anndata(source, target)

    counts, metadata = patient_pseudobulk(data)
    counts.to_csv(destination / "patient_pseudobulk_raw_counts.csv.gz", compression="gzip")
    metadata.to_csv(destination / "patient_pseudobulk_metadata.csv")
    deg = run_paired_deg(counts, metadata, args.n_cpus)
    deg.to_csv(destination / "metastasis_vs_primary_all_genes.csv.gz", index=False)

    signatures = make_signatures(deg, sorted(set(args.signature_sizes)))
    signature_rows = []
    for name, genes in signatures.items():
        subset = deg.set_index("gene").loc[genes].reset_index()
        subset.insert(0, "signature", name)
        subset.insert(1, "rank", np.arange(1, len(subset) + 1))
        signature_rows.append(subset)
        subset.to_csv(destination / f"{name}_genes.csv", index=False)
    pd.concat(signature_rows, ignore_index=True).to_csv(
        destination / "metastasis_signature_genes_long.csv", index=False
    )

    primary = data[data.obs["side"].eq("primary").to_numpy()].copy()
    uc.compute_ucell_scores(
        primary, signatures=signatures, chunk_size=args.ucell_chunk_size
    )
    score_columns = [f"{name}_UCell" for name in signatures]
    scores = primary.obs[
        ["patient_id", "sample_id_exact", "observation_id"] + score_columns
    ].reset_index(drop=True)
    scores.to_csv(destination / "primary_cell_ucell_scores.csv.gz", index=False)

    long = gate_occurrences(manifest, ot_root, args.budget_tag, scores)
    if long[score_columns].isna().any().any():
        raise RuntimeError("Some primary gate occurrences lack UCell scores")
    long.to_csv(destination / "primary_gate_occurrence_ucell_scores.csv.gz", index=False)
    pair, patient, tests = summarize_scores(long, score_columns)
    pair.to_csv(destination / "pair_median_ucell_scores.csv", index=False)
    patient.to_csv(destination / "patient_median_ucell_scores.csv", index=False)
    tests.to_csv(destination / "paired_gate_tests.csv", index=False)

    violin_figure(
        long, score_columns, dataset, "Cell occurrences", destination / "01_cell_ucell_violin"
    )
    violin_figure(
        patient, score_columns, dataset, "Patient-level medians",
        destination / "02_patient_median_ucell_violin",
    )
    report = {
        "dataset": dataset,
        "signature_discovery": "patient-paired metastasis malignant versus all primary malignant",
        "signature_ranking": "positive PyDESeq2 Wald statistic; OT gate not used",
        "signature_sizes": sorted(set(args.signature_sizes)),
        "ucell_engine": "pyUCell",
        "patient_n": int(metadata["patient_id"].nunique()),
        "pair_n": int(manifest["pair_id"].nunique()),
        "primary_unique_cell_n": int(primary.n_obs),
        "primary_gate_occurrence_n": int(len(long)),
        "inference_unit": "patient median across exact primary-metastasis pairs",
        "internal_validation_warning": "Signatures and gate validation use the same dataset.",
    }
    (destination / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (destination / "VALIDATION_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(tests.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
