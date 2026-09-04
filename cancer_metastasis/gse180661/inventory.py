"""Create patient/site availability and cell-type composition figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cancer_metastasis.common import load_exact_side, read_file_record


def annotation_values(data) -> np.ndarray:
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            return data.obs[column].astype(str).to_numpy()
    return np.repeat("unannotated", data.n_obs)


def role(record: dict) -> str:
    value = str(record.get("site_class", "")).lower()
    if "primary" in value:
        return "primary"
    if any(word in value for word in ("metasta", "non-primary", "non_primary")):
        return "metastasis"
    if record["source_pair_ids"] and not record["target_pair_ids"]:
        return "primary"
    if record["target_pair_ids"] and not record["source_pair_ids"]:
        return "metastasis"
    return "unresolved"


def sample_inventory(converted_root: Path, malignant_annotation: str,
                     dataset: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    pattern = f"{dataset}/*/*.h5ad" if dataset else "*/*/*.h5ad"
    records = [read_file_record(path) for path in sorted(converted_root.glob(pattern))]
    groups: dict[tuple[str, str, str, str], list[str]] = {}
    for record in records:
        sample_role = role(record)
        for sample in record["samples"]:
            key = (record["dataset_id"], record["patient_id"], str(sample), sample_role)
            groups.setdefault(key, []).append(record["path"])
    sample_rows = []
    composition_rows = []
    for (dataset, patient, sample, sample_role), paths in sorted(groups.items()):
        data = load_exact_side(sorted(paths), sample)
        annotations = annotation_values(data)
        counts = pd.Series(annotations).value_counts()
        malignant_n = int(np.sum(annotations == malignant_annotation))
        sample_rows.append({
            "dataset_id": dataset, "patient_id": patient, "sample": sample,
            "anatomical_site": sample, "sample_role": sample_role,
            "total_cell_n": int(data.n_obs), "malignant_cell_n": malignant_n,
            "nonmalignant_cell_n": int(data.n_obs - malignant_n),
            "source_file_n": len(paths), "source_h5ads_json": json.dumps(paths),
        })
        for annotation, count in counts.items():
            composition_rows.append({
                "dataset_id": dataset, "patient_id": patient, "sample": sample,
                "anatomical_site": sample, "sample_role": sample_role,
                "cell_type": str(annotation), "cell_n": int(count),
                "cell_fraction": float(count / data.n_obs),
                "total_cell_n": int(data.n_obs),
            })
    return pd.DataFrame(sample_rows), pd.DataFrame(composition_rows)


def save(figure, output: Path, stem: str):
    figure.tight_layout()
    figure.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def availability_figure(samples: pd.DataFrame, minimum_malignant: int, output: Path):
    patients = sorted(samples["patient_id"].unique())
    sites = sorted(samples["anatomical_site"].unique())
    patient_y = {value: index for index, value in enumerate(patients)}
    site_x = {value: index for index, value in enumerate(sites)}
    figure, axis = plt.subplots(figsize=(max(12, 0.55 * len(sites) + 5),
                                         max(8, 0.3 * len(patients) + 2)))
    colors = {"primary": "#2C7BB6", "metastasis": "#D7191C", "unresolved": "#888888"}
    maximum = max(float(samples["total_cell_n"].max()), 1.0)
    for sample_role, table in samples.groupby("sample_role", sort=True):
        sizes = 30 + 520 * np.sqrt(table["total_cell_n"].to_numpy() / maximum)
        axis.scatter(table["anatomical_site"].map(site_x), table["patient_id"].map(patient_y),
                     s=sizes, c=colors.get(sample_role, "#888888"), alpha=0.72,
                     edgecolors="white", linewidths=0.5, label=sample_role.title())
    insufficient = samples[samples["malignant_cell_n"].lt(minimum_malignant)]
    axis.scatter(insufficient["anatomical_site"].map(site_x),
                 insufficient["patient_id"].map(patient_y), marker="x", s=42,
                 c="black", linewidths=0.8,
                 label=f"Malignant cells < {minimum_malignant}")
    axis.set_xticks(range(len(sites)), sites, rotation=45, ha="right")
    axis.set_yticks(range(len(patients)), patients)
    axis.invert_yaxis()
    axis.set_xlabel("Anatomical sample/site")
    axis.set_ylabel("Patient")
    axis.set_title("Patient-resolved primary and metastatic sample availability")
    axis.grid(color="#EEEEEE", linewidth=0.5)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(figure, output, "01_patient_site_availability")


def composition_colors(cell_types: list[str]):
    cmap = plt.get_cmap("tab20")
    return {value: cmap(index % 20) for index, value in enumerate(cell_types)}


def patient_composition_figure(composition: pd.DataFrame, output: Path,
                               colors: dict[str, tuple]):
    grouped = composition.groupby(
        ["patient_id", "sample_role", "cell_type"], as_index=False
    )["cell_n"].sum()
    patients = sorted(grouped["patient_id"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(16, max(8, 0.3 * len(patients) + 2)),
                                sharey=True)
    for axis, sample_role in zip(axes, ("primary", "metastasis")):
        table = grouped[grouped["sample_role"].eq(sample_role)]
        pivot = table.pivot_table(index="patient_id", columns="cell_type", values="cell_n",
                                  fill_value=0).reindex(patients, fill_value=0)
        left = np.zeros(len(pivot))
        for cell_type in colors:
            values = pivot.get(cell_type, pd.Series(0, index=pivot.index)).to_numpy()
            axis.barh(np.arange(len(pivot)), values, left=left, color=colors[cell_type],
                      height=0.78, label=cell_type)
            left += values
        axis.set_title(sample_role.title())
        axis.set_xlabel("Cell number")
        axis.set_yticks(np.arange(len(patients)), patients)
        axis.invert_yaxis()
    axes[0].set_ylabel("Patient")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="center left",
                  bbox_to_anchor=(1.0, 0.5), fontsize=8)
    figure.suptitle("Patient-level author-annotated cell-type composition")
    save(figure, output, "02a_patient_level_cell_type_counts")


def site_composition_figure(composition: pd.DataFrame, output: Path,
                            colors: dict[str, tuple]):
    keys = ["patient_id", "sample_role", "sample", "cell_type"]
    grouped = composition.groupby(keys, as_index=False).agg(
        cell_n=("cell_n", "sum"), total_cell_n=("total_cell_n", "max")
    )
    grouped["label"] = (
        grouped["patient_id"].astype(str) + " | " + grouped["sample_role"].astype(str)
        + " | " + grouped["sample"].astype(str)
    )
    order = grouped[["patient_id", "sample_role", "sample", "label"]].drop_duplicates().sort_values(
        ["patient_id", "sample_role", "sample"], kind="stable"
    )["label"].tolist()
    pivot = grouped.pivot_table(index="label", columns="cell_type", values="cell_n",
                                fill_value=0).reindex(order)
    fractions = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    figure, axis = plt.subplots(figsize=(14, max(10, 0.25 * len(order) + 2)))
    left = np.zeros(len(fractions))
    for cell_type in colors:
        values = fractions.get(cell_type, pd.Series(0, index=fractions.index)).to_numpy()
        axis.barh(np.arange(len(fractions)), values, left=left, color=colors[cell_type],
                  height=0.8, label=cell_type)
        left += values
    totals = pivot.sum(axis=1).astype(int)
    for index, total in enumerate(totals):
        axis.text(1.01, index, f"n={total:,}", va="center", fontsize=6)
    axis.set_yticks(np.arange(len(order)), order, fontsize=6)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.12)
    axis.set_xlabel("Within-sample cell fraction")
    axis.set_ylabel("Patient | compartment | sample/site")
    axis.set_title("Patient-by-site author-annotated cell-type composition")
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    save(figure, output, "02b_patient_site_level_cell_type_fractions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("converted_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--dataset", default="GSE180661")
    parser.add_argument("--malignant-annotation", default="Ovarian.cancer.cell")
    parser.add_argument("--minimum-malignant-cells", type=int, default=20)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    root = args.converted_root / args.dataset
    if not root.is_dir():
        raise FileNotFoundError(root)
    samples, composition = sample_inventory(
        root.parent, args.malignant_annotation, dataset=args.dataset
    )
    samples = samples[samples["dataset_id"].eq(args.dataset)].copy()
    composition = composition[composition["dataset_id"].eq(args.dataset)].copy()
    if samples.empty:
        raise RuntimeError(f"No samples found for {args.dataset}")
    samples["malignant_evaluable"] = samples["malignant_cell_n"].ge(
        args.minimum_malignant_cells
    )
    samples.to_csv(args.output_root / "sample_inventory.csv", index=False)
    composition.to_csv(args.output_root / "sample_cell_type_composition.csv.gz",
                       index=False, compression="gzip")
    colors = composition_colors(sorted(composition["cell_type"].unique()))
    availability_figure(samples, args.minimum_malignant_cells, args.output_root)
    patient_composition_figure(composition, args.output_root, colors)
    site_composition_figure(composition, args.output_root, colors)
    report = {
        "dataset_id": args.dataset, "patient_n": int(samples["patient_id"].nunique()),
        "sample_n": len(samples), "total_cell_n": int(samples["total_cell_n"].sum()),
        "malignant_evaluable_sample_n": int(samples["malignant_evaluable"].sum()),
        "cell_type_n": int(composition["cell_type"].nunique()),
    }
    (args.output_root / "inventory_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
