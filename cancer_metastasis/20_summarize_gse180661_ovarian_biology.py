"""Create a compact biological summary of the GSE180661 ovarian-cancer analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_figure(fig, output_root: Path, stem: str) -> None:
    fig.savefig(output_root / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def winner_cell_summary(winners: pd.DataFrame, cell_root: Path) -> pd.DataFrame:
    rows = []
    for row in winners.itertuples(index=False):
        path = (
            cell_root
            / row.pair_id
            / "scope_malignant"
            / "budget_0.95"
            / "cell_confidence.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        cells = pd.read_csv(path)
        cells = cells[cells["method"].eq("M4-E") & cells["side"].eq("target")]
        if cells.empty:
            raise RuntimeError(f"No M4-E target cells in {path}")
        rows.append({
            "dataset_id": row.dataset_id,
            "patient_id": row.patient_id,
            "target_sample": row.target_sample,
            "winning_source_sample": row.source_sample,
            "target_malignant_cell_n": len(cells),
            "target_rejection_rate": float(cells["rejected"].mean()),
            "target_mean_rejection_score": float(
                cells["normalized_rejection_score"].mean()
            ),
            "target_median_rejection_score": float(
                cells["normalized_rejection_score"].median()
            ),
            "target_q25_rejection_score": float(
                cells["normalized_rejection_score"].quantile(0.25)
            ),
            "target_q75_rejection_score": float(
                cells["normalized_rejection_score"].quantile(0.75)
            ),
        })
    return pd.DataFrame(rows)


def make_figures(groups: pd.DataFrame, output_root: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})

    source_counts = groups["winning_source_sample"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(source_counts.index, source_counts.values, color="#356a9a")
    ax.set(
        xlabel="Metastatic target groups won",
        ylabel="Candidate primary sampling site",
        title="M4-E primary-site compatibility winners",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_root, "01_primary_origin_winner_frequency")

    ordered_sites = (
        groups.groupby("target_sample")["target_rejection_rate"]
        .median()
        .sort_values()
        .index.tolist()
    )
    values = [
        100 * groups.loc[groups["target_sample"].eq(site), "target_rejection_rate"]
        for site in ordered_sites
    ]
    fig, ax = plt.subplots(figsize=(8.2, max(4.8, 0.52 * len(ordered_sites))))
    positions = np.arange(1, len(ordered_sites) + 1)
    ax.boxplot(values, positions=positions, vert=False, showfliers=False)
    for position, site, site_values in zip(positions, ordered_sites, values):
        jitter = np.linspace(-0.09, 0.09, len(site_values)) if len(site_values) > 1 else [0]
        ax.scatter(site_values, position + jitter, s=28, color="#c44e52", zorder=3)
    ax.set(
        yticks=positions,
        yticklabels=ordered_sites,
        xlabel="M4-E rejected target malignant cells (%)",
        ylabel="Metastatic sampling site",
        title="Primary-incompatible malignant state by metastatic site",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_root, "02_target_rejection_by_metastatic_site")

    stability = pd.Series({
        "Exact sample": groups["recommended_range_exact_robust"].mean(),
        "Left/right collapsed": groups["recommended_range_laterality_robust"].mean(),
        "Broad compartment": groups["recommended_range_compartment_robust"].mean(),
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(stability.index, 100 * stability.values, color=["#8172b2", "#55a868", "#4c72b0"])
    for bar, value in zip(bars, stability.values):
        ax.text(bar.get_x() + bar.get_width() / 2, 100 * value + 1.5, f"{100*value:.1f}%", ha="center")
    ax.set(
        ylim=(0, 108),
        ylabel="Winner agreement after source-cap change (%)",
        title="Biological resolution of origin robustness",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_root, "03_origin_robustness_by_resolution")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    robust = groups["recommended_range_exact_robust"].astype(bool)
    for state, label, color in ((True, "Exact winner stable", "#4c72b0"), (False, "Exact winner changed", "#dd8452")):
        use = groups[robust.eq(state)]
        ax.scatter(
            100 * use["target_rejection_rate"],
            100 * use["baseline_margin_rate"],
            s=35 + 0.015 * use["target_malignant_cell_n"],
            alpha=0.75,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set(
        xlabel="Rejected target malignant cells (%)",
        ylabel="Winner margin over second candidate (percentage points)",
        title="Target novelty and confidence in the selected primary",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_root, "04_target_rejection_vs_origin_margin")


def summarize(
    biological_root: Path,
    cell_root: Path,
    sensitivity_root: Path,
    output_root: Path,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=False)
    winners = pd.read_csv(biological_root / "m4e_origin_group_winners.csv")
    robustness = pd.read_csv(sensitivity_root / "origin_group_robustness.csv")
    cells = winner_cell_summary(winners, cell_root)
    groups = cells.merge(
        robustness,
        on=["dataset_id", "patient_id", "target_sample"],
        how="left",
        validate="one_to_one",
    )
    groups.to_csv(output_root / "target_group_biological_summary.csv", index=False)

    source_frequency = (
        groups.groupby("winning_source_sample", as_index=False)
        .agg(target_group_n=("target_sample", "size"), patient_n=("patient_id", "nunique"))
        .sort_values("target_group_n", ascending=False)
    )
    source_frequency.to_csv(output_root / "primary_origin_winner_frequency.csv", index=False)

    target_site = (
        groups.groupby("target_sample", as_index=False)
        .agg(
            target_group_n=("patient_id", "size"),
            patient_n=("patient_id", "nunique"),
            malignant_cell_n=("target_malignant_cell_n", "sum"),
            median_target_rejection_rate=("target_rejection_rate", "median"),
            mean_target_rejection_rate=("target_rejection_rate", "mean"),
            median_target_rejection_score=("target_median_rejection_score", "median"),
        )
        .sort_values(["patient_n", "median_target_rejection_rate"], ascending=False)
    )
    target_site.to_csv(output_root / "metastatic_site_rejection_summary.csv", index=False)

    exact_changed = groups[~groups["recommended_range_exact_robust"].astype(bool)].copy()
    exact_changed.to_csv(output_root / "source_cap_changed_exact_winners.csv", index=False)

    report = {
        "dataset_id": "GSE180661",
        "analysis_scope": "author-annotated Ovarian.cancer.cell only",
        "origin_inference_method": "M4-E",
        "patient_n": int(groups["patient_id"].nunique()),
        "multi_primary_target_group_n": len(groups),
        "target_malignant_cell_n_across_unique_patient_targets": int(
            groups["target_malignant_cell_n"].sum()
        ),
        "median_group_target_rejection_rate": float(groups["target_rejection_rate"].median()),
        "recommended_source_cap_sensitivity": "0.95 versus 0.90, target cap fixed at 0.95",
        "exact_origin_winner_stable_n": int(groups["recommended_range_exact_robust"].sum()),
        "laterality_collapsed_winner_stable_n": int(
            groups["recommended_range_laterality_robust"].sum()
        ),
        "anatomical_compartment_winner_stable_n": int(
            groups["recommended_range_compartment_robust"].sum()
        ),
        "interpretation": (
            "M4-E ranks primary malignant expression-state compatibility for each metastatic "
            "sample. Rejected target cells represent states not well explained by the selected "
            "primary under the fitted model. They are not a measured fraction of metastatic "
            "lineages, and origin calls require CNV, mutation, or lineage validation."
        ),
    }
    (output_root / "analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    make_figures(groups, output_root)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("biological_root", type=Path)
    parser.add_argument("cell_root", type=Path)
    parser.add_argument("sensitivity_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    summarize(
        args.biological_root,
        args.cell_root,
        args.sensitivity_root,
        args.output_root,
    )


if __name__ == "__main__":
    main()
