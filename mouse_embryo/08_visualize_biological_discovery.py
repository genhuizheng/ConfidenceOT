"""Create biological-discovery figures from the full three-stage MOSTA run.

The figures answer four biological questions: which annotated populations are
candidate disappearances or emergences, whether those signals reproduce across
section pairs, where the rejected bins lie in embryo space, and where balanced
OT would force those bins to match.  Calibration validity is used as a filter;
it is not presented as a biological result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIR_PATTERN = re.compile(r"^(E\d+\.\d+)_(.+)_to_(E\d+\.\d+)_(.+)$")
CONFIDENCE_METHOD = "Calibrated | M4-R / UOT"
TRADITIONAL_METHOD = "Traditional Balanced OT"
STAGES = ("E9.5 → E10.5", "E10.5 → E11.5")
SOURCE_COLOR = "#d55e00"
TARGET_COLOR = "#0072b2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--consistency-threshold", type=float, default=0.25)
    parser.add_argument("--max-labels", type=int, default=14)
    parser.add_argument("--heldout-top-genes", type=int, default=12)
    parser.add_argument(
        "--skip-heldout-validation", action="store_true",
        help="Skip the independent held-out-gene panel (useful only for legacy smoke data).",
    )
    return parser.parse_args()


def pair_metadata(pair_id: str) -> dict[str, str]:
    match = PAIR_PATTERN.match(pair_id)
    if match is None:
        raise ValueError(f"Unrecognized pair id: {pair_id}")
    source_stage, source_sample, target_stage, target_sample = match.groups()
    return {
        "source_stage": source_stage,
        "source_sample": source_sample,
        "target_stage": target_stage,
        "target_sample": target_sample,
        "stage_transition": f"{source_stage} → {target_stage}",
    }


def save_figure(
    figure: plt.Figure,
    destination: Path,
    dpi: int,
    *,
    panels: list[tuple[str, plt.Axes]] | None = None,
) -> None:
    figure.savefig(destination.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    if panels:
        panel_root = destination.parent / "panels"
        panel_root.mkdir(parents=True, exist_ok=True)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for panel_name, axis in panels:
            extent = axis.get_tightbbox(renderer).transformed(figure.dpi_scale_trans.inverted())
            extent = extent.expanded(1.06, 1.10)
            panel_destination = panel_root / f"{destination.name}__{panel_name}"
            figure.savefig(panel_destination.with_suffix(".png"), dpi=dpi, bbox_inches=extent)
            figure.savefig(panel_destination.with_suffix(".pdf"), bbox_inches=extent)
    plt.close(figure)


def read_pair_tables(run_root: Path):
    certificates: list[dict[str, object]] = []
    collections: dict[str, list[pd.DataFrame]] = {
        "rejection": [], "transitions": [], "forced": [], "cells": [], "background": [],
    }
    pair_roots: dict[str, Path] = {}
    filenames = {
        "rejection": "population_rejection.csv",
        "transitions": "population_transitions.csv",
        "forced": "traditional_forced_matches.csv",
        "cells": "cell_confidence.csv",
        "background": "preanalysis_exclusions.csv",
    }
    for success in sorted(run_root.glob("pairs/*/analysis/SUCCESS")):
        pair_root = success.parents[1]
        pair_id = pair_root.name
        analysis = pair_root / "analysis"
        metadata = pair_metadata(pair_id)
        pair_roots[pair_id] = pair_root
        with (analysis / "calibration.json").open() as handle:
            certificate = json.load(handle)
        certificates.append({
            "pair_id": pair_id,
            **metadata,
            "calibration_valid": bool(certificate["calibration_valid"]),
            "source_raw": float(certificate["validation_source_raw_acceptance"]),
            "target_raw": float(certificate["validation_target_raw_acceptance"]),
        })
        for key, filename in filenames.items():
            table = pd.read_csv(analysis / filename)
            table.insert(0, "pair_id", pair_id)
            for name, value in metadata.items():
                table[name] = value
            collections[key].append(table)
    if not certificates:
        raise FileNotFoundError(f"No completed pair analyses found under {run_root}")
    return (
        pd.DataFrame(certificates),
        {key: pd.concat(parts, ignore_index=True) for key, parts in collections.items()},
        pair_roots,
    )


def annotation_composition(rejection: pd.DataFrame) -> pd.DataFrame:
    records = []
    for side in ("source", "target"):
        stage_column = f"{side}_stage"
        sample_column = f"{side}_sample"
        selected = rejection[rejection.side.eq(side)][
            [stage_column, sample_column, "annotation", "n"]
        ].rename(columns={stage_column: "stage", sample_column: "sample"})
        selected = selected.drop_duplicates(["stage", "sample", "annotation"])
        selected["sample_n"] = selected.groupby(["stage", "sample"])["n"].transform("sum")
        selected["fraction"] = selected["n"] / selected["sample_n"]
        records.append(selected)
    composition = pd.concat(records, ignore_index=True)
    return composition.groupby(["stage", "annotation"], as_index=False).agg(
        mean_fraction=("fraction", "mean"),
        section_sd=("fraction", "std"),
        section_n=("sample", "nunique"),
    )


def rejection_summary(
    rejection: pd.DataFrame,
    certificates: pd.DataFrame,
    consistency_threshold: float,
) -> pd.DataFrame:
    valid = certificates.set_index("pair_id")["calibration_valid"]
    selected = rejection[rejection.method.eq(CONFIDENCE_METHOD)].copy()
    selected["calibration_valid"] = selected.pair_id.map(valid).fillna(False)
    rows = []
    for keys, group in selected.groupby(["stage_transition", "side", "annotation"]):
        certified = group[group.calibration_valid]
        values = certified.rejected_fraction if not certified.empty else pd.Series(dtype=float)
        rows.append({
            "stage_transition": keys[0], "side": keys[1], "annotation": keys[2],
            "all_pair_mean": float(group.rejected_fraction.mean()),
            "all_pair_n": int(group.pair_id.nunique()),
            "certified_mean": float(values.mean()) if not values.empty else np.nan,
            "certified_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "certified_pair_n": int(certified.pair_id.nunique()),
            "certified_consistency": float(np.mean(values >= consistency_threshold))
            if not values.empty else np.nan,
        })
    return pd.DataFrame(rows)


def build_fate_candidates(
    composition: pd.DataFrame,
    rejection: pd.DataFrame,
) -> pd.DataFrame:
    composition_lookup = composition.set_index(["stage", "annotation"])["mean_fraction"]
    rows = []
    for transition in STAGES:
        source_stage, target_stage = transition.split(" → ")
        subset = rejection[rejection.stage_transition.eq(transition)]
        source = subset[subset.side.eq("source")].set_index("annotation")
        target = subset[subset.side.eq("target")].set_index("annotation")
        annotations = sorted(
            set(composition.loc[composition.stage.eq(source_stage), "annotation"])
            | set(composition.loc[composition.stage.eq(target_stage), "annotation"])
        )
        for annotation in annotations:
            source_fraction = float(composition_lookup.get((source_stage, annotation), 0.0))
            target_fraction = float(composition_lookup.get((target_stage, annotation), 0.0))
            source_rejection = float(source.loc[annotation, "certified_mean"]) if annotation in source.index else 0.0
            target_rejection = float(target.loc[annotation, "certified_mean"]) if annotation in target.index else 0.0
            source_rejection = 0.0 if np.isnan(source_rejection) else source_rejection
            target_rejection = 0.0 if np.isnan(target_rejection) else target_rejection
            log2_fold_change = float(np.log2((target_fraction + 1e-4) / (source_fraction + 1e-4)))
            rows.append({
                "stage_transition": transition, "annotation": annotation,
                "source_fraction": source_fraction, "target_fraction": target_fraction,
                "log2_target_over_source_fraction": log2_fold_change,
                "source_certified_rejection": source_rejection,
                "target_certified_rejection": target_rejection,
                "directional_rejection": target_rejection - source_rejection,
                "disappearance_score": source_rejection * max(-log2_fold_change, 0.0),
                "emergence_score": target_rejection * max(log2_fold_change, 0.0),
            })
    return pd.DataFrame(rows)


def plot_fate_candidates(candidates: pd.DataFrame, destination: Path, dpi: int, max_labels: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for axis, transition in zip(axes, STAGES):
        values = candidates[candidates.stage_transition.eq(transition)].copy()
        magnitude = np.maximum(values.source_certified_rejection, values.target_certified_rejection)
        sizes = 35 + 550 * np.sqrt(np.maximum(values.source_fraction, values.target_fraction))
        image = axis.scatter(
            values.log2_target_over_source_fraction, values.directional_rejection,
            c=magnitude, s=sizes, cmap="magma", vmin=0, vmax=1,
            edgecolor="#333333", linewidth=0.45, alpha=0.9,
        )
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.axvline(0, color="#777777", linewidth=0.8)
        axis.axvspan(axis.get_xlim()[0], 0, ymin=0, ymax=0.5, color=SOURCE_COLOR, alpha=0.035)
        axis.axvspan(0, axis.get_xlim()[1], ymin=0.5, ymax=1, color=TARGET_COLOR, alpha=0.035)
        label_score = np.maximum(values.disappearance_score, values.emergence_score)
        for index in np.argsort(label_score.to_numpy())[-max_labels:]:
            row = values.iloc[index]
            axis.annotate(
                row.annotation,
                (row.log2_target_over_source_fraction, row.directional_rejection),
                xytext=(4, 4), textcoords="offset points", fontsize=8,
            )
        axis.set_title(transition)
        axis.set_xlabel("Annotation abundance change, log2(target/source)")
        axis.grid(alpha=0.15)
        axis.text(0.02, 0.03, "candidate disappearance", color=SOURCE_COLOR,
                  transform=axis.transAxes, fontsize=10)
        axis.text(0.98, 0.97, "candidate emergence", color=TARGET_COLOR,
                  transform=axis.transAxes, fontsize=10, ha="right", va="top")
    axes[0].set_ylabel("Directional rejection: target − source")
    figure.colorbar(image, ax=axes, label="Certified population rejection fraction",
                    fraction=0.025, pad=0.02)
    figure.suptitle("Developmental disappearance and emergence candidates", fontsize=16)
    figure.subplots_adjust(left=0.08, right=0.91, bottom=0.14, top=0.87, wspace=0.18)
    save_figure(
        figure, destination, dpi,
        panels=[
            ("e9p5_to_e10p5", axes[0]),
            ("e10p5_to_e11p5", axes[1]),
        ],
    )


def plot_rejection_reproducibility(
    summary: pd.DataFrame,
    destination: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 9))
    last = None
    for axis, transition in zip(axes, STAGES):
        subset = summary[summary.stage_transition.eq(transition)]
        source = subset[subset.side.eq("source")].set_index("annotation")
        target = subset[subset.side.eq("target")].set_index("annotation")
        annotations = sorted(set(source.index) | set(target.index), key=lambda label: max(
            float(source.loc[label, "certified_mean"]) if label in source.index else 0,
            float(target.loc[label, "certified_mean"]) if label in target.index else 0,
        ))
        matrix = np.full((len(annotations), 4), np.nan)
        for row, annotation in enumerate(annotations):
            if annotation in source.index:
                matrix[row, 0] = source.loc[annotation, "certified_mean"]
                matrix[row, 1] = source.loc[annotation, "certified_consistency"]
            if annotation in target.index:
                matrix[row, 2] = target.loc[annotation, "certified_mean"]
                matrix[row, 3] = target.loc[annotation, "certified_consistency"]
        last = axis.imshow(matrix, cmap="magma", vmin=0, vmax=1, aspect="auto")
        axis.set_yticks(range(len(annotations)), annotations, fontsize=8)
        axis.set_xticks(range(4), ["Source\nmean", "Source\nconsistency", "Target\nmean", "Target\nconsistency"])
        axis.set_title(transition)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                if np.isfinite(matrix[row, column]):
                    color = "black" if matrix[row, column] > 0.68 else "white"
                    axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center",
                              fontsize=6.5, color=color)
    figure.colorbar(last, ax=axes, label="Fraction", fraction=0.025, pad=0.02)
    figure.suptitle("Cross-section reproducibility of developmental rejection", fontsize=16)
    figure.subplots_adjust(left=0.14, right=0.90, bottom=0.08, top=0.90, wspace=0.45)
    save_figure(
        figure, destination, dpi,
        panels=[
            ("e9p5_to_e10p5", axes[0]),
            ("e10p5_to_e11p5", axes[1]),
        ],
    )


def aggregate_transition(table: pd.DataFrame, pairs: list[str], method: str):
    selected = table[table.pair_id.isin(pairs) & table.method.eq(method)]
    source_labels = sorted(selected.source_annotation.unique())
    target_labels = sorted(selected.target_annotation.unique())
    matrices = []
    for pair_id in pairs:
        pair = selected[selected.pair_id.eq(pair_id)]
        if pair.empty:
            continue
        matrices.append(pair.pivot(
            index="source_annotation", columns="target_annotation",
            values="source_conditional_probability",
        ).reindex(index=source_labels, columns=target_labels, fill_value=0).to_numpy())
    return source_labels, target_labels, np.mean(matrices, axis=0)


def plot_certified_transitions(
    certificates: pd.DataFrame,
    transitions: pd.DataFrame,
    destination: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(19, 13))
    for row, transition in enumerate(STAGES):
        pairs = certificates.loc[
            certificates.stage_transition.eq(transition) & certificates.calibration_valid, "pair_id"
        ].tolist()
        source_labels, target_labels, traditional = aggregate_transition(
            transitions, pairs, TRADITIONAL_METHOD
        )
        _, _, confidence = aggregate_transition(transitions, pairs, CONFIDENCE_METHOD)
        difference = confidence - traditional
        limit = max(float(np.nanpercentile(np.abs(difference), 99)), 0.05)
        for column, (matrix, title, cmap, vmin, vmax) in enumerate((
            (traditional, "Traditional OT", "magma", 0, 1),
            (confidence, "ConfidenceOT retained", "magma", 0, 1),
            (difference, "ConfidenceOT − Traditional", "coolwarm", -limit, limit),
        )):
            axis = axes[row, column]
            image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            axis.set_xticks(range(len(target_labels)), target_labels, rotation=60, ha="right", fontsize=6.5)
            axis.set_yticks(range(len(source_labels)), source_labels if column == 0 else [], fontsize=7.5)
            axis.set_title(f"{transition} — {title}\ncertified pairs n={len(pairs)}", fontsize=10)
            if column == 0:
                axis.set_ylabel("Source annotation")
            axis.set_xlabel("Target annotation")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle("Developmental transitions after excluding unsupported cells", fontsize=16)
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.91, hspace=0.55, wspace=0.34)
    panel_names = []
    for row, transition in enumerate(STAGES):
        stage_slug = transition.lower().replace(".", "p").replace(" → ", "_to_")
        for column, method_slug in enumerate(("traditional_ot", "confidenceot", "difference")):
            panel_names.append((f"{stage_slug}__{method_slug}", axes[row, column]))
    save_figure(figure, destination, dpi, panels=panel_names)


def forced_matrix(table: pd.DataFrame, pairs: list[str], side: str, max_rows: int = 12):
    selected = table[table.pair_id.isin(pairs) & table.rejection_side.eq(side)]
    scores = selected.groupby("rejected_annotation")["transported_mass"].sum().nlargest(max_rows)
    rows = scores.index.tolist()
    columns = sorted(selected.traditional_forced_partner_annotation.unique())
    matrices = []
    for pair_id in pairs:
        pair = selected[selected.pair_id.eq(pair_id)]
        matrices.append(pair.pivot(
            index="rejected_annotation", columns="traditional_forced_partner_annotation",
            values="conditional_probability",
        ).reindex(index=rows, columns=columns, fill_value=0).to_numpy())
    return rows, columns, np.mean(matrices, axis=0)


def plot_forced_matches(
    certificates: pd.DataFrame,
    forced: pd.DataFrame,
    destination: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 13))
    for row, transition in enumerate(STAGES):
        pairs = certificates.loc[
            certificates.stage_transition.eq(transition) & certificates.calibration_valid, "pair_id"
        ].tolist()
        for column, side in enumerate(("source", "target")):
            labels, partners, matrix = forced_matrix(forced, pairs, side)
            axis = axes[row, column]
            image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=max(np.nanpercentile(matrix, 99), 0.1),
                                aspect="auto")
            axis.set_yticks(range(len(labels)), labels, fontsize=8)
            axis.set_xticks(range(len(partners)), partners, rotation=60, ha="right", fontsize=7)
            direction = "destinations" if side == "source" else "putative precursors"
            axis.set_title(f"{transition}: rejected {side} → forced {direction}", fontsize=10)
            axis.set_xlabel("Traditional OT partner annotation")
            axis.set_ylabel(f"Rejected {side} annotation")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle("Where balanced OT forces biologically unsupported matches", fontsize=16)
    figure.subplots_adjust(left=0.13, right=0.96, bottom=0.16, top=0.91, hspace=0.55, wspace=0.38)
    panel_names = []
    for row, transition in enumerate(STAGES):
        stage_slug = transition.lower().replace(".", "p").replace(" → ", "_to_")
        for column, side in enumerate(("source", "target")):
            panel_names.append((f"{stage_slug}__{side}", axes[row, column]))
    save_figure(figure, destination, dpi, panels=panel_names)


def consensus_cells(cells: pd.DataFrame, certificates: pd.DataFrame, transition: str, side: str):
    valid_pairs = set(certificates.loc[
        certificates.stage_transition.eq(transition) & certificates.calibration_valid, "pair_id"
    ])
    sample_column = f"{side}_sample"
    selected = cells[
        cells.stage_transition.eq(transition) & cells.side.eq(side) & cells.pair_id.isin(valid_pairs)
    ].copy()
    return selected.groupby(
        [sample_column, "observation_id", "annotation", "spatial_x", "spatial_y"], as_index=False
    ).agg(rejection_frequency=("rejected", "mean"), partner_pair_n=("pair_id", "nunique")).rename(
        columns={sample_column: "sample"}
    )


def plot_spatial_consensus(
    certificates: pd.DataFrame,
    cells: pd.DataFrame,
    background: pd.DataFrame,
    transition: str,
    destination: Path,
    dpi: int,
) -> None:
    source = consensus_cells(cells, certificates, transition, "source")
    target = consensus_cells(cells, certificates, transition, "target")
    samples = [source["sample"].unique().tolist(), target["sample"].unique().tolist()]
    columns = max(max(map(len, samples)), 1)
    figure, axes = plt.subplots(2, columns, figsize=(3.2 * columns, 6.8), squeeze=False)
    last = None
    for row, (side, values) in enumerate((("source", source), ("target", target))):
        for column in range(columns):
            axis = axes[row, column]
            if column >= len(samples[row]):
                axis.axis("off")
                continue
            sample = samples[row][column]
            section = values[values["sample"].eq(sample)]
            background_sample = background[
                background.side.eq(side)
                & background[f"{side}_sample"].eq(sample)
                & background.stage_transition.eq(transition)
            ].drop_duplicates("observation_id")
            axis.scatter(background_sample.spatial_x, background_sample.spatial_y, s=2,
                         color="#d8d8d8", linewidth=0, rasterized=True)
            last = axis.scatter(
                section.spatial_x, section.spatial_y, c=section.rejection_frequency,
                s=3.5, cmap="magma", vmin=0, vmax=1, linewidth=0, rasterized=True,
            )
            top = section.groupby("annotation")["rejection_frequency"].mean().nlargest(2)
            subtitle = ", ".join(f"{name} {value:.2f}" for name, value in top.items())
            partners = int(section.partner_pair_n.max()) if not section.empty else 0
            axis.set_title(f"{side.title()} {sample}; partners={partners}\n{subtitle}", fontsize=8.5)
            axis.set_aspect("equal", adjustable="datalim")
            axis.invert_yaxis()
            axis.set_xticks([])
            axis.set_yticks([])
    if last is not None:
        figure.colorbar(last, ax=axes, label="Rejection frequency across certified partner sections",
                        fraction=0.018, pad=0.015)
    figure.suptitle(f"Spatial consensus of developmental exclusion — {transition}", fontsize=15)
    figure.subplots_adjust(left=0.03, right=0.92, bottom=0.05, top=0.88, hspace=0.30, wspace=0.10)
    panel_names = []
    for row, side in enumerate(("source", "target")):
        for column, sample in enumerate(samples[row]):
            panel_names.append((f"{side}__{sample.lower().replace('.', 'p')}", axes[row, column]))
    save_figure(figure, destination, dpi, panels=panel_names)


def heldout_effect(values: np.ndarray, rejected: np.ndarray) -> np.ndarray:
    rejected_values = values[rejected]
    retained_values = values[~rejected]
    pooled = values.std(axis=0, ddof=1)
    pooled[pooled < 1e-6] = 1.0
    return (rejected_values.mean(axis=0) - retained_values.mean(axis=0)) / pooled


def plot_heldout_validation(
    certificates: pd.DataFrame,
    pair_roots: dict[str, Path],
    destination: Path,
    dpi: int,
    top_gene_n: int,
) -> pd.DataFrame:
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    output_rows = []
    for row, transition in enumerate(STAGES):
        candidates = certificates[
            certificates.stage_transition.eq(transition) & certificates.calibration_valid
        ].copy()
        candidates["worst_raw"] = candidates[["source_raw", "target_raw"]].max(axis=1)
        pair_id = candidates.sort_values(["worst_raw", "pair_id"]).iloc[0].pair_id
        prepared_path = pair_roots[pair_id] / "preparation" / "prepared_pair.npz"
        prepared = np.load(prepared_path, allow_pickle=False)
        cells = pd.read_csv(pair_roots[pair_id] / "analysis" / "cell_confidence.csv")
        genes = prepared["heldout_genes"].astype(str)
        for column, side in enumerate(("source", "target")):
            axis = axes[row, column]
            values = prepared[f"{side}_heldout_log"].astype(np.float64)
            cell_side = cells[cells.side.eq(side)].reset_index(drop=True)
            population = cell_side.groupby("annotation").agg(
                n=("rejected", "size"), rejected_n=("rejected", "sum"),
                rejected_fraction=("rejected", "mean"),
            )
            eligible = population[(population.rejected_n >= 5) & ((population.n - population.rejected_n) >= 5)]
            annotation = eligible.rejected_fraction.idxmax() if not eligible.empty else "all annotations"
            mask = np.ones(len(cell_side), dtype=bool) if annotation == "all annotations" else cell_side.annotation.eq(annotation).to_numpy()
            rejected = cell_side.loc[mask, "rejected"].to_numpy(dtype=bool)
            if rejected.sum() < 2 or (~rejected).sum() < 2:
                axis.axis("off")
                axis.set_title(
                    f"{transition}; {side}\ninsufficient rejected/retained bins",
                    fontsize=9,
                )
                continue
            effects = heldout_effect(values[mask], rejected)
            selected = np.argsort(np.abs(effects))[-top_gene_n:]
            order = selected[np.argsort(effects[selected])]
            colors = np.where(effects[order] >= 0, TARGET_COLOR, SOURCE_COLOR)
            axis.barh(np.arange(len(order)), effects[order], color=colors)
            axis.set_yticks(np.arange(len(order)), genes[order], fontsize=8)
            axis.axvline(0, color="#444444", linewidth=0.8)
            axis.set_xlabel("Standardized expression: rejected − retained")
            axis.set_title(f"{transition}; {side}; {annotation}\n{pair_id}", fontsize=9)
            axis.grid(axis="x", alpha=0.15)
            for gene_index in order:
                output_rows.append({
                    "stage_transition": transition, "pair_id": pair_id, "side": side,
                    "annotation": annotation, "heldout_gene": genes[gene_index],
                    "standardized_rejected_minus_retained": effects[gene_index],
                    "rejected_n": int(rejected.sum()), "retained_n": int((~rejected).sum()),
                })
    figure.suptitle("Independent held-out gene programs in rejected cells", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    panel_names = []
    for row, transition in enumerate(STAGES):
        stage_slug = transition.lower().replace(".", "p").replace(" → ", "_to_")
        for column, side in enumerate(("source", "target")):
            panel_names.append((f"{stage_slug}__{side}", axes[row, column]))
    save_figure(figure, destination, dpi, panels=panel_names)
    return pd.DataFrame(output_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    certificates, tables, pair_roots = read_pair_tables(args.run_root)
    composition = annotation_composition(tables["rejection"])
    summary = rejection_summary(tables["rejection"], certificates, args.consistency_threshold)
    candidates = build_fate_candidates(composition, summary)

    certificates.to_csv(args.output_dir / "analysis_certificate_scope.csv", index=False)
    composition.to_csv(args.output_dir / "annotation_abundance_by_stage.csv", index=False)
    summary.to_csv(args.output_dir / "population_rejection_reproducibility.csv", index=False)
    candidates.sort_values(
        ["stage_transition", "disappearance_score", "emergence_score"], ascending=[True, False, False]
    ).to_csv(args.output_dir / "developmental_fate_candidates.csv", index=False)

    plot_fate_candidates(candidates, args.output_dir / "01_developmental_fate_candidates", args.dpi,
                         args.max_labels)
    plot_rejection_reproducibility(summary, args.output_dir / "02_rejection_reproducibility", args.dpi)
    plot_certified_transitions(certificates, tables["transitions"],
                               args.output_dir / "03_certified_transition_landscape", args.dpi)
    plot_forced_matches(certificates, tables["forced"],
                        args.output_dir / "04_traditional_ot_forced_matches", args.dpi)
    for number, transition in enumerate(STAGES, start=5):
        slug = transition.lower().replace(".", "p").replace(" → ", "_to_")
        plot_spatial_consensus(
            certificates, tables["cells"], tables["background"], transition,
            args.output_dir / f"{number:02d}_spatial_consensus_{slug}", args.dpi,
        )
    if not args.skip_heldout_validation:
        heldout = plot_heldout_validation(
            certificates, pair_roots, args.output_dir / "07_heldout_gene_validation",
            args.dpi, args.heldout_top_genes,
        )
        heldout.to_csv(args.output_dir / "heldout_gene_validation.csv", index=False)
    manifest = sorted(str(path.relative_to(args.output_dir)) for path in args.output_dir.rglob("*.png"))
    (args.output_dir / "figure_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Completed pairs: {len(certificates)}")
    print(f"Certified pairs used for primary biology: {int(certificates.calibration_valid.sum())}")
    print(f"Biological figures: {args.output_dir.resolve()}")
    print("\n".join(manifest))


if __name__ == "__main__":
    main()
