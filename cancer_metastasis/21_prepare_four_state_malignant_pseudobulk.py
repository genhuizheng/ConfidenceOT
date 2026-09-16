"""Prepare patient-level four-state malignant pseudobulks for GSE180661.

ConfidenceOT gates both sides of every selected primary--metastasis pair.  This
workflow therefore retains four malignant states: primary retained/rejected and
metastasis retained/rejected.

**What a cell must satisfy to enter a pseudobulk.**  Three requirements, and
only the middle one has changed:

1. *One lesion per patient.*  Retention is a property of a (primary cell,
   metastasis) pair, not of the cell: 64% of this dataset's primary cells carry
   different labels against different lesions of the same patient, and the
   per-pair retained fraction ranges from 0.07 to 0.67.  Requiring a cell to be
   retained against every lesion left 6 cells out of 1,332 for the first
   patient, and every softer collapsing rule -- majority, any-site -- states a
   claim the data does not make, because "most of this patient's lesions" is
   not a biological object and the partner count varies per patient.  So one
   lesion is named per patient, the one with the most cells, and each primary
   cell is gated exactly once against it.  ``retained`` then means compatible
   with that lesion, which is a claim the design supports.  Size is fixed
   before any gate is fitted, so the choice cannot be steered by the result.

   The site-consensus machinery is kept and still runs; with one lesion it is
   satisfied trivially on the primary side, and it continues to ensure that a
   metastatic cell paired with several primaries contributes one vector.
   ``--metastasis-selection all`` restores the previous behaviour.

2. *Calibration.*  The pair's M4-E calibration must have found a feasible cost
   and produced a clean fit with a valid inference certificate.  This replaces
   the cap-robustness requirement described below.  ``calibration_m4r_clean``
   is deliberately **not** required: this workflow reads M4-E only, and the
   older conflated flag failed on 249 of 252 pan-cancer pairs for an M4-R
   reason that says nothing about the M4-E gate.

3. *Origin selection.*  When a patient has several primary samples for one
   metastasis, the origin ranking can select one as the likely source.  That
   ranking is now optional, because the table it reads was computed under the
   rotation null with the rejection budget enforced and cannot be reused for a
   gate calibrated the other way.  Without it, every primary sample enters on
   its own terms.

**Why cap-robustness is gone.**  The requirement used to be that a cell keep
its label between the 0.85 and 0.90 source-cap fits.  That was a reasonable
hedge while the cap decided the answer -- and it did: under the rotation null
any cap produced a rejection rate equal to that cap.  The budget is no longer
enforced, and the gate is now exactly its own calibrated rule
(``forced_in_share_of_retained`` is 0.000 and ``sign_rule_concordance`` 1.000
at every quartile on GSE180661), so there is no cap for a label to be robust
to.  The two configurations that exist now, the unconstrained gate and the one
held to a rejection interval, are not two caps of the same kind: requiring
agreement between them would exclude precisely the cells where the interval
bit, which filters on the prior instead of hedging against it.  They are
therefore run as two separate analyses -- pass each gate root in turn and
compare the results -- rather than intersected into one.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from common import expression_matrix, gene_keys, load_exact_side


MALIGNANT = "Ovarian.cancer.cell"

# The fields a usable M4-E calibration must all carry. M4-R cleanliness is
# excluded on purpose: nothing here reads the reversible gate.
M4E_CALIBRATION_FIELDS = (
    "calibration_feasible_cost_found",
    "calibration_m4e_clean",
    "calibration_m4e_inference_valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("gate_root", type=Path,
                        help="Completed pair output root supplying the gate")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--index", type=int, help="Patient index")
    parser.add_argument("--print-patient-count", action="store_true")
    parser.add_argument(
        "--sensitivity-root", type=Path, default=None,
        help="Second pair output root. When given, a cell must keep its label "
             "across both roots, which is the pre-2026-09 cap-robust rule. "
             "Left out, the gate root decides alone.",
    )
    parser.add_argument("--sensitivity-label", default="source090")
    parser.add_argument(
        "--robustness-csv", type=Path, default=None,
        help="Origin-group robustness table. When given, one primary sample is "
             "selected per metastasis. Left out, every primary sample enters.",
    )
    parser.add_argument(
        "--metastasis-selection", choices=("largest", "all"), default="largest",
        help="'largest' names one lesion per patient and gates each primary "
             "cell once against it. 'all' keeps every lesion and requires a "
             "cell to carry the same label against all of them.",
    )
    parser.add_argument("--malignant-annotation", default=MALIGNANT)
    parser.add_argument("--include-exact-winner-unstable", action="store_true")
    parser.add_argument(
        "--allow-invalid-calibration", action="store_true",
        help="Keep pairs whose M4-E calibration did not converge",
    )
    args = parser.parse_args()
    if args.sensitivity_root is not None and args.robustness_csv is None:
        # analysis_groups reads the winner columns out of the robustness table,
        # and the cap-robust rule was only ever defined alongside them.
        raise ValueError("--sensitivity-root requires --robustness-csv")
    return args


def pair_id(patient: str, source: str, target: str) -> str:
    return f"{patient}__{source}__{target}"


def analysis_groups(
    manifest: pd.DataFrame,
    robustness: pd.DataFrame | None,
    sensitivity_label: str,
) -> pd.DataFrame:
    keys = ["dataset_id", "patient_id", "target_sample"]
    if robustness is None:
        # Every primary sample enters on its own terms. Selecting one origin
        # per metastasis needs the origin ranking, and that table was computed
        # under the rotation null with the budget enforced, so it cannot be
        # reused for a gate calibrated the other way.
        columns = [*keys, "source_sample"]
        groups = (
            manifest[columns].astype(str).drop_duplicates()
            .sort_values(columns, kind="stable").reset_index(drop=True)
        )
        groups["candidate_primary_n"] = (
            groups.groupby(keys, sort=False)["source_sample"].transform("size")
        )
        groups["origin_group_type"] = "every_primary"
        # The exact-winner filter has nothing to select between here, so it is
        # satisfied rather than bypassed.
        groups["recommended_range_exact_robust"] = True
        return groups
    sensitivity_column = (
        f"{re.sub(r'[^A-Za-z0-9]+', '_', sensitivity_label).lower()}_winner"
    )
    rows: list[dict] = []
    for values, table in manifest.groupby(keys, sort=True, dropna=False):
        record = dict(zip(keys, values))
        sources = sorted(table["source_sample"].astype(str).unique())
        record["candidate_primary_n"] = len(sources)
        if len(sources) == 1:
            record.update({
                "baseline_winner": sources[0],
                sensitivity_column: sources[0],
                "recommended_range_exact_robust": True,
                "origin_group_type": "single_primary_compatibility",
            })
        else:
            match = robustness.copy()
            for key, value in record.items():
                if key in keys:
                    match = match[match[key].astype(str).eq(str(value))]
            if len(match) != 1:
                raise RuntimeError(f"Could not uniquely resolve origin group: {record}")
            selected = match.iloc[0]
            record.update({
                "baseline_winner": str(selected["baseline_winner"]),
                sensitivity_column: str(selected[sensitivity_column]),
                "recommended_range_exact_robust": bool(
                    selected["recommended_range_exact_robust"]
                ),
                "origin_group_type": "multi_primary_origin_ranking",
            })
        rows.append(record)
    groups = pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)
    # One name for the selected primary in either mode, so the caller does not
    # have to know which one produced it.
    groups["source_sample"] = groups["baseline_winner"].astype(str)
    return groups


def designate_metastasis(groups: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep the largest metastatic sample per patient and drop the rest.

    Cell count is fixed before any gate is fitted, so naming the largest lesion
    cannot be steered by the result, and the larger lesion is the one whose
    transport plan rests on more observations.
    """
    sizes = (
        manifest[["patient_id", "target_sample", "target_n"]]
        .assign(
            patient_id=lambda frame: frame["patient_id"].astype(str),
            target_sample=lambda frame: frame["target_sample"].astype(str),
        )
        .groupby(["patient_id", "target_sample"], as_index=False)["target_n"].max()
        .sort_values(
            ["patient_id", "target_n", "target_sample"],
            ascending=[True, False, True], kind="stable",
        )
    )
    selected = sizes.drop_duplicates("patient_id", keep="first")
    chosen = set(zip(selected["patient_id"], selected["target_sample"]))
    keep = [
        (str(patient), str(target)) in chosen
        for patient, target in zip(groups["patient_id"], groups["target_sample"])
    ]
    return groups[keep].copy()


def target_never_rejects(caps: set) -> bool:
    """Whether every pair was fitted with a target that cannot reject."""
    values = [
        float(cap) for cap in caps
        if cap is not None and np.isfinite(pd.to_numeric(cap, errors="coerce"))
    ]
    return bool(values) and all(value == 0.0 for value in values)


def usable_contrasts(
    definitions: list[tuple[str, str, str]],
    cell_n: dict[str, int],
    one_sided: bool,
) -> tuple[list[tuple[str, str, str]], list[dict]]:
    """Split contrasts into those with cells on both sides and those without.

    A contrast with an empty side would reach PyDESeq2 as an all-zero column,
    which it does not reject but cannot say anything about either.
    """
    kept: list[tuple[str, str, str]] = []
    dropped: list[dict] = []
    for contrast, case, reference in definitions:
        empty = [state for state in (case, reference) if not cell_n[state]]
        if not empty:
            kept.append((contrast, case, reference))
            continue
        if one_sided and empty == ["metastasis_rejected"]:
            # Not a failure. Rejection is one-sided by design here, so the
            # metastatic side has no rejected state to contrast against.
            reason = "one_sided_rejection"
        elif all(state.endswith("_nonmalignant") for state in empty):
            # The depth-equalised H5ADs carry only the malignant cells the
            # analysis selected, so there is no same-compartment background.
            reason = "malignant_only_input"
        else:
            reason = "no_cells"
        dropped.append({"contrast": contrast, "empty_states": empty, "reason": reason})
    return kept, dropped


def one_result_file(root: Path, pair: str, name: str) -> Path:
    matches = sorted((root / pair).glob(f"scope_malignant/*/{name}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} for {pair} under {root}; found {len(matches)}")
    return matches[0]


def read_gate(root: Path, pair: str, side: str, prefix: str) -> pd.DataFrame:
    path = one_result_file(root, pair, "cell_confidence.csv")
    table = pd.read_csv(
        path,
        usecols=["method", "side", "observation_id", "rejected",
                 "normalized_rejection_score", "signed_rejection_margin"],
    )
    table = table[table["method"].eq("M4-E") & table["side"].eq(side)].copy()
    table = table.drop(columns=["method", "side"]).rename(columns={
        "rejected": f"{prefix}_rejected",
        "normalized_rejection_score": f"{prefix}_rejection_score",
        "signed_rejection_margin": f"{prefix}_signed_margin",
    })
    if table["observation_id"].duplicated().any():
        raise RuntimeError(f"Duplicate {side} observation IDs in {path}")
    return table


def read_run(root: Path, pair: str) -> dict:
    with one_result_file(root, pair, "run.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def read_hvg(root: Path, pair: str) -> set[str]:
    return {str(value) for value in read_run(root, pair).get("hvg", [])}


def calibration_refusal(run: dict) -> str | None:
    """Name the reason this pair's M4-E gate is not calibrated, or None.

    A pair fitted with ``--fixed-rejection-cost`` has no calibration to
    validate; its flags are all false for that reason rather than because
    anything failed, so it is accepted and the mode is reported instead.
    """
    if str(run.get("calibration_null", "none")) == "none":
        return None
    failed = [name for name in M4E_CALIBRATION_FIELDS if not bool(run.get(name))]
    return ",".join(failed) if failed else None


def single_gate_status(gate: pd.DataFrame) -> pd.DataFrame:
    """Label cells from one gate root, with no second root to agree with."""
    table = gate.copy()
    table["pair_robust_status"] = np.where(
        table["baseline_rejected"].to_numpy(bool), "rejected", "retained"
    )
    return table


def cap_robust_gate(baseline: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    joined = baseline.merge(
        sensitivity, on="observation_id", how="inner", validate="one_to_one"
    )
    sensitivity_rejected = next(
        column for column in joined if column.endswith("_rejected")
        and column != "baseline_rejected"
    )
    joined["pair_robust_status"] = np.select(
        [
            joined["baseline_rejected"] & joined[sensitivity_rejected],
            ~joined["baseline_rejected"] & ~joined[sensitivity_rejected],
        ],
        ["rejected", "retained"],
        default="cap_discordant",
    )
    return joined


def consensus_classification(
    records: pd.DataFrame, expected_occurrences: dict[str, int]
) -> pd.DataFrame:
    """Collapse repeated pair occurrences without duplicating biological cells."""
    rows = []
    for entity_id, table in records.groupby("entity_id", sort=False):
        sample = str(table["sample"].iloc[0])
        statuses = set(table["pair_robust_status"])
        observed = int(table["pair_id"].nunique())
        expected = int(expected_occurrences[sample])
        if observed != expected:
            status = "incomplete_pair_coverage"
        elif statuses == {"retained"}:
            status = "retained"
        elif statuses == {"rejected"}:
            status = "rejected"
        else:
            status = "site_or_cap_discordant"
        rows.append({
            "entity_id": entity_id,
            "sample": sample,
            "observation_id": str(table["observation_id"].iloc[0]),
            "consensus_status": status,
            "observed_pair_n": observed,
            "expected_pair_n": expected,
            "mean_baseline_rejection_score": float(
                table["baseline_rejection_score"].mean()
            ),
        })
    return pd.DataFrame(rows)


def annotation_values(data) -> np.ndarray:
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            return data.obs[column].astype(str).to_numpy()
    raise KeyError("H5AD has no cell-type annotation column")


def gene_symbols(data) -> np.ndarray:
    symbols = np.asarray(data.var_names.astype(str), dtype=str)
    if "gene_symbol" in data.var:
        candidate = data.var["gene_symbol"].astype(str).str.strip().to_numpy(dtype=str)
        valid = ~pd.Series(candidate).str.lower().isin(
            {"", "na", "n/a", "nan", "none", "null", "<na>"}
        ).to_numpy()
        symbols = np.where(valid, candidate, symbols)
    return np.asarray([value.strip() for value in symbols], dtype=str)


def collapse_gene_symbols(data, ot_features: set[str]):
    matrix = sparse.csr_matrix(expression_matrix(data), dtype=np.float64)
    if matrix.data.size and (
        matrix.data.min() < 0 or not np.allclose(matrix.data, np.round(matrix.data))
    ):
        raise ValueError("Patient pseudobulk requires non-negative integer raw counts")
    symbols = gene_symbols(data)
    keys = gene_keys(data)
    lookup: dict[str, int] = {}
    inverse = np.empty(len(symbols), dtype=np.int64)
    ordered: list[str] = []
    for index, symbol in enumerate(symbols):
        if symbol not in lookup:
            lookup[symbol] = len(ordered)
            ordered.append(symbol)
        inverse[index] = lookup[symbol]
    aggregation = sparse.csr_matrix(
        (np.ones(len(symbols)), (np.arange(len(symbols)), inverse)),
        shape=(len(symbols), len(ordered)),
    )
    collapsed = (matrix @ aggregation).tocsr()
    used = np.zeros(len(ordered), dtype=bool)
    for index, key in enumerate(keys):
        if str(key) in ot_features:
            used[inverse[index]] = True
    return collapsed, np.asarray(ordered, dtype=str), used


def add_vector(destination: defaultdict[str, int], genes: np.ndarray, values: np.ndarray):
    for gene, value in zip(genes, values):
        if value:
            destination[str(gene)] += int(value)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv)
    robustness = (
        pd.read_csv(args.robustness_csv) if args.robustness_csv is not None else None
    )
    groups = analysis_groups(manifest, robustness, args.sensitivity_label)
    if not args.include_exact_winner_unstable:
        groups = groups[groups["recommended_range_exact_robust"]].copy()
    if args.metastasis_selection == "largest":
        groups = designate_metastasis(groups, manifest)
    patients = sorted(groups["patient_id"].astype(str).unique())
    if args.print_patient_count:
        print(len(patients))
        return
    if args.index is None:
        raise ValueError("--index is required unless --print-patient-count is used")
    if args.index < 0 or args.index >= len(patients):
        raise IndexError(f"--index {args.index} outside 0..{len(patients) - 1}")
    patient = patients[args.index]
    patient_groups = groups[groups["patient_id"].astype(str).eq(patient)].copy()
    sensitivity_column = (
        f"{re.sub(r'[^A-Za-z0-9]+', '_', args.sensitivity_label).lower()}_winner"
    )
    output = args.output_root / "patients" / f"{args.index:03d}_{patient}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PSEUDOBULK_READY").is_file():
        print(f"SKIP completed patient={patient}")
        return

    source_records: list[pd.DataFrame] = []
    target_records: list[pd.DataFrame] = []
    source_paths: dict[str, list[str]] = {}
    target_paths: dict[str, list[str]] = {}
    source_pair_counts: dict[str, int] = defaultdict(int)
    target_pair_counts: dict[str, int] = defaultdict(int)
    ot_features: set[str] = set()
    pair_rows = []
    excluded_pairs: list[dict[str, str]] = []
    calibration_nulls: set[str] = set()
    target_caps: set = set()
    for _, group in patient_groups.iterrows():
        source = str(group["source_sample"])
        if robustness is not None:
            sensitivity_source = str(group[sensitivity_column])
            if source != sensitivity_source:
                raise RuntimeError(
                    "Exact-unstable winner entered the main four-state analysis"
                )
        target = str(group["target_sample"])
        pair = pair_id(patient, source, target)
        match = manifest[manifest["pair_id"].astype(str).eq(pair)]
        if len(match) != 1:
            raise RuntimeError(f"Manifest did not uniquely resolve {pair}")
        row = match.iloc[0]

        # Checked before any counter moves: consensus_classification compares
        # the pairs a cell was seen in against the pairs it was expected in, so
        # an excluded pair must never be expected.
        run = read_run(args.gate_root, pair)
        calibration_nulls.add(str(run.get("calibration_null", "unknown")))
        target_caps.add(run.get("target_rejection_budget_cap"))
        refusal = calibration_refusal(run)
        if refusal is not None and not args.allow_invalid_calibration:
            excluded_pairs.append({"pair_id": pair, "reason": refusal})
            continue

        source_paths.setdefault(source, json.loads(str(row["source_h5ads_json"])))
        target_paths.setdefault(target, json.loads(str(row["target_h5ads_json"])))
        source_pair_counts[source] += 1
        target_pair_counts[target] += 1
        for side, sample, records in (
            ("source", source, source_records), ("target", target, target_records)
        ):
            primary = read_gate(args.gate_root, pair, side, "baseline")
            gate = (
                cap_robust_gate(
                    primary,
                    read_gate(
                        args.sensitivity_root, pair, side, args.sensitivity_label
                    ),
                )
                if args.sensitivity_root is not None
                else single_gate_status(primary)
            )
            gate.insert(0, "pair_id", pair)
            gate.insert(1, "sample", sample)
            gate.insert(2, "entity_id", sample + "::" + gate["observation_id"].astype(str))
            records.append(gate)
        ot_features |= read_hvg(args.gate_root, pair)
        if args.sensitivity_root is not None:
            ot_features |= read_hvg(args.sensitivity_root, pair)
        pair_rows.append({"pair_id": pair, "source_sample": source,
                          "target_sample": target})

    if len(calibration_nulls) > 1:
        # Mixing a rotation-null root with a within-side one would average two
        # different thresholds into one pseudobulk without saying so.
        raise RuntimeError(
            f"{patient}: pairs disagree on the calibration null: "
            f"{sorted(calibration_nulls)}"
        )
    if not pair_rows:
        report = {
            "patient_id": patient,
            "selected_pair_n": 0,
            "candidate_pair_n": int(len(patient_groups)),
            "excluded_pair_n": len(excluded_pairs),
            "excluded_pairs": excluded_pairs,
            "reason": "no pair carried a usable M4-E calibration",
        }
        (output / "diagnostics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
        print(f"SKIP patient={patient} no usable pair", flush=True)
        return

    source_long = pd.concat(source_records, ignore_index=True)
    target_long = pd.concat(target_records, ignore_index=True)
    source_cells = consensus_classification(source_long, source_pair_counts)
    target_cells = consensus_classification(target_long, target_pair_counts)
    source_cells.insert(0, "side", "primary")
    target_cells.insert(0, "side", "metastasis")
    classification = pd.concat([source_cells, target_cells], ignore_index=True)
    classification.to_csv(
        output / "four_state_cell_classification.csv.gz", index=False, compression="gzip"
    )

    counts: dict[str, defaultdict[str, int]] = {
        state: defaultdict(int) for state in (
            "primary_retained", "primary_rejected", "primary_nonmalignant",
            "metastasis_retained", "metastasis_rejected", "metastasis_nonmalignant",
        )
    }
    cell_n = {state: 0 for state in counts}
    used_for_ot: set[str] = set()
    for side, paths_by_sample, cells, prefix in (
        ("primary", source_paths, source_cells, "primary"),
        ("metastasis", target_paths, target_cells, "metastasis"),
    ):
        for sample, paths in paths_by_sample.items():
            data = load_exact_side(paths, sample)
            annotations = annotation_values(data)
            matrix, genes, used = collapse_gene_symbols(data, ot_features)
            used_for_ot.update(genes[used])
            lookup = {str(value): index for index, value in enumerate(data.obs_names)}
            sample_cells = cells[cells["sample"].eq(sample)]
            for status in ("retained", "rejected"):
                ids = sample_cells.loc[
                    sample_cells["consensus_status"].eq(status), "observation_id"
                ].astype(str).tolist()
                missing = [value for value in ids if value not in lookup]
                if missing:
                    raise KeyError(f"{len(missing)} {side} gate IDs absent from {sample}")
                indices = np.asarray([lookup[value] for value in ids], dtype=np.int64)
                values = np.asarray(matrix[indices].sum(axis=0)).ravel()
                state = f"{prefix}_{status}"
                add_vector(counts[state], genes, values)
                cell_n[state] += len(indices)
            background = np.flatnonzero(annotations != args.malignant_annotation)
            values = np.asarray(matrix[background].sum(axis=0)).ravel()
            state = f"{prefix}_nonmalignant"
            add_vector(counts[state], genes, values)
            cell_n[state] += len(background)

    core = [
        ("primary_rejected_vs_primary_retained", "primary_rejected", "primary_retained"),
        ("metastasis_rejected_vs_metastasis_retained", "metastasis_rejected", "metastasis_retained"),
        ("metastasis_retained_vs_primary_retained", "metastasis_retained", "primary_retained"),
        ("metastasis_rejected_vs_primary_retained", "metastasis_rejected", "primary_retained"),
        ("metastasis_rejected_vs_primary_rejected", "metastasis_rejected", "primary_rejected"),
        ("metastasis_retained_vs_primary_rejected", "metastasis_retained", "primary_rejected"),
    ]
    background = [
        ("primary_retained_vs_primary_nonmalignant", "primary_retained", "primary_nonmalignant"),
        ("primary_rejected_vs_primary_nonmalignant", "primary_rejected", "primary_nonmalignant"),
        ("metastasis_retained_vs_metastasis_nonmalignant", "metastasis_retained", "metastasis_nonmalignant"),
        ("metastasis_rejected_vs_metastasis_nonmalignant", "metastasis_rejected", "metastasis_nonmalignant"),
    ]
    one_sided = target_never_rejects(target_caps)
    core, core_dropped = usable_contrasts(core, cell_n, one_sided)
    background, background_dropped = usable_contrasts(background, cell_n, one_sided)
    dropped_contrasts = core_dropped + background_dropped
    if not core and not background:
        report = {
            "patient_id": patient,
            "selected_pair_n": len(pair_rows),
            "state_cell_n": cell_n,
            "dropped_contrasts": dropped_contrasts,
            "reason": "every contrast had an empty side",
        }
        (output / "diagnostics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
        print(f"SKIP patient={patient} no usable contrast", flush=True)
        return

    genes = sorted(set().union(*(set(value) for value in counts.values())))
    records = []
    vectors = []
    for family, definitions in (("malignant_four_state", core),
                                ("same_compartment_background", background)):
        for contrast, case, reference in definitions:
            for comparison_status, state in (("case", case), ("reference", reference)):
                sample_id = f"{patient}__{contrast}__{comparison_status}"
                records.append({
                    "sample_id": sample_id, "patient_id": patient,
                    "group_id": patient,
                    "contrast": contrast, "comparison_status": comparison_status,
                    "cell_set": state, "cell_n": cell_n[state],
                    "contrast_family": family,
                    "selected_pair_n": len(pair_rows),
                })
                vectors.append([counts[state].get(gene, 0) for gene in genes])
    pseudobulk = pd.DataFrame(
        np.asarray(vectors, dtype=np.int64),
        index=pd.Index([record["sample_id"] for record in records], name="sample_id"),
        columns=genes,
    )
    pseudobulk.to_csv(output / "pseudobulk_raw_counts.csv.gz", compression="gzip")
    pd.DataFrame(records).to_csv(output / "pseudobulk_sample_metadata.csv", index=False)
    pd.DataFrame({
        "gene": genes,
        "used_for_ot": [gene in used_for_ot for gene in genes],
    }).to_csv(output / "pseudobulk_gene_metadata.csv.gz", index=False, compression="gzip")
    pd.DataFrame(pair_rows).to_csv(output / "selected_pairs.csv", index=False)
    report = {
        "patient_id": patient,
        "selected_pair_n": len(pair_rows),
        "candidate_pair_n": int(len(patient_groups)),
        "gate_root": str(args.gate_root),
        "calibration_null": sorted(calibration_nulls),
        "origin_selection": (
            "origin_ranking_winner" if robustness is not None else "every_primary"
        ),
        "metastasis_selection": args.metastasis_selection,
        "one_sided_rejection": one_sided,
        "target_rejection_budget_cap": sorted(
            str(cap) for cap in target_caps
        ),
        "dropped_contrasts": dropped_contrasts,
        "cap_robustness": (
            f"label must agree with {args.sensitivity_root}"
            if args.sensitivity_root is not None
            else "not required; the budget is reported rather than enforced"
        ),
        "calibration_requirement": (
            "allowed to fail" if args.allow_invalid_calibration
            else " and ".join(M4E_CALIBRATION_FIELDS)
        ),
        "excluded_pair_n": len(excluded_pairs),
        "excluded_pairs": excluded_pairs,
        "exact_winner_robust_only": not args.include_exact_winner_unstable,
        "state_cell_n": cell_n,
        "source_consensus_definition": (
            "counted once; same gate across every selected metastatic partner"
        ),
        "excluded_source_status_counts": source_cells["consensus_status"].value_counts().to_dict(),
        "excluded_target_status_counts": target_cells["consensus_status"].value_counts().to_dict(),
        "core_contrasts": [value[0] for value in core],
        "background_contrasts": [value[0] for value in background],
    }
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "PSEUDOBULK_READY").write_text("ready\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
