"""Phase 0: what the three MOSTA runs actually contain, read from their files (read-only).

    python mouse_embryo/biology/31_audit_runs.py --out <analysis root> [--run ds=<root> ...] [--probe]

For every run (default bio_common.RUNS): expected pairs (manifest), pairs with
SUCCESS, FAILED markers, unfinished staging / claims, and per pair which files
exist (per-bin tables, per-bin partner distributions, potentials, coupling
record and its read-back check).  From pair_metrics.csv: M4-E / M4-R / balanced
inner and outer convergence and cycles.  From calibration.json: feasibility,
M4-E calibration cleanliness, held-out M4-R validation, inference validity and
the warnings.  From representations/*/representation.json: the preprocessing
of every pair, checked to be one configuration per run.  --probe also rebuilds
one coupling block of the smallest pair of every run with potentials
(mosta_coupling.PairCoupling, SHA-256 verified) and compares its row masses
with the saved per-bin totals.

Writes audit/RUN_COMPLETENESS.tsv, audit/METHOD_VALIDITY_AUDIT.tsv,
audit/RUN_STATUS_REPORT.md, audit/PREPROCESSING_AUDIT.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bio_common import RUNS, Outputs  # noqa: E402

PAIR_FILES = ("SUCCESS", "bins_source.csv.gz", "bins_target.csv.gz", "transport_source.npz", "transport_target.npz",
              "annotation_fractions.csv", "annotation_transitions.csv.gz", "pair_metrics.csv", "calibration.json",
              "run.json", "potentials.npz", "coupling.json", "coupling_check.json")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def audit_run(name: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    info: dict = {"run": name, "root": str(root), "exists": root.is_dir()}
    if not root.is_dir():
        return pd.DataFrame(), pd.DataFrame(), info
    pairs = pd.read_csv(root / "manifest" / "pairs.csv")
    info["expected_pairs"] = len(pairs)
    info["expected_by_kind"] = pairs["kind"].value_counts().to_dict()
    rows, validity = [], []
    # The collect step's copies (audit/), used only for a pair whose own files are absent (a partial local copy).
    collected_metrics = pd.read_csv(root / "audit" / "pair_metrics.csv") if (root / "audit" / "pair_metrics.csv").is_file() else None
    collected_calibration = (pd.read_csv(root / "audit" / "calibration.csv").set_index("pair_id")
                             if (root / "audit" / "calibration.csv").is_file() else None)
    for _, pair in pairs.iterrows():
        pid = pair["pair_id"]
        directory = root / "pairs" / pid
        row = {"run": name, "pair_id": pid, "kind": pair["kind"], "group": pair["group"],
               "source_sample": pair["source_sample"], "target_sample": pair["target_sample"],
               "failed_marker": (root / "pairs" / f"{pid}.FAILED").is_file(),
               "representation_success": (root / "representations" / pid / "REPRESENTATION_SUCCESS").is_file()}
        for f in PAIR_FILES:
            row[f"has_{f}"] = (directory / f).is_file()
        check = read_json(directory / "coupling_check.json")
        row["coupling_check_pass"] = None if check is None else bool(check.get("pass"))
        representation = read_json(root / "representations" / pid / "representation.json")
        row["preprocessing_label"] = None if representation is None else representation.get("preprocessing", {}).get("label")
        row["preprocessing_json"] = None if representation is None else json.dumps(representation.get("preprocessing"), sort_keys=True)
        row["status"] = ("completed" if row["has_SUCCESS"] else "failed" if row["failed_marker"]
                         else "incomplete" if directory.exists() else "missing")
        rows.append(row)
        calibration = read_json(directory / "calibration.json")
        if calibration is None and collected_calibration is not None and pid in collected_calibration.index:
            c = collected_calibration.loc[pid]
            calibration = {k: c[k] for k in ("feasible_cost_found", "m4e_calibration_clean", "m4r_validation_clean",
                                             "m4e_inference_valid", "calibration_valid", "selection_status")}
            calibration.update(validation_aggregate_valid=c["validation_aggregate_valid"],
                               validation_source_raw_acceptance=c["validation_source_raw_acceptance"],
                               validation_target_raw_acceptance=c["validation_target_raw_acceptance"],
                               warning_messages=[w for w in str(c["warnings"]).split(" | ") if w and w != "nan"])
            row["validity_source"] = "audit/ copy"
        else:
            row["validity_source"] = "pair files"
        calibration = calibration or {}
        if row["has_pair_metrics.csv"]:
            metrics = pd.read_csv(directory / "pair_metrics.csv")
        elif collected_metrics is not None:
            metrics = collected_metrics[collected_metrics["pair_id"] == pid]
        else:
            metrics = pd.DataFrame()
        for _, m in metrics.iterrows():
            validity.append({
                "run": name, "pair_id": pid, "kind": pair["kind"], "group": pair["group"], "method": m["method"],
                "inner_converged": bool(m["inner_converged"]), "outer_converged": bool(m["outer_converged"]),
                "cycle_detected": bool(m["cycle_detected"]), "cycle_length": int(m["cycle_length"]),
                "outer_iterations": int(m["outer_iterations"]),
                "source_rejected": float(m["source_final_rejection_rate"]),
                "target_rejected": float(m["target_final_rejection_rate"]),
                "source_budget_override_rate": float(m["source_budget_override_rate"]),
                "target_budget_override_rate": float(m["target_budget_override_rate"]),
                "rejection_cost": float(m["rejection_cost"]), "cost_scale": float(m["cost_scale"]),
                "device": m.get("device"), "dtype": m.get("dtype"),
                "feasible_cost_found": calibration.get("feasible_cost_found"),
                "m4e_calibration_clean": calibration.get("m4e_calibration_clean"),
                "m4r_validation_clean": calibration.get("m4r_validation_clean"),
                "validation_aggregate_valid": calibration.get("validation_aggregate_valid"),
                "m4e_inference_valid": calibration.get("m4e_inference_valid"),
                "calibration_valid": calibration.get("calibration_valid"),
                "validation_source_raw_acceptance": calibration.get("validation_source_raw_acceptance"),
                "validation_target_raw_acceptance": calibration.get("validation_target_raw_acceptance"),
                "selection_status": calibration.get("selection_status"),
                "warnings": " | ".join(calibration.get("warning_messages") or []),
            })
    completeness = pd.DataFrame(rows)
    info["status_counts"] = completeness["status"].value_counts().to_dict()
    info["preprocessing_labels"] = completeness["preprocessing_label"].value_counts(dropna=False).to_dict()
    info["preprocessing_configurations"] = int(completeness["preprocessing_json"].dropna().nunique())
    info["preprocessing"] = (json.loads(completeness["preprocessing_json"].dropna().iloc[0])
                             if completeness["preprocessing_json"].notna().any() else None)
    for f in PAIR_FILES:
        info[f"pairs_with_{f}"] = int(completeness[f"has_{f}"].sum())
    info["coupling_check_pass"] = int((completeness["coupling_check_pass"] == True).sum())  # noqa: E712
    info["coupling_check_fail"] = int((completeness["coupling_check_pass"] == False).sum())  # noqa: E712
    for marker in ("equivalence/EQUIVALENCE_PASS", "equivalence/EQUIVALENCE_FAIL",
                   "coupling_verification/COUPLING_PASS", "coupling_verification/COUPLING_FAIL"):
        info[marker] = (root / marker).is_file()
    info["equivalence_summary"] = read_json(root / "equivalence" / "summary.json")
    info["equalisation"] = read_json(root / "equalisation" / "report.json")
    claims = root / "claims"
    info["leftover_claims"] = sorted(p.name for p in claims.iterdir()) if claims.is_dir() else []
    tmp = root / "tmp"
    info["leftover_staging"] = sorted(str(p.relative_to(tmp)) for p in tmp.glob("*/*")) if tmp.is_dir() else []
    info["audit_present"] = (root / "audit").is_dir()
    info["package_present"] = (root / "package").is_dir()
    return completeness, pd.DataFrame(validity), info


def probe(name: str, root: Path, completeness: pd.DataFrame) -> dict:
    """Rebuild the M4-E and balanced row masses of the smallest pair with potentials from the saved files."""
    import mosta_coupling

    usable = completeness[completeness["has_potentials.npz"] & completeness["has_SUCCESS"]]
    if usable.empty:
        return {"run": name, "probe": "no completed pair with potentials"}
    sizes = pd.read_csv(root / "equalisation" / "per_slice.csv").set_index("sample")["bins_analysed"]
    usable = usable.assign(elements=[sizes[s] * sizes[t] for s, t in zip(usable["source_sample"], usable["target_sample"])])
    pid = usable.sort_values("elements")["pair_id"].iloc[0]
    coupling = mosta_coupling.PairCoupling(root / "pairs" / pid, root=root, device="cpu")
    bins = pd.read_csv(root / "pairs" / pid / "bins_source.csv.gz")
    out = {"run": name, "pair_id": pid, "elements": int(usable["elements"].min())}
    for method, support, column in (("M4-E", "all", "co_total_mass"), ("M4-E", "active", "co_partner_mass"),
                                    ("balanced", "all", "bot_partner_mass")):
        rows, _ = coupling.marginals(method, support=support)
        expected = bins[column].to_numpy(dtype=np.float64)
        scale = float(np.max(np.abs(expected))) or 1.0
        out[f"{method}_{support}_max_rel_diff_vs_7digit_table"] = float(np.max(np.abs(rows - expected)) / scale)
    return out


def report(infos: list[dict], completeness: pd.DataFrame, validity: pd.DataFrame, probes: list[dict]) -> str:
    lines = ["# MOSTA run status (read from the files)", ""]
    for info in infos:
        name = info["run"]
        lines += [f"## {name}", "", f"- root: `{info['root']}`"]
        if not info["exists"]:
            lines += ["- **root does not exist**", ""]
            continue
        sc = info["status_counts"]
        lines += [f"- expected pairs: {info['expected_pairs']} ({info['expected_by_kind']})",
                  f"- completed (SUCCESS): {sc.get('completed', 0)}; failed: {sc.get('failed', 0)}; "
                  f"incomplete: {sc.get('incomplete', 0)}; never started: {sc.get('missing', 0)}",
                  f"- preprocessing labels: {info['preprocessing_labels']}; distinct configurations: {info['preprocessing_configurations']}",
                  f"- pairs with per-bin tables: {info['pairs_with_bins_source.csv.gz']} source / {info['pairs_with_bins_target.csv.gz']} target",
                  f"- pairs with per-bin partner-annotation distributions (transport_*.npz): "
                  f"{info['pairs_with_transport_source.npz']} source / {info['pairs_with_transport_target.npz']} target",
                  f"- pairs with potentials.npz: {info['pairs_with_potentials.npz']}; coupling read-back pass / fail: "
                  f"{info['coupling_check_pass']} / {info['coupling_check_fail']}",
                  f"- EQUIVALENCE_PASS {info['equivalence/EQUIVALENCE_PASS']}, EQUIVALENCE_FAIL {info['equivalence/EQUIVALENCE_FAIL']}, "
                  f"COUPLING_PASS {info['coupling_verification/COUPLING_PASS']}, COUPLING_FAIL {info['coupling_verification/COUPLING_FAIL']}",
                  f"- audit/ {info['audit_present']}, package/ {info['package_present']}; leftover claims {len(info['leftover_claims'])}, "
                  f"leftover staging {len(info['leftover_staging'])}"]
        v = validity[validity["run"] == name]
        if len(v):
            lines += ["", "| method | pairs | inner converged | outer converged | cycle detected |", "|---|---|---|---|---|"]
            for method, g in v.groupby("method"):
                lines.append(f"| {method} | {len(g)} | {int(g['inner_converged'].sum())} | {int(g['outer_converged'].sum())} | "
                             f"{int(g['cycle_detected'].sum())} |")
            c = v.drop_duplicates("pair_id")
            lines += ["", f"- calibration: feasible cost found {int(c['feasible_cost_found'].eq(True).sum())}/{len(c)}; "
                      f"M4-E calibration clean {int(c['m4e_calibration_clean'].eq(True).sum())}; held-out M4-R validation "
                      f"aggregate >= acceptance {int(c['validation_aggregate_valid'].eq(True).sum())}; M4-R validation fits clean "
                      f"{int(c['m4r_validation_clean'].eq(True).sum())}; **M4-E inference valid "
                      f"{int(c['m4e_inference_valid'].eq(True).sum())}**; calibration valid (M4-R) {int(c['calibration_valid'].eq(True).sum())}",
                      "- M4-E inference valid = a feasible cost was found, the M4-E acceptance curves were monotone, and the "
                      "mean held-out raw acceptance of the M4-R validation fits on the 5 + 5 held-out within-side nulls reached "
                      "the within-side minimum (src/confidenceot/calibration.py). It is a property of the calibration, not of the "
                      "observed fit.",
                      "- warnings: " + "; ".join(f"{k} ({n})" for k, n in
                                                 c["warnings"].str.split(r" \| ").explode().value_counts().items() if k)]
            by_kind = c.groupby("kind")["m4e_inference_valid"].apply(lambda s: f"{int(s.eq(True).sum())}/{len(s)}")
            lines.append("- M4-E inference valid by kind: " + ", ".join(f"{k} {x}" for k, x in by_kind.items()))
        lines.append("")
    if probes:
        lines += ["## Coupling rebuild probe", "", "| run | pair | rebuilt vs saved per-bin mass (max relative difference) |", "|---|---|---|"]
        for p in probes:
            diffs = ", ".join(f"{k}: {v:.2e}" for k, v in p.items() if k.endswith("table"))
            lines.append(f"| {p['run']} | {p.get('pair_id', '')} | {diffs or p.get('probe', '')} |")
        lines += ["", "The saved per-bin tables are written with 7 significant digits, so differences near 1e-7 are the table's rounding.", ""]
    return "\n".join(lines)


def preprocessing_report(infos: list[dict]) -> str:
    lines = ["# Preprocessing of each MOSTA run (from the runs' own records)", ""]
    for info in infos:
        if not info.get("exists"):
            continue
        lines += [f"## {info['run']}", "", "Representation configuration (representations/<pair>/representation.json, "
                  f"{info['preprocessing_configurations']} distinct configuration(s) over the run):", "", "```json",
                  json.dumps(info["preprocessing"], indent=2), "```", "", "Equalisation record (equalisation/report.json):",
                  "", "```json", json.dumps(info["equalisation"], indent=2), "```", ""]
    lines += ["Cost, scale and solver settings are fixed in mouse_embryo/mosta_common.py and 23_run_mosta_pair.py: "
              "UOT backbone, epsilon 0.1, lambda 1, tolerance 1e-4, within-side acceptance 0.99, budget 0.95 not enforced, "
              "calibration on min(2000, N, M) label-blind bins per side with 5 + 5 within-side split nulls; balanced OT is "
              "entropic with epsilon 0.1 and budget 0 on the same cost; cost scale = median over 1,000,000 sampled cross-side pairs.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], help="name=root; default the three runs in bio_common.RUNS")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.run) if args.run else {k: str(v) for k, v in RUNS.items()}
    out = Outputs(args.out)
    tables, validities, infos, probes = [], [], [], []
    for name, root in runs.items():
        completeness, validity, info = audit_run(name, Path(root))
        print(f"{name}: {info.get('status_counts')}", flush=True)
        tables.append(completeness); validities.append(validity); infos.append(info)
        if args.probe and len(completeness) and completeness["has_potentials.npz"].any():
            probes.append(probe(name, Path(root), completeness))
    completeness = pd.concat(tables, ignore_index=True)
    validity = pd.concat(validities, ignore_index=True)
    out.table(completeness.drop(columns=["preprocessing_json"]), "audit/RUN_COMPLETENESS.tsv")
    out.table(validity, "audit/METHOD_VALIDITY_AUDIT.tsv")
    out.json(infos, "audit/run_info.json")
    if probes:
        out.table(pd.DataFrame(probes), "audit/coupling_probe.tsv")
    out.text(report(infos, completeness, validity, probes), "audit/RUN_STATUS_REPORT.md")
    out.text(preprocessing_report(infos), "audit/PREPROCESSING_AUDIT.md")


if __name__ == "__main__":
    main()
