"""Why each LTL331 benchmark fit is or is not M4-E inference valid (runs on TACC; reads each row's calibration.json).

    python realdata_benchmark/16_export_calibration_flags.py --root ROOT

PDX_GATE_COUPLING_METRICS.tsv keeps only the conjunction m4e_inference_valid = feasible_cost_found and
m4e_calibration_clean and validation_aggregate_valid (confidenceot/calibration.py). For every row of
ROOT/worklist_ltl331.tsv this writes each part from the fit's calibration.json (05_ writes it under
ROOT/ltl331/<config>/pair<i>/fits/pair<i>/<config>/out/<pair>/scope_malignant/budget_*/), with:
  max_source_raw_acceptance / max_target_raw_acceptance  highest within-side acceptance on the cost grid
  validation_*_raw_acceptance                           held-out M4-R acceptance at the selected cost
  validation_fits_terminal                              held-out fits that hit the outer cap or cycled
  invalid_reason   first failing part: no_feasible_cost (no grid cost reaches the acceptance on both sides),
                   calibration_fit_terminal_or_nonmonotone (m4e_calibration_clean false), heldout_acceptance_short
                   (aggregate held-out acceptance below the requirement); empty when valid
Writes ROOT/PDX_CALIBRATION_FLAGS.tsv. Refuses if it exists. Fitting success is a different column (the fit itself
exits 0 whether or not the calibration is valid).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--worklist", default="worklist_ltl331.tsv", help="another LTL331 worklist writes PDX_CALIBRATION_FLAGS_<stem>.tsv")
    a = p.parse_args()
    tag = "" if a.worklist == "worklist_ltl331.tsv" else "_" + Path(a.worklist).stem
    target = a.root / f"PDX_CALIBRATION_FLAGS{tag}.tsv"
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    w = pd.read_csv(a.root / a.worklist, sep="\t", dtype=str)
    rows = []
    for _, r in w.iterrows():
        files = sorted(Path(r.out).glob("fits/pair*/*/out/*/scope_malignant/budget_*/calibration.json"))
        rec = {"config_id": r.config_id, "pair_index": int(r.pair_index), "pair_id": r.pair_id, "files": len(files)}
        if len(files) == 1:
            c = json.loads(files[0].read_text())
            val = c.get("validation") or []
            rec.update({k: c.get(k) for k in ("acceptance_requirement", "selection_status", "refinement_method", "rejection_cost",
                                               "feasible_cost_found", "m4e_calibration_clean", "source_monotone", "target_monotone",
                                               "validation_aggregate_valid", "validation_source_raw_acceptance",
                                               "validation_target_raw_acceptance", "m4e_inference_valid", "m4r_validation_clean")})
            rec.update(max_source_raw_acceptance=float(np.max(c["source_raw_acceptance_curve"])),
                       max_target_raw_acceptance=float(np.max(c["target_raw_acceptance_curve"])),
                       max_grid_cost=float(np.max(c["curve_costs"])), validation_fits=len(val),
                       validation_fits_terminal=sum(not (v["inner_converged"] and v["outer_converged"]) or v["cycle_detected"] for v in val),
                       warning_messages=" | ".join(c.get("warning_messages") or []))
            rec["invalid_reason"] = ("" if c["m4e_inference_valid"] else "no_feasible_cost" if not c["feasible_cost_found"] else
                                     "calibration_fit_terminal_or_nonmonotone" if not c["m4e_calibration_clean"] else
                                     "heldout_acceptance_short" if not c["validation_aggregate_valid"] else "other")
        rows.append(rec)
    out = pd.DataFrame(rows); out.to_csv(target, sep="\t", index=False)
    print(out.groupby("files").size().to_string())
    if "invalid_reason" in out:
        print(out.invalid_reason.replace("", "valid").value_counts().to_string())
    print(f"-> {target}")


if __name__ == "__main__":
    main()
