"""Count the malignant compartment on both sides of every pair in a manifest.

Before running a grid, know how much of it can run at all. A pair whose
source side has no cell the uniform call names malignant is not evaluable in
malignant scope, and ``02_run_pair.py`` stops on it with

    scope=malignant input subset is not evaluable: source=0, target=2801

That is not a fault in the run. It is the inferCNV abstention that
``38_abstained_samples_report.py`` describes: where the diploid control
collapsed, every origin-lineage cell in the file is written ``undetermined``,
because no CNV cluster separated from the reference and no threshold could be
set. The method assessed those cells and declined to name them. Counting them
as non-malignant would move them into the distribution without saying so, so
they are counted in their own column here and never folded into either side.

The counterfactual is reported for the same reason it is not applied:
``evaluable_if_undetermined_folded`` says how many pairs the strict filter
costs, which is the number that decides whether recovering an abstaining
sample is worth the work. It is a measurement of the filter, not a proposal
to change it.

Read from ``obs`` alone, in backed mode. The expression matrix decides
nothing about how many cells carry a label, and loading it for 252 pairs
across both sides would cost hours to learn nothing. The sample selection,
the multi-file concatenation and the malignant mask are the same operations
the runner performs, and none of them changes a cell count.

    python cancer_metastasis/39_count_malignant_cells.py MANIFEST.csv OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--malignant-column", default="malignant")
    parser.add_argument("--malignant-value", default="malignant")
    parser.add_argument("--undetermined-value", default="undetermined")
    parser.add_argument(
        "--minimum-cells-per-side", type=int, default=20,
        help="The runner's --minimum-scope-cells, so evaluable here means "
             "evaluable there")
    return parser.parse_args()


def side_paths(row: pd.Series, side: str) -> list[str]:
    """The same resolution 02_run_pair.py makes, json list first."""
    column = f"{side}_h5ads_json"
    if column in row and pd.notna(row[column]):
        return [str(value) for value in json.loads(str(row[column]))]
    return [str(row[f"{side}_h5ad"])]


def side_counts(paths: list[str], sample: str, column: str, malignant: str,
                undetermined: str) -> dict:
    """Cells of each kind on one side, read from obs without touching X."""
    import anndata as ad

    total = 0
    malignant_n = 0
    undetermined_n = 0
    missing_column = False
    for path in paths:
        backed = ad.read_h5ad(path, backed="r")
        try:
            obs = backed.obs
        finally:
            backed.file.close()
        if "sample_id" in obs:
            obs = obs[obs["sample_id"].astype(str).to_numpy() == sample]
        if obs.shape[0] == 0:
            continue
        total += int(obs.shape[0])
        if column not in obs:
            # A file from a source that predates the uniform call. Recorded
            # rather than raised: one such deposit must not stop the count
            # for the other eleven, and the runner would need
            # --include-annotation for it anyway.
            missing_column = True
            continue
        values = obs[column].astype(str).to_numpy()
        malignant_n += int(np.sum(values == malignant))
        undetermined_n += int(np.sum(values == undetermined))
    return {"total_n": total, "malignant_n": malignant_n,
            "undetermined_n": undetermined_n,
            "other_n": total - malignant_n - undetermined_n,
            "malignant_column_missing": missing_column}


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for position, (_, row) in enumerate(manifest.iterrows()):
        record = {"index": position, "pair_id": row.get("pair_id")}
        for key in ("dataset_id", "patient_id", "source_sample",
                    "target_sample"):
            if key in manifest.columns:
                record[key] = row.get(key)
        for side in ("source", "target"):
            try:
                counts = side_counts(
                    side_paths(row, side), str(row[f"{side}_sample"]),
                    args.malignant_column, args.malignant_value,
                    args.undetermined_value)
            except Exception as error:  # noqa: BLE001 - reported, not raised
                # One unreadable file must not cost the other 251 pairs.
                counts = {"total_n": -1, "malignant_n": -1,
                          "undetermined_n": -1, "other_n": -1,
                          "malignant_column_missing": False}
                record[f"{side}_error"] = f"{type(error).__name__}: {error}"
            for key, value in counts.items():
                record[f"{side}_{key}"] = value
        records.append(record)
        print(f"[{position + 1}/{len(manifest)}] {record['pair_id']}  "
              f"source={record['source_malignant_n']}  "
              f"target={record['target_malignant_n']}", flush=True)

    pairs = pd.DataFrame(records)
    minimum = args.minimum_cells_per_side
    pairs["evaluable"] = (pairs["source_malignant_n"] >= minimum) & \
                         (pairs["target_malignant_n"] >= minimum)
    # A measurement of what the strict filter costs, not a proposal. The
    # abstained cells were assessed and declined, so folding them in would
    # change what the source distribution is, not merely how many cells it
    # has.
    pairs["evaluable_if_undetermined_folded"] = (
        (pairs["source_malignant_n"] + pairs["source_undetermined_n"] >= minimum)
        & (pairs["target_malignant_n"] + pairs["target_undetermined_n"] >= minimum))
    pairs["blocked_by"] = np.select(
        [pairs["evaluable"],
         (pairs["source_malignant_n"] < minimum) &
         (pairs["target_malignant_n"] < minimum),
         pairs["source_malignant_n"] < minimum],
        ["", "both", "source"], default="target")

    pairs.to_csv(args.output_root / "malignant_cell_counts.csv", index=False)

    group = "dataset_id" if "dataset_id" in pairs else "pair_id"
    by_dataset = pairs.groupby(group).agg(
        pairs_n=("pair_id", "size"),
        evaluable_n=("evaluable", "sum"),
        recoverable_n=("evaluable_if_undetermined_folded", "sum"),
        source_malignant_median=("source_malignant_n", "median"),
        target_malignant_median=("target_malignant_n", "median"),
        source_malignant_total=("source_malignant_n", "sum"),
        target_malignant_total=("target_malignant_n", "sum"),
        undetermined_total=("source_undetermined_n", "sum"),
    ).reset_index()
    by_dataset["recoverable_n"] -= by_dataset["evaluable_n"]
    by_dataset = by_dataset.rename(
        columns={"recoverable_n": "blocked_only_by_abstention_n"})
    by_dataset.to_csv(
        args.output_root / "malignant_cell_counts_by_dataset.csv", index=False)

    evaluable = int(pairs["evaluable"].sum())
    recoverable = int(pairs["evaluable_if_undetermined_folded"].sum()) - evaluable
    print()
    print(f"pairs                      {len(pairs)}")
    print(f"evaluable in malignant scope  {evaluable} "
          f"({evaluable / max(len(pairs), 1):.1%})")
    print(f"blocked, source side only     "
          f"{int(pairs['blocked_by'].eq('source').sum())}")
    print(f"blocked, target side only     "
          f"{int(pairs['blocked_by'].eq('target').sum())}")
    print(f"blocked, both sides           "
          f"{int(pairs['blocked_by'].eq('both').sum())}")
    print(f"blocked only by abstention    {recoverable}  "
          f"(would pass if undetermined counted; reported, not applied)")
    if "source_error" in pairs or "target_error" in pairs:
        broken = pairs[pairs.filter(like="_error").notna().any(axis=1)]
        print(f"unreadable sides              {len(broken)}")
    print()
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(by_dataset.to_string(index=False))
    print()
    print(f"wrote {args.output_root / 'malignant_cell_counts.csv'}")
    print(f"wrote {args.output_root / 'malignant_cell_counts_by_dataset.csv'}")


if __name__ == "__main__":
    main()
