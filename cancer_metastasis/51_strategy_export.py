"""Export package for the five CNV / RNA gate strategies: every cell's state, the counts, the
diagnostics and the DEG/GSEA tables, small enough to download and plot anywhere.

    python cancer_metastasis/51_strategy_export.py --manifest UNIFORM.csv --manifest PROSTATE.csv \\
        --datasets GSE180661 GSE225857 GSE271675 --gates-root .../cnv_rna_strategies_infercnvpy \\
        --acceptances 0.99 0.95 0.90 --umap-root .../reference_umap_cycling_cnv_acc0.99 \\
        [--umap-root .../strategy_umap] --downstream-root .../downstream_cnv_rna_strategies --out EXPORT \\
        [--eligible-manifest .../pancancer_20260924/pair_manifest_eligible.csv]

Pairs: the --manifest files are the ones the strategy OT ran on (submit_cnv_rna_strategies.sh:
factorial_20260929/uniform.csv and prepared_GSE271675_20260916/pair_manifest_eligible.csv), so
every exported pair is one the gates were asked to run. An --eligible-manifest is only reported
against: its pairs for these datasets that are not in --manifest are listed in
pairs_outside_ot_manifest.tsv, and nothing is added to or removed from the export because of it.

Strategies (gate roots <gates-root>/acc<a>/<strategy>/arm_A/<block>/ranknm256_noscale_ds_cos):
  rna, cnv, rna_cnv                 single: retain / reject
  cnv_then_rna, rna_then_cnv        two-step, final state: exclude (step 1 rejected) /
                                    reject (step 1 retained, step 2 rejected) / retain (both)
  cnv_then_rna_step1 = cnv, rna_then_cnv_step1 = rna: step 1, retain / reject
A cell a completed gate did not score is "not_gated"; a pair a strategy did not complete is
"not_run" for all its cells, and pair_status.tsv says why (no SUCCESS). M4-E only.

Coordinates: one frozen joint UMAP per pair (48_reference_umap.py), read from the first
--umap-root that has the pair. They are never recomputed here. Every strategy's gated cells
must be reference cells; a pair where they are not is reported in pair_status.tsv
(umap_status) and its cells keep NaN coordinates.

Writes to OUT:
  cells/<dataset>/<pair>.csv.gz   one row per reference cell: cell_uid, pair, dataset, patient,
                                  side (primary / metastasis), sample, observation_id,
                                  annotation (the cell's own label), umap_1, umap_2,
                                  total_counts, detected_genes, then for each acceptance a:
                                  <strategy>__acc<a> (state) and score__<strategy>__acc<a>
                                  (r = -signed rejection margin of that gate)
  state_counts.tsv                pair x acceptance x strategy x side x state: cells, fraction
  pair_status.tsv                 pair x acceptance x strategy: completed, budget dir, umap_status
  pair_metrics.tsv                every completed gate's pair_metrics.csv (M4-E and M4-R)
  pairs_outside_ot_manifest.tsv   with --eligible-manifest: its pairs the OT manifest does not list
  deg_gsea/acc<a>/<side>/<strategy>/<block>/<accession>/<label>/  the downstream files a
                                  figure needs (DONE/FAILED, reports, the all-gene DEG table,
                                  the GSEA table, pseudobulk sample metadata), copied as they are
  MANIFEST.tsv, MD5SUMS, README.txt
No coupling matrix is exported; they stay beside the gates on TACC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "ranknm256_noscale_ds_cos"
SINGLE = ("rna", "cnv", "rna_cnv")
TWO_STEP = {"cnv_then_rna": "cnv", "rna_then_cnv": "rna"}
STRATEGY_COLUMNS = ("rna", "cnv", "rna_cnv", "cnv_then_rna_step1", "cnv_then_rna",
                    "rna_then_cnv_step1", "rna_then_cnv")
SIDE = {"source": "primary", "target": "metastasis"}
DEG_FILES = {"DONE", "FAILED", "manifest_trimmed.csv", "pydeseq2_report.json", "pydeseq2_all_gene_discovery.csv",
             "all_gene_discovery_gsea_results.csv", "gsea_summary.csv", "gsea_report.json",
             "pseudobulk_sample_metadata.csv"}
BLOCKS = {"GSE271675": "prostate"}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(name: str) -> str:  # 48_reference_umap.safe, so a pair finds its reference directory
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def gate(root: Path, pair_id: str) -> tuple[pd.DataFrame | None, str]:
    """M4-E rows of a completed gate, keyed by (side, observation_id); None if not completed."""
    done = sorted((root / pair_id).glob("scope_malignant/*/SUCCESS"))
    if len(done) != 1:
        return None, "no SUCCESS" if not done else f"{len(done)} SUCCESS"
    table = pd.read_csv(done[0].parent / "cell_confidence.csv",
                        usecols=lambda c: c in {"method", "side", "observation_id", "retained",
                                                "signed_rejection_margin", "annotation", "sample_id"})
    table = table[table["method"].astype(str).eq("M4-E")].copy()
    table["observation_id"] = table["observation_id"].astype(str)
    table["retained"] = table["retained"].astype(bool)
    return table.set_index(["side", "observation_id"]), done[0].parent.name


def read_manifests(paths: list[Path], datasets: list[str]) -> pd.DataFrame:
    table = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    table = table[table["dataset_id"].astype(str).isin(datasets)]
    if "eligible" in table:
        table = table[table["eligible"].astype(bool)]
    return table.drop_duplicates(["dataset_id", "pair_id"]).sort_values(["dataset_id", "pair_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--gates-root", type=Path, required=True)
    parser.add_argument("--acceptances", nargs="+", required=True)
    parser.add_argument("--umap-root", type=Path, action="append", required=True)
    parser.add_argument("--downstream-root", type=Path)
    parser.add_argument("--eligible-manifest", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "MANIFEST.tsv").exists():
        sys.exit(f"{args.out}/MANIFEST.tsv exists; the export is not rewritten")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifests(args.manifest, args.datasets)
    if args.eligible_manifest:
        wider = read_manifests(args.eligible_manifest, args.datasets)
        known = set(zip(manifest["dataset_id"].astype(str), manifest["pair_id"].astype(str)))
        outside = wider[[(str(d), str(p)) not in known for d, p in zip(wider["dataset_id"], wider["pair_id"])]]
        outside = outside[["dataset_id", "pair_id", "patient_id"]].assign(
            reason="in " + ", ".join(map(str, args.eligible_manifest)) + "; not in the OT manifest "
                   + ", ".join(map(str, args.manifest)))
        outside.to_csv(args.out / "pairs_outside_ot_manifest.tsv", sep="\t", index=False)
        print(f"{len(outside)} eligible-manifest pairs are not in the OT manifest: "
              f"{outside.groupby('dataset_id').size().to_dict()} (pairs_outside_ot_manifest.tsv)", flush=True)
    counts_rows, status_rows, metrics = [], [], []

    for row in manifest.to_dict("records"):
        pair, dataset, patient = str(row["pair_id"]), str(row["dataset_id"]), str(row["patient_id"])
        block = BLOCKS.get(dataset, "uniform")
        reference_path = next((root / "pairs" / safe(pair) / "reference_umap.csv.gz" for root in args.umap_root
                               if (root / "pairs" / safe(pair) / "reference_umap.csv.gz").exists()), None)
        gates_by_acc = {}
        for acc in args.acceptances:
            gates_by_acc[acc] = {}
            for strategy in SINGLE + tuple(TWO_STEP):
                root = args.gates_root / f"acc{acc}" / strategy / "arm_A" / block / LABEL
                table, note = gate(root, pair)
                gates_by_acc[acc][strategy] = table
                status_rows.append({"pair_id": pair, "dataset_id": dataset, "patient_id": patient, "acceptance": acc,
                                    "strategy": strategy, "completed": table is not None, "note": note})
                if table is not None:
                    path = next((root / pair).glob("scope_malignant/*/pair_metrics.csv"), None)
                    if path is not None:
                        metrics.append(pd.read_csv(path).assign(acceptance=acc, strategy=strategy))
        # Reference cells: the frozen UMAP's, or, without one, the union of what any gate scored.
        if reference_path is not None:
            cells = pd.read_csv(reference_path)
            cells["observation_id"] = cells["observation_id"].astype(str)
            umap_status = f"reused from {reference_path.parents[2]}"
        else:
            keys = set()
            for by in gates_by_acc.values():
                for table in by.values():
                    if table is not None:
                        keys |= set(table.index)
            cells = pd.DataFrame(sorted(keys), columns=["side", "observation_id"])
            for column in ("umap_1", "umap_2", "total_counts", "detected_genes"):
                cells[column] = np.nan
            cells["cell_uid"] = [f"{dataset}:{patient}:{pair}:{s}:{o}" for s, o in zip(cells["side"], cells["observation_id"])]
            cells["sample"] = ""
            umap_status = "no reference UMAP for this pair"
        key = pd.MultiIndex.from_arrays([cells["side"], cells["observation_id"]])
        outside = 0
        annotation = pd.Series(np.nan, index=key, dtype=object)
        for acc, by in gates_by_acc.items():
            for strategy, table in by.items():
                if table is None:
                    continue
                outside += int((~table.index.isin(key)).sum())
                if "annotation" in table:
                    annotation = annotation.combine_first(table["annotation"].astype(str).reindex(key))
        if outside and reference_path is not None:
            umap_status = f"{outside} gated cells are not reference cells"
        out = pd.DataFrame({"cell_uid": cells["cell_uid"].to_numpy(), "pair_id": pair, "dataset_id": dataset,
                            "patient_id": patient, "side": cells["side"].map(SIDE).to_numpy(),
                            "sample": cells.get("sample", pd.Series([""] * len(cells))).to_numpy(),
                            "observation_id": cells["observation_id"].to_numpy(), "annotation": annotation.to_numpy(),
                            "umap_1": cells["umap_1"].to_numpy(), "umap_2": cells["umap_2"].to_numpy(),
                            "total_counts": cells["total_counts"].to_numpy(), "detected_genes": cells["detected_genes"].to_numpy()})
        for acc, by in gates_by_acc.items():
            def single_state(table):
                if table is None:
                    return np.full(len(key), "not_run", dtype=object), np.full(len(key), np.nan)
                retained = table["retained"].reindex(key).to_numpy(dtype=object)
                missing = pd.isna(retained)
                state = np.where(missing, "not_gated", np.where(retained == True, "retain", "reject"))  # noqa: E712
                return state.astype(object), (-table["signed_rejection_margin"].reindex(key)).to_numpy(float)
            states = {}
            for strategy in SINGLE:
                states[strategy], score = single_state(by[strategy])
                out[f"{strategy}__acc{acc}"], out[f"score__{strategy}__acc{acc}"] = states[strategy], score
            for strategy, first in TWO_STEP.items():
                step1 = states[first]
                step2, score = single_state(by[strategy])
                final = np.where(step1 == "reject", "exclude",
                                 np.where(step1 == "retain", np.where(np.isin(step2, ("retain", "reject")), step2,
                                                                      np.char.add("step2_", step2.astype(str))), step1))
                out[f"{strategy}_step1__acc{acc}"] = step1
                out[f"{strategy}__acc{acc}"] = final
                out[f"score__{strategy}__acc{acc}"] = score
            for column in STRATEGY_COLUMNS:
                values = out[f"{column}__acc{acc}"]
                for side, part in values.groupby(out["side"]):
                    tally = part.value_counts()
                    for state, n in tally.items():
                        counts_rows.append({"pair_id": pair, "dataset_id": dataset, "patient_id": patient,
                                            "acceptance": acc, "strategy": column, "side": side, "state": state,
                                            "cells": int(n), "fraction": float(n / len(part))})
        destination = args.out / "cells" / dataset
        destination.mkdir(parents=True, exist_ok=True)
        out.to_csv(destination / f"{safe(pair)}.csv.gz", index=False, compression="gzip")
        for record in status_rows:
            if record["pair_id"] == pair:
                record["umap_status"] = umap_status
        print(f"{dataset} {pair}: {len(out)} cells; {umap_status}", flush=True)

    pd.DataFrame(counts_rows).to_csv(args.out / "state_counts.tsv", sep="\t", index=False)
    pd.DataFrame(status_rows).to_csv(args.out / "pair_status.tsv", sep="\t", index=False)
    if metrics:
        pd.concat(metrics, ignore_index=True).to_csv(args.out / "pair_metrics.tsv", sep="\t", index=False)

    copied = 0
    if args.downstream_root is not None and args.downstream_root.exists():
        for acc in args.acceptances:
            base = args.downstream_root / f"acc{acc}"
            for path in sorted(base.rglob("*")) if base.exists() else []:
                if path.is_file() and path.name in DEG_FILES and "four_state/patients" not in path.as_posix():
                    target = args.out / "deg_gsea" / f"acc{acc}" / path.relative_to(base)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    copied += 1

    (args.out / "README.txt").write_text(__doc__, encoding="utf-8")
    files = sorted(p for p in args.out.rglob("*") if p.is_file() and p.name not in {"MANIFEST.tsv", "MD5SUMS"})
    pd.DataFrame({"file": [str(p.relative_to(args.out)) for p in files], "bytes": [p.stat().st_size for p in files]}).to_csv(
        args.out / "MANIFEST.tsv", sep="\t", index=False)
    (args.out / "MD5SUMS").write_text("".join(f"{md5(p)}  {p.relative_to(args.out)}\n" for p in files), encoding="utf-8")
    print(json.dumps({"pairs": len(manifest), "deg_gsea_files": copied, "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
