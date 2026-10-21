"""The prostate infercnvpy profiles, re-exported in Faming Zhao's cell names, for 02_run_pair.py.

    python cancer_metastasis/50_prostate_cnv_faming_names.py --manifest PROSTATE_MANIFEST \\
        --cnv-root .../cnv_profiles/infercnvpy_patient --out .../cnv_profiles/infercnvpy_patient_faming

The prostate pairs are Faming Zhao's cells, named Pat1_Tu1_<barcode>, in the per-specimen h5ads
the manifest points to. The infercnvpy units were built from the collection's files, so their
export (<cnv-root>/GSE271675/<patient>/cells.tsv, cnv.f32, genes.tsv; the observation cells
only) names cells GSM..._<barcode> with the specimen in source_file
(Patient1__primary__Patient1_TU1.h5ad). 02_run_pair.py looks a cell up by name, so it finds no
prostate cell there.

This joins the two on (specimen, barcode), the rule 45_build_prostate_cycle_filter.py uses for
the cycle filter: Pat1_Tu1 <-> Patient1_TU1, Pat2_LN4 <-> Patient2_LN4, and the 16-base
barcode with its -N suffix. It writes, per patient, the same three files in Faming's names:
  cells.tsv   cell, source_file (the collection file the profile came from), barcode
              (= Faming's cell name, the column 02_run_pair.py indexes)
  cnv.f32     the matched rows of the original cnv.f32, unchanged, in the h5ads' cell order
  genes.tsv   copied
and mapping_report.tsv: per specimen, the cells, the analysed cells (02_run_pair.py's rule: the
first of cell_type, annotation, celltype, cell_type_final in the given annotations), and how
many of each have a profile.

Nothing is filtered or imputed. A cell without a profile is left out of the export and counted.
EXPORT_COMPLETE is written only when every analysed cell of every specimen has a profile,
because 02_run_pair.py stops on any analysed cell without one. Otherwise the report is written
and the script exits 1. The original export is only read. An output with EXPORT_COMPLETE is
not overwritten.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = "GSE271675"
BARCODE = re.compile(r"[ACGT]{16}(?:-\d+)?")
LABEL_COLUMNS = ("cell_type", "annotation", "celltype", "cell_type_final")  # 02_run_pair.labels


def barcode_of(name: str) -> str | None:
    found = BARCODE.search(str(name))
    return found.group(0) if found else None


def collection_token(specimen: str) -> str | None:
    """Pat1_Tu1 -> Patient1_TU1, Pat2_LN4 -> Patient2_LN4 (45_'s rule)."""
    match = re.fullmatch(r"Pat(\d+)_(Tu|LN)(\d+)", str(specimen))
    return f"Patient{match.group(1)}_{'TU' if match.group(2) == 'Tu' else 'LN'}{match.group(3)}" if match else None


def labels_of(obs: pd.DataFrame) -> np.ndarray:
    for column in LABEL_COLUMNS:
        if column in obs:
            return obs[column].astype(str).to_numpy()
    return np.repeat("unannotated", len(obs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cnv-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--annotations", default="Epithelial|Basal Epithelial|Neuroendocrine")
    args = parser.parse_args()
    import anndata as ad

    if (args.out / "EXPORT_COMPLETE").exists():
        sys.exit(f"{args.out}/EXPORT_COMPLETE exists; not overwritten")
    analysed_labels = set(args.annotations.split("|"))
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["dataset_id"].astype(str) == DATASET]
    report_rows, complete = [], True
    for patient, rows in manifest.groupby("patient_id"):
        unit = args.cnv_root / DATASET / str(patient)
        cells = pd.read_csv(unit / "cells.tsv", sep="\t", dtype=str)
        genes = pd.read_csv(unit / "genes.tsv", sep="\t", dtype=str)
        values = np.memmap(unit / "cnv.f32", dtype=np.float32, mode="r", shape=(len(cells), len(genes)))
        cells["token"] = cells["source_file"].str.replace(r"\.h5ad$", "", regex=True).str.split("__").str[-1]
        cells["bc"] = [barcode_of(name) for name in cells["barcode"]]
        if cells["bc"].isna().any():
            raise ValueError(f"{unit}: {int(cells['bc'].isna().sum())} CNV cells have no 16-base barcode")
        if cells.duplicated(["token", "bc"]).any():
            raise ValueError(f"{unit}: a barcode repeats within one specimen")
        lookup = pd.Series(np.arange(len(cells)), index=pd.MultiIndex.from_frame(cells[["token", "bc"]]))

        paths = sorted(set(rows["source_h5ad"].astype(str)) | set(rows["target_h5ad"].astype(str)))
        names, sources, positions = [], [], []
        for path in paths:
            data = ad.read_h5ad(path, backed="r")
            obs_names = data.obs_names.astype(str).to_numpy()
            analysed = np.isin(labels_of(data.obs), list(analysed_labels))
            data.file.close()
            specimen = Path(path).stem
            token = collection_token(specimen)
            if token is None:
                raise ValueError(f"{path}: specimen {specimen!r} is not Pat<N>_Tu<M> or Pat<N>_LN<M>")
            keys = pd.MultiIndex.from_arrays([[token] * len(obs_names), [barcode_of(n) for n in obs_names]])
            found = lookup.reindex(keys).to_numpy()
            matched = ~pd.isna(found)
            report_rows.append({"patient_id": patient, "specimen": specimen, "collection_token": token,
                                "cells": len(obs_names), "cells_with_profile": int(matched.sum()),
                                "analysed": int(analysed.sum()),
                                "analysed_with_profile": int((analysed & matched).sum())})
            if (analysed & ~matched).any():
                complete = False
            names += list(obs_names[matched])
            sources += list(cells["source_file"].to_numpy()[found[matched].astype(int)])
            positions += list(found[matched].astype(int))
        if len(set(names)) != len(names):
            raise ValueError(f"{patient}: a Faming cell name repeats across specimens")
        target = args.out / DATASET / str(patient)
        target.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cell": names, "source_file": sources, "barcode": names}).to_csv(
            target / "cells.tsv", sep="\t", index=False)
        np.asarray(values[np.asarray(positions, dtype=np.int64)], dtype=np.float32).tofile(target / "cnv.f32")
        shutil.copyfile(unit / "genes.tsv", target / "genes.tsv")
        print(f"{patient}: {len(names)} cells exported from {len(paths)} specimens", flush=True)

    report = pd.DataFrame(report_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out / "mapping_report.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(report.to_string(index=False))
    total, covered = int(report["analysed"].sum()), int(report["analysed_with_profile"].sum())
    print(f"analysed cells with a profile: {covered} of {total}")
    if not complete:
        sys.exit("some analysed cells have no CNV profile; EXPORT_COMPLETE not written (see mapping_report.tsv)")
    (args.out / "EXPORT_COMPLETE").write_text(str(args.manifest.resolve()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
