"""A cycle filter for GSE271675 in Faming Zhao's cell names.

    python cancer_metastasis/45_build_prostate_cycle_filter.py OUT_TSV_GZ

The collection's cycle/ot_cell_filter.tsv.gz names GSE271675 cells as the
collection's files do (GSM..._barcode) and carries the collection's malignant
call, which GSE271675 does not have. The prostate pairs are Faming Zhao's cells,
named Pat1_Tu1_barcode, with his labels. This writes the same table for them, so
02_run_pair.py --cell-filter reads it unchanged.

Each of his cells is found by barcode in the collection file of its own specimen
(Pat1_Tu1 -> Patient1__primary__Patient1_TU1) and takes that cell's phase from
cycle/GSE271675.cells.tsv.gz -- the scoring every other deposit's phase comes
from. malignant is his analysed labels, AuthorLabel Epithelial, Basal Epithelial
and Neuroendocrine, the compartment the prostate pairs were run on. keep_arm_b
drops the malignant cells outside G1, as the collection's rule does; keep_arm_d
equals it. Every analysed cell must get a phase, or nothing is written.
"""
import os
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd

LAB = os.environ.get("PROSTATE_H5AD", "/scratch/10119/ghzheng/AllPatients_Malignant_harmony.h5ad")
BASE = os.environ.get("CNV_BASE", "/scratch/10119/ghzheng/primary_metastatic_cancer")
ANALYSED = {"Epithelial", "Basal Epithelial", "Neuroendocrine"}
BARCODE = re.compile(r"[ACGT]{16}(?:-\d+)?")


def barcode_of(name):
    found = BARCODE.search(str(name))
    return found.group(0) if found else None


def collection_token(specimen):
    m = re.fullmatch(r"Pat(\d+)_(Tu|LN)(\d+)", str(specimen))
    return f"Patient{m.group(1)}_{'TU' if m.group(2) == 'Tu' else 'LN'}{m.group(3)}" if m else None


def main():
    out = sys.argv[1]
    lab = ad.read_h5ad(LAB, backed="r")
    fz = lab.obs[["Patient", "Specimen", "AuthorLabel"]].astype(str).copy()
    fz["cell_id"] = lab.obs_names.astype(str)
    lab.file.close()
    fz["token"] = fz["Specimen"].map(collection_token)
    fz["barcode"] = [barcode_of(n) for n in fz["cell_id"]]

    cycle = pd.read_csv(os.path.join(BASE, "cycle", "GSE271675.cells.tsv.gz"), sep="\t", dtype=str)
    cycle["token"] = cycle["source_file"].str.replace(r"\.h5ad$", "", regex=True).str.split("__").str[-1]
    cycle["barcode"] = [barcode_of(n) for n in cycle["cell_id"]]
    if cycle.duplicated(["token", "barcode"]).any():
        sys.exit("a barcode repeats within one collection file of the cycle table")
    table = fz.merge(cycle[["token", "barcode", "s_score", "g2m_score", "cell_cycle_phase"]],
                     on=["token", "barcode"], how="left", validate="one_to_one")
    table["malignant"] = np.where(table["AuthorLabel"].isin(ANALYSED), "malignant", "non_malignant")
    missing = table["cell_cycle_phase"].isna() & (table["malignant"] == "malignant")
    print(f"{len(table)} cells; {int((table['malignant'] == 'malignant').sum())} analysed; "
          f"analysed cells without a phase: {int(missing.sum())}")
    if missing.any():
        print(table.loc[missing, "Specimen"].value_counts().to_string())
        sys.exit("not every analysed cell has a phase; nothing written")

    cycling = table["cell_cycle_phase"].notna() & (table["cell_cycle_phase"] != "G1")
    table["keep_arm_a"] = True
    table["keep_arm_b"] = ~((table["malignant"] == "malignant") & cycling)
    table["keep_arm_c"] = True
    table["keep_arm_d"] = table["keep_arm_b"]
    table["deposit"] = "GSE271675"
    table = table.rename(columns={"Specimen": "source_file"})
    columns = ["deposit", "source_file", "cell_id", "AuthorLabel", "malignant", "s_score",
               "g2m_score", "cell_cycle_phase", "keep_arm_a", "keep_arm_b", "keep_arm_c", "keep_arm_d"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # Renamed into place, so a file at OUT is always a whole one: the
    # submission reuses it when it exists.
    table[columns].to_csv(out + ".partial", sep="\t", index=False, compression="gzip")
    os.replace(out + ".partial", out)
    analysed = table[table["malignant"] == "malignant"]
    print("\nanalysed cells removed by keep_arm_b, per specimen:")
    print(analysed.groupby("source_file")["keep_arm_b"].agg(cells="size", removed=lambda k: int((~k).sum()))
          .assign(share=lambda f: (f["removed"] / f["cells"]).round(3)).to_string())
    print(f"\ntotal removed {int((~analysed['keep_arm_b']).sum())} of {len(analysed)} "
          f"({(~analysed['keep_arm_b']).mean():.3f}); written {out}")


if __name__ == "__main__":
    main()
