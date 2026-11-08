"""Build the CRISPR-side inputs of every scored unit once, locally, before any coupling is read (MATRIX_METRICS.md).

    python lineage_validation/crispr_matrix_benchmark_20261108/01_build_bundles.py --inputs INPUTS --out BUNDLES \
        [--units L23__KPTracer_3457__3457_Apc_T4__family_metastases ...]

INPUTS is the extracted crispr_inputs_20261108.tgz (allele tables, UNITS.tsv). One pickle per unit, BUNDLES/<unit>.pkl:
  source_ids, target_ids (the OT cells, in a canonical order; the scorer aligns every coupling to it by id), target
  site, pre-equalisation depth where it is known locally;
  lineages: per lineage present in the unit, its source / target rows, the Lineage object (events, weights, observed
  intBCs, L2 intBC-allele partitions), the UPGMA partition, and the cross-side agreement s / evaluable of its block;
  quinn_clone: the clone of every cell (Quinn L1);
  tree (L2/L3 cases only): L2-tree subclone groups and labels per family and level, and the L3-tree ground truth with
  the tree distances of every scored metastatic cell (level3_cell_lineage, unchanged code).
Units: the 9 L2/L3 cases and every KP L1 pair with same_met_family. Refuses to overwrite a bundle.
"""

from __future__ import annotations

import argparse
import glob
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "lineage_validation" / "level3_cell_lineage"))
import matrix_core as mc  # noqa: E402

L2 = REPO / "lineage_validation" / "level2_pairwise_ot"
L2_RUNS = L2 / "data" / "full_20261022_204810" / "full"
KPX = REPO / "results" / "kp_tracer_lineage_validation" / "export" / "tables"
UNLABELLED, OTHER = "family unlabelled", "other"


def load_tables(inputs: Path):
    kp = mc.read_allele_table(inputs / "data/kp_tracer/KPTracer-Data/KPTracer.alleleTable.FINAL.txt", "KPTracer")
    unf = mc.read_allele_table(inputs / "data/kp_tracer/KPTracer-Data/KPTracer.alleleTable.unfiltered.txt", "KPTracer")
    unf = unf[~unf.cell.isin(set(kp.cell))]          # MATRIX_METRICS.md clarification (2): FINAL first, unfiltered for the rest
    kp = pd.concat([kp.assign(from_unfiltered=False), unf.assign(from_unfiltered=True)], ignore_index=True)
    tabs = {"KPTracer": (kp, mc.indel_priors(kp))}
    for f in sorted(glob.glob(str(inputs / "data/quinn2021_GSE161363/converted/lineage/*/*alleleTable*.txt.gz"))):
        mouse = Path(f).parent.name
        t = mc.read_allele_table(f, "Quinn2021")
        tabs[f"Quinn2021:{mouse}"] = (t, mc.indel_priors(t))
    return tabs


def map_ids(ot: pd.Series, table_cells: pd.Index) -> pd.Series:
    """OT id -> allele-table cellBC: exact, else without the '<site>.' prefix (Quinn 10k); NaN when absent."""
    exact = ot.where(ot.isin(table_cells))
    strip = ot.str.split(".", n=1).str[-1]
    alt = strip.where(strip.isin(table_cells))
    out = exact.fillna(alt)
    if out.dropna().duplicated().any():
        raise SystemExit("two OT cells map to one allele-table cell")
    return out


def tree_parts(case: pd.Series, cells: pd.DataFrame) -> dict:
    """L2-tree inputs (as level2_pairwise_ot) and L3-tree ground truth + distances (as level3_cell_lineage)."""
    import l3_core as c3
    fam_cells = pd.read_csv(L2 / "inputs" / "family_cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    subs = pd.read_csv(L2 / "inputs" / "subclones.tsv", sep="\t", dtype={"level": int})
    src = cells[cells.side == "source"].cell.to_numpy(); tgt = cells[cells.side == "target"].cell.to_numpy()
    l2 = {}
    for fam in case["families"].split(";"):
        table = fam_cells[fam_cells["name"].eq(fam)].set_index("cell_id")
        for level in (1, 2):
            allowed = sorted(subs.loc[subs["name"].eq(fam) & subs["level"].eq(level), "subclone"])
            if not allowed:
                continue
            lab = table[f"L{level}"]
            g = np.array([(lab[c] if lab[c] in allowed else UNLABELLED) if c in lab.index else
                          (UNLABELLED if case["dataset"] == "KPTracer" else OTHER) for c in src], dtype=object)
            labels = lab.replace("", "ambiguous").reindex(tgt).to_numpy(dtype=object)
            l2[(fam, level)] = {"source_group": g, "allowed": allowed, "target_label": labels}
    fams, allt, n_src = c3.ground_truth(case, cells[["side", "cell", "site"]], c3.tree_files())
    l3 = {}
    for fam, (ft, s_rows, ftab) in fams.items():
        sc = ftab[ftab["status"].isin(("evaluable", "ambiguous", "not_informative"))].reset_index(drop=True)
        D = np.empty((len(sc), len(s_rows)), dtype=np.int16)
        for k, node in enumerate(sc["target_node"].astype(int)):
            D[k] = ft.distances(node)
        l3[fam] = {"s_rows": s_rows, "D": D, "targets": {"target_col": sc["target_col"].to_numpy(np.int64),
                   "cell": sc["cell"].to_numpy(dtype=str), "status": sc["status"].to_numpy(dtype=str)}}
    # plain numpy / dict only, so that the pickle loads under another pandas version on TACC
    return {"l2": l2, "l3": l3, "status_counts": allt["status"].value_counts().to_dict()}


def build(unit: pd.Series, tabs: dict, kp_cells: pd.DataFrame, kp_depth: pd.Series) -> dict:
    if unit.level == "L23":
        case = pd.read_csv(L2 / "inputs" / "cases.tsv", sep="\t", dtype=str).set_index("case_id").loc[unit.pair_or_case]
        case = case.copy(); case["case_id"] = unit.pair_or_case
        cells = pd.read_csv(L2_RUNS / unit.pair_or_case / "cells.tsv.gz", sep="\t", dtype=str)
        depth = pd.to_numeric(cells.total_counts_raw, errors="coerce").to_numpy()
    else:
        case = None
        c = kp_cells[kp_cells.pair_id == unit.pair_or_case]
        cells = pd.DataFrame({"side": c.side, "cell": c.cell_id, "site": c.tumor_id})
        depth = cells.cell.map(kp_depth).to_numpy(float)
    key = "KPTracer" if unit.dataset == "KPTracer" else f"Quinn2021:{unit.mouse}"
    t, prior = tabs[key]
    src = cells[cells.side == "source"].reset_index(drop=True); tgt = cells[cells.side == "target"].reset_index(drop=True)
    cell_lin = t.drop_duplicates("cell").set_index("cell")["lineage"]
    ms, mt = map_ids(src.cell, cell_lin.index), map_ids(tgt.cell, cell_lin.index)
    ls, lt = ms.map(cell_lin), mt.map(cell_lin)
    out = {"unit_id": unit.unit_id, "level": unit.level, "dataset": unit.dataset, "mouse": unit.mouse,
           "source_ids": src.cell.to_numpy(dtype=str), "target_ids": tgt.cell.to_numpy(dtype=str),
           "target_site": tgt.site.to_numpy(dtype=str),
           "source_depth": depth[(cells.side == "source").to_numpy()], "target_depth": depth[(cells.side == "target").to_numpy()],
           "source_lineage": ls.to_numpy(dtype=object), "target_lineage": lt.to_numpy(dtype=object),
           "source_in_table": ms.notna().to_numpy(), "target_in_table": mt.notna().to_numpy(), "lineages": {}}
    if "from_unfiltered" in t:
        fu = t.drop_duplicates("cell").set_index("cell")["from_unfiltered"]
        out["source_from_unfiltered"] = ms.map(fu).fillna(False).to_numpy(bool)
        out["target_from_unfiltered"] = mt.map(fu).fillna(False).to_numpy(bool)
    for lin in sorted(set(ls.dropna()) & set(lt.dropna())):
        rs = np.flatnonzero((ls == lin).to_numpy()); rt = np.flatnonzero((lt == lin).to_numpy())
        L = mc.Lineage(t[t.lineage == lin], np.r_[ms.iloc[rs].to_numpy(), mt.iloc[rt].to_numpy()],
                       np.r_[np.repeat("source", len(rs)), np.repeat("target", len(rt))], prior)
        rows_s, rows_t = np.arange(len(rs)), len(rs) + np.arange(len(rt))
        s, ok = L.agreement(rows_s, rows_t)
        t0 = time.perf_counter()
        up = L.upgma_partition()
        parts = {k: v for k, v in L.partitions.items()}
        if (up >= 0).any():
            parts["upgma"] = up
        out["lineages"][lin] = {"source_rows": rs, "target_rows": rt, "s": s, "ok": ok, "partitions": parts,
                                "founder_characters": L.founder_characters, "events": len(L.events), "intbcs": len(L.intbcs),
                                "upgma_seconds": round(time.perf_counter() - t0, 1)}
    if unit.dataset != "KPTracer":
        out["quinn_clone"] = (ls.to_numpy(dtype=object), lt.to_numpy(dtype=object))
    if unit.level == "L23":
        out["tree"] = tree_parts(case, cells)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--units", nargs="*")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(a.inputs / "lineage_preprocessing_factorial_20261103" / "UNITS.tsv", sep="\t", dtype=str)
    units = units[(units.level == "L23") | (units.same_met_family == "True")]
    if a.units:
        units = units[units.unit_id.isin(a.units)]
    tabs = load_tables(a.inputs)
    kp_cells = pd.read_csv(KPX / "cells.csv.gz", usecols=["label", "pair_id", "tumor_id", "cell_id", "side"])
    kp_cells = kp_cells[kp_cells.label == "ranknm256_noscale_ds_cos"]
    lin = pd.read_csv(KPX / "cell_lineage.csv.gz", usecols=["cell_id", "predownsample_total_counts"]).drop_duplicates("cell_id")
    kp_depth = lin.set_index("cell_id")["predownsample_total_counts"]
    for _, u in units.iterrows():
        f = a.out / f"{u.unit_id}.pkl"
        if f.exists():
            print(f"{u.unit_id}: exists, kept"); continue
        t0 = time.perf_counter()
        b = build(u, tabs, kp_cells, kp_depth)
        with open(f, "wb") as fh:
            pickle.dump(b, fh, protocol=4)
        nl = len(b["lineages"])
        print(f"{u.unit_id}: {len(b['source_ids'])} x {len(b['target_ids'])} cells, {nl} lineages, "
              f"{sum(len(v['partitions']) for v in b['lineages'].values())} partitions, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
