"""Read-only audit for the lineage ground-truth validation of ConfidenceOT (KP-Tracer, Quinn 2021).

    python lineage_validation/00_audit.py --out AUDIT_DIR

Nothing here fits OT, reads a gate against lineage, or writes outside AUDIT_DIR. Lineage
labels are only counted, to see which experiments the data can support. For each dataset
and mouse it writes:

  sites.tsv            every site (tumour or tissue): role (primary / metastasis /
                       pre-injection), cells, cells usable for OT (KP: malignant; Quinn:
                       all A549), cells with a lineage label, eligibility (>= 20 usable
                       cells, the project's rule)
  kp_families.tsv      KP-Tracer: each metastasis, its MetFamily and which primaries of the
                       mouse share it
  tree_coverage.tsv    each lineage tree: leaves, how many leaves are cells of the
                       expression data, and per site how many usable cells are tree leaves
  quinn_clone_by_site.tsv  Quinn: annotated cells per LineageGroup x tissue, per mouse
  cell_label_classes.tsv   per mouse and metastatic site, the target cells in each label
                       class for a pooled-source design: matchable (its lineage is
                       observed among the candidate source cells), unobserved_in_source
                       (labelled, but its lineage has no cell in the source), ambiguous
                       (no lineage label / not a tree leaf)
  feasibility.tsv      per mouse: experiments A, B, C1, C2, C3 eligible or N/A, with reason
  existing_results.tsv what earlier runs already exist (KP OT labels, KP couplings,
                       infercnvpy CNV per unit)
  audit.json, report.txt
The eligibility rules are fixed here, before any result is read: a site is eligible with
>= 20 usable cells; a lineage is "observed" at a site with >= 1 labelled cell there (a
sensitivity column uses >= 5).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MIN_CELLS = 20
OBSERVED = (1, 5)

_spec = importlib.util.spec_from_file_location("d1", REPO / "kp_lineage" / "validation_v2" / "04_lineage_concordance.py")
d1 = importlib.util.module_from_spec(_spec)
sys.modules["d1"] = d1
_spec.loader.exec_module(d1)


def obs_of(path: Path) -> pd.DataFrame:
    import anndata as ad
    data = ad.read_h5ad(path, backed="r")
    obs = data.obs.copy()
    obs.index = obs.index.astype(str)
    data.file.close()
    return obs


# --- KP-Tracer ------------------------------------------------------------------------------

def audit_kp(tracer: Path, root: Path, out: Path, say) -> dict:
    tumours = pd.read_csv(root / "prepared" / "kptracer_tumours.csv", dtype=str)
    manifest = pd.read_csv(root / "prepared" / "pair_manifest_eligible.csv", dtype=str)
    mice = sorted(manifest["patient_id"].unique(), key=lambda m: (len(m), m))
    site_rows, family_rows, tree_rows, class_rows, feas_rows = [], [], [], [], []
    tree_files = sorted((tracer / "trees").glob("*_tree.nwk"))
    for mouse in mice:
        mt = tumours[tumours["mouse"].eq(mouse) & tumours["kind"].ne("not_a_tumour")].copy()
        cells = {}
        for tumour in mt["tumor"]:
            path = root / "prepared" / "h5ad" / f"{tumour}.h5ad"
            obs = obs_of(path)
            cells[tumour] = obs
        stems = [p.name[: -len("_tree.nwk")] for p in tree_files if p.name.startswith(f"{mouse}_")]
        trees = {s: d1.Tree(d1.tree_path(tracer, s, "cassiopeia")) for s in stems}
        leaves = {s: set(t.leaf_of) for s, t in trees.items()}
        all_leaves = set().union(*leaves.values()) if leaves else set()
        for tumour, obs in cells.items():
            kind = mt.loc[mt["tumor"].eq(tumour), "kind"].iloc[0]
            malignant = obs.index[obs["malignant"].astype(str).eq("malignant")]
            labelled = int(pd.Index(malignant).isin(list(all_leaves)).sum())
            site_rows.append({"dataset": "KPTracer", "mouse": mouse, "site": tumour, "role": kind,
                              "cells": int(len(obs)), "usable_cells": int(len(malignant)),
                              "lineage_labelled_usable": labelled,
                              "met_family": mt.loc[mt["tumor"].eq(tumour), "met_family"].iloc[0],
                              "eligible": bool(len(malignant) >= MIN_CELLS)})
        for stem, tree in trees.items():
            row = {"dataset": "KPTracer", "mouse": mouse, "tree": stem, "leaves": len(tree.leaves),
                   **tree.resolution}
            in_data = 0
            for tumour, obs in cells.items():
                malignant = obs.index[obs["malignant"].astype(str).eq("malignant")]
                hit = int(pd.Index(malignant).isin(list(leaves[stem])).sum())
                in_data += hit
                row[f"cells_from_{tumour}"] = hit
            row["leaves_that_are_usable_cells"] = in_data
            tree_rows.append(row)
        sites = pd.DataFrame([r for r in site_rows if r["mouse"] == mouse])
        primaries = sites[(sites["role"] == "primary") & sites["eligible"]]
        mets = sites[(sites["role"] == "metastasis") & sites["eligible"]]
        for met in mets.to_dict("records"):
            related = primaries[primaries["met_family"] == met["met_family"]]["site"].tolist()
            family_rows.append({"dataset": "KPTracer", "mouse": mouse, "metastasis": met["site"],
                                "met_family": met["met_family"], "eligible_primaries": len(primaries),
                                "related_primaries": ";".join(related),
                                "unrelated_primaries": ";".join(sorted(set(primaries["site"]) - set(related)))})
            obs = cells[met["site"]]
            malignant = obs.index[obs["malignant"].astype(str).eq("malignant")]
            on_tree = pd.Index(malignant).isin(list(all_leaves))
            if related:
                matchable, unobserved = int(on_tree.sum()), 0
            else:
                matchable, unobserved = 0, int(on_tree.sum())
            class_rows.append({"dataset": "KPTracer", "mouse": mouse, "target_site": met["site"],
                               "design": "all eligible primaries pooled as source",
                               "label_basis": "MetFamily of the target tumour; cell on a lineage tree",
                               "matchable": matchable, "unobserved_in_source": unobserved,
                               "ambiguous": int((~on_tree).sum())})
        n_p, n_m = len(primaries), len(mets)
        fam = pd.DataFrame([r for r in family_rows if r["mouse"] == mouse])
        a_auroc = int(((fam["related_primaries"] != "") & (fam["unrelated_primaries"] != "")).sum()) if len(fam) else 0
        b_rows = []
        for p in primaries.to_dict("records"):
            rel = int((mets["met_family"] == p["met_family"]).sum())
            b_rows.append((p["site"], rel, n_m - rel))
        b_ok = [b for b in b_rows if n_m >= 2]
        b_auroc = sum(1 for _, r, u in b_rows if r >= 1 and u >= 1)
        feas_rows.append({
            "dataset": "KPTracer", "mouse": mouse, "eligible_primaries": n_p, "eligible_metastases": n_m,
            "A": "eligible" if n_p >= 2 and n_m >= 1 else "N/A: fewer than 2 eligible primaries",
            "A_metastases_with_related_and_unrelated_primaries": a_auroc,
            "B": "eligible" if n_m >= 2 and n_p >= 1 else "N/A: fewer than 2 eligible metastases",
            "B_primaries_with_related_and_unrelated_metastases": b_auroc,
            "C1": "eligible" if b_ok else "N/A: fewer than 2 eligible metastases",
            "C2": "eligible" if n_p >= 2 and n_m >= 1 else "N/A: fewer than 2 eligible primaries",
            "C3": "eligible" if n_p >= 2 and n_m >= 2 else "N/A: needs >= 2 eligible primaries and >= 2 metastases",
            "cell_level_truth": "tree relatedness and MetFamily; no per-cell clone label beyond the tumour/family",
        })
        say(f"KP {mouse}: {n_p} eligible primaries, {n_m} eligible metastases, trees {stems}")
    return {"sites": site_rows, "families": family_rows, "trees": tree_rows, "classes": class_rows,
            "feasibility": feas_rows}


# --- Quinn 2021 ----------------------------------------------------------------------------

def newick_leaves(text: str) -> list[str]:
    """Leaf names of a newick string, without a full parser: tokens right after '(' or ','."""
    names = re.findall(r"[(,]\s*([^():,;\s\[\]]+)", text)
    return [n for n in names if n]


def audit_quinn(data: Path, root: Path, out: Path, say) -> dict:
    converted = data / "converted"
    site_rows, tree_rows, clone_rows, class_rows, feas_rows = [], [], [], [], []
    for path in sorted((converted / "h5ad").glob("*.h5ad")):
        mouse = path.stem
        obs = obs_of(path)
        sample = obs["sampleID"].astype(object) if "sampleID" in obs else pd.Series(np.nan, index=obs.index)
        lg = obs["LineageGroup"].astype(object) if "LineageGroup" in obs else pd.Series(np.nan, index=obs.index)
        labelled = sample.notna() & lg.notna()
        if mouse == "LM0":
            site_rows.append({"dataset": "Quinn2021", "mouse": mouse, "site": "LM0", "role": "pre-injection",
                              "cells": int(len(obs)), "usable_cells": int(len(obs)),
                              "lineage_labelled_usable": int(lg.notna().sum()), "eligible": False,
                              "note": "cells before injection, not a tumour site"})
            if "LG_Present" in obs:
                say(f"Quinn LM0 LG_Present: {obs['LG_Present'].astype(str).value_counts().to_dict()}")
            continue
        tissue = sample.astype(str).str.replace(r"^\d+k", "", regex=True)
        for site in sorted(sample.dropna().unique()):
            inside = sample.eq(site)
            short = re.sub(r"^\d+k", "", str(site))
            site_rows.append({"dataset": "Quinn2021", "mouse": mouse, "site": site,
                              "role": "primary (injection site)" if short == "LL" else "metastasis or other tissue",
                              "cells": int(inside.sum()), "usable_cells": int(inside.sum()),
                              "lineage_labelled_usable": int((inside & lg.notna()).sum()),
                              "eligible": bool(inside.sum() >= MIN_CELLS)})
        site_rows.append({"dataset": "Quinn2021", "mouse": mouse, "site": "(no meta row)", "role": "unlabelled",
                          "cells": int(sample.isna().sum()), "usable_cells": int(sample.isna().sum()),
                          "lineage_labelled_usable": 0, "eligible": False,
                          "note": "kept in OT fits but neither site nor clone is known"})
        table = pd.crosstab(lg[labelled].astype(str), sample[labelled].astype(str))
        for clone, row in table.iterrows():
            clone_rows.append({"dataset": "Quinn2021", "mouse": mouse, "LineageGroup": clone, **row.to_dict(),
                               "total": int(row.sum())})
        primary = [s for s in table.columns if re.sub(r"^\d+k", "", s) == "LL"]
        others = [s for s in table.columns if s not in primary and table[s].sum() >= MIN_CELLS]
        for target in table.columns:
            if target in primary:
                continue
            for k in OBSERVED:
                in_source = set(table.index[table[primary].sum(axis=1) >= k]) if primary else set()
                cells_t = lg[labelled & sample.eq(target)].astype(str)
                class_rows.append({"dataset": "Quinn2021", "mouse": mouse, "target_site": target,
                                   "design": "LL (injection site) as source", "observed_rule": f">= {k} LL cells",
                                   "label_basis": "LineageGroup (authors)",
                                   "matchable": int(cells_t.isin(in_source).sum()),
                                   "unobserved_in_source": int((~cells_t.isin(in_source)).sum()),
                                   "ambiguous": int((sample.eq(target) & lg.isna()).sum())})
        # Trees: one file per LineageGroup as deposited; leaves matched to the h5ad cell names.
        tree_dir = converted / "lineage" / mouse / "trees"
        names = set(obs.index)
        for tree_file in sorted(p for p in tree_dir.rglob("*") if p.is_file()):
            try:
                text = tree_file.read_text(errors="replace")
                leaves = newick_leaves(text) if "(" in text else []
            except Exception as error:  # noqa: BLE001 - reported
                leaves, text = [], f"unreadable: {error}"
            match = re.search(r"(?i)lg[_]?(\d+)", tree_file.name)
            hits = [l for l in leaves if l in names]
            tree_rows.append({"dataset": "Quinn2021", "mouse": mouse, "tree": str(tree_file.relative_to(tree_dir)),
                              "LineageGroup": match.group(1) if match else "", "leaves": len(leaves),
                              "leaves_that_are_cells": len(hits),
                              "example_leaf": leaves[0] if leaves else "", "bytes": tree_file.stat().st_size})
        eligible_primary = [s for s in primary if table[s].sum() >= MIN_CELLS]
        feas_rows.append({
            "dataset": "Quinn2021", "mouse": mouse, "eligible_primaries": len(eligible_primary),
            "eligible_metastases": len(others),
            "A": "N/A: one injection site (LL) per mouse; no second primary exists",
            "A_metastases_with_related_and_unrelated_primaries": 0,
            "B": "eligible" if eligible_primary and len(others) >= 2 else "N/A: needs LL and >= 2 eligible tissues",
            "B_primaries_with_related_and_unrelated_metastases": "no binary site truth: every tissue may share clones with LL",
            "C1": "eligible" if eligible_primary and len(others) >= 2 else "N/A",
            "C2": "N/A: one primary site", "C3": "N/A: one primary site",
            "cell_level_truth": "per-cell LineageGroup (clone) and a tree per clone; seeding is multidirectional",
        })
        say(f"Quinn {mouse}: tissues {table.columns.tolist()}, clones {len(table)}, unlabelled {int(sample.isna().sum())}")
    return {"sites": site_rows, "trees": tree_rows, "clones": clone_rows, "classes": class_rows,
            "feasibility": feas_rows}


def existing(kp_root: Path, quinn_root: Path) -> list[dict]:
    rows = []
    for label_dir in sorted((kp_root / "ot" / "kptracer").glob("*")):
        rows.append({"what": f"KP OT {label_dir.name}", "units": len(list(label_dir.glob("*/scope_malignant/*/SUCCESS")))})
    coupling = kp_root / "validation_v3" / "coupling"
    for label_dir in sorted(p for p in coupling.glob("*") if p.is_dir() and p.name != "report"):
        rows.append({"what": f"KP couplings {label_dir.name}", "units": len(list(label_dir.glob("*/scope_malignant/*/coupling_m4e.npz")))})
    rows.append({"what": "KP coupling reproduction SUCCESS", "units": int((coupling / "report" / "SUCCESS").exists())})
    rows.append({"what": "KP infercnvpy mice", "units": len(list((kp_root / "cnv_profiles" / "infercnvpy_mouse" / "KPTracer").glob("*/SUCCESS")))})
    rows.append({"what": "Quinn infercnvpy experiments", "units": len(list((quinn_root / "cnv_profiles" / "infercnvpy_experiment" / "Quinn2021").glob("*/SUCCESS")))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tracer", type=Path, default=Path("/scratch/10119/ghzheng/OT_project/data/kp_tracer/KPTracer-Data"))
    parser.add_argument("--kp-root", type=Path, default=Path("/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot"))
    parser.add_argument("--quinn-data", type=Path, default=Path("/scratch/10119/ghzheng/OT_project/data/quinn2021_GSE161363"))
    parser.add_argument("--quinn-root", type=Path, default=Path("/scratch/10119/ghzheng/OT_project/quinn2021_confidenceot"))
    args = parser.parse_args()
    if (args.out / "SUCCESS").exists():
        sys.exit(f"{args.out}/SUCCESS exists; the audit is not rerun")
    args.out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(text: str) -> None:
        print(text, flush=True)
        lines.append(text)

    kp = audit_kp(args.tracer, args.kp_root, args.out, say)
    quinn = audit_quinn(args.quinn_data, args.quinn_root, args.out, say)
    pd.DataFrame(kp["sites"] + quinn["sites"]).to_csv(args.out / "sites.tsv", sep="\t", index=False)
    pd.DataFrame(kp["families"]).to_csv(args.out / "kp_families.tsv", sep="\t", index=False)
    pd.DataFrame(kp["trees"] + quinn["trees"]).to_csv(args.out / "tree_coverage.tsv", sep="\t", index=False)
    pd.DataFrame(quinn["clones"]).to_csv(args.out / "quinn_clone_by_site.tsv", sep="\t", index=False)
    pd.DataFrame(kp["classes"] + quinn["classes"]).to_csv(args.out / "cell_label_classes.tsv", sep="\t", index=False)
    feas = pd.DataFrame(kp["feasibility"] + quinn["feasibility"])
    feas.to_csv(args.out / "feasibility.tsv", sep="\t", index=False)
    pd.DataFrame(existing(args.kp_root, args.quinn_root)).to_csv(args.out / "existing_results.tsv", sep="\t", index=False)
    summary = {
        "rules": {"min_cells_per_site": MIN_CELLS, "lineage_observed_at_site": list(OBSERVED)},
        "feasibility_counts": {exp: feas[exp].astype(str).str.startswith("eligible").groupby(feas["dataset"]).sum().to_dict()
                               for exp in ("A", "B", "C1", "C2", "C3")},
    }
    (args.out / "audit.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (args.out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "SUCCESS").write_text("done\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
