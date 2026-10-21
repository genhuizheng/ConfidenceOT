"""Build the OT inputs of the lineage validation: pooled sides and manifests (VALIDATION_PROTOCOL.md §4).

    python lineage_validation/01_build_inputs.py --out INPUTS_DIR

Reads only the prepared data; writes INPUTS_DIR. No lineage label is written into any OT
input: the H5ADs written here carry the expression, `sample_id` (the side), `site` (the
tumour or tissue of each cell), `malignant` and, for KP-Tracer, the authors' `cell_type`;
MetFamily, LineageGroup and trees stay out. (A single KP-Tracer tumour side reuses the
equalised file the 104-pair runs used; its obs also holds met_family, which 02_run_pair.py
never reads.)

KP-Tracer (dataset KPTracer, one row per OT run):
  D_A  source = all eligible primaries of the mouse pooled, target = one eligible metastasis
  D_B  source = one eligible primary, target = all eligible metastases pooled (mice with >= 2)
  D_C3 source = all eligible primaries pooled, target = all eligible metastases pooled (>= 2, >= 2)
  Eligible = >= 20 malignant cells in the equalised per-tumour file the 104-pair runs used.
  Pooled sides are the concatenation of those equalised files (manifest *_ds) and of the
  prepared raw files (manifest *_raw, required by the pipeline to cover the same pairs).
Quinn 2021 (dataset Quinn2021):
  D_B  source = LL, target = all eligible tissues pooled; cells = those with an author meta
  row; eligible = >= 20 such cells. Only raw files are written: the equalised ones come from
  27_downsample_counts.py run on quinn_D_B_raw.csv (submit_runs.sh does this).

Writes cell_sites.tsv.gz (dataset, mouse, side, site, cell: every cell of every side, so
the scoring can tell which site a pooled cell came from), manifests/<block>_{raw,ds}.csv (block = kp_D_A, kp_D_B, kp_D_C3, quinn_D_B),
h5ad/<dataset>/{raw,ds}/<file>.h5ad, sides.tsv (every side: its sites and cell counts) and
SUCCESS. Refuses if SUCCESS exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

MIN_CELLS = 20
KP_ROOT = Path("/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot")
QUINN = Path("/scratch/10119/ghzheng/OT_project/data/quinn2021_GSE161363/converted/h5ad")


CELL_ROWS: list = []


def record(dataset: str, mouse: str, side: str, obs_names, sites) -> None:
    CELL_ROWS.append(pd.DataFrame({"dataset": dataset, "mouse": mouse, "side": side,
                                   "site": np.asarray(sites, dtype=object), "cell": np.asarray(obs_names, dtype=str)}))


def read(path: Path) -> ad.AnnData:
    data = ad.read_h5ad(path)
    data.obs_names = data.obs_names.astype(str)
    return data


def pooled(paths: list[Path], sample: str, site_of: list[str], dataset: str, patient: str) -> ad.AnnData:
    parts = []
    for path, site in zip(paths, site_of):
        part = read(path)
        cell_type = part.obs["cell_type"].astype(str).to_numpy() if "cell_type" in part.obs else "unannotated"
        part.obs = pd.DataFrame({"sample_id": sample, "site": site, "malignant": part.obs["malignant"].astype(str).to_numpy(),
                                 "cell_type": cell_type, "patient_id": patient, "dataset_id": dataset}, index=part.obs_names)
        parts.append(part)
    data = ad.concat(parts, join="inner", merge="same")
    if data.obs_names.duplicated().any():
        raise ValueError(f"{sample}: a cell name repeats across the pooled sites")
    return data


def single(path: Path, sample: str, site: str, dataset: str, patient: str) -> ad.AnnData:
    data = read(path)
    data.obs = pd.DataFrame({"sample_id": sample, "site": site, "malignant": data.obs["malignant"].astype(str).to_numpy(),
                             "patient_id": patient, "dataset_id": dataset}, index=data.obs_names)
    return data


def row(pair_id, dataset, patient, block, src_sample, tgt_sample, src_path, tgt_path, src_sites, tgt_sites):
    return {"pair_id": pair_id, "dataset_id": dataset, "patient_id": patient, "design": block,
            "source_sample": src_sample, "target_sample": tgt_sample,
            "source_h5ad": str(src_path), "target_h5ad": str(tgt_path),
            "source_h5ads_json": json.dumps([str(src_path)]), "target_h5ads_json": json.dumps([str(tgt_path)]),
            "source_sites": ";".join(src_sites), "target_sites": ";".join(tgt_sites), "eligible": True,
            "skip_reason": ""}


def build_kp(out: Path, sides: list) -> dict[str, list]:
    tumours = pd.read_csv(KP_ROOT / "prepared" / "kptracer_tumours.csv", dtype=str)
    ds_manifest = pd.read_csv(KP_ROOT / "downsampled" / "pair_manifest_downsampled.csv", dtype=str)
    ds_path = {}
    for side in ("source", "target"):
        for sample, path in zip(ds_manifest[f"{side}_sample"], ds_manifest[f"{side}_h5ad"]):
            ds_path[sample] = Path(path)
    mice = sorted(pd.read_csv(KP_ROOT / "prepared" / "pair_manifest_eligible.csv", dtype=str)["patient_id"].unique(),
                  key=lambda m: (len(m), m))
    rows = {"kp_D_A": [], "kp_D_B": [], "kp_D_C3": []}
    for kind in ("raw", "ds"):
        (out / "h5ad" / "KPTracer" / kind).mkdir(parents=True, exist_ok=True)
    for mouse in mice:
        mt = tumours[tumours["mouse"].eq(mouse) & tumours["kind"].isin(["primary", "metastasis"])]
        count = {}
        for tumour in mt["tumor"]:
            if tumour in ds_path:
                obs = ad.read_h5ad(ds_path[tumour], backed="r").obs
                count[tumour] = int(obs["malignant"].astype(str).eq("malignant").sum())
        prim = [t for t in mt.loc[mt["kind"].eq("primary"), "tumor"] if count.get(t, 0) >= MIN_CELLS]
        mets = [t for t in mt.loc[mt["kind"].eq("metastasis"), "tumor"] if count.get(t, 0) >= MIN_CELLS]
        raw = lambda t: KP_ROOT / "prepared" / "h5ad" / f"{t}.h5ad"
        written = {}

        def write_pool(name, members):
            if name in written:
                return written[name]
            paths = {}
            for kind, src in (("raw", [raw(t) for t in members]), ("ds", [ds_path[t] for t in members])):
                data = pooled(src, name, members, "KPTracer", mouse)
                path = out / "h5ad" / "KPTracer" / kind / f"{name}.h5ad"
                data.write_h5ad(path, compression="lzf")
                paths[kind] = path
                if kind == "ds":
                    record("KPTracer", mouse, name, data.obs_names, data.obs["site"].to_numpy())
                    sides.append({"dataset": "KPTracer", "mouse": mouse, "side": name, "sites": ";".join(members),
                                  "cells": int(data.n_obs),
                                  "malignant": int(data.obs["malignant"].eq("malignant").sum())})
                del data
            written[name] = paths
            return paths

        def write_single(tumour):
            # A single tumour keeps its own files: the equalised one the 104-pair runs used.
            if tumour not in written:
                names = ad.read_h5ad(ds_path[tumour], backed="r").obs_names
                record("KPTracer", mouse, tumour, names, [tumour] * len(names))
                written[tumour] = {"raw": raw(tumour), "ds": ds_path[tumour]}
            return written[tumour]

        if len(prim) >= 2 and mets:
            pool = write_pool(f"{mouse}_primaries", prim)
            for met in mets:
                m = write_single(met)
                rows["kp_D_A"].append((f"{mouse}__D_A__primaries__{met}", mouse, f"{mouse}_primaries", met, pool, m, prim, [met]))
        if len(mets) >= 2 and prim:
            pool = write_pool(f"{mouse}_metastases", mets)
            for p in prim:
                s = write_single(p)
                rows["kp_D_B"].append((f"{mouse}__D_B__{p}__metastases", mouse, p, f"{mouse}_metastases", s, pool, [p], mets))
        if len(prim) >= 2 and len(mets) >= 2:
            ps, ms = write_pool(f"{mouse}_primaries", prim), write_pool(f"{mouse}_metastases", mets)
            rows["kp_D_C3"].append((f"{mouse}__D_C3__primaries__metastases", mouse, f"{mouse}_primaries", f"{mouse}_metastases", ps, ms, prim, mets))
        print(f"KP {mouse}: primaries {prim}, metastases {mets}", flush=True)
    return rows


def build_quinn(out: Path, sides: list) -> dict[str, list]:
    rows = {"quinn_D_B": []}
    (out / "h5ad" / "Quinn2021" / "raw").mkdir(parents=True, exist_ok=True)
    for path in sorted(QUINN.glob("*.h5ad")):
        mouse = path.stem
        if mouse == "LM0":
            continue
        data = read(path)
        site = data.obs["sampleID"].astype(object)
        labelled = site.notna().to_numpy()
        counts = site[labelled].astype(str).value_counts()
        primary = [s for s in counts.index if s.replace(mouse, "").replace("30k", "") == "LL" or s == "LL"]
        if len(primary) != 1:
            raise ValueError(f"{mouse}: expected one LL site, found {primary}")
        ll = primary[0]
        others = sorted(s for s in counts.index if s != ll and counts[s] >= MIN_CELLS)

        def subset(sites, sample):
            keep = site.isin(sites).to_numpy()
            part = data[keep].copy()
            part.obs = pd.DataFrame({"sample_id": sample, "site": site[keep].astype(str).to_numpy(), "malignant": "malignant",
                                     "patient_id": mouse, "dataset_id": "Quinn2021"}, index=part.obs_names)
            file = out / "h5ad" / "Quinn2021" / "raw" / f"{sample}.h5ad"
            part.write_h5ad(file, compression="lzf")
            record("Quinn2021", mouse, sample, part.obs_names, part.obs["site"].to_numpy())
            sides.append({"dataset": "Quinn2021", "mouse": mouse, "side": sample, "sites": ";".join(sites),
                          "cells": int(part.n_obs), "malignant": int(part.n_obs)})
            return file

        if counts[ll] >= MIN_CELLS and len(others) >= 2:
            src = subset([ll], f"{mouse}_LL")
            tgt = subset(others, f"{mouse}_metastases")
            rows["quinn_D_B"].append((f"{mouse}__D_B__LL__metastases", mouse, f"{mouse}_LL", f"{mouse}_metastases",
                                      {"raw": src}, {"raw": tgt}, [ll], others))
        print(f"Quinn {mouse}: LL {counts.get(ll, 0)} cells, targets {others}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kp-root", type=Path, default=KP_ROOT)
    parser.add_argument("--quinn-h5ad", type=Path, default=QUINN)
    args = parser.parse_args()
    globals().update(KP_ROOT=args.kp_root, QUINN=args.quinn_h5ad)
    if (args.out / "SUCCESS").exists():
        sys.exit(f"{args.out}/SUCCESS exists; the inputs are not rebuilt")
    (args.out / "manifests").mkdir(parents=True, exist_ok=True)
    sides: list = []
    blocks = {**build_kp(args.out, sides), **build_quinn(args.out, sides)}
    for block, entries in blocks.items():
        dataset = "Quinn2021" if block.startswith("quinn") else "KPTracer"
        for kind in ("raw", "ds"):
            if dataset == "Quinn2021" and kind == "ds":
                continue
            table = pd.DataFrame([row(pid, dataset, mouse, block, ss, ts, sp[kind], tp[kind], s_sites, t_sites)
                                  for pid, mouse, ss, ts, sp, tp, s_sites, t_sites in entries])
            table.to_csv(args.out / "manifests" / f"{block}_{kind}.csv", index=False)
        print(f"{block}: {len(entries)} runs", flush=True)
    pd.concat(CELL_ROWS, ignore_index=True).drop_duplicates(["dataset", "side", "cell"]).to_csv(
        args.out / "cell_sites.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(sides).drop_duplicates(["dataset", "side"]).to_csv(args.out / "sides.tsv", sep="\t", index=False)
    (args.out / "SUCCESS").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()
