"""Raw counts of exactly the cells the equalised run analyses, so that the logcpm arm and ranknm256_noscale_ds_cos
see the same cells.

    python ltl331_nepc/03_raw_on_ds_cells.py --raw-manifest manifest_raw.csv --ds-manifest DS/pair_manifest_downsampled.csv \
        --out RAWQC_DIR

27_downsample_counts.py applies the project's QC (>= 1000 UMIs, >= 500 genes, <= 20% mitochondrial) and writes
equalised files with the cells that pass. A label that does not equalise reads the original counts, which still
hold every cell; this writes, per time point, the original counts of the equalised file's cells in its order
(as MOSTA's 22_ does for its logcpm arm). Writes RAWQC_DIR/h5ad/<sample>.h5ad and RAWQC_DIR/pair_manifest_rawqc.csv
(the ds manifest's pairs with the paths swapped). Refuses if the manifest exists; stops if a cell is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--ds-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out_manifest = args.out / "pair_manifest_rawqc.csv"
    if out_manifest.exists():
        raise SystemExit(f"{out_manifest} exists")
    raw, ds = pd.read_csv(args.raw_manifest), pd.read_csv(args.ds_manifest)
    if set(raw["pair_id"]) != set(ds["pair_id"]):
        raise SystemExit("the raw and equalised manifests list different pairs")
    (args.out / "h5ad").mkdir(parents=True, exist_ok=True)
    raw_path = {r[f"{s}_sample"]: r[f"{s}_h5ad"] for _, r in raw.iterrows() for s in ("source", "target")}
    ds_path = {r[f"{s}_sample"]: r[f"{s}_h5ad"] for _, r in ds.iterrows() for s in ("source", "target")}
    written = {}
    for sample, dpath in sorted(ds_path.items()):
        keep = ad.read_h5ad(dpath, backed="r").obs_names
        full = ad.read_h5ad(raw_path[sample])
        missing = keep.difference(full.obs_names)
        if len(missing):
            raise SystemExit(f"{sample}: {len(missing)} equalised cells are not in the original file")
        part = full[keep].copy()
        path = args.out / "h5ad" / f"{sample}.h5ad"
        part.write_h5ad(path, compression="lzf")
        written[sample] = str(path)
        print(f"{sample}: {len(keep)} of {full.n_obs} cells (the equalised file's cells, original counts)", flush=True)
    m = ds.copy()
    for side in ("source", "target"):
        m[f"{side}_h5ad"] = m[f"{side}_sample"].map(written)
        m[f"{side}_h5ads_json"] = m[f"{side}_h5ad"].map(lambda p: json.dumps([p]))
    m.to_csv(out_manifest, index=False)


if __name__ == "__main__":
    main()
