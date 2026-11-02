"""Worklist of the author-provided Harmony arm of LTL331 (runs on TACC; small).

    python realdata_benchmark/19_build_author_harmony_worklist.py --root ROOT --embedding AUTHOR_EMBEDDINGS_LTL331.npz \
        --key X_harmony [--costs cos sqeuc]

ROOT is the joint-benchmark root that holds worklist_ltl331.tsv (the 64-configuration arm; read, never changed). For
each of its 7 pairs and each cost: one row, config_id author_harmony|<KEY>|<cost>, the same manifests / run root /
converted dir / acceptance, plus author_embedding and author_key (10_tacc_worker.py passes them to 05_), out under
ROOT/ltl331_author_harmony/. Writes ROOT/worklist_ltl331_author_harmony.tsv; refuses if it exists. Stops if KEY is not in
the npz or any cell of the 7 pairs' frozen equalised manifest has no coordinates (the arm uses exactly those cells).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--embedding", type=Path, required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--costs", nargs="*", default=["cos", "sqeuc"])
    a = p.parse_args()
    target = a.root / "worklist_ltl331_author_harmony.tsv"
    if target.exists():
        sys.exit(f"{target} exists; not rebuilt")
    z = np.load(a.embedding, allow_pickle=False)
    if a.key not in z.files:
        sys.exit(f"{a.key} not in {a.embedding} ({z.files})")
    E = pd.DataFrame(z[a.key], index=pd.Index(z["cellid"].astype(str)))
    base = pd.read_csv(a.root / "worklist_ltl331.tsv", sep="\t", dtype=str).drop_duplicates("pair_index")
    man = pd.read_csv(base.manifest_ds.iloc[0])
    missing = 0
    for _, r in man.iterrows():
        for side in ("source_h5ad", "target_h5ad"):
            f = Path(r[side]); f = f if f.exists() else Path(base.run_root.iloc[0]) / "inputs" / "ds" / "h5ad" / f.name
            ids = ad.read_h5ad(f, backed="r").obs_names.astype(str)
            missing += int(E.reindex(ids).isna().any(axis=1).sum())
    if missing:
        sys.exit(f"{missing} cells of the frozen equalised manifest have no {a.key} coordinates")
    rows = []
    for _, r in base.sort_values("pair_index", key=lambda s: s.astype(int)).iterrows():
        for cost in a.costs:
            cid = f"author_harmony|{a.key}|{cost}"
            rows.append({**r.to_dict(), "config_id": cid, "author_embedding": str(a.embedding.resolve()), "author_key": a.key,
                         "out": str(a.root / "ltl331_author_harmony" / cid.replace("|", "__") / f"pair{r.pair_index}")})
    pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
    print(pd.DataFrame(rows)[["config_id", "pair_index", "pair_id"]].to_string(index=False))
    print(f"{len(rows)} rows -> {target}")


if __name__ == "__main__":
    main()
