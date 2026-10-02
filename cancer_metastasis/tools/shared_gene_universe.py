"""The genes every deposit of a merged analysis measured, for 13_'s --gene-universe.

A pseudobulk written by ``21_prepare_four_state_malignant_pseudobulk.py`` lists
only the genes that patient had counts for, and ``13_run_paired_pydeseq2.py``
takes the union over patients and fills the rest with 0. Inside one deposit
that is right: a gene missing from a patient's table had no counts there. A
merged analysis breaks it, because a gene another deposit never measured is
then read as zero counts in that deposit's patients -- and colorectal's three
deposits measure 18,005, 33,660 and 42,875 gene symbols, of which 16,951 are
common to all three.

So the measured set is read off the matrices, not off the counts: the gene
symbols of every h5ad a deposit's pairs read, collapsed with ``21_``'s own
``gene_symbols`` so the names match the pseudobulk's columns. Within a deposit
a gene is measured if any of its files carries it -- the same reading the
single-deposit runs already make -- and across deposits only the genes every
deposit measured are kept.

Usage:

    python cancer_metastasis/tools/shared_gene_universe.py MANIFEST_CSV OUT.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent


def load_pseudobulk_module():
    """21_ itself, so the gene symbols are collapsed exactly as it collapses them."""
    # 21_ imports common, and common imports confidenceot from src/.
    sys.path.insert(0, str(HERE.parent / "src"))
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(
        "pseudobulk", HERE / "21_prepare_four_state_malignant_pseudobulk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deposit_genes(manifest: pd.DataFrame, pseudobulk) -> dict[str, dict[str, object]]:
    """Per deposit: the symbols in any of its files, in every file, and the file count."""
    import anndata as ad

    found: dict[str, dict[str, object]] = {}
    for deposit, table in manifest.groupby(manifest["dataset_id"].astype(str)):
        paths: set[str] = set()
        for column in ("source_h5ads_json", "target_h5ads_json"):
            for value in table[column]:
                paths.update(json.loads(str(value)))
        per_file = []
        for path in sorted(paths):
            with warnings.catch_warnings():
                # Duplicate var names are expected; 21_ sums them by symbol.
                warnings.simplefilter("ignore", UserWarning)
                data = ad.read_h5ad(path, backed="r")
            per_file.append(set(pseudobulk.gene_symbols(data)))
            data.file.close()
        found[deposit] = {"any": set.union(*per_file),
                          "every": set.intersection(*per_file),
                          "files": len(per_file)}
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_txt", type=Path)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    found = deposit_genes(manifest, load_pseudobulk_module())
    for deposit, record in found.items():
        print(f"{deposit}: {len(record['any'])} gene symbols in any of "
              f"{record['files']} file(s), {len(record['every'])} in every one")
    shared = set.intersection(*(record["any"] for record in found.values()))
    if not shared:
        raise SystemExit("the deposits share no gene symbol")
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.write_text("\n".join(sorted(shared)) + "\n", encoding="utf-8")
    print(f"{len(shared)} genes measured in every deposit -> {args.output_txt}")


if __name__ == "__main__":
    main()
