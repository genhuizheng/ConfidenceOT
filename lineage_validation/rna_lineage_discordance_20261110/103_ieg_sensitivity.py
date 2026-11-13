"""PLAN_v3_deg.md immediate-early-gene sensitivity: the hallmark prerank of every real (non-random) contrast rerun on
its saved Wilcoxon ranking with the immediate-early genes removed; same gseapy parameters as 60_deg_gsea.py.
Writes RUN/gsea_hallmark_noieg.tsv.

    python lineage_validation/rna_lineage_discordance_20261110/103_ieg_sensitivity.py \
        --run results/rna_lineage_discordance_20261110/deg_v3/deg_gsea_raw \
        --contrasts results/rna_lineage_discordance_20261110/deg_v3/contrasts.tsv \
        --gmt-mouse results/reference/msigdb_mouse/mh.all.2025.1.Mm.symbols.gmt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

IEG = ["Fos", "Fosb", "Jun", "Junb", "Jund", "Egr1", "Egr2", "Egr3", "Ier2", "Ier3", "Ier5", "Atf3", "Dusp1", "Nr4a1",
       "Klf2", "Klf4", "Klf6", "Zfp36", "Btg2", "Cyr61", "Ccn1", "Hspa1a", "Hspa1b"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--contrasts", type=Path, required=True)
    p.add_argument("--gmt-mouse", type=Path, required=True)
    a = p.parse_args()
    import gseapy
    c = pd.read_csv(a.contrasts, sep="\t", dtype={"mouse": str})
    real = c[~c.type.str.contains("_rand")]
    out = []
    for i, r in enumerate(real.itertuples()):
        deg = pd.read_csv(a.run / "deg" / f"{r.contrast_id}.tsv.gz", sep="\t")
        deg = deg[~deg.names.isin(IEG)]
        rnk = deg[["names", "scores"]].drop_duplicates("names").set_index("names")["scores"].sort_values(ascending=False)
        res = gseapy.prerank(rnk=rnk, gene_sets=str(a.gmt_mouse), permutation_num=1000, seed=20261110, min_size=15, max_size=500,
                             threads=4, outdir=None, verbose=False).res2d
        res = res.rename(columns={"Term": "term", "NES": "nes", "FDR q-val": "fdr"})
        res["contrast_id"] = r.contrast_id
        out.append(res[["contrast_id", "term", "nes", "fdr"]])
        print(f"{i + 1}/{len(real)} {r.contrast_id}", flush=True)
    pd.concat(out).to_csv(a.run / "gsea_hallmark_noieg.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
