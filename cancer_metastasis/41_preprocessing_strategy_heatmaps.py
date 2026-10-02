"""Two heatmaps: every preprocessing strategy against the genes and the
pathways it calls.

The question these answer is not which strategy is best. It is **how much of
the biological conclusion is a property of the preprocessing choice**. If one
gene list survives all thirty-two cells of the factorial, the conclusion is
robust to a decision nobody has evidence about. If every cell gives a
different list, the list was never a finding.

That framing matters because the project's own standard -- a configuration
must not be chosen because it produces a more interesting biological result --
would otherwise be violated by the very act of drawing this figure. It is not
violated, because nothing here selects a configuration. A row is not better
for having more coloured cells; a *column* that is coloured in every row is
what the figure is looking for.

**The columns are a union, never an intersection.** A gene enters the figure
if any strategy calls it, and every strategy then shows what it actually found
there -- including nothing. Intersecting would keep only the genes all
strategies agree on, which is a picture of the agreement with the
disagreement deleted: the strategies that dissent would be invisible, and so
would the genes they dissent about. The union is larger and harder to read and
it is the only version that can be wrong in a way you can see.

Two matrices per readout, because a cell has two things to say:

``value``
    The signed effect: ``log2_fold_change`` for genes, ``NES`` for pathways.
    Present whether or not the strategy called it significant, because "this
    strategy saw the same direction but missed the threshold" and "this
    strategy saw the opposite direction" are different failures and an
    unsigned significance map cannot tell them apart.

``called``
    Whether that strategy called it at its own FDR threshold. Drawn as a mark
    over the value, so the reader sees agreement in direction and agreement in
    significance as separate things.

Grey is absence and white is zero. They are different claims and a diverging
colour map puts white in the middle, so a figure that let them share a colour
would hide the silence it exists to show.

Rows are ordered by the factorial, not alphabetically: normalisation outermost
then the binary tags, so the effect of one axis is a comparison between
adjacent rows rather than a hunt across the figure.

    python cancer_metastasis/41_preprocessing_strategy_heatmaps.py OUT_DIR \\
        --deg LABEL=path/to/pydeseq2_all_gene_discovery.csv \\
        --gsea LABEL=path/to/xxx_gsea_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# The names 13_run_paired_pydeseq2.py and 15_run_pydeseq2_gseapy.py write,
# looked up rather than assumed: 13 renames baseMean/log2FoldChange/padj and
# 15 renames Term/FDR q-val, so the raw PyDESeq2 and GSEApy spellings never
# reach a file.
DEG_KEY, DEG_VALUE, DEG_FDR = "gene", "log2_fold_change", "fdr"
GSEA_KEY, GSEA_VALUE, GSEA_FDR = "pathway", "NES", "fdr"

# The 2^5 factorial, in factorial order. The four cells that put library-size
# normalisation and rank encoding together have no name in the production API
# and are a bit-for-bit identity with rank alone; with the equalisation axis
# there are eight of them, and tests/test_benchmark_identity.py asserts the
# identity rather than this figure pretending to show it.
#
# rank256 is a normalisation of its own: it divides each gene by its nonzero
# median over both sides before ranking, which ranknm256 does not. Without it
# here its eight arms fell to the bottom of the figure in read order.
STEMS = ("raw", "logcpm", "rank256", "ranknm256")
TAGS = (("noscale", "scale_genes"), ("ds", "equalise_depth"), ("cos", "cosine"))


def factorial_order() -> list[str]:
    """The thirty-two labels, ordered by the factorial."""
    order = []
    for stem in STEMS:
        for bits in range(8):
            parts = [stem]
            for position, (tag, _) in enumerate(TAGS):
                if bits & (1 << (len(TAGS) - 1 - position)):
                    parts.append(tag)
            order.append("_".join(parts))
    return order


def labelled(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"expected LABEL=PATH, found {value!r}")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("the label must not be empty")
    return label.strip(), Path(path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--deg", type=labelled, action="append", default=[],
                        metavar="LABEL=CSV",
                        help="pydeseq2_all_gene_discovery.csv for one strategy")
    parser.add_argument("--gsea", type=labelled, action="append", default=[],
                        metavar="LABEL=CSV",
                        help="<contrast>_gsea_results.csv for one strategy")
    parser.add_argument("--max-fdr", type=float, default=0.05)
    parser.add_argument(
        "--top", type=int, default=60,
        help="Columns to draw, chosen by how many strategies call them and "
             "then by effect size. The full union is always written to CSV; "
             "this only bounds the figure, because a union over two dozen "
             "strategies runs to thousands of genes and a heatmap that wide "
             "is not a readable object.")
    parser.add_argument("--title", default="")
    return parser.parse_args()


def read_one(path: Path, key: str, value: str, fdr: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = [column for column in (key, value, fdr) if column not in table]
    if missing:
        raise SystemExit(
            f"{path} is missing {missing}; it has {list(table.columns)[:12]}")
    table = table[[key, value, fdr]].copy()
    table[key] = table[key].astype(str)
    if table[key].duplicated().any():
        repeated = table.loc[table[key].duplicated(), key].unique()[:5]
        raise SystemExit(f"{path} repeats {key}: {list(repeated)}")
    return table.set_index(key)


def build(sources: list[tuple[str, Path]], key: str, value: str, fdr: str,
          max_fdr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strategy-by-feature value and called matrices, over the union."""
    tables = {label: read_one(path, key, value, fdr)
              for label, path in sources}
    # The union, built as a sorted set so the column order does not depend on
    # which strategy happened to be read first.
    union = sorted(set().union(*(set(table.index) for table in tables.values())))
    rows = [label for label in factorial_order() if label in tables]
    rows += [label for label in tables if label not in rows]
    values = pd.DataFrame(np.nan, index=rows, columns=union, dtype=float)
    called = pd.DataFrame(False, index=rows, columns=union, dtype=bool)
    for label in rows:
        table = tables[label]
        shared = table.index.intersection(union)
        values.loc[label, shared] = pd.to_numeric(
            table.loc[shared, value], errors="coerce").to_numpy()
        significance = pd.to_numeric(table.loc[shared, fdr], errors="coerce")
        called.loc[label, shared] = (significance < max_fdr).to_numpy()
    return values, called


def draw(values: pd.DataFrame, called: pd.DataFrame, out: Path, *,
         title: str, value_name: str, top: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Ordered by how many strategies call it, then by the median effect. A
    # column every strategy calls is the robust end; a column one strategy
    # calls is the fragile end. Putting them next to each other is the point.
    rank = pd.DataFrame({
        "calls": called.sum(axis=0),
        "effect": values.abs().median(axis=0, skipna=True),
    }).sort_values(["calls", "effect"], ascending=[False, False])
    keep = rank.index[:top]
    shown, marks = values[keep], called[keep]

    height = max(2.4, 0.34 * len(shown) + 1.5)
    width = max(6.0, 0.20 * len(keep) + 3.0)
    figure, axes = plt.subplots(figsize=(width, height))
    limit = float(np.nanmax(np.abs(shown.to_numpy()))) if shown.size else 1.0
    limit = limit if np.isfinite(limit) and limit > 0 else 1.0
    # Absence has to look different from zero. A diverging map puts white in
    # the middle, so a strategy that never saw a gene and a strategy that saw
    # no effect would render identically -- and half of what this figure is
    # for is showing which strategies were silent about what.
    palette = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#d9d9d9")
    image = axes.imshow(shown.to_numpy(), aspect="auto", cmap=palette,
                        vmin=-limit, vmax=limit, interpolation="nearest")
    # Significance as a mark over the value, not as a second colour scale:
    # direction and significance are different claims.
    y, x = np.nonzero(marks.to_numpy())
    axes.scatter(x, y, s=9.0, c="black", marker="o", linewidths=0)

    axes.set_xticks(range(len(keep)))
    axes.set_xticklabels(keep, rotation=90, fontsize=5.5)
    axes.set_yticks(range(len(shown)))
    axes.set_yticklabels(shown.index, fontsize=7, family="monospace")
    axes.set_xlabel(f"{len(keep)} of {values.shape[1]} in the union, "
                    f"ordered by how many strategies call them")
    if title:
        axes.set_title(title, fontsize=9)
    bar = figure.colorbar(image, ax=axes, fraction=0.02, pad=0.01)
    bar.set_label(value_name, fontsize=7)
    bar.ax.tick_params(labelsize=6)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)


def report(values: pd.DataFrame, called: pd.DataFrame) -> dict:
    """How much of the answer is the preprocessing choice."""
    n_strategies = len(called)
    per_feature = called.sum(axis=0)
    unanimous = int((per_feature == n_strategies).sum())
    single = int((per_feature == 1).sum())
    # Direction disagreement among the strategies that called it at all.
    conflicting = 0
    for feature in values.columns:
        signs = np.sign(values.loc[called[feature], feature].dropna())
        if len(signs) > 1 and len(set(signs.tolist())) > 1:
            conflicting += 1
    return {
        "strategies": n_strategies,
        "union_size": int(values.shape[1]),
        "called_by_every_strategy": unanimous,
        "called_by_exactly_one": single,
        "called_with_conflicting_direction": conflicting,
        "per_strategy_called_n": {label: int(called.loc[label].sum())
                                  for label in called.index},
    }


def main() -> None:
    args = parse_args()
    if not args.deg and not args.gsea:
        raise SystemExit("pass at least one --deg or --gsea")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"max_fdr": args.max_fdr}

    for name, sources, key, value, fdr, value_name in (
        ("deg", args.deg, DEG_KEY, DEG_VALUE, DEG_FDR, "log2 fold change"),
        ("gsea", args.gsea, GSEA_KEY, GSEA_VALUE, GSEA_FDR, "NES"),
    ):
        if not sources:
            continue
        values, called = build(sources, key, value, fdr, args.max_fdr)
        values.to_csv(args.out_dir / f"{name}_value_matrix.csv")
        called.to_csv(args.out_dir / f"{name}_called_matrix.csv")
        draw(values, called, args.out_dir / f"{name}_heatmap.png",
             title=args.title or f"{name.upper()} by preprocessing strategy",
             value_name=value_name, top=args.top)
        summary[name] = report(values, called)
        block = summary[name]
        print(f"=== {name} ===")
        print(f"strategies                        {block['strategies']}")
        print(f"union                             {block['union_size']}")
        print(f"called by every strategy          "
              f"{block['called_by_every_strategy']}")
        print(f"called by exactly one             "
              f"{block['called_by_exactly_one']}")
        print(f"called with conflicting direction "
              f"{block['called_with_conflicting_direction']}")
        print()

    (args.out_dir / "strategy_heatmap_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
