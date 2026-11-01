"""Summaries of the LTL331 ConfidenceOT run read from its files (local, after the TACC download): gates by the authors'
cell types, depth, agreement between the two representations, the coupling.

    python ltl331_nepc/05_gate_summary.py --run-root RUN_ROOT --converted CONVERTED --out OUT

RUN_ROOT is the downloaded ltl331_confidenceot/ (runs/acc0.99/ltl331/<label>/<pair>/scope_malignant/<budget>/,
inputs/ds/predownsample_depth.csv.gz); CONVERTED holds samples.tsv and h5ad/ (obs only is read). Cell types are the
authors' coarse_celltype (harmony object, joined by cell name); the integer seurat_clusters / orig.ident columns are
stored codes and are not used. Every pair, side, method and label is summarised; nothing is selected.

Tables (OUT/tables/):
  pair_metrics.tsv          every pair_metrics.csv row with label and pair
  gate_by_celltype.tsv      rejected fraction per label, method, pair, side, coarse_celltype (n cells)
  gate_by_phase.tsv         the same per cell-cycle phase (velocity object; 227 cells have none)
  depth_auc.tsv             AUC of the cell's pre-equalisation depth for rejected vs retained, per pair-side, all
                            cells and within each cell type (>= 50 cells, both groups present)
  depth_stratified.tsv      rejected fraction per depth quintile (within pair-side) x cell type
  latent_time_rho.tsv       Spearman rho between latent_time and signed_rejection_margin per pair-side-type (>= 50)
  arm_agreement.tsv         per pair-side: cells rejected in both labels, agreement, Cohen's kappa (M4-E)
  coupling_rows.tsv         per label, method, pair: median normalised row entropy of P (1 = uniform over targets),
                            mass in the retained x retained block
  coupling_type_mass.tsv    coupling mass between source and target cell types, full P and the retained block,
                            as share of each target type's mass and as enrichment over the independent coupling
Figures (OUT/figures/, one panel per file, PDF + PNG): rejected fraction per pair (source / target) per label and
method; rejected fraction by cell type per label (one heatmap per side); depth-stratified rejected fraction per
pair-side; row-entropy per pair.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

TYPES = ["PRAD", "Basal", "Other", "NE"]
LABELS = ["ranknm256_noscale_ds_cos", "logcpm"]


def pair_dirs(run_root: Path) -> list[tuple[str, str, Path]]:
    out = []
    for f in sorted(glob.glob(str(run_root / "runs" / "acc0.99" / "ltl331" / "*" / "*" / "scope_malignant" / "*" / "pair_metrics.csv"))):
        d = Path(f).parent
        out.append((d.parents[2].name, d.parents[1].name.replace("LTL331__", ""), d))
    return out


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def short(pair: str) -> str:
    a, b = pair.split("__")
    return f"{a.split('_')[0]}→{b.split('_')[0]}"


def kappa(x: np.ndarray, y: np.ndarray) -> float:
    po = np.mean(x == y); px, py = x.mean(), y.mean()
    pe = px * py + (1 - px) * (1 - py)
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    tables, figs = a.out / "tables", a.out / "figures"
    tables.mkdir(parents=True)

    obs = []
    for f in sorted((a.converted / "h5ad").glob("*.h5ad")):
        x = ad.read_h5ad(f, backed="r"); obs.append(x.obs[["sample_id", "coarse_celltype", "phase", "latent_time"]].copy()); x.file.close()
    obs = pd.concat(obs)
    obs["coarse_celltype"] = obs["coarse_celltype"].astype(str); obs["phase"] = obs["phase"].astype(object)
    depth = pd.read_csv(a.run_root / "inputs" / "ds" / "predownsample_depth.csv.gz").set_index("observation_id")["predownsample_total_counts"]

    dirs = pair_dirs(a.run_root)
    metrics, cells, coup_rows, coup_mass = [], [], [], []
    for label, pair, d in dirs:
        m = pd.read_csv(d / "pair_metrics.csv"); m.insert(0, "pair", pair); m.insert(0, "label", label); metrics.append(m)
        c = pd.read_csv(d / "cell_confidence.csv", usecols=["method", "side", "observation_id", "rejected", "signed_rejection_margin"])
        c.insert(0, "pair", pair); c.insert(0, "label", label); cells.append(c)
        for method in ["M4-E", "M4-R"]:
            f = d / f"coupling_{method.lower().replace('-', '')}.npz"
            if not f.is_file():
                continue
            P = np.load(f)["coupling"]
            s = c[(c.method == method) & (c.side == "source")]; t = c[(c.method == method) & (c.side == "target")]
            if P.shape != (len(s), len(t)):
                sys.exit(f"{f}: coupling {P.shape} vs {len(s)} x {len(t)} cells")
            R = P / P.sum(1, keepdims=True)
            H = -(R * np.log(np.where(R > 0, R, 1.0))).sum(1) / np.log(P.shape[1])
            rs, rt = ~s.rejected.to_numpy(bool), ~t.rejected.to_numpy(bool)
            coup_rows.append({"label": label, "method": method, "pair": pair, "source_n": P.shape[0], "target_n": P.shape[1],
                              "median_normalised_row_entropy": float(np.median(H)), "total_mass": float(P.sum()),
                              "retained_block_mass_fraction": float(P[np.ix_(rs, rt)].sum() / P.sum())})
            st = obs.loc[s.observation_id, "coarse_celltype"].to_numpy(); tt = obs.loc[t.observation_id, "coarse_celltype"].to_numpy()
            for block, (ms, mt) in {"full": (np.ones_like(rs), np.ones_like(rt)), "retained": (rs, rt)}.items():
                Q = P[np.ix_(ms, mt)]; qs, qt = st[ms], tt[mt]
                M = np.array([[Q[np.ix_(qs == x, qt == y)].sum() for y in TYPES] for x in TYPES])
                ind = np.outer(M.sum(1), M.sum(0)) / M.sum() if M.sum() > 0 else np.full_like(M, np.nan)
                for i, x in enumerate(TYPES):
                    for j, y in enumerate(TYPES):
                        coup_mass.append({"label": label, "method": method, "pair": pair, "block": block, "source_type": x, "target_type": y,
                                          "mass": M[i, j], "share_of_target_type": M[i, j] / M[:, j].sum() if M[:, j].sum() > 0 else np.nan,
                                          "enrichment_over_independent": M[i, j] / ind[i, j] if ind[i, j] > 0 else np.nan})
    metrics = pd.concat(metrics); metrics.to_csv(tables / "pair_metrics.tsv", sep="\t", index=False)
    cells = pd.concat(cells, ignore_index=True)
    missing = ~cells.observation_id.isin(obs.index)
    if missing.any():
        sys.exit(f"{missing.sum()} gated cells have no obs row")
    cells = cells.join(obs[["coarse_celltype", "phase", "latent_time"]], on="observation_id")
    cells["depth"] = cells.observation_id.map(depth)
    pd.DataFrame(coup_rows).to_csv(tables / "coupling_rows.tsv", sep="\t", index=False)
    pd.DataFrame(coup_mass).to_csv(tables / "coupling_type_mass.tsv", sep="\t", index=False)

    keys = ["label", "method", "pair", "side"]
    g = cells.groupby(keys + ["coarse_celltype"]).rejected.agg(n="size", rejected_fraction="mean").reset_index()
    g.to_csv(tables / "gate_by_celltype.tsv", sep="\t", index=False)
    cells.assign(phase=cells.phase.fillna("none")).groupby(keys + ["phase"]).rejected.agg(n="size", rejected_fraction="mean").reset_index() \
        .to_csv(tables / "gate_by_phase.tsv", sep="\t", index=False)

    auc, strat, rho = [], [], []
    for k, grp in cells.groupby(keys):
        for t, sub in [("all", grp)] + list(grp.groupby("coarse_celltype")):
            y = sub.rejected.to_numpy(int)
            if len(sub) >= 50 and 0 < y.mean() < 1:
                auc.append(dict(zip(keys, k), coarse_celltype=t, n=len(sub), rejected_fraction=y.mean(),
                                auc_depth_rejected=roc_auc_score(y, sub.depth)))
            if t != "all":
                lt = sub.dropna(subset=["latent_time"])
                if len(lt) >= 50:
                    rho.append(dict(zip(keys, k), coarse_celltype=t, n=len(lt),
                                    spearman_latent_time_vs_margin=spearmanr(lt.latent_time, lt.signed_rejection_margin)[0]))
        q = grp.assign(depth_quintile=pd.qcut(grp.depth, 5, labels=False))
        s = q.groupby(["depth_quintile", "coarse_celltype"]).rejected.agg(n="size", rejected_fraction="mean").reset_index()
        for col, v in zip(keys, k):
            s.insert(0, col, v)
        strat.append(s)
    pd.DataFrame(auc).to_csv(tables / "depth_auc.tsv", sep="\t", index=False)
    strat = pd.concat(strat); strat.to_csv(tables / "depth_stratified.tsv", sep="\t", index=False)
    pd.DataFrame(rho).to_csv(tables / "latent_time_rho.tsv", sep="\t", index=False)

    agree = []
    for method in ["M4-E", "M4-R"]:
        w = cells[cells.method == method].pivot_table(index=["pair", "side", "observation_id"], columns="label", values="rejected").dropna()
        for (pair, side), sub in w.groupby(level=[0, 1]):
            x, y = sub[LABELS[0]].to_numpy(int), sub[LABELS[1]].to_numpy(int)
            agree.append({"method": method, "pair": pair, "side": side, "n": len(sub), f"rejected_{LABELS[0]}": x.mean(),
                          f"rejected_{LABELS[1]}": y.mean(), "rejected_both": np.mean(x & y), "agreement": np.mean(x == y),
                          "cohen_kappa": kappa(x, y)})
    pd.DataFrame(agree).to_csv(tables / "arm_agreement.tsv", sep="\t", index=False)

    # figures, one panel per file
    pairs = sorted(cells.pair.unique())
    for label in LABELS:
        for method in ["M4-E", "M4-R"]:
            m = metrics[(metrics.label == label) & (metrics.method == method)].set_index("pair").loc[pairs]
            fig, ax = plt.subplots(figsize=(4.2, 2.6)); xs = np.arange(len(pairs))
            ax.bar(xs - 0.2, m.source_final_rejection_rate, 0.4, label="source", color="#4C72B0")
            ax.bar(xs + 0.2, m.target_final_rejection_rate, 0.4, label="target", color="#DD8452")
            ax.set_xticks(xs); ax.set_xticklabels([short(p) for p in pairs], rotation=45, ha="right", fontsize=7)
            ax.set_ylim(0, 1); ax.set_ylabel("rejected fraction"); ax.set_title(f"{label}, {method}", fontsize=8)
            ax.legend(frameon=False, fontsize=7); ax.spines[["top", "right"]].set_visible(False)
            save(fig, figs, f"rejected_per_pair__{label}__{method}")
        for side in ["source", "target"]:
            h = g[(g.label == label) & (g.method == "M4-E") & (g.side == side)].pivot(index="pair", columns="coarse_celltype", values="rejected_fraction").reindex(index=pairs, columns=TYPES)
            n = g[(g.label == label) & (g.method == "M4-E") & (g.side == side)].pivot(index="pair", columns="coarse_celltype", values="n").reindex(index=pairs, columns=TYPES)
            fig, ax = plt.subplots(figsize=(3.4, 3.0))
            im = ax.imshow(h.to_numpy(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
            for i in range(h.shape[0]):
                for j in range(h.shape[1]):
                    if not np.isnan(h.iat[i, j]):
                        ax.text(j, i, f"{h.iat[i, j]:.2f}\n{int(n.iat[i, j])}", ha="center", va="center", fontsize=5.5,
                                color="white" if h.iat[i, j] < 0.6 else "black")
            ax.set_xticks(range(len(TYPES))); ax.set_xticklabels(TYPES, fontsize=7)
            ax.set_yticks(range(len(pairs))); ax.set_yticklabels([short(p) for p in pairs], fontsize=7)
            ax.set_title(f"M4-E {side} rejected fraction (cells)\n{label}", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, label="rejected fraction")
            save(fig, figs, f"rejected_by_celltype__{label}__M4-E__{side}")
        for (pair, side), s in strat[(strat.label == label) & (strat.method == "M4-E")].groupby(["pair", "side"]):
            fig, ax = plt.subplots(figsize=(2.8, 2.2))
            for t, col in zip(TYPES, ["#4C72B0", "#55A868", "#8172B3", "#C44E52"]):
                u = s[(s.coarse_celltype == t) & (s.n >= 10)]
                if len(u):
                    ax.plot(u.depth_quintile + 1, u.rejected_fraction, "o-", ms=3, lw=1, color=col, label=t)
            ax.set_ylim(-0.02, 1.02); ax.set_xticks(range(1, 6)); ax.set_xlabel("pre-equalisation depth quintile", fontsize=7)
            ax.set_ylabel("rejected fraction", fontsize=7); ax.set_title(f"{short(pair)} {side}, {label}", fontsize=7)
            ax.legend(frameon=False, fontsize=6); ax.spines[["top", "right"]].set_visible(False)
            save(fig, figs / "depth_stratified", f"depth_stratified__{label}__{pair}__{side}")
    cr = pd.DataFrame(coup_rows)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    for (label, method), u in cr.groupby(["label", "method"]):
        u = u.set_index("pair").loc[pairs]
        ax.plot(range(len(pairs)), u.median_normalised_row_entropy, "o-", ms=3, lw=1, label=f"{label} {method}")
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels([short(p) for p in pairs], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("median row entropy / log(n target)", fontsize=7); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=6); ax.spines[["top", "right"]].set_visible(False)
    save(fig, figs, "coupling_row_entropy")
    print(f"{len(dirs)} pair directories, {cells.observation_id.nunique()} cells; tables and figures in {a.out}")


if __name__ == "__main__":
    main()
