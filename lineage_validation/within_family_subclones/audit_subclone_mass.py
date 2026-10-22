"""Read-only numerical audit of the subclone mass calculation (classify_subclones.py).

    python lineage_validation/within_family_subclones/audit_subclone_mass.py --defs DEFS --export EXPORT \\
        [--family KPTracer_3515_3515_Lkb1_T1 ...] [--strategies rna cnv cnv_then_rna] [--out FILE.txt]

For each family, strategy and method, on the saved couplings and gates of the export:
  1. mass in the raw coupling: total, rows of rejected / excluded source cells, columns of rejected / excluded
     target cells, the retained x retained block
  2. alignment of coupling rows / columns with cell_confidence.csv, tested from the coupling alone: in M4-E every
     entry touching a gated cell has the same cost c, so the columns of rejected targets (and the rows of rejected
     sources) form an exact rank-one block; the residual of log P on that block is ~0 only if the rejected flags sit
     on the right rows. The same test on as many randomly chosen retained rows / columns is the negative control
  3. two-step: step-1-excluded cells absent from the step-2 gate and carrying zero mass in the extended matrix
  4. subclone labels: every source cell has one label (no cell in two clades)
  5. normalisation, as classify() computes it: for retained targets with retained-source mass, the group
     weights over all groups (subclones + family unlabelled + other) sum to 1 and the within-family weights sum
     to 1; for rejected / excluded targets, what the output table holds
  6. baselines per source subclone: share of retained cells (what the size reference and the heatmap's top row
     used) against share of retained transport mass sum_{i in g} rho_i a_i (a = P 1); and the mean within-family
     weight per true subclone
  7. how far each retained target's within-family profile is from the independent coupling: mean total
     variation between q~_{.j} and the retained-mass share, at cell and subclone level, and the effective
     number of source partners exp(H(q_{.j}))
Prints everything; writes nothing but --out.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import lv_common as lv  # noqa: E402

spec = importlib.util.spec_from_file_location("classify_subclones", HERE / "classify_subclones.py")
cs = importlib.util.module_from_spec(spec)
sys.modules["classify_subclones"] = cs
spec.loader.exec_module(cs)


def rank_one_residual(block: np.ndarray) -> float:
    """max |log B - (r_i + c_j)| after double centring; 0 for an exact rank-one positive block."""
    if block.shape[0] < 2 or block.shape[1] < 2 or (block <= 0).any():
        return float("nan")
    L = np.log(block)
    R = L - L.mean(axis=1, keepdims=True) - L.mean(axis=0, keepdims=True) + L.mean()
    return float(np.abs(R).max())


def audit_one(export, defs, name, row, strategy, method, rng):
    block, pair = cs.run_for(row)
    run = cs.load_export_run(export, strategy, block, pair, method)
    P, rho, tau = run["P"], run["rho"], run["tau"]
    gate_dir = export / "ot" / "gates" / block / pair / strategy
    gate = lv.read_gate(gate_dir, method)
    src_gate, tgt_gate = gate[gate["side"].eq("source")], gate[gate["side"].eq("target")]
    raw = np.load(export / "couplings" / block / pair / strategy / f"coupling_{method.lower().replace('-', '')}.npz")["coupling"]
    print(f"\n=== {name} | {pair} | {strategy} | {method}")
    print(f"raw coupling {raw.shape}, gate rows {len(src_gate)} x {len(tgt_gate)}, duplicate ids "
          f"{int(src_gate['observation_id'].duplicated().sum())} / {int(tgt_gate['observation_id'].duplicated().sum())}")
    # 1. mass
    rs, rt = src_gate["retained"].to_numpy(), tgt_gate["retained"].to_numpy()
    m = raw.sum()
    print(f"[1] total mass {m:.6g}; rows of rejected sources {raw[~rs].sum() / m:.4f} of it; columns of rejected targets "
          f"{raw[:, ~rt].sum() / m:.4f}; retained x retained block {raw[np.ix_(rs, rt)].sum() / m:.4f}; "
          f"zero entries {int((raw == 0).sum())}; min entry {raw.min():.3g}")
    # 2. alignment by the rank-one structure of the gated block
    out = {}
    for label, mask, axis in (("rejected target columns", ~rt, 1), ("rejected source rows", ~rs, 0)):
        k = int(mask.sum())
        if k < 2:
            print(f"[2] {label}: {k} (too few for the rank-one test)")
            continue
        blockm = raw[:, mask] if axis == 1 else raw[mask, :]
        pool = np.flatnonzero(~mask)
        pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
        control = raw[:, pick] if axis == 1 else raw[pick, :]
        out[label] = (rank_one_residual(blockm), rank_one_residual(control))
        print(f"[2] {label} ({k}): log-residual of the rank-one fit {out[label][0]:.2e}; "
              f"random retained {'columns' if axis == 1 else 'rows'} (control) {out[label][1]:.2e}")
    # 3. two-step
    if strategy in lv.STEP1:
        base = lv.read_gate(export / "ot" / "gates" / block / pair / lv.STEP1[strategy], "M4-E")
        bs, bt = base[base["side"].eq("source")], base[base["side"].eq("target")]
        ex_s = set(bs.loc[~bs["retained"], "observation_id"]); ex_t = set(bt.loc[~bt["retained"], "observation_id"])
        in_step2 = ex_s & set(src_gate["observation_id"]) | ex_t & set(tgt_gate["observation_id"])
        excl_rows = np.isin(run["source"], list(ex_s)); excl_cols = np.isin(run["target"], list(ex_t))
        print(f"[3] step 1 excluded {len(ex_s)} source / {len(ex_t)} target cells; of them in the step-2 gate {len(in_step2)}; "
              f"mass on their rows {P[excl_rows].sum():.3g}, columns {P[:, excl_cols].sum():.3g}; "
              f"rho on them {int(rho[excl_rows].sum())}, tau {int(tau[excl_cols].sum())}")
    # 4. labels
    cells = pd.read_csv(defs / "cells" / f"{name}.csv.gz", dtype=str, keep_default_na=False)
    subs = pd.read_csv(defs / "subclones.tsv", sep="\t", dtype=str)
    subs = subs[subs["name"] == name]
    g1 = subs.loc[subs["level"] == "1", "subclone"].tolist()
    dup = int(cells["cell_id"].duplicated().sum())
    l1 = dict(zip(cells["cell_id"], cells["L1"]))
    if row["dataset"] == "KPTracer":
        meta = pd.concat([pd.read_csv(p, dtype=str) for p in sorted((export / "meta").glob("KPTracer_meta_*.csv.gz"))])
        fam = dict(zip(meta["cell_id"], meta["MetFamily"]))
        in_family = lambda c: fam.get(c, "") == row["family_or_clone"]
        other = lambda c: False
    else:
        lab = pd.read_csv(export / "labels" / "cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
        lab = lab[(lab["dataset"] == "Quinn2021") & (lab["mouse"] == str(row["mouse"]))].set_index("cell")["lineage"]
        clone = row["family_or_clone"][2:]
        in_family = lambda c: lab.get(c, "") == clone
        other = lambda c: lab.get(c, "") != clone
    src = [cs.OTHER if other(c) else (l1.get(c, "") if l1.get(c, "") in g1 else cs.UNLABELLED) for c in run["source"]]
    tgt = [l1.get(c, "ambiguous") if in_family(c) else None for c in run["target"]]
    print(f"[4] cells table rows {len(cells)}, duplicate cell ids {dup}; level-1 labels per source cell: "
          + ", ".join(f"{g} {src.count(g)}" for g in g1 + [cs.UNLABELLED, cs.OTHER]))
    # 5. normalisation as classify() computes it
    table, summary, _, _ = cs.classify(run, src, tgt, g1, 1, None)
    groups = g1 + [cs.UNLABELLED, cs.OTHER]
    gindex = {g: k for k, g in enumerate(groups)}
    gs = np.array([gindex[x] for x in src])
    GP = lv.onehot(gs, len(groups), rho.astype(float)) @ P
    D = GP.sum(axis=0)
    ok = tau & (D > 0)
    full = GP[:, ok] / D[ok]
    fam_rows = [gindex[g] for g in g1] + [gindex[cs.UNLABELLED]]
    within = table.loc[ok, [f"S__{g}" for g in g1] + [f"S__{cs.UNLABELLED}"]].to_numpy(float)
    fam_tab = table[[c is not None for c in tgt]]
    rejected_rows = fam_tab[~fam_tab["state"].eq("retained")]
    print(f"[5] retained targets with retained-source mass {int(ok.sum())}: max |sum over all groups - 1| {np.abs(full.sum(0) - 1).max():.2e}; "
          f"max |within-family sum - 1| {np.abs(within.sum(1) - 1).max():.2e}; "
          f"sum(subclones + unlabelled + other) equals D_j: {np.allclose(GP[:, ok].sum(0), D[ok])}")
    if len(rejected_rows):
        held = rejected_rows[[f"S__{g}" for g in g1] + [f"S__{cs.UNLABELLED}"]].to_numpy(float).sum(1)
        print(f"    family targets not retained {len(rejected_rows)} (outcome {rejected_rows['outcome'].value_counts().to_dict()}): "
              f"the table's S__ columns sum to {held.min():.3f}-{held.max():.3f} for them (not zeroed); family_weight "
              f"{rejected_rows['family_weight'].min():.3f}-{rejected_rows['family_weight'].max():.3f}")
    # 6. baselines
    a = P.sum(axis=1) / P.sum()
    fam_src = np.isin(gs, fam_rows)
    ra = rho * a
    count_share = {g: int(((gs == gindex[g]) & rho).sum()) for g in g1}
    tc = sum(count_share.values()) + int(((gs == gindex[cs.UNLABELLED]) & rho).sum())
    mass_share = {g: float(ra[gs == gindex[g]].sum()) for g in g1 + [cs.UNLABELLED]}
    tm = sum(mass_share.values())
    print("[6] per source subclone: retained cells / share of retained family cells / share of retained transport mass")
    for g in g1 + [cs.UNLABELLED]:
        n = count_share.get(g, int(((gs == gindex[g]) & rho).sum()))
        print(f"    {g:18s} {n:6d}  {n / tc if tc else np.nan:.3f}  {mass_share[g] / tm if tm else np.nan:.3f}")
    fam_t = table[(table["true"].isin(g1)) & table["state"].eq("retained")]
    print("    mean within-family weight of retained metastatic cells, by true subclone (rows) x source group (columns):")
    print(fam_t.groupby("true")[[f"S__{g}" for g in g1] + [f"S__{cs.UNLABELLED}"]].mean().round(3).to_string())
    # 7. distance from independence
    keep_cols = np.flatnonzero(ok & np.array([t in g1 for t in tgt]))
    if len(keep_cols):
        rows = np.flatnonzero(fam_src & rho)
        Q = P[np.ix_(rows, keep_cols)]
        Q = Q / Q.sum(axis=0, keepdims=True)
        indep = ra[rows] / ra[rows].sum()
        tv_cell = 0.5 * np.abs(Q - indep[:, None]).sum(axis=0)
        ent = -(np.where(Q > 0, Q * np.log(Q), 0)).sum(axis=0)
        sub_share = np.array([mass_share[g] / tm for g in g1 + [cs.UNLABELLED]])
        tv_sub = 0.5 * np.abs(within[np.isin(np.flatnonzero(ok), keep_cols)] - sub_share).sum(axis=1)
        print(f"[7] {len(keep_cols)} retained labelled metastatic cells, {len(rows)} retained family source cells: "
              f"total variation from the independent coupling, cell level mean {tv_cell.mean():.3f} (median {np.median(tv_cell):.3f}); "
              f"subclone level mean {tv_sub.mean():.3f}; effective source partners exp(H) median {np.exp(np.median(ent)):.0f} of {len(rows)}")
    acc = summary["accuracy"]
    size_mass = max(g1, key=lambda g: mass_share[g])
    scored = table[table["outcome"].isin(["correct", "incorrect"])]
    print(f"    accuracy {acc:.3f}; size reference by retained cells {summary['size_reference_subclone']} "
          f"{summary['size_reference_accuracy']:.3f}; by retained mass {size_mass} {(scored['true'] == size_mass).mean():.3f}; "
          f"chance by cells {summary['chance_accuracy']:.3f}, by mass {np.mean([mass_share[t] / tm for t in scored['true']]) if len(scored) else np.nan:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--defs", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--family", nargs="+", default=["KPTracer_3515_3515_Lkb1_T1", "Quinn2021_10k_LG2"])
    parser.add_argument("--strategies", nargs="+", default=["rna", "cnv", "cnv_then_rna"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    feas = pd.read_csv(args.defs / "feasibility.tsv", sep="\t", dtype={"mouse": str})
    rng = np.random.default_rng(20261022)
    buf = io.StringIO()
    with redirect_stdout(buf):
        for name in args.family:
            row = next(r for _, r in feas.iterrows() if f"{r['dataset']}_{r['mouse']}_{r['family_or_clone']}" == name)
            for strategy in args.strategies:
                for method in lv.METHODS:
                    audit_one(args.export, args.defs, name, row, strategy, method, rng)
    text = buf.getvalue()
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
