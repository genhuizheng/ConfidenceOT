"""One organ case from the per-bin tables of 32_: gates, cost, markers, balanced-OT remapping, controls.

    python mouse_embryo/biology/33_organ_case.py --out <analysis root> --organ-set heart \
        [--runs ds logcpm] [--focal-group E11.5->E12.5] [--support-threshold 0.5]

Unit of analysis: one annotation x one pair x one side x one run x one method.
Nothing is filtered: every pair with organ bins is in every table, with its
calibration and convergence flags beside it (audit/METHOD_VALIDITY_AUDIT.tsv
from 31_); same-stage pairs are the controls of the adjacent ones.

Fixed in advance (not tuned on the results):
* support classes of a partner annotation: same annotation, curated successor
  (literature/moscot_curated_transitions.tsv when obtained), other
  (unclassified, not wrong);
* "high balanced-OT support" = the bin's balanced conditional probability of
  same + curated >= --support-threshold (default 0.5, a majority);
* marker-positive = at least one count of the gene in the equalised bin;
* near cutoff = |margin| <= 5% of the rejection cost;
* cutoff sensitivity = the final decision costs re-thresholded at 0.9, 0.95,
  1.05 and 1.1 x the rejection cost.  This is a static re-thresholding of one
  fit, not a refit (the gates of a refit would move with the coupling).

Writes tables under analysis/<organ_set>/ and one figure per panel (PDF + PNG)
under figures/<organ_set>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bio_common import STAGES, Outputs, curated_transitions, markers, organ_sets, support_classes  # noqa: E402

GROUPS = {  # gate groups on identical bins of one pair side
    "E_retained": lambda d: d.m4e_retained == 1, "E_rejected": lambda d: d.m4e_retained == 0,
    "R_retained": lambda d: d.m4r_retained == 1, "R_rejected": lambda d: d.m4r_retained == 0,
    "both_retained": lambda d: (d.m4e_retained == 1) & (d.m4r_retained == 1),
    "both_rejected": lambda d: (d.m4e_retained == 0) & (d.m4r_retained == 0),
    "E_only_retained": lambda d: (d.m4e_retained == 1) & (d.m4r_retained == 0),
    "R_only_retained": lambda d: (d.m4e_retained == 0) & (d.m4r_retained == 1),
    "all": lambda d: np.ones(len(d), dtype=bool),
}
KEYS = ["run", "annotation", "pair_id", "kind", "group", "side", "own_sample", "partner_sample", "own_stage", "partner_stage"]
SENSITIVITY = (0.9, 0.95, 1.05, 1.1)
RETAINED, REJECTED, GREY = "#1A9E8F", "#E4572E", "#C8C8C8"


def group_label(row) -> str:
    return row["group"] if row["kind"] == "adjacent" else f"{row['own_stage']} same-stage"


def load(out_root: Path, runs: list[str], organ: str) -> pd.DataFrame:
    frames = []
    for run in runs:
        path = out_root / "organ_bins" / run / organ / "bins.csv.gz"
        if path.is_file():
            frames.append(pd.read_csv(path, low_memory=False))
        else:
            print(f"no {path}; run {run} left out of this case", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_support(out_root: Path, bins: pd.DataFrame, organ: str, threshold: float) -> pd.DataFrame:
    """Per bin: same / curated / other probability under balanced OT, the M4-E coupling and (if rebuilt) M4-R."""
    curated = curated_transitions()
    parts = []
    for (run, pid, side), group in bins.groupby(["run", "pair_id", "side"], sort=False):
        group = group.copy()
        direction = "forward" if side == "source" else "backward"
        for kind, folder, key in (("bot", "distributions", "bot"), ("co", "distributions", "co"),
                                  ("m4r", "distributions_m4r", "m4r")):
            path = out_root / "organ_bins" / run / organ / folder / f"{pid}__{side}.npz"
            for c in ("same", "curated", "other"):
                group[f"{kind}_{c}"] = np.nan
            if not path.is_file():
                continue
            with np.load(path) as z:
                rows = pd.Series(np.arange(len(z["bin_row"])), index=z["bin_row"])
                d = z[key][rows.loc[group["bin_row"]].to_numpy()].astype(np.float64)
                partners = list(z["partner_annotations"])
            for annotation in group["annotation"].unique():
                members = (group["annotation"] == annotation).to_numpy()
                classes = support_classes(annotation, partners, curated, direction)
                for c in ("same", "curated", "other"):
                    cols = classes == c
                    values = d[members][:, cols].sum(axis=1) if cols.any() else np.zeros(members.sum())
                    values[~np.isfinite(d[members]).all(axis=1)] = np.nan
                    group.loc[members, f"{kind}_{c}"] = values
        parts.append(group)
    bins = pd.concat(parts, ignore_index=True)
    for kind in ("bot", "co", "m4r"):
        bins[f"{kind}_supported"] = bins[f"{kind}_same"] + bins[f"{kind}_curated"]
    bins["bot_high_support"] = np.where(bins["bot_supported"].notna(), bins["bot_supported"] >= threshold, np.nan)
    return bins


def gate_tables(bins: pd.DataFrame, validity: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, agreement = [], []
    for key, d in bins.groupby(KEYS, sort=False):
        base = dict(zip(KEYS, key))
        base.update(n_bins=len(d), partner_annotation_bins=int(d["partner_annotation_bins"].iloc[0]),
                    partner_has_annotation=bool(d["partner_has_annotation"].iloc[0]),
                    rejection_cost=float(d["rejection_cost"].iloc[0]))
        for m in ("m4e", "m4r"):
            base[f"{m}_retained"] = int(d[f"{m}_retained"].sum())
            base[f"{m}_rejected"] = int((d[f"{m}_retained"] == 0).sum())
            base[f"{m}_rejected_fraction"] = float((d[f"{m}_retained"] == 0).mean())
            base[f"{m}_raw_rejected_fraction"] = float((d[f"{m}_raw_retained"] == 0).mean())
            base[f"{m}_budget_overridden"] = int(d[f"{m}_budget_overridden"].sum())
            base[f"{m}_no_partner"] = int((d[f"{m}_decision_cost"] == 0).sum())
        rows.append(base)
        e, r = d["m4e_retained"] == 1, d["m4r_retained"] == 1
        union = int((e | r).sum())
        agreement.append({**{k: base[k] for k in KEYS}, "n_bins": len(d),
                          "both_retained": int((e & r).sum()), "both_rejected": int((~e & ~r).sum()),
                          "E_only_retained": int((e & ~r).sum()), "R_only_retained": int((~e & r).sum()),
                          "agreement_fraction": float((e == r).mean()),
                          "retained_jaccard": float((e & r).sum() / union) if union else np.nan})
    table, agree = pd.DataFrame(rows), pd.DataFrame(agreement)
    if validity is not None:
        flags = validity.pivot_table(index=["run", "pair_id"], columns="method",
                                     values=["outer_converged", "cycle_detected"], aggfunc="first")
        flags.columns = [f"{m.replace('-', '').lower()}_{v}" for v, m in flags.columns]
        flags = flags.join(validity.drop_duplicates(["run", "pair_id"]).set_index(["run", "pair_id"])[
            ["m4e_inference_valid", "calibration_valid", "validation_aggregate_valid"]]).reset_index()
        keep = [c for c in flags.columns if c.startswith(("run", "pair_id", "m4e_outer", "m4r_outer", "m4r_cycle",
                                                          "m4e_inference", "calibration_valid", "validation"))]
        table = table.merge(flags[keep], on=["run", "pair_id"], how="left")
        agree = agree.merge(flags[keep], on=["run", "pair_id"], how="left")
    return table, agree


def remapping(bins: pd.DataFrame) -> pd.DataFrame:
    """Balanced OT (and each gate's own coupling) on exactly the bins of each gate group."""
    rows = []
    for key, d in bins.groupby(KEYS, sort=False):
        for name, rule in GROUPS.items():
            members = d[np.asarray(rule(d), dtype=bool)]
            row = {**dict(zip(KEYS, key)), "gate_group": name, "n_bins": len(members)}
            for kind in ("bot", "co", "m4r"):
                for c in ("same", "curated", "other", "supported"):
                    values = members[f"{kind}_{c}"]
                    row[f"{kind}_{c}_mean"] = float(values.mean()) if values.notna().any() else np.nan
                row[f"{kind}_n_with_distribution"] = int(members[f"{kind}_same"].notna().sum())
            row["bot_high_support_fraction"] = (float(members["bot_high_support"].astype(float).mean())
                                                if members["bot_high_support"].notna().any() else np.nan)
            row["bot_spread_median"] = float(members["bot_partner_spread"].median()) if len(members) else np.nan
            row["co_spread_median"] = float(members["co_partner_spread"].median()) if len(members) else np.nan
            # paired, identical bins: the gate's own coupling against balanced OT on its retained bins
            for kind in ("co", "m4r"):
                both = members[f"{kind}_supported"].notna() & members["bot_supported"].notna()
                row[f"{kind}_minus_bot_supported_paired"] = (
                    float((members.loc[both, f"{kind}_supported"] - members.loc[both, "bot_supported"]).mean())
                    if both.any() else np.nan)
                row[f"{kind}_minus_bot_n_paired"] = int(both.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def four_way(bins: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for key, d in bins.groupby(KEYS, sort=False):
        known = d["bot_supported"].notna()
        for m in ("m4e", "m4r"):
            for state, flag in (("retained", 1), ("rejected", 0)):
                for support in ("high", "low"):
                    hit = known & (d[f"{m}_retained"] == flag) & ((d["bot_supported"] >= threshold) == (support == "high"))
                    rows.append({**dict(zip(KEYS, key)), "method": m, "gate": state, "bot_support": support,
                                 "threshold": threshold, "n_bins": int(hit.sum()),
                                 "fraction_of_bins_with_distribution": float(hit.sum() / known.sum()) if known.any() else np.nan,
                                 "bot_supported_mean": float(d.loc[hit, "bot_supported"].mean()) if hit.any() else np.nan})
    return pd.DataFrame(rows)


def rejected_audit(bins: pd.DataFrame, validity_table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Rejected bins split by hypothesis type (A absent / B present, low BOT support / D present, high BOT support)
    with technical flags (E): pair not inference-valid, M4-R not converged, bin below the equalisation depth."""
    flags = validity_table.set_index(["run", "pair_id", "annotation", "side"])
    rows = []
    for key, d in bins.groupby(KEYS, sort=False):
        k = dict(zip(KEYS, key))
        f = flags.loc[(k["run"], k["pair_id"], k["annotation"], k["side"])]
        for m in ("m4e", "m4r"):
            rej = d[d[f"{m}_retained"] == 0]
            present = bool(d["partner_has_annotation"].iloc[0])
            high = rej["bot_supported"] >= threshold
            rows.append({**k, "method": m, "rejected": len(rej),
                         "type_A_partner_absent": len(rej) if not present else 0,
                         "type_B_present_bot_low": int((present & ~high & rej["bot_supported"].notna()).sum()),
                         "type_D_present_bot_high": int((present & high).sum()),
                         "no_bot_distribution": int(rej["bot_supported"].isna().sum()),
                         "flag_pair_not_inference_valid": (np.nan if pd.isna(f.get("m4e_inference_valid", np.nan))
                                                           else not bool(f["m4e_inference_valid"])),
                         "flag_m4r_not_outer_converged": (np.nan if m != "m4r" or pd.isna(f.get("m4r_outer_converged", np.nan))
                                                          else not bool(f["m4r_outer_converged"])),
                         "below_equalisation_depth_bins": int(rej["untouched_by_equalisation"].fillna(0).astype(bool).sum()),
                         "near_cutoff_bins": int((rej[f"{m}_margin"].abs() <= 0.05 * rej["rejection_cost"]).sum()),
                         "relative_margin_median": float((rej[f"{m}_margin"] / rej["rejection_cost"]).median()) if len(rej) else np.nan})
    return pd.DataFrame(rows)


def cutoff_and_markers(bins: pd.DataFrame, out_root: Path, organ: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Decision cost against the cutoff; marker expression against cost and gate; static cutoff sensitivity."""
    from scipy.stats import mannwhitneyu, spearmanr

    genes = markers().query("organ_set == @organ")
    expression = {}
    for sample in bins["own_sample"].unique():
        path = out_root / "expression" / f"{sample}.csv.gz"
        if path.is_file():
            expression[sample] = pd.read_csv(path).set_index("bin_id")
    marker_rows, cost_rows, sens_rows = [], [], []
    for key, d in bins.groupby(KEYS, sort=False):
        k = dict(zip(KEYS, key))
        x = expression.get(k["own_sample"])
        for m in ("m4e", "m4r"):
            rel = d[f"{m}_margin"] / d["rejection_cost"]
            has_partner = d[f"{m}_decision_cost"] != 0
            cost_rows.append({**k, "method": m, "n_bins": len(d), "no_partner_bins": int((~has_partner).sum()),
                              "relative_margin_q10": float(rel.quantile(0.1)), "relative_margin_median": float(rel.median()),
                              "relative_margin_q90": float(rel.quantile(0.9)),
                              "near_cutoff_fraction": float((rel.abs() <= 0.05).mean()),
                              "decision_cost_median": float(d[f"{m}_decision_cost"].median()),
                              "rejection_cost": float(d["rejection_cost"].iloc[0])})
            base = float((d[f"{m}_retained"] == 0).mean())
            for factor in SENSITIVITY:
                rejected = ~(d[f"{m}_decision_cost"] < factor * d["rejection_cost"]) & has_partner
                rejected |= ~has_partner & (d[f"{m}_retained"] == 0)
                sens_rows.append({**k, "method": m, "cutoff_factor": factor, "official_rejected_fraction": base,
                                  "static_rejected_fraction": float(rejected.mean())})
            if x is None:
                continue
            for _, g in genes.iterrows():
                column = f"{g['gene']}__log1p_cp10k"
                if column not in x.columns:
                    marker_rows.append({**k, "method": m, "gene": g["gene"], "program": g["program"], "gene_present": False})
                    continue
                values = x[column].reindex(d["bin_id"]).to_numpy()
                counts = x[f"{g['gene']}__count"].reindex(d["bin_id"]).to_numpy()
                retained = (d[f"{m}_retained"] == 1).to_numpy()
                ok = np.isfinite(values) & has_partner.to_numpy()
                rho = spearmanr(values[ok], rel[ok]).statistic if ok.sum() > 10 and np.std(values[ok]) > 0 else np.nan
                auc = np.nan
                if retained.any() and (~retained).any() and np.isfinite(values).all():
                    auc = mannwhitneyu(values[~retained], values[retained]).statistic / ((~retained).sum() * retained.sum())
                positive = counts > 0
                marker_rows.append({
                    **k, "method": m, "gene": g["gene"], "program": g["program"], "gene_present": True,
                    "n_bins": len(d), "positive_fraction": float(positive.mean()),
                    "mean_retained": float(values[retained].mean()) if retained.any() else np.nan,
                    "mean_rejected": float(values[~retained].mean()) if (~retained).any() else np.nan,
                    "auc_rejected_vs_retained": float(auc),
                    "spearman_expression_vs_relative_margin": float(rho),
                    "rejected_fraction_marker_positive": float((~retained[positive]).mean()) if positive.any() else np.nan,
                    "rejected_fraction_marker_negative": float((~retained[~positive]).mean()) if (~positive).any() else np.nan,
                    "n_marker_positive": int(positive.sum())})
    return pd.DataFrame(marker_rows), pd.DataFrame(cost_rows), pd.DataFrame(sens_rows)


def summarise(table: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
    g = table.groupby(by, dropna=False)[value]
    return g.agg(n_pairs="size", mean="mean", sd="std", median="median", min="min", max="max").reset_index()


# ------------------------------------------------------------------ figures

def save(out: Outputs, fig, relative: str) -> None:
    import matplotlib.pyplot as plt

    for ext in ("pdf", "png"):
        target = out.path(f"{relative}.{ext}")
        fig.savefig(target, dpi=300, bbox_inches="tight")
        out.record(f"{relative}.{ext}")
    plt.close(fig)


def order_key(label: str) -> tuple:
    stage = label.split("->")[0].split(" ")[0]
    return (STAGES.index(stage) if stage in STAGES else 99, "same-stage" in label)


def fig_rejection(out: Outputs, table: pd.DataFrame, organ: str, run: str, annotation: str) -> None:
    import matplotlib.pyplot as plt

    d = table[(table["run"] == run) & (table["annotation"] == annotation)].copy()
    if d.empty:
        return
    d["label"] = d.apply(group_label, axis=1)
    labels = sorted(d["label"].unique(), key=order_key)
    for m, name in (("m4e", "M4-E"), ("m4r", "M4-R")):
        fig, ax = plt.subplots(figsize=(max(4, 0.42 * len(labels)), 2.6))
        for j, side in enumerate(("source", "target")):
            for i, label in enumerate(labels):
                v = d[(d["label"] == label) & (d["side"] == side)][f"{m}_rejected_fraction"]
                if v.empty:
                    continue
                x = i + (j - 0.5) * 0.36
                ax.bar(x, v.mean(), width=0.34, color=["#4C72B0", "#DD8452"][j], alpha=0.35 if "same" in label else 0.85,
                       yerr=v.std() if len(v) > 1 else None, error_kw={"lw": 0.6})
                ax.scatter(np.full(len(v), x), v, s=4, color="#222222", zorder=3, linewidths=0)
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
        ax.set_ylabel(f"{name} rejected fraction")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{annotation} bins, {run}: {name} (source blue, target orange; pale = same-stage control)")
        save(out, fig, f"figures/{organ}/{run}/rejection_by_transition_{annotation.replace(' ', '_')}_{m}")


def fig_cost(out: Outputs, bins: pd.DataFrame, organ: str, run: str, annotation: str, focal: str) -> None:
    import matplotlib.pyplot as plt

    d = bins[(bins["run"] == run) & (bins["annotation"] == annotation) & (bins["group"] == focal) & (bins["kind"] == "adjacent")]
    if d.empty:
        return
    for m, name in (("m4e", "M4-E"), ("m4r", "M4-R")):
        for side in ("source", "target"):
            s = d[(d["side"] == side) & (d[f"{m}_decision_cost"] != 0)]
            if s.empty:
                continue
            rel = s[f"{m}_decision_cost"] / s["rejection_cost"]
            fig, ax = plt.subplots(figsize=(3.2, 2.4))
            bins_edges = np.linspace(np.nanpercentile(rel, 0.5), np.nanpercentile(rel, 99.5), 60)
            ax.hist(rel[s[f"{m}_retained"] == 1], bins=bins_edges, color=RETAINED, alpha=0.8, label="retained")
            ax.hist(rel[s[f"{m}_retained"] == 0], bins=bins_edges, color=REJECTED, alpha=0.7, label="rejected")
            ax.axvline(1.0, color="#222222", lw=0.8, ls="--")
            ax.set_xlabel("decision cost / rejection cost (cutoff = 1)")
            ax.set_ylabel("bins")
            ax.set_title(f"{annotation} {side}, {focal}, {run}, {name}\n{s['pair_id'].nunique()} pairs, "
                         f"{len(s)} bins ({int((d[f'{m}_decision_cost'] == 0).sum())} without partner mass not shown)")
            ax.legend(frameon=False)
            save(out, fig, f"figures/{organ}/{run}/decision_cost_{annotation.replace(' ', '_')}_{focal.replace('->', '_to_')}_{side}_{m}")


def fig_markers(out: Outputs, bins: pd.DataFrame, out_root: Path, organ: str, run: str, annotation: str, focal: str) -> None:
    import matplotlib.pyplot as plt

    d = bins[(bins["run"] == run) & (bins["annotation"] == annotation) & (bins["group"] == focal) & (bins["kind"] == "adjacent")]
    genes = markers().query("organ_set == @organ")["gene"].tolist()
    for side in ("source", "target"):
        s = d[d["side"] == side]
        frames = []
        for sample, g in s.groupby("own_sample"):
            path = out_root / "expression" / f"{sample}.csv.gz"
            if path.is_file():
                x = pd.read_csv(path).set_index("bin_id")
                frames.append(g.join(x, on="bin_id"))
        if not frames:
            continue
        s = pd.concat(frames)
        for m, name in (("m4e", "M4-E"), ("m4r", "M4-R")):
            t = s[s[f"{m}_decision_cost"] != 0].copy()
            t["rel"] = t[f"{m}_margin"] / t["rejection_cost"]
            t["q"] = pd.qcut(t["rel"], 10, labels=False, duplicates="drop")
            fig, ax = plt.subplots(figsize=(3.4, 2.5))
            for gene in genes:
                column = f"{gene}__log1p_cp10k"
                if column not in t.columns:
                    continue
                means = t.groupby("q").agg(rel=("rel", "median"), value=(column, "mean"))
                ax.plot(means["rel"], means["value"], marker="o", ms=2.5, lw=0.9, label=gene)
            ax.axvline(0.0, color="#222222", lw=0.8, ls="--")
            ax.set_xlabel("relative margin (decile medians; > 0 rejected)")
            ax.set_ylabel("mean log1p(CP10k), equalised counts")
            ax.set_title(f"{annotation} {side}, {focal}, {run}, {name}")
            ax.legend(frameon=False, fontsize=5.5, ncol=2)
            save(out, fig, f"figures/{organ}/{run}/markers_vs_margin_{annotation.replace(' ', '_')}_{focal.replace('->', '_to_')}_{side}_{m}")


def fig_remap(out: Outputs, remap: pd.DataFrame, organ: str, run: str, annotation: str, focal: str) -> None:
    import matplotlib.pyplot as plt

    d = remap[(remap["run"] == run) & (remap["annotation"] == annotation) & (remap["group"] == focal) & (remap["kind"] == "adjacent")]
    if d.empty or d["bot_same_mean"].isna().all():
        return
    order = ["E_retained", "E_rejected", "R_retained", "R_rejected", "both_retained", "both_rejected", "E_only_retained", "R_only_retained"]
    for side in ("source", "target"):
        s = d[(d["side"] == side) & (d["n_bins"] > 0) & (d["bot_n_with_distribution"] > 0)]
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        for i, g in enumerate(order):
            v = s[s["gate_group"] == g]
            if v.empty:
                continue
            w = v["n_bins"].to_numpy()  # bins pooled over pairs: weighted by bins
            bottom = 0.0
            for c, colour in (("same", "#3B6FB6"), ("curated", "#8FB3E0"), ("other", "#D9D9D9")):
                value = np.nansum(v[f"bot_{c}_mean"].to_numpy() * w) / w[np.isfinite(v[f"bot_{c}_mean"].to_numpy())].sum()
                ax.bar(i, value, bottom=bottom, color=colour, width=0.7, label=c if i == 0 else None)
                bottom += value
            ax.text(i, 1.01, f"{int(w.sum())}", ha="center", fontsize=5)
        ax.set_xticks(range(len(order)), [o.replace("_", " ") for o in order], rotation=45, ha="right")
        ax.set_ylabel("balanced-OT conditional mass")
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{annotation} {side}, {focal}, {run}: balanced OT on each gate group (bins above bars)")
        ax.legend(frameon=False, fontsize=5.5, loc="lower right")
        save(out, fig, f"figures/{organ}/{run}/balanced_remap_{annotation.replace(' ', '_')}_{focal.replace('->', '_to_')}_{side}")


def fig_spatial(out: Outputs, bins: pd.DataFrame, root: Path | None, organ: str, run: str, annotation: str, focal: str,
                table: pd.DataFrame) -> pd.DataFrame:
    """Representative pair of the focal transition: the pair whose M4-E source rejected fraction is the median
    (lower median for an even count).  Every candidate is listed in the returned table."""
    import matplotlib.pyplot as plt

    t = table[(table["run"] == run) & (table["annotation"] == annotation) & (table["group"] == focal)
              & (table["kind"] == "adjacent") & (table["side"] == "source")].sort_values("m4e_rejected_fraction")
    if t.empty:
        return pd.DataFrame()
    pick = t.iloc[(len(t) - 1) // 2]["pair_id"]
    t = t.assign(selected=t["pair_id"] == pick, rule="median M4-E source rejected fraction (lower median)")
    d = bins[(bins["run"] == run) & (bins["pair_id"] == pick) & (bins["annotation"] == annotation)]
    for side in ("source", "target"):
        s = d[d["side"] == side]
        if s.empty:
            continue
        context = None
        if root is not None:
            from bio_common import analysed_bins
            context = analysed_bins(root, s["own_sample"].iloc[0])[["x", "y"]]
        panels = [("m4e_retained", "M4-E gate"), ("m4r_retained", "M4-R gate"), ("bot_supported", "balanced OT same+curated")]
        for column, title in panels:
            if s[column].isna().all():
                continue
            fig, ax = plt.subplots(figsize=(3.0, 3.0))
            if context is not None:
                ax.scatter(context["x"], context["y"], s=0.3, color=GREY, linewidths=0, rasterized=True)
            if column.endswith("retained"):
                colours = np.where(s[column] == 1, RETAINED, REJECTED)
                ax.scatter(s["x"], s["y"], s=0.8, c=colours, linewidths=0, rasterized=True)
            else:
                sc = ax.scatter(s["x"], s["y"], s=0.8, c=s[column], cmap="viridis", vmin=0, vmax=1, linewidths=0, rasterized=True)
                fig.colorbar(sc, ax=ax, shrink=0.6)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{pick} {side}\n{annotation}: {title} ({run})")
            save(out, fig, f"figures/{organ}/{run}/spatial_{annotation.replace(' ', '_')}_{pick}_{side}_{column}")
    return t


def fig_preprocessing(out: Outputs, table: pd.DataFrame, organ: str, annotation: str, runs: list[str]) -> None:
    import matplotlib.pyplot as plt

    if len(runs) < 2:
        return
    a, b = runs[:2]
    d = table[table["annotation"] == annotation]
    for m, name in (("m4e", "M4-E"), ("m4r", "M4-R")):
        wide = d.pivot_table(index=["pair_id", "side", "kind"], columns="run", values=f"{m}_rejected_fraction").dropna()
        if a not in wide or b not in wide or wide.empty:
            continue
        fig, ax = plt.subplots(figsize=(2.8, 2.8))
        for kind, colour in (("adjacent", "#3B6FB6"), ("same_stage", "#BBBBBB")):
            w = wide.xs(kind, level="kind") if kind in wide.index.get_level_values("kind") else None
            if w is not None:
                ax.scatter(w[a], w[b], s=5, color=colour, linewidths=0, label=f"{kind} ({len(w)})")
        ax.plot([0, 1], [0, 1], color="#222222", lw=0.6, ls="--")
        ax.set_xlabel(f"{a} rejected fraction")
        ax.set_ylabel(f"{b} rejected fraction")
        ax.set_title(f"{annotation}, {name}: pair sides")
        ax.legend(frameon=False, fontsize=5.5)
        save(out, fig, f"figures/{organ}/robustness_{a}_vs_{b}_{annotation.replace(' ', '_')}_{m}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--organ-set", required=True)
    parser.add_argument("--runs", nargs="+", default=["ds", "logcpm"])
    parser.add_argument("--run-root", action="append", default=[], help="name=root, for the spatial context")
    parser.add_argument("--focal-group", default=None, help="adjacent transition for the case figures; default every one")
    parser.add_argument("--support-threshold", type=float, default=0.5)
    args = parser.parse_args()
    out = Outputs(args.out)
    organ = args.organ_set
    roots = {k: Path(v) for k, v in (item.split("=", 1) for item in args.run_root)}
    bins = load(args.out, args.runs, organ)
    if bins.empty:
        raise SystemExit(f"no per-bin tables for {organ}; run 32_ first")
    runs = [r for r in args.runs if r in set(bins["run"])]
    bins = add_support(args.out, bins, organ, args.support_threshold)
    validity_path = args.out / "audit" / "METHOD_VALIDITY_AUDIT.tsv"
    validity = pd.read_csv(validity_path, sep="\t") if validity_path.is_file() else None
    table, agree = gate_tables(bins, validity)
    remap = remapping(bins)
    four = four_way(bins, args.support_threshold)
    rejected = rejected_audit(bins, table, args.support_threshold)
    marker_table, cost_table, sens_table = cutoff_and_markers(bins, args.out, organ)
    base = f"analysis/{organ}"
    out.table(table, f"{base}/RETAINED_REJECTED_BY_ORGAN.tsv")
    for m in ("m4e", "m4r"):
        summary = summarise(table.assign(label=table.apply(group_label, axis=1)), f"{m}_rejected_fraction",
                            ["run", "annotation", "side", "kind", "label", "partner_has_annotation"])
        out.table(summary, f"{base}/rejected_fraction_summary_{m}.tsv")
    out.table(agree, f"{base}/M4E_M4R_GATE_AGREEMENT.tsv")
    out.table(remap, f"{base}/BALANCED_REMAPPING_BY_CELLTYPE.tsv")
    out.table(four, f"{base}/DEVELOPMENTAL_TRANSITION_SUPPORT.tsv")
    out.table(rejected, f"{base}/REJECTED_POPULATION_AUDIT.tsv")
    out.table(four[(four["gate"] == "rejected") & (four["bot_support"] == "high")], f"{base}/OVER_REJECTION_AUDIT.tsv")
    out.table(marker_table, f"{base}/MARKER_CUTOFF_ANALYSIS.tsv")
    out.table(cost_table, f"{base}/decision_cost_summary.tsv")
    out.table(sens_table, f"{base}/cutoff_sensitivity_static.tsv")
    focal = [args.focal_group] if args.focal_group else sorted(bins.loc[bins["kind"] == "adjacent", "group"].unique(), key=order_key)
    selections = []
    for run in runs:
        for annotation in sorted(bins["annotation"].unique()):
            fig_rejection(out, table, organ, run, annotation)
            for group in focal:
                fig_cost(out, bins, organ, run, annotation, group)
                fig_markers(out, bins, args.out, organ, run, annotation, group)
                fig_remap(out, remap, organ, run, annotation, group)
                selections.append(fig_spatial(out, bins, roots.get(run), organ, run, annotation, group, table))
    for annotation in sorted(bins["annotation"].unique()):
        fig_preprocessing(out, table, organ, annotation, runs)
    selections = [s for s in selections if len(s)]
    if selections:
        out.table(pd.concat(selections, ignore_index=True), f"{base}/PAIR_SELECTION.tsv")
    print(f"{organ}: {len(bins)} bin rows, {bins['pair_id'].nunique()} pairs, runs {runs}", flush=True)


if __name__ == "__main__":
    main()
