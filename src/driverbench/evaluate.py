"""Score every dataset in data/ with every scorer and compute driver-recovery metrics.

Metrics per (dataset, decision, method):
  auprc_all        drivers vs every other gene (markers, progenitor, mediators, nulls)
  auprc_vs_markers drivers vs downstream genes of the same decision only  <- headline metric
  prec_at_k        precision in the top-k genes, k = number of drivers
  best_driver_rank rank of the highest-scored driver (1 = top)
Chance level for auprc_* is the driver prevalence in the respective gene set (reported).

Usage:  python -m driverbench.evaluate [--methods ...] [--pattern 'BF_*']
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from driverbench.baselines import SCORERS, DecisionContext
from driverbench.probe_interventional import score_interventional_probe

from driverbench.method_timing import score_timing_cf, score_onset_only, score_timing_cf_fast, score_timing_cf_stress, score_timing_cf_v2

from driverbench.competitors import score_cellrank
SCORERS = {**SCORERS, "CellRank": lambda a, ctx: score_cellrank(a, ctx, f"syn_{a.uns['model']}_rep{a.uns['rep']}_{ctx.decision}"),
           "PROBE_interventional_linear": score_interventional_probe,
           "OURS_timing_cf": score_timing_cf, "ABL_onset_only": score_onset_only,
           "OURS_timing_cf_fast": score_timing_cf_fast, "OURS_timing_cf_stress": score_timing_cf_stress,
           "OURS_v2": score_timing_cf_v2}

from driverbench.paths import ROOT, SYNTH
TERMINAL_TIME_QUANTILE = 0.7  # cells in the last 30% of time that have committed = "annotated terminal types"


def contexts(a: ad.AnnData, seed: int):
    for d, fates in a.uns["fate_genes"].items():
        fates = list(fates)
        parents = dict(a.uns.get("parents", {"D2": ["D1", "A"]}))   # old CBF files: D2 within A
        if d not in parents:
            cells = np.ones(a.n_obs, bool)                           # whole dataset
        else:  # nested decision: analyse the parent lineage (assumed annotated)
            pd_, pf = parents[d]
            cells = (a.obs[f"fate_{pd_}"].astype(object) == pf).values
        o = a.obs
        late = o["sim_time"] >= o.loc[cells, "sim_time"].quantile(TERMINAL_TIME_QUANTILE)
        term = o[f"fate_{d}"].astype(object).where(late & o[f"committed_{d}"].astype(bool))
        yield DecisionContext(decision=d, fates=fates, terminal=term, cells=cells, seed=seed)


def metrics(scores: pd.Series, var: pd.DataFrame, decision: str, n_tiebreak: int = 50) -> dict:
    """Ties are broken at random (drivers come first in var order, so a stable sort would
    flatter saturated scores); metrics are averaged over n_tiebreak random tie-breaks."""
    drv = (var["role"] == "driver") & (var["decision"] == decision)
    down = var["role"].isin(["marker", "mediator"]) & (var["decision"] == decision)
    s = scores.reindex(var.index).astype(float).values
    fin = np.isfinite(s)
    s = np.where(fin, s, (s[fin].min() if fin.any() else 0) - 1)
    k = int(drv.sum())
    sub = (drv | down).values
    kind = var["kind"] if "kind" in var else pd.Series("", index=var.index)
    early_t = down & kind.isin(["fast", "repressed"])
    sub_fast = (drv | early_t).values
    y = drv.values
    rng = np.random.default_rng(0)
    out = {"auprc_all": [], "auprc_vs_markers": [], "auprc_vs_fast_targets": [], "prec_at_k": [], "best_driver_rank": []}
    for _ in range(n_tiebreak):
        r = rankdata(s) + rng.random(len(s)) * 0.5      # jitter only reorders exact ties
        order = np.argsort(-r)
        out["auprc_all"].append(average_precision_score(y, r))
        out["auprc_vs_markers"].append(average_precision_score(y[sub], r[sub]))
        out["auprc_vs_fast_targets"].append(average_precision_score(y[sub_fast], r[sub_fast]) if early_t.any() else np.nan)
        out["prec_at_k"].append(y[order[:k]].mean())
        out["best_driver_rank"].append(np.where(y[order])[0][0] + 1)
    res = {m: float(np.mean(v)) for m, v in out.items()}
    res["chance_all"] = float(y.mean())
    res["chance_vs_markers"] = float(y[sub].mean())
    res["chance_vs_fast_targets"] = float(y[sub_fast].mean()) if early_t.any() else np.nan
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(SCORERS))
    ap.add_argument("--pattern", default="*.h5ad")
    ap.add_argument("--out", default="results/baseline_metrics.csv")
    args = ap.parse_args()
    rows = []
    for f in sorted(SYNTH.glob(args.pattern)):
        a = ad.read_h5ad(f)
        for ctx in contexts(a, seed=int(a.uns["rep"])):
            for name in args.methods:
                t0 = time.time()
                try:
                    sc = SCORERS[name](a, ctx)
                    r = metrics(sc, a.var, ctx.decision)
                    top = list(sc.sort_values(ascending=False).index[:5])
                except Exception as e:  # keep the sweep going, record the failure
                    r, top = {"error": repr(e)[:200]}, []
                rows.append({"dataset": f.stem, "model": a.uns["model"], "rep": a.uns["rep"],
                             "decision": ctx.decision, "method": name, **r,
                             "top5": " ".join(top), "seconds": round(time.time() - t0, 2)})
            print(f"done {f.stem} {ctx.decision}", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / args.out          # relative paths are taken from the repository root
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    cols = ["auprc_vs_markers", "chance_vs_markers", "auprc_vs_fast_targets", "chance_vs_fast_targets",
            "auprc_all", "chance_all", "prec_at_k", "best_driver_rank"]
    summ = df.groupby(["model", "decision", "method"])[cols].agg(["mean", "std"]).round(3)
    summ.to_csv(out.with_name(out.stem + "_summary.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.groupby(["model", "decision", "method"])[cols].mean().round(3))


if __name__ == "__main__":
    main()
