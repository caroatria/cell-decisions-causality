"""Evaluate methods on the nested-tree networks (data_tree/, simulate_tree.py), 17 Nov 2026.

Every split (decision) of every network is scored separately, with the same decision contexts and metrics as the
main synthetic benchmark (evaluate.contexts / evaluate.metrics): headline metric = AUPRC drivers vs markers.
Nested decisions are analysed within their parent lineage, as for CBF and HSC. The frozen version 2 method is used
unchanged (nothing was tuned on these networks).
Output: results/tree_depth.csv  (one row per network x decision x method), appended per network, so an interrupted
run resumes where it stopped.
Usage: python experiments/tree_depth/eval_tree.py [--methods ...] [--pattern 'T08_*']
"""
from __future__ import annotations
import argparse, time, warnings
from pathlib import Path
import anndata as ad, pandas as pd
warnings.filterwarnings("ignore")
from driverbench.evaluate import contexts, metrics
from driverbench.baselines import SCORERS as BASE
from driverbench.method_timing import score_timing_cf_v2
from driverbench.competitors import score_cellrank

from driverbench.paths import ROOT, TREE
OUT = ROOT / "results" / "tree_depth.csv"
SCORERS = {"OURS_v2": score_timing_cf_v2,
           "CellRank": lambda a, ctx: score_cellrank(a, ctx, f"tree_{a.uns['model']}_{ctx.decision}"),
           "rf_shap_fixed": BASE["rf_shap_fixed"], "de_terminal": BASE["de_terminal"], "fate_corr": BASE["fate_corr"]}

COLS = ["dataset", "p", "network_depth", "n_splits", "decision", "decision_level", "n_cells", "n_terminal", "method",
        "auprc_all", "auprc_vs_markers", "auprc_vs_fast_targets", "prec_at_k", "best_driver_rank", "chance_all",
        "chance_vs_markers", "chance_vs_fast_targets", "seconds", "error"]


def decision_depth(a, d: str) -> int:
    par = {k: list(v) for k, v in a.uns.get("parents", {}).items()}
    n = 1
    while d in par:
        d = par[d][0]; n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(SCORERS))
    ap.add_argument("--pattern", default="T*.h5ad")
    args = ap.parse_args()
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev.dataset, prev.decision, prev.method))
    for f in sorted((TREE).glob(args.pattern)):
        a = ad.read_h5ad(f)
        # Trajectories are stored every 0.2 time units, so the t = 0 column (all cells at the shared initial state,
        # 4-21 cells per network) forms a tight clump that splits the kNN graph and breaks DPT. Drop it, as the finer
        # time grid of the main benchmark effectively does.
        a = a[a.obs["sim_time"].values > 0].copy()
        rows = []
        for ctx in contexts(a, seed=0):
            n_term = ctx.terminal[ctx.cells].value_counts()
            for m in args.methods:
                if (f.stem, ctx.decision, m) in done:
                    continue
                t0 = time.time()
                try:
                    if len(n_term) < 2 or n_term.min() < 5:
                        raise ValueError(f"too few terminal cells {n_term.to_dict()}")
                    r = metrics(SCORERS[m](a, ctx), a.var, ctx.decision)
                except Exception as e:
                    r = {"error": repr(e)[:200]}
                rows.append({"dataset": f.stem, "p": a.uns["p"], "network_depth": int(a.uns["depth"]),
                             "n_splits": len(a.uns["fate_genes"]), "decision": ctx.decision,
                             "decision_level": decision_depth(a, ctx.decision), "n_cells": int(ctx.cells.sum()),
                             "n_terminal": int(n_term.sum()), "method": m, **r, "seconds": round(time.time() - t0, 1)})
        if rows:   # fixed column order, so appends from different networks always line up
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"done {f.stem} ({len(a.uns['fate_genes'])} decisions)", flush=True)


if __name__ == "__main__":
    main()
