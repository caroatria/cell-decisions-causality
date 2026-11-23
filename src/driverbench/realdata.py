"""Real-data validation on Paul et al. 2015 (mouse myeloid progenitors), literature ground truth.

Two decisions:
  ErMk : MEP -> erythroid vs megakaryocyte
  MoNeu: GMP -> monocyte vs neutrophil
Ground truth = known regulators of each decision compiled from the literature (see LITERATURE below;
please review). Real ground truth is incomplete, so we report precision@k and the rank of each
known regulator rather than driver-vs-marker AUPRC.

Usage: python -m driverbench.realdata path/to/paul15.h5ad
"""
from __future__ import annotations
import sys, time, warnings
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
from driverbench.baselines import SCORERS as BASE, DecisionContext
from driverbench.method_timing import score_timing_cf, score_timing_cf_fast
from driverbench.competitors import score_cefcon, score_celloracle, score_cellrank
import os
from driverbench.paths import RESULTS

# Decision-specific regulators with strong experimental support (core) and broader literature (extended).
# Gene symbols as in Paul15 (PU.1 = Sfpi1). TO REVIEW by a domain expert before any paper use.
LITERATURE = {
    "ErMk": {
        "core": ["Klf1", "Fli1", "Myb"],                       # Klf1/Fli1 antagonism; Myb favours Ery over Mk
        "extended": ["Gata1", "Zfpm1", "Nfe2", "Tal1", "Meis1", "Runx1", "Gfi1b", "Pbx1", "Ikzf1"],
    },
    "MoNeu": {
        "core": ["Irf8", "Cebpe", "Gfi1", "Klf4"],             # Irf8/Klf4 -> Mo; Cebpe/Gfi1 -> Neu
        "extended": ["Sfpi1", "Spi1", "Cebpa", "Egr1", "Egr2", "Nab2", "Zeb2", "Cebpb", "Mafb"],
    },
}
HB_GENES = ["Hbb-b1", "Hba-a2", "Hbb-b2", "Car2"]   # used only to pick the most mature erythroid clusters
LINEAGES = {
    "ErMk": {"progenitor": ["7MEP"], "fates": {"Ery": None, "Mk": ["8Mk"]},
             "lineage": ["1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery", "7MEP", "8Mk"]},
    "MoNeu": {"progenitor": ["9GMP", "10GMP"], "fates": {"Mo": ["14Mo", "15Mo"], "Neu": ["16Neu", "17Neu"]},
              "lineage": ["9GMP", "10GMP", "14Mo", "15Mo", "16Neu", "17Neu"]},
}
METHODS = {
    # fast version (300 candidate genes, 300 undecided cells): ~1,500 genes here; same accuracy as exact on the benchmark
    "OURS_timing_cf_fast": lambda a, ctx: score_timing_cf_fast(a, ctx, candidates=300, max_cells=300, n_jobs=2),
    "fate_corr": BASE["fate_corr"],
    "fate_corr_decision_region": BASE["fate_corr_decision_region"],
    "absorption_corr_decision_region": BASE["absorption_corr_decision_region"],
    "de_terminal": BASE["de_terminal"],
    "rf_shap_fixed": BASE["rf_shap_fixed"],
    "obs_counterfactual": BASE["obs_counterfactual"],
    "random": BASE["random"],
    # published competitors (3 Nov), own environments, see competitors.py
    "CEFCON": lambda a, ctx: score_cefcon(a, ctx, f"paul15_{ctx.decision}"),
    "CellOracle": lambda a, ctx: score_celloracle(a, ctx, f"paul15_{ctx.decision}"),
    "CellRank": lambda a, ctx: score_cellrank(a, ctx, f"paul15_{ctx.decision}"),
}


def prepare(path: str, n_hvg: int = 1500) -> ad.AnnData:
    a = ad.read_h5ad(path)
    X = a.X
    if np.asarray(X[:50].toarray() if hasattr(X, "toarray") else X[:50]).max() > 50:  # raw counts
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    lit = {g for d in LITERATURE.values() for v in d.values() for g in v} | set(HB_GENES)
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat")
    keep = a.var["highly_variable"].values | a.var_names.isin(list(lit))
    a = a[:, keep].copy()
    a.X = np.asarray(a.X.toarray() if hasattr(a.X, "toarray") else a.X, dtype=np.float32)
    for k in [k for k in list(a.layers) if k is not None]:   # newer anndata lists X as layers[None]
        del a.layers[k]
    return a


def context_for(a: ad.AnnData, decision: str) -> tuple[ad.AnnData, DecisionContext, dict]:
    spec = LINEAGES[decision]
    cl = a.obs["paul15_clusters"].astype(str)
    sub = a[cl.isin(spec["lineage"]).values].copy()
    cl = sub.obs["paul15_clusters"].astype(str)
    fates = dict(spec["fates"])
    if "Ery" in fates and fates["Ery"] is None:   # most mature erythroid clusters = highest haemoglobin expression
        hb = [g for g in HB_GENES if g in sub.var_names]
        ery = [c for c in spec["lineage"] if c.endswith("Ery")]
        means = {c: float(np.asarray(sub[(cl == c).values][:, hb].X).mean()) for c in ery}
        fates["Ery"] = sorted(means, key=means.get, reverse=True)[:2]
    term = pd.Series(np.nan, index=sub.obs_names, dtype=object)
    for f, cls in fates.items():
        term[cl.isin(cls).values] = f
    # root: progenitor cell closest to the progenitor centroid (stands in for an annotated stem cell)
    sc.pp.pca(sub, n_comps=20)
    prog = cl.isin(spec["progenitor"]).values
    cen = sub.obsm["X_pca"][prog].mean(0)
    d = np.linalg.norm(sub.obsm["X_pca"] - cen, axis=1); d[~prog] = np.inf
    sub.obs["sim_time"] = 1.0
    sub.obs.iloc[int(np.argmin(d)), sub.obs.columns.get_loc("sim_time")] = 0.0
    ctx = DecisionContext(decision=decision, fates=list(fates), terminal=term,
                          cells=np.ones(sub.n_obs, bool), seed=0)
    return sub, ctx, fates


def evaluate(scores: pd.Series, decision: str) -> dict:
    lit = LITERATURE[decision]
    core = [g for g in lit["core"] if g in scores.index]
    allg = [g for g in lit["core"] + lit["extended"] if g in scores.index]
    r = scores.rank(ascending=False, method="average")
    out = {"n_genes": len(scores), "n_core": len(core), "n_all": len(allg),
           "expected_core_hits@20_random": round(20 * len(core) / len(scores), 3)}
    for k in (10, 20, 50):
        top = set(scores.sort_values(ascending=False).index[:k])
        out[f"core_hits@{k}"] = len(top & set(core))
        out[f"all_hits@{k}"] = len(top & set(allg))
    out["core_median_rank"] = float(r[core].median()) if core else np.nan
    out["all_median_rank"] = float(r[allg].median()) if allg else np.nan
    out["ranks"] = " ".join(f"{g}:{int(r[g])}" for g in allg)
    return out


def main(path):
    a = prepare(path)
    rows = []
    for decision in LINEAGES:
        sub, ctx, fates = context_for(a, decision)
        print(f"{decision}: {sub.n_obs} cells x {sub.n_vars} genes; terminal fates {fates}; "
              f"terminal cells {ctx.terminal.notna().sum()}", flush=True)
        only = [m for m in os.environ.get("METHODS", "").split(",") if m]
        for name, fn in METHODS.items():
            if only and name not in only:
                continue
            t0 = time.time()
            s = fn(sub, ctx)
            rows.append({"decision": decision, "method": name, **evaluate(s, decision),
                         "top10": " ".join(s.sort_values(ascending=False).index[:10]),
                         "seconds": round(time.time() - t0, 1)})
            print(f"  {name:32s} core@20={rows[-1]['core_hits@20']}/{rows[-1]['n_core']} "
                  f"all@50={rows[-1]['all_hits@50']}/{rows[-1]['n_all']} "
                  f"median rank core={rows[-1]['core_median_rank']:.0f}  ({rows[-1]['seconds']}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / os.environ.get("OUT", "paul15_literature_eval.csv").removeprefix("results/"), index=False)
    return df


if __name__ == "__main__":
    main(sys.argv[1])
