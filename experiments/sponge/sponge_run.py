"""Run the frozen method (version 2: K=3, perfate, logreg readout) on the sponge data.
Discovery run: there is no ground truth, so this only writes driver scores; nothing is evaluated.

Same recipe as nestorowa_eval.py:
  - preprocessing  = paul15_eval.prepare (log-normalise if raw counts, 1,500 HVGs)
  - terminal cells = whole annotated terminal clusters
  - root           = stem-cluster cell closest to the stem centroid in PCA (or highest CytoTRACE, ROOT=cytotrace)
  - method         = OURS_v2_TF (regulators and candidates = TFs; the real-data default) if a TF list is
                     given, otherwise OURS_v2_fast (300 candidates, all genes as regulators)

Usage (from the repository root, conda env driver-benchmark):
  python experiments/sponge/sponge_run.py PATH/TO/sponge.h5ad --cluster-key leiden --tfs PATH/TO/sponge_tfs.txt
Options (all have defaults from the current annotation, which is NOT final -- skin is 0 or 4):
  --stem 5  --skin 0  --sclero 2,3  --digest 6  --choano 12
  --fates choano,sclero,digest,skin    which fates form the decision (one multi-fate decision)
  --pairwise                           instead: one 2-fate decision per pair of fates
  --lineage all|annotated              cells used: all cells (default, like Nestorowa) or only
                                       stem + transition (--transition 1) + terminal clusters
  --root centroid|cytotrace --cytotrace-key ct_pseudotime
  --methods OURS_v2_TF,OURS_v2_fast,fate_corr,...   (baselines from baselines.SCORERS also allowed)
             OURS_v2_TF = TFs only (default); OURS_v2_fast = top 300 genes by fate correlation, all genes as
             regulators; OURS_v2_c300TF = top 300 + all TFs; OURS_v2_all = every gene (slow)
  --tag sponge   prefix of all output files; use a new tag to keep earlier results
Experiment options (8 Nov 2026; defaults = the frozen method):
  --k 3                          rollout steps K
  --cell-weight uniform          uniform | fate_region (own undecided cells per fate) | fate_open | pseudotime
  --score-mode perfate           perfate (rank on P(fate) shifts) | perfate_logodds (rank on log-odds shifts)
  --tree                         lineage tree instead of one multi-fate decision:
                                   tree1: stem -> choano vs branch (branch = sclero + digest + skin clusters)
                                   tree2: transition -> the other fates, on transition + those clusters only,
                                          root = transition cell closest to the stem centroid
  --seeds 0,1,2  --subsample 0.8       robustness: repeat on random cell subsamples
Output: results/sponge_scores_<decision>_s<seed>.csv (gene x method), results/sponge_top.csv, and for our
methods results/sponge_perfate_<method>_<decision>_s<seed>.csv (per-fate shifts; see fate_table)"""
import argparse, itertools, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc
warnings.filterwarnings("ignore")
from driverbench import realdata as P15
from driverbench.baselines import SCORERS as BASE, DecisionContext
from driverbench.method_timing import score_timing_cf_v2
from driverbench.probe_interventional import candidate_genes
from driverbench.paths import REAL, ROOT
def ints(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def args_():
    p = argparse.ArgumentParser()
    p.add_argument("h5ad")
    p.add_argument("--cluster-key", default="leiden")
    p.add_argument("--tfs", default=None, help="text file, one gene id per line (e.g. Pfam DNA-binding-domain hits)")
    p.add_argument("--stem", default="5"); p.add_argument("--transition", default="1")
    p.add_argument("--choano", default="12"); p.add_argument("--sclero", default="2,3")
    p.add_argument("--digest", default="6"); p.add_argument("--skin", default="0")
    p.add_argument("--fates", default="choano,sclero,digest,skin")
    p.add_argument("--pairwise", action="store_true")
    p.add_argument("--lineage", default="all", choices=["all", "annotated"])
    p.add_argument("--root", default="centroid", choices=["centroid", "cytotrace"])
    p.add_argument("--cytotrace-key", default="ct_pseudotime")
    p.add_argument("--methods", default=None)
    p.add_argument("--n-hvg", type=int, default=1500)
    p.add_argument("--seeds", default="0"); p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--tag", default="sponge")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--cell-weight", default="uniform", choices=["uniform", "fate_region", "fate_open", "pseudotime"])
    p.add_argument("--score-mode", default="perfate", choices=["perfate", "perfate_logodds"])
    p.add_argument("--tree", action="store_true")
    return p.parse_known_args()[0]      # known_args: fig_sponge.py adds its own options on top


def load(a_):
    a = sc.read_h5ad(a_.h5ad)
    if a_.cluster_key not in a.obs:
        raise SystemExit(f"--cluster-key {a_.cluster_key!r} not in obs; columns: {list(a.obs.columns)}")
    a.obs[a_.cluster_key] = a.obs[a_.cluster_key].astype(str)
    tfs_raw = set()
    if a_.tfs:
        tfs_raw = {l.strip().split()[0] for l in open(a_.tfs) if l.strip() and not l.startswith("#")}
    # keep every TF through the HVG filter (prepare keeps HVGs + "literature" genes; put the TFs there)
    P15.LITERATURE["_sponge_tfs"] = {"core": sorted(tfs_raw & set(a.var_names))}
    a.layers.clear() if hasattr(a.layers, "clear") else [a.layers.pop(k) for k in list(a.layers) if k is not None]
    a.uns = {}                                     # prepare drops layers anyway; keeps the temp copy small
    tmp = REAL / f"_{a_.tag}_tmp.h5ad"
    tmp.parent.mkdir(exist_ok=True)
    a.write_h5ad(tmp)
    a = P15.prepare(str(tmp), n_hvg=a_.n_hvg)
    tmp.unlink()
    return a, [g for g in a.var_names if g in tfs_raw]


def context(a, a_, fates, seed, spec=None):
    """spec (tree mode): {"name", "groups": {fate: clusters}, "cells": clusters kept (None = as --lineage), "entry": bool}"""
    cl = a.obs[a_.cluster_key].values
    groups = {"choano": ints(a_.choano), "sclero": ints(a_.sclero), "digest": ints(a_.digest), "skin": ints(a_.skin)}
    stem, trans = ints(a_.stem), ints(a_.transition)
    if spec is not None:
        groups = spec["groups"]
        if spec.get("entry"):                                   # root = transition cell closest to the stem centroid
            sc.pp.pca(a, n_comps=20)
            ent = a.obsm["X_pca"][np.isin(cl, stem)].mean(0)
            d_ = np.linalg.norm(a.obsm["X_pca"] - ent, axis=1); d_[~np.isin(cl, trans)] = np.inf
            entry_cell = a.obs_names[int(np.argmin(d_))]
        if spec.get("cells"):
            a = a[np.isin(cl, spec["cells"])].copy(); cl = a.obs[a_.cluster_key].values
    used = sorted(set(stem + trans + [c for f in fates for c in groups[f]]))
    if spec is not None and spec.get("cells"):
        used = sorted(set(spec["cells"]))
    missing = [c for c in used if c not in set(cl)]
    if missing:
        raise SystemExit(f"clusters {missing} not in obs[{a_.cluster_key!r}] (values: {sorted(set(cl))})")
    if a_.lineage == "annotated" and not (spec is not None and spec.get("cells")):
        a = a[np.isin(cl, used)].copy(); cl = a.obs[a_.cluster_key].values
    if a_.subsample < 1:
        rng = np.random.default_rng(seed)
        a = a[np.sort(rng.choice(a.n_obs, int(a_.subsample * a.n_obs), replace=False))].copy()
        cl = a.obs[a_.cluster_key].values
    term = pd.Series(np.nan, index=a.obs_names, dtype=object)
    for f in fates:
        term[np.isin(cl, groups[f])] = f
    is_stem = np.isin(cl, stem)
    if spec is not None and spec.get("entry"):
        ri = int(np.where(a.obs_names == entry_cell)[0][0]) if entry_cell in set(a.obs_names) else \
             int(np.argmin(np.where(np.isin(cl, trans), 0, 1)))    # entry cell dropped by subsampling: any transition cell
    elif a_.root == "cytotrace":
        # expects a pseudotime-like column (low = undifferentiated). For a potency score (high = stem),
        # pass a column holding 1 - score.
        v = a.obs[a_.cytotrace_key].values.astype(float)
        ri = int(np.argmin(np.where(is_stem, v, np.inf)))
    else:
        sc.pp.pca(a, n_comps=20)
        d = np.linalg.norm(a.obsm["X_pca"] - a.obsm["X_pca"][is_stem].mean(0), axis=1); d[~is_stem] = np.inf
        ri = int(np.argmin(d))
    a.obs["sim_time"] = 1.0; a.obs.iloc[ri, a.obs.columns.get_loc("sim_time")] = 0.0   # methods take the root from here
    name = spec["name"] if spec is not None else "stem_to_" + "_".join(fates)
    ctx = DecisionContext(decision=name, fates=list(fates), terminal=term, cells=np.ones(a.n_obs, bool), seed=seed)
    return a, ctx


def fate_table(b, ctx, a_, s, sh):
    """One row per scored gene (candidates only), sorted by driver score.
    dP_<fate>:  mean increase in P(fate) of undecided cells when the gene is clamped to its typical level in that
                fate (90th percentile in the fate's terminal cells) and fast-forwarded K=3 steps.
    pct_<fate>: per-fate rank percentile of dP among scored genes; score = max over fates (the perfate rule),
                best_fate = the fate where that maximum is reached.
    logFC_<fate>_vs_stem: mean expression (log-normalised) in the fate's terminal cells minus in stem cells,
                so you can see whether the gene goes up or down in that fate."""
    fates = list(ctx.fates)
    dP = sh[[f"prob_{f}" for f in fates]].copy(); dP.columns = fates
    dP = dP.dropna(how="all")
    ranked_on = "dL" if getattr(a_, "score_mode", "perfate") == "perfate_logodds" else "dP"
    dL = sh[[f"logodds_{f}" for f in fates]].copy(); dL.columns = fates; dL = dL.loc[dP.index]
    pct = (dL if ranked_on == "dL" else dP).rank(pct=True)          # per-fate ranking actually used by the score
    X = np.asarray(b[:, dP.index].X)
    cl = b.obs[a_.cluster_key].values
    ref = np.isin(cl, ints(a_.stem)); ref_name = "stem"
    if not ref.any():                                   # tree2 has no stem cells: compare with the transition cells
        ref = np.isin(cl, ints(a_.transition)); ref_name = "transition"
    stem = X[ref].mean(0)
    lfc = pd.DataFrame({f: X[(ctx.terminal == f).values].mean(0) - stem for f in fates}, index=dP.index)
    t = pd.DataFrame({"score": s.reindex(dP.index), "best_fate": pct.idxmax(axis=1)})
    t["dP_best"] = [dP.at[g, f] for g, f in zip(t.index, t.best_fate)]
    t["logFC_best_vs_stem"] = [lfc.at[g, f] for g, f in zip(t.index, t.best_fate)]
    t = t.join(dP.add_prefix("dP_")).join(dL.add_prefix("dL_")).join(pct.add_prefix("pct_")).join(lfc.add_suffix("_vs_stem").add_prefix("logFC_"))
    t = t.sort_values("score", ascending=False)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    t["ranked_on"] = ranked_on
    t["logFC_reference"] = ref_name
    t.index.name = "gene"
    return t


def main():
    a_ = args_()
    a, tfs = load(a_)
    print(f"{a.n_obs} cells x {a.n_vars} genes after prepare; {len(tfs)} TFs kept", flush=True)
    kw = dict(max_cells=300, n_jobs=a_.n_jobs, return_shifts=True,  # ours: (scores, per-fate shifts)
              K=a_.k, cell_weight=a_.cell_weight, score_mode=a_.score_mode)
    print(f"settings: K={a_.k}, cell_weight={a_.cell_weight}, score_mode={a_.score_mode}, tree={a_.tree}", flush=True)
    methods = {
        "OURS_v2_TF": lambda a, c: score_timing_cf_v2(a, c, candidates=tfs, regulators=tfs, **kw),
        "OURS_v2_c300TF": lambda a, c: score_timing_cf_v2(a, c, candidates=sorted(set(candidate_genes(a, c, 300)) | set(tfs)), **kw),
        "OURS_v2_fast": lambda a, c: score_timing_cf_v2(a, c, candidates=300, **kw),
        "OURS_v2_all": lambda a, c: score_timing_cf_v2(a, c, candidates=None, **kw),   # every gene: slow
        **{k: f for k, f in BASE.items() if not k.startswith("ORACLE")},
    }
    chosen = ints(a_.methods) if a_.methods else (["OURS_v2_TF"] if tfs else ["OURS_v2_fast"])
    if not tfs and any(m in ("OURS_v2_TF", "OURS_v2_c300TF") for m in chosen):
        raise SystemExit("OURS_v2_TF / OURS_v2_c300TF need --tfs (no TFs found in the data otherwise)")
    fates = ints(a_.fates)
    if a_.tree:
        gr = {"choano": ints(a_.choano), "sclero": ints(a_.sclero), "digest": ints(a_.digest), "skin": ints(a_.skin)}
        rest = [f for f in fates if f != "choano"]
        decisions = [(["choano", "branch"], {"name": "tree1_stem_to_choano_vs_branch",
                                             "groups": {"choano": gr["choano"], "branch": [c for f in rest for c in gr[f]]}}),
                     (rest, {"name": "tree2_transition_to_" + "_".join(rest), "groups": gr, "entry": True,
                             "cells": sorted(set(ints(a_.transition) + [c for f in rest for c in gr[f]]))})]
    else:
        decisions = [(list(p), None) for p in itertools.combinations(fates, 2)] if a_.pairwise else [(fates, None)]
    top = []
    for fs, spec in decisions:
        for seed in map(int, ints(a_.seeds)):
            b, ctx = context(a, a_, fs, seed, spec)
            print(f"\n== {ctx.decision} seed {seed}: {b.n_obs} cells, terminal {ctx.terminal.value_counts().to_dict()}, "
                  f"root {b.obs_names[int(np.argmin(b.obs.sim_time.values))]}", flush=True)
            cols = {}
            for m in chosen:
                t0 = time.time(); s = methods[m](b, ctx)
                sh = None
                if isinstance(s, tuple):
                    s, sh = s
                cols[m] = s
                ss = s.reindex(tfs) if (m.endswith("_TF") and tfs) else s
                t10 = ss.sort_values(ascending=False).index[:20].tolist()
                print(f"{m:16s} {time.time()-t0:6.0f}s  top 20: {' '.join(t10)}", flush=True)
                top.append({"decision": ctx.decision, "seed": seed, "method": m, "top20": " ".join(t10)})
                if sh is not None:
                    tab = fate_table(b, ctx, a_, s, sh)
                    tab.to_csv(ROOT / "results" / f"{a_.tag}_perfate_{m}_{ctx.decision}_s{seed}.csv")
                    base = sh.attrs.get("baseline_fate_prob", {})
                    print("  undecided cells after fast-forward, mean P(fate): "
                          + ", ".join(f"{f} {p:.2f}" for f, p in base.items()), flush=True)
                    print(tab.head(20)[["rank", "best_fate", "dP_best", "logFC_best_vs_stem"]]
                          .to_string(float_format=lambda x: f"{x:.3f}"), flush=True)
            out = ROOT / "results" / f"{a_.tag}_scores_{ctx.decision}_s{seed}.csv"
            pd.DataFrame(cols).to_csv(out)
            pd.DataFrame(top).to_csv(ROOT / "results" / f"{a_.tag}_top.csv", index=False)
    print("EXIT 0")


if __name__ == "__main__":
    main()
