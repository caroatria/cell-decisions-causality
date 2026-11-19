"""Lineage-tracing test on LARRY (Weinreb et al. 2020, in vitro), neutrophil vs monocyte.
Everything here was fixed BEFORE running any method; method code frozen (version 2); run once.

Ground truth (uses clones; methods never see clones or time points):
  clonal fate of a day-2 progenitor = fraction Neu among its clone's Neu+Mo descendants at days 4/6
  (clones with >= 2 Neu+Mo descendants that make up >= 50% of the clone's differentiated descendants).
  oracle score of a gene = |Spearman(day-2 expression, clonal fate)| over these day-2 cells
  early predictors = top 50 genes by oracle score; late markers = top 200 genes by |log fold change|
  between mature Neu and Mo cells that are not in the oracle top 200.
Metrics: AUPRC early predictors vs late markers (the benchmark's driver-vs-marker metric), Spearman between
method scores and oracle scores, literature regulators (Mo: Irf8, Klf4 ...; Neu: Cebpe, Gfi1 ...).
Methods' input: Neu/Mo lineage = cells annotated Neutrophil/Monocyte + undifferentiated cells whose 30
nearest annotated cells (PCA) are >= 50% Neu/Mo; random 8,000 cells (stratified by annotation, seed 0);
terminal = annotated Neutrophil / Monocyte; root = day-2 lineage cell closest to the day-2 centroid.
Usage: python larry_eval.py   -> results/larry_eval.csv, results/larry_oracle.csv"""
import time, warnings, os, json
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
import paul15_eval as P15
from baselines import SCORERS as BASE, DecisionContext
from method_timing import score_timing_cf_fast, score_timing_cf_v2
from probe_interventional import candidate_genes
from competitors import score_cefcon, score_celloracle, score_cellrank
from nestorowa_eval import tfs_in
ROOT = Path(__file__).resolve().parent
DEC = "MoNeu"          # literature list as for Paul15 (Spi1 is the gene symbol here)
N_CELLS = 8000
SEED = int(os.environ.get("SEED", 0))                 # cell-subsample seed (run 1 used 0)
TAG = os.environ.get("TAG", "")                       # "" = original output names; else *_<TAG>.csv + scores
ONLY = [m for m in os.environ.get("METHODS", "").split(",") if m]   # optional subset of methods


def lognorm(a):
    a = a.copy(); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a); return a


def build():
    full = ad.read_h5ad(ROOT / "data_real" / "larry_invitro.h5ad")
    ann = full.obs["Cell type annotation"].astype(str).values
    tp = full.obs["Time point"].astype(float).values
    L = lognorm(full)
    sc.pp.highly_variable_genes(L, n_top_genes=2000, flavor="seurat"); sc.pp.pca(L, n_comps=30, mask_var="highly_variable")
    annotated = ann != "Undifferentiated"
    nn = NearestNeighbors(n_neighbors=30).fit(L.obsm["X_pca"][annotated])
    idx = nn.kneighbors(L.obsm["X_pca"][~annotated], return_distance=False)
    frac = np.isin(ann[annotated][idx], ["Neutrophil", "Monocyte"]).mean(1)
    lineage = np.isin(ann, ["Neutrophil", "Monocyte"]); lineage[np.where(~annotated)[0][frac >= 0.5]] = True
    # ---- clonal ground truth
    C = full.obsm["clones"].tocsr()
    clone = np.where(np.asarray(C.sum(1)).ravel() > 0, np.asarray(C.argmax(1)).ravel(), -1)
    late = (tp > 2) & (clone >= 0)
    df = pd.DataFrame({"clone": clone[late], "ann": ann[late]})
    cnt = pd.crosstab(df.clone, df.ann)
    nm = cnt.get("Neutrophil", 0) + cnt.get("Monocyte", 0)
    diff = cnt.drop(columns=["Undifferentiated"], errors="ignore").sum(1)
    ok = (nm >= 2) & (nm >= 0.5 * diff)
    bias = (cnt.get("Neutrophil", 0) / nm.replace(0, np.nan))[ok]
    d2 = (tp == 2) & (ann == "Undifferentiated") & np.isin(clone, bias.index)
    fate = bias.reindex(clone[d2]).values
    return full, L, lineage, d2, fate


def main():
    full, L, lineage, d2, fate = build()
    ann = full.obs["Cell type annotation"].astype(str).values; tp = full.obs["Time point"].astype(float).values
    print(f"lineage cells {lineage.sum()}, day-2 cells with clonal fate {d2.sum()} "
          f"(Neu-biased >0.5: {(fate > 0.5).sum()}, Mo-biased <0.5: {(fate < 0.5).sum()})", flush=True)
    rng = np.random.default_rng(SEED)
    li = np.where(lineage)[0]
    sel = np.sort(np.concatenate([rng.choice(li[ann[li] == t], int(round(N_CELLS * (ann[li] == t).mean())), replace=False)
                                  for t in np.unique(ann[li])]))
    sub_raw = full[sel].copy(); del sub_raw.obsm["clones"]
    tmp = Path("/tmp/larry_sub.h5ad"); sub_raw.write_h5ad(tmp)
    a = P15.prepare(str(tmp))
    genes = list(a.var_names); tfs = tfs_in(genes)
    # ---- reference sets on the evaluated genes
    X2 = L[d2][:, genes].X; X2 = X2.toarray() if hasattr(X2, "toarray") else X2
    rho = np.array([spearmanr(X2[:, j], fate)[0] if X2[:, j].std() > 0 else 0.0 for j in range(len(genes))])
    oracle = pd.Series(np.nan_to_num(np.abs(rho)), index=genes)
    Lt = L[np.isin(ann, ["Neutrophil", "Monocyte"])][:, genes]
    Xt = Lt.X.toarray(); isneu = (Lt.obs["Cell type annotation"] == "Neutrophil").values
    lfc = pd.Series(np.abs(Xt[isneu].mean(0) - Xt[~isneu].mean(0)), index=genes)
    pd.DataFrame({"oracle_abs_spearman": oracle, "signed_spearman_with_Neu": rho, "terminal_abs_lfc": lfc}).to_csv(ROOT / "results" / (f"larry_oracle_{TAG}.csv" if TAG else "larry_oracle.csv"))

    def refsets(space, n_early, n_late, excl):
        o, l = oracle[space].sort_values(ascending=False), lfc[space].sort_values(ascending=False)
        early = list(o.index[:n_early]); late = [g for g in l.index if g not in set(o.index[:excl])][:n_late]
        return early, late
    REF = {"all genes": (genes, *refsets(genes, 50, 200, 200)), "TFs": (tfs, *refsets(tfs, 10, 30, 30))}
    print(f"{len(genes)} genes, {len(tfs)} TFs. oracle top 15: {' '.join(oracle.sort_values(ascending=False).index[:15])}", flush=True)
    print(f"oracle top TFs: {' '.join(oracle[tfs].sort_values(ascending=False).index[:10])}", flush=True)

    term = pd.Series(np.where(np.isin(a.obs["Cell type annotation"], ["Neutrophil", "Monocyte"]),
                              a.obs["Cell type annotation"].map({"Neutrophil": "Neu", "Monocyte": "Mo"}), None),
                     index=a.obs_names, dtype=object)
    sc.pp.pca(a, n_comps=20)
    early2 = (a.obs["Time point"].astype(float) == 2).values & term.isna().values
    dd = np.linalg.norm(a.obsm["X_pca"] - a.obsm["X_pca"][early2].mean(0), axis=1); dd[~early2] = np.inf
    a.obs["sim_time"] = 1.0; a.obs.iloc[int(np.argmin(dd)), a.obs.columns.get_loc("sim_time")] = 0.0
    a.obs = a.obs[["Cell type annotation", "sim_time"]]          # methods see no time points / clones
    ctx = DecisionContext(decision=DEC, fates=["Mo", "Neu"], terminal=term, cells=np.ones(a.n_obs, bool), seed=0)
    print(f"method input: {a.n_obs} cells, terminal {term.value_counts().to_dict()}", flush=True)
    del full, L, sub_raw; import gc; gc.collect()             # free ~3 GB before methods (competitors run as subprocesses)
    TAGC = f"larry_seed{SEED}"
    methods = {
        "OURS_v2_fast": lambda: score_timing_cf_v2(a, ctx, candidates=300, max_cells=300, n_jobs=2),
        "OURS_v1_fast": lambda: score_timing_cf_fast(a, ctx, candidates=300, max_cells=300, n_jobs=2),
        "OURS_v2_TF": lambda: score_timing_cf_v2(a, ctx, candidates=tfs, regulators=tfs, max_cells=300, n_jobs=2),
        # wider candidate filters (2 Nov): 1000 genes, or the top 300 plus every TF (regulators = all genes)
        "OURS_v2_c1000": lambda: score_timing_cf_v2(a, ctx, candidates=1000, max_cells=300, n_jobs=2),
        "OURS_v2_c300TF": lambda: score_timing_cf_v2(a, ctx, candidates=sorted(set(candidate_genes(a, ctx, 300)) | set(tfs)),
                                                     max_cells=300, n_jobs=2),
        # direct-effect removal (3 Nov): subtract the K=0 shift ("sub") or mask the clamped gene at readout ("mask")
        **{f"OURS_v2{d}_TF": (lambda d=d: score_timing_cf_v2(a, ctx, candidates=tfs, regulators=tfs, max_cells=300,
                                                            n_jobs=2, score_mode=f"perfate_{d}")) for d in ("sub", "mask")},
        **{f"OURS_v2{d}_c300TF": (lambda d=d: score_timing_cf_v2(a, ctx, candidates=sorted(set(candidate_genes(a, ctx, 300)) | set(tfs)),
                                                                max_cells=300, n_jobs=2, score_mode=f"perfate_{d}")) for d in ("sub", "mask")},
        # published competitors (3 Nov), own environments, see competitors.py
        "CEFCON": lambda: score_cefcon(a, ctx, TAGC),
        "CellOracle": lambda: score_celloracle(a, ctx, TAGC),
        "CellRank": lambda: score_cellrank(a, ctx, TAGC),
        **{k: (lambda f=f: f(a, ctx)) for k, f in BASE.items() if not k.startswith("ORACLE")},
    }
    if ONLY:
        methods = {k: v for k, v in methods.items() if k in ONLY}
    if TAG:
        json.dump({sp: {"pool": list(p), "early": e, "late": l} for sp, (p, e, l) in REF.items()},
                  open(ROOT / "results" / f"larry_refsets_{TAG}.json", "w"))
    scores = {}
    rows = []
    if os.environ.get("RESUME"):                                   # keep finished methods, run the rest
        prev = pd.read_csv(ROOT / "results" / "larry_eval.csv"); rows = prev.to_dict("records")
        methods = {k: v for k, v in methods.items() if k not in set(prev["method"])}
    for name, fn in methods.items():
        t0 = time.time(); s = fn(); sec = round(time.time() - t0, 1); scores[name] = s
        for space in (["TFs"] if name.endswith("_TF") else ["all genes", "TFs"]):
            pool, early, late = REF[space]
            ss = s.reindex(pool).fillna(s.min() - 1)
            jit = np.random.default_rng(0).random(len(ss)) * 1e-9      # break exact ties at random
            sub = ss.reindex(early + late).values + jit[:len(early + late)]
            r = {"method": name, "ranked_among": space,
                 "auprc_early_vs_late": average_precision_score(np.r_[np.ones(len(early)), np.zeros(len(late))], sub),
                 "chance": len(early) / (len(early) + len(late)),
                 "spearman_with_oracle": spearmanr(ss.values, oracle[pool].values)[0],
                 "early_in_top50": len(set(ss.sort_values(ascending=False).index[:50]) & set(early)),
                 **P15.evaluate(ss, DEC), "top10": " ".join(ss.sort_values(ascending=False).index[:10]), "seconds": sec}
            rows.append(r)
            print(f"{name:32s} [{space:9s}] AUPRC early-vs-late={r['auprc_early_vs_late']:.2f} (chance {r['chance']:.2f}) "
                  f"rho(oracle)={r['spearman_with_oracle']:.2f} early@50={r['early_in_top50']} lit core@20={r['core_hits@20']}/{r['n_core']} "
                  f"lit ranks: {r['ranks']}", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results" / (f"larry_eval_{TAG}.csv" if TAG else "larry_eval.csv"), index=False)
        if TAG:
            pd.DataFrame(scores).to_csv(ROOT / "results" / f"larry_scores_{TAG}.csv")
    print("EXIT 0")


if __name__ == "__main__":
    main()
