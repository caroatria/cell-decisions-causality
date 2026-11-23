"""Independent real-data test on Nestorowa et al. 2016 (mouse HSPCs, Smart-seq2, 1,656 cells).
Decision: HSC -> erythroid/megakaryocyte-primed (MegE) vs lympho-myeloid-primed (LyMy) progenitors, the first
major split. Everything below was fixed BEFORE running any method; the method code is frozen (version 2,
K=3, perfate, logreg readout). Run once.

Cells (paper clusters x sorting gates): stem = purple (mostly LT-HSC); transition = deeppink (MPPs);
terminal MegE = darkturquoise cells sorted as MEP (287); terminal LyMy = gold cells sorted as GMP or LMPP (254).
All other cells (incl. CMPs) are non-terminal. Root = LT-HSC in purple closest to the purple centroid.
Ground truth: literature regulators of this split (TO REVIEW by a domain expert).
Usage: python experiments/real_data/nestorowa_eval.py      -> results/nestorowa_eval.csv"""
import time, warnings, os
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
warnings.filterwarnings("ignore")
from driverbench import realdata as P15
from driverbench.baselines import SCORERS as BASE, DecisionContext
from driverbench.method_timing import score_timing_cf_fast, score_timing_cf_v2
from driverbench.probe_interventional import candidate_genes
from driverbench.competitors import score_cefcon, score_celloracle, score_cellrank
from driverbench.paths import REAL, ROOT, MOUSE_TFS
DEC = "MegE_LyMy"
P15.LITERATURE[DEC] = {
    "core": ["Gata1", "Klf1", "Tal1", "Spi1", "Cebpa", "Ikzf1"],   # Gata1/Pu.1 antagonism; Klf1, Scl/Tal1 -> MegE; Cebpa, Ikaros -> LyMy
    "extended": ["Gata2", "Zfpm1", "Nfe2", "Lmo2", "Myb", "Gfi1b", "Fli1", "Irf8", "Gfi1", "Cebpb", "Satb1"],
}


def tfs_in(var_names):
    raw = {l.strip() for l in open(MOUSE_TFS) if l.strip()}
    return [g for g in var_names if g in raw]


def context(a):
    a = a[a.obs.cluster != "grey"].copy()
    cl, gate = a.obs.cluster.astype(str).values, a.obs.gate.astype(str).values
    term = pd.Series(np.nan, index=a.obs_names, dtype=object)
    term[(cl == "darkturquoise") & (gate == "MEP")] = "MegE"
    term[(cl == "gold") & np.isin(gate, ["GMP", "LMPP"])] = "LyMy"
    sc.pp.pca(a, n_comps=20)
    stem = (cl == "purple") & (gate == "LTHSC")
    d = np.linalg.norm(a.obsm["X_pca"] - a.obsm["X_pca"][cl == "purple"].mean(0), axis=1); d[~stem] = np.inf
    a.obs["sim_time"] = 1.0; a.obs.iloc[int(np.argmin(d)), a.obs.columns.get_loc("sim_time")] = 0.0
    ctx = DecisionContext(decision=DEC, fates=["MegE", "LyMy"], terminal=term, cells=np.ones(a.n_obs, bool), seed=0)
    return a, ctx


def main():
    a = P15.prepare(str(REAL / "nestorowa16.h5ad"))
    a, ctx = context(a)
    tfs = tfs_in(list(a.var_names))
    lit = P15.LITERATURE[DEC]["core"] + P15.LITERATURE[DEC]["extended"]
    print(f"{a.n_obs} cells x {a.n_vars} genes, {len(tfs)} TFs; terminal {ctx.terminal.value_counts().to_dict()}; "
          f"literature genes missing: {[g for g in lit if g not in a.var_names]}; not TFs: {[g for g in lit if g in a.var_names and g not in tfs]}", flush=True)
    TAGC = "nestorowa"
    methods = {
        "OURS_v2_fast": lambda: score_timing_cf_v2(a, ctx, candidates=300, max_cells=300, n_jobs=2),
        "OURS_v1_fast": lambda: score_timing_cf_fast(a, ctx, candidates=300, max_cells=300, n_jobs=2),
        "OURS_v2_TF": lambda: score_timing_cf_v2(a, ctx, candidates=tfs, regulators=tfs, n_jobs=2),
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
    ONLY = [m for m in os.environ.get("METHODS", "").split(",") if m]
    if ONLY:
        methods = {k: v for k, v in methods.items() if k in ONLY}
    out = os.environ.get("OUT", "nestorowa_eval.csv")
    rows = []
    for name, fn in methods.items():
        t0 = time.time(); s = fn(); sec = round(time.time() - t0, 1)
        for space in (["TFs"] if name.endswith("_TF") else ["all genes", "TFs"]):
            ss = s if space == "all genes" else s.reindex(tfs)
            r = {"method": name, "ranked_among": space, **P15.evaluate(ss, DEC),
                 "top10": " ".join(ss.sort_values(ascending=False).index[:10]), "seconds": sec}
            rows.append(r)
            print(f"{name:32s} [{space:9s}] core@20={r['core_hits@20']}/{r['n_core']} all@20={r['all_hits@20']}/{r['n_all']} "
                  f"median core={r['core_median_rank']:.0f} (random ~{r['n_genes']/2:.0f})  ranks: {r['ranks']}", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results" / out, index=False)
    print("EXIT 0")


if __name__ == "__main__":
    main()
