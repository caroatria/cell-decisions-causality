"""Runs inside the CEFCON environment (tools/envs/cefcon). Called by competitors.py; do not import elsewhere.
Input  <dir>/input.h5ad (written by competitors.export): log-normalised X; obs['fate'] (terminal fate or
       'progenitor'); obs['dpt']; obs['P_<fate>'] (out-of-sample fate probability, same classifier as all methods).
Output <dir>/cefcon_scores.csv: one row per gene, one column per lineage (CEFCON gene score) + 'score'.
Lineage l = terminal cells of fate l + non-terminal cells with P_l >= 0.3 (undecided cells are in both lineages),
pseudotime = DPT from the shared root (CEFCON's own examples use Slingshot; we pass the pseudotime every method uses).
Lineage DE (CEFCON's optional node weights) = |mean(terminal l) - mean(earliest 20% of the lineage)|.
Genes constant within any lineage are dropped (CEFCON's correlation step fails on them).
Settings: epochs=350 (CEFCON default), repeats=1 (default 3; CPU budget), at most MAX_CELLS cells per lineage,
solver SCIP (free; the paper used Gurobi), edge_threshold_avgDegree=8 (default)."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, cefcon as cf

d = Path(sys.argv[1]); cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
MAX_CELLS, EPOCHS, REPEATS, SEED = cfg.get("max_cells", 2000), cfg.get("epochs", 350), cfg.get("repeats", 1), cfg.get("seed", 0)
a = sc.read_h5ad(d / "input.h5ad")
a.X = np.asarray(a.X, dtype=np.float32)
fates = json.loads(a.uns["fates_json"]) if "fates_json" in a.uns else sorted(set(a.obs["fate"]) - {"progenitor"})
rng = np.random.default_rng(SEED)
if cfg.get("prior_tfs"):    # 17 Nov 2026, synthetic data: TF-list prior = every listed TF -> every other gene
    net = pd.DataFrame([(t, g, "directed") for t in cfg["prior_tfs"] if t in a.var_names for g in a.var_names if g != t],
                       columns=["from", "to", "edge_type"])
else:
    net = pd.read_csv(Path(cfg["cefcon_root"]) / "prior_data" / f"network_{cfg.get('species', 'mouse')}.csv")
pt = a.obs["dpt"].values.astype(float)
keep = np.zeros(a.n_obs, bool)
for f in fates:
    inl = (a.obs["fate"].values == f) | ((a.obs["fate"].values == "progenitor") & (a.obs[f"P_{f}"].values >= 0.3))
    idx = np.where(inl)[0]
    if len(idx) > MAX_CELLS:
        idx = np.sort(rng.choice(idx, MAX_CELLS, replace=False))
    sel = np.zeros(a.n_obs, bool); sel[idx] = True
    a.obs[f] = np.where(sel, pt, np.nan)
    keep |= sel
    early = sel & (pt <= np.quantile(pt[sel], 0.2))
    term = sel & (a.obs["fate"].values == f)
    a.var[f"{f}_logFC"] = np.abs(a.X[term].mean(0) - a.X[early].mean(0))
a = a[keep].copy(); a.uns["lineages"] = list(fates)
# CEFCON's correlation step fails when a gene is constant within a lineage (e.g. Paul15 Mk: 62 unexpressed genes);
# drop such genes (they get the lowest score)
const = np.zeros(a.n_vars, bool)
for f in fates:
    const |= np.asarray(a.X[a.obs[f].notna().values].std(0) == 0).ravel()
a = a[:, ~const].copy()
print(f"dropped {const.sum()} genes constant within a lineage", flush=True)
for c in [c for c in a.obs.columns if c not in fates]:
    del a.obs[c]
t0 = time.time()
data = cf.data_preparation(a, net)
out = {}
for l, dl in data.items():
    m = cf.NetModel(epochs=EPOCHS, repeats=REPEATS, cuda=-1, seed=SEED)
    m.run(dl, showProgressBar=False)
    r = m.get_cefcon_results(edge_threshold_avgDegree=8)
    r.gene_influence_score(); r.driver_regulators(solver="SCIP")
    inf = r.influence_score["influence_score"]
    drv = set(r.driver_regulator.index[r.driver_regulator["is_driver_regulator"]])
    # CEFCON's ranking: driver regulators first (by influence score), then all other network genes by influence
    out[l] = inf + 1000.0 * inf.index.isin(drv)
    print(f"lineage {l}: {dl.n_obs} cells, {len(inf)} network genes, {len(drv)} drivers; top: {list(r.driver_regulator.index[:8])}", flush=True)
S = pd.DataFrame(out)
S["score"] = S.rank(pct=True).max(axis=1)          # each lineage contributes its top genes equally
S.to_csv(d / "cefcon_scores.csv")
print(f"done in {time.time() - t0:.0f}s", flush=True)
