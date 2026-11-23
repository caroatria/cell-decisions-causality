"""Runs inside the CellOracle environment (tools/envs/celloracle). Called by competitors.py.
Input  <dir>/input.h5ad (see competitors.export). Output <dir>/celloracle_scores.csv.
Standard CellOracle pipeline (tutorial settings): mouse scATAC-atlas base GRN; KNN imputation (k = 2.5% of cells);
cluster-specific GRNs (clusters = terminal fates + progenitors; alpha=10; p<0.001, top 2,000 links);
fit_GRN_for_simulation(alpha=10). Then, for every regulator in the base GRN that is among the genes, an in-silico
knockout (expression 0, n_propagation=3). Readout: the same fate classifier as every method (standardise + logistic
regression C=0.1 on terminal cells, here on CellOracle's imputed expression), applied to the undecided cells before and
after the knockout. Gene score = max over fates of |mean change in P(fate)|. Genes not simulated get NaN."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, celloracle as co
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

d = Path(sys.argv[1]); cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
a = sc.read_h5ad(d / "input.h5ad")
MAXC = int(cfg.get("max_cells", 3000))     # KNN imputation with k=2.5% of 8,000 cells exceeds 8 GB; tutorials use ~3k cells
if a.n_obs > MAXC:                          # stratified by fate label, fixed seed
    rng = np.random.default_rng(0); keep = []
    for f, idx in a.obs.groupby("fate").indices.items():
        keep += list(rng.choice(idx, size=max(1, round(len(idx) * MAXC / a.n_obs)), replace=False))
    a = a[np.sort(keep)].copy(); print(f"subsampled to {a.n_obs} cells", flush=True)
a.X = np.asarray(a.X, dtype=np.float32)
fates = json.loads(a.uns["fates_json"])
a.obs["fate"] = a.obs["fate"].astype("category")
a.layers["raw_count"] = np.expm1(a.X)      # CellOracle's variability step reads this layer; = counts per 10,000
t0 = time.time()
if cfg.get("prior_tfs"):    # 17 Nov 2026, synthetic data: TF-list prior = every listed TF may regulate every other gene
    tfl = [t for t in cfg["prior_tfs"] if t in a.var_names]
    genes_ = list(a.var_names)
    base = pd.DataFrame({"peak_id": [f"peak_{g}" for g in genes_], "gene_short_name": genes_})
    for t in tfl:
        base[t] = [0.0 if g == t else 1.0 for g in genes_]
else:
    base = co.data.load_mouse_scATAC_atlas_base_GRN()
_cv = co.Oracle.score_cv_vs_mean
def _cv_safe(self, N, **kw):   # 17 Nov 2026: CellOracle's QC gene selection indexes past the end with <100 genes
    try:                       # (synthetic data); fall back to "all genes variable". No effect on real data.
        return _cv(self, N, **kw)
    except IndexError:
        self.cv_mean_selected_genes = np.array(self.adata.var_names)
co.Oracle.score_cv_vs_mean = _cv_safe
oracle = co.Oracle()
oracle.import_anndata_as_normalized_count(adata=a, cluster_column_name="fate", embedding_name="X_pca2")
oracle.import_TF_data(TF_info_matrix=base)
oracle.perform_PCA()
elbow = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_)) > 0.002))[0]   # tutorial rule
n_comps = int(min(elbow[0], 50)) if len(elbow) else 30
k = max(int(0.025 * oracle.adata.shape[0]), 5)
oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k * 8, b_maxl=k * 4, n_jobs=2)
links = oracle.get_links(cluster_name_for_GRN_unit="fate", alpha=10, verbose_level=0)
links.filter_links(p=0.001, weight="coef_abs", threshold_number=2000)
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)
genes = list(oracle.adata.var_names)
regs = sorted(set(base.columns[2:]) & set(genes))
# readout: same classifier design as the benchmark (terminal cells -> out-of-sample for progenitors)
Xi = oracle.adata.layers["imputed_count"]; Xi = Xi.toarray() if hasattr(Xi, "toarray") else np.asarray(Xi)
fate = oracle.adata.obs["fate"].astype(str).values
tm = fate != "progenitor"
scl = StandardScaler().fit(Xi[tm]); clf = LogisticRegression(C=0.1, max_iter=2000).fit(scl.transform(Xi[tm]), fate[tm])
order = [list(clf.classes_).index(f) for f in fates]
region = oracle.adata.obs["region"].values.astype(bool)
P0 = clf.predict_proba(scl.transform(Xi[region]))[:, order]
print(f"{oracle.adata.shape[0]} cells, {len(genes)} genes, {len(regs)} regulators to knock out, {region.sum()} undecided cells; "
      f"setup {time.time() - t0:.0f}s", flush=True)
rows = {}
for i, g in enumerate(regs):
    if float(np.asarray(Xi[:, genes.index(g)]).max()) <= 0:
        continue
    try:
        oracle.simulate_shift(perturb_condition={g: 0.0}, n_propagation=3, ignore_warning=True)
    except ValueError as e:                     # CellOracle refuses genes without enough GRN connections: no effect
        rows[g] = {**{f"dP_{f}": 0.0 for f in fates}, "score": 0.0}; continue
    Xs = oracle.adata.layers["simulated_count"]; Xs = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)
    P1 = clf.predict_proba(scl.transform(Xs[region]))[:, order]
    dP = (P1 - P0).mean(0)
    rows[g] = {**{f"dP_{f}": dP[j] for j, f in enumerate(fates)}, "score": float(np.abs(dP).max())}
    if i % 25 == 0:
        print(f"  {i}/{len(regs)} {g} {rows[g]['score']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
pd.DataFrame(rows).T.to_csv(d / "celloracle_scores.csv")
print(f"done in {time.time() - t0:.0f}s", flush=True)
