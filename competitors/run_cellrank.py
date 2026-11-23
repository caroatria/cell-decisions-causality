"""Runs inside the CellRank environment (tools/envs/cellrank; CellRank 2.0.6, Weiler et al. 2024). Called by competitors.py.
Input  <dir>/input.h5ad (see competitors.export). Output <dir>/cellrank_scores.csv.
Standard CellRank 2 pipeline without RNA velocity: PseudotimeKernel on the shared DPT (from the shared root; soft
thresholding, the default), 15-NN graph on 20 PCs (as everywhere else), GPCCA estimator with the terminal states set to the
annotated terminal cells (the same labels every method gets), fate probabilities, then compute_lineage_drivers (Pearson
correlation of each gene with each fate probability over all cells, CellRank's default). Gene score = max over fates of
the correlation (CellRank reports positive correlations as drivers towards a fate)."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, cellrank as cr

d = Path(sys.argv[1])
a = sc.read_h5ad(d / "input.h5ad")
a.X = np.asarray(a.X, dtype=np.float32)
fates = json.loads(a.uns["fates_json"])
t0 = time.time()
sc.pp.pca(a, n_comps=min(20, a.n_vars - 1, a.n_obs - 1))
sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
pk = cr.kernels.PseudotimeKernel(a, time_key="dpt").compute_transition_matrix(threshold_scheme="soft", show_progress_bar=False)
g = cr.estimators.GPCCA(pk)
term = a.obs["fate"].astype(str).where(a.obs["fate"].astype(str) != "progenitor")
g.set_terminal_states(pd.Series(pd.Categorical(term, categories=fates), index=a.obs_names))
try:
    g.compute_fate_probabilities(show_progress_bar=False)
except ValueError:          # iterative solver did not converge (seen on a few tree networks): CellRank's own suggested fallback
    g.compute_fate_probabilities(solver="direct", show_progress_bar=False)
drv = g.compute_lineage_drivers(lineages=fates)
corr = drv[[f"{f}_corr" for f in fates]].fillna(0.0)
out = corr.copy(); out["score"] = corr.max(axis=1)
out.to_csv(d / "cellrank_scores.csv")
print(f"{a.n_obs} cells, {a.n_vars} genes, fates {fates}; done in {time.time() - t0:.0f}s", flush=True)
