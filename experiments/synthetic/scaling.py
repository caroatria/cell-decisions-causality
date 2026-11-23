"""Runtime scaling of OURS_timing_cf (one decision), per step.

genes: BF_rep0 enlarged with noisy copies of existing genes (realistic correlation), 500 cells
cells: real extra cells, taken as 1 / 4 / 8 snapshots per BF_rep0 trajectory (needs sims/BF/rep0)
fast:  same gene sizes up to 4000 with the speed-ups (100 candidate genes, 150 undecided cells)
Writes results/scaling_genes.csv, results/scaling_cells.csv, results/scaling_fast_genes.csv.
The first call of a session includes a one-off ~25 s numba compilation inside DPT; a warm-up call
is run first so the numbers are steady-state.

Usage: python experiments/synthetic/scaling.py
"""
import cProfile, pstats, time, warnings
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from driverbench import networks as N
from driverbench import simulate as S
from driverbench.evaluate import contexts
from driverbench.method_timing import score_timing_cf, score_timing_cf_fast

from driverbench.paths import ROOT, SYNTH
def profile(a, fn=score_timing_cf):
    ctx = next(contexts(a, 0))
    pr = cProfile.Profile(); t0 = time.perf_counter(); pr.enable()
    fn(a, ctx)
    pr.disable(); tot = time.perf_counter() - t0
    cum = {"lasso_fit_s": 0.0, "logreg_fit_s": 0.0, "rollout_s": 0.0, "dpt_s": 0.0}
    for (fname, _, name), v in pstats.Stats(pr).stats.items():
        if name == "fit" and "coordinate_descent" in fname: cum["lasso_fit_s"] += v[3]
        elif name == "fit" and "logistic" in fname: cum["logreg_fit_s"] += v[3]
        elif name == "rollout": cum["rollout_s"] += v[3]
        elif name == "dpt": cum["dpt_s"] += v[3]
    return {"genes": a.n_vars, "cells": a.n_obs, "total_s": round(tot, 1), **{k: round(v, 1) for k, v in cum.items()}}


def enlarge_genes(base, G, rng):
    X = base.X; extra = G - X.shape[1]
    idx = rng.integers(0, X.shape[1], extra)
    Xg = np.hstack([X, X[:, idx] + rng.normal(0, 0.3, (X.shape[0], extra))]).clip(0).astype(np.float32)
    var = pd.concat([base.var, pd.DataFrame({"role": "null", "decision": "", "kind": "null"},
                                            index=[f"x{i}" for i in range(extra)])])
    a = ad.AnnData(Xg, obs=base.obs.copy(), var=var); a.uns.update(base.uns)
    return a


def main():
    base = ad.read_h5ad(SYNTH / "BF_rep0.h5ad")
    rng = np.random.default_rng(0)
    profile(base)  # warm-up (numba compilation)
    rows = [profile(enlarge_genes(base, G, rng)) for G in (100, 200, 400, 800)]
    pd.DataFrame(rows).to_csv(ROOT / "results" / "scaling_genes.csv", index=False); print(pd.DataFrame(rows))
    simdir = ROOT / "sims" / "BF" / "rep0" / "BF" / "simulations"
    m = N.bifurcation()
    rows = [profile(S.build_snapshot(m, simdir, 0, cells_per_traj=c)) for c in (1, 4, 8)]
    pd.DataFrame(rows).to_csv(ROOT / "results" / "scaling_cells.csv", index=False); print(pd.DataFrame(rows))
    fast = lambda a, ctx: score_timing_cf_fast(a, ctx, candidates=100, max_cells=150, n_jobs=2)
    rows = []
    for G in (100, 200, 400, 800):
        rows.append({"variant": "exact version", **profile(enlarge_genes(base, G, rng))})
    for G in (200, 400, 800, 2000, 4000):
        rows.append({"variant": "fast: 100 candidate genes, 150 cells", **profile(enlarge_genes(base, G, rng), fast)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "scaling_fast_genes.csv", index=False)


if __name__ == "__main__":
    main()
