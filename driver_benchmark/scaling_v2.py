"""Runtime scaling of version 2 (K=3, per-fate ranking), one decision of BF_rep0 (5 Nov 2026).
Same protocol as scaling.py (which measured version 1, K=15):
  genes: BF_rep0 enlarged with noisy copies of existing genes, 500 cells; exact version vs the real-data
         settings (300 candidate genes, at most 300 undecided cells, 2 parallel jobs)
  cells: 1 / 4 / 8 snapshots per BF_rep0 trajectory (500 / 2,000 / 4,000 cells), exact version
A warm-up call is run first (one-off numba compilation inside DPT).
Writes results/scaling_v2_genes.csv and results/scaling_v2_cells.csv."""
import time, warnings
from pathlib import Path
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
import networks as N, simulate as S
from evaluate import contexts
from method_timing import score_timing_cf_v2
from scaling import enlarge_genes
ROOT = Path(__file__).resolve().parent
import anndata as ad

def timed(a, fn):
    ctx = next(contexts(a, 0)); t0 = time.perf_counter(); fn(a, ctx)
    return {"genes": a.n_vars, "cells": a.n_obs, "total_s": round(time.perf_counter() - t0, 1)}

exact = lambda a, ctx: score_timing_cf_v2(a, ctx)
fast = lambda a, ctx: score_timing_cf_v2(a, ctx, candidates=300, max_cells=300, n_jobs=2)
base = ad.read_h5ad(ROOT / "data" / "BF_rep0.h5ad"); rng = np.random.default_rng(0)
timed(base, exact)  # warm-up
rows = []
for G in (100, 200, 400, 800, 1600):
    rows.append({"variant": "exact", **timed(enlarge_genes(base, G, rng), exact)}); print(rows[-1], flush=True)
for G in (400, 800, 1600, 3200):
    rows.append({"variant": "fast (300 genes, 300 cells)", **timed(enlarge_genes(base, G, rng), fast)}); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "scaling_v2_genes.csv", index=False)
simdir = ROOT / "sims" / "BF" / "rep0" / "BF" / "simulations"; m = N.bifurcation()
rows = []
for c in (1, 4, 8):
    rows.append({"variant": "exact", **timed(S.build_snapshot(m, simdir, 0, cells_per_traj=c), exact)}); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "scaling_v2_cells.csv", index=False)
print("EXIT 0")
