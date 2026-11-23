"""Ablations of version 2 (K=3, perfate) on the held-out synthetic decisions (same set as eval_v2.py:
BF/TF replicates 3-4, CBF, HSC; 28 decisions). Writes results/ablation_v2.csv.
Variants:
  v2            reference (identical settings to results/v2_heldout.csv, K=3 perfate)
  K0            no rollout: clamp and read the fate predictor directly = observational counterfactual
                with our predictor (only the direct effect of the clamped gene)
  region_all    interventions on all non-terminal cells instead of the top-half-entropy decision region
  pt_true       simulator time instead of DPT (upper bound; not available on real data)
  pt_pca_dist   naive pseudotime: distance from the root in PCA space
  pt_dpt_root   DPT from a perturbed root (random one of the root's 20 nearest cells), seeds 0-2
"""
import os, glob, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
warnings.filterwarnings("ignore")
from driverbench.evaluate import contexts, metrics
from driverbench.method_timing import score_timing_cf_v2
from driverbench.paths import ROOT, SYNTH
VARIANTS = {
    "v2": {}, "K0": {"K": 0}, "region_all": {"region_q": 0.0},
    "pt_true": {"pseudotime": "true"}, "pt_pca_dist": {"pseudotime": "pca_dist"},
    **{f"pt_dpt_root_s{s}": {"pseudotime": "dpt_root", "seed": s} for s in range(3)},
}
rows = []
for f in sorted(glob.glob(str(SYNTH / "*.h5ad"))):
    name = Path(f).stem; model, rep = name.split("_rep"); rep = int(rep)
    if os.environ.get("ONLY_NEW") and not (model in ("BF", "TF") and int(rep) >= 5):
        continue                                            # ONLY_NEW=1: extra replicates 5-12 only (5 Nov 2026)
    if model in ("BF", "TF") and rep <= 2:
        continue
    a = ad.read_h5ad(f)
    for ctx in contexts(a, seed=0):
        if name == "HSC_rep2" and ctx.decision == "D2":
            continue
        for v, kw in VARIANTS.items():
            s = score_timing_cf_v2(a, ctx, **kw)
            rows.append({"dataset": name, "model": model, "decision": ctx.decision, "variant": v, **metrics(s, a.var, ctx.decision)})
    print(name, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / ("ablation_v2" + ("_new" if os.environ.get("ONLY_NEW") else "") + ".csv"), index=False)
print("EXIT 0")
