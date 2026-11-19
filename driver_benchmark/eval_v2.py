"""Evaluate the version-2 settings (K forward steps + scoring rule), chosen on dev data only
(BF + TF replicates 0-2; results/horizon_benchmark.csv), on all decisions NOT used for that choice:
BF/TF replicates 3-4, CBF (held out), HSC (curated, never tuned on). K=3 is the choice, K=5 a sensitivity check.
Writes results/v2_heldout.csv."""
import os, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
warnings.filterwarnings("ignore")
from evaluate import contexts, metrics
from probe_interventional import score_interventional_probe
from method_timing import timing_mask
from score_modes import modes_from_shifts
ROOT = Path(__file__).resolve().parent
rows = []
for f in sorted(glob.glob(str(ROOT / "data" / "*.h5ad"))):
    name = Path(f).stem; model, rep = name.split("_rep"); rep = int(rep)
    if os.environ.get("ONLY_NEW") and not (model in ("BF", "TF") and int(rep) >= 5):
        continue                                            # ONLY_NEW=1: extra replicates 5-12 only (5 Nov 2026)
    if model in ("BF", "TF") and rep <= 2:
        continue                                            # dev data used for the choice
    a = ad.read_h5ad(f)
    for ctx in contexts(a, seed=0):
        if name == "HSC_rep2" and ctx.decision == "D2":
            continue
        for K in (3, 5):
            _, sh = score_interventional_probe(a, ctx, allowed_fn=timing_mask(0.0, 0.2), K=K, return_shifts=True)
            for m, s in modes_from_shifts(sh, ctx.fates, np.ones(a.n_vars, bool)).items():
                rows.append({"dataset": name, "model": model, "decision": ctx.decision, "K": K, "mode": m,
                             **metrics(s, a.var, ctx.decision)})
    print(name, flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / ("v2_heldout" + ("_new" if os.environ.get("ONLY_NEW") else "") + ".csv"), index=False)
print("EXIT 0")
