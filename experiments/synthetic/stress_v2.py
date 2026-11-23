"""Fast vs exact for version 2 on the 28 held-out decisions (5 Nov 2026): the speed-ups with deliberately strict
settings (interventions on the top 30 genes by fate correlation, at most 100 undecided cells) against the exact
method (results/v2_heldout.csv, K=3 per-fate). Writes results/fast_vs_exact_v2.csv."""
import os, glob, time, warnings
from pathlib import Path
import pandas as pd, anndata as ad
warnings.filterwarnings("ignore")
from driverbench.evaluate import contexts, metrics
from driverbench.method_timing import score_timing_cf_v2
from driverbench.paths import ROOT, SYNTH
ex = pd.read_csv(ROOT / "results" / "v2_heldout.csv"); ex = ex[(ex.K == 3) & (ex["mode"] == "perfate")]
rows = []
for f in sorted(glob.glob(str(SYNTH / "*.h5ad"))):
    name = Path(f).stem; model, rep = name.split("_rep")
    if os.environ.get("ONLY_NEW") and not (model in ("BF", "TF") and int(rep) >= 5):
        continue                                            # ONLY_NEW=1: extra replicates 5-12 only (5 Nov 2026)
    if model in ("BF", "TF") and int(rep) <= 2: continue
    a = ad.read_h5ad(f)
    for ctx in contexts(a, seed=0):
        if name == "HSC_rep2" and ctx.decision == "D2": continue
        t0 = time.time(); s = score_timing_cf_v2(a, ctx, candidates=30, max_cells=100, n_jobs=1); sec = time.time() - t0
        e = ex[(ex.dataset == name) & (ex.decision == ctx.decision)].iloc[0]
        rows.append({"dataset": name, "model": model, "decision": ctx.decision, "auprc_exact": e.auprc_vs_markers,
                     "auprc_fast": metrics(s, a.var, ctx.decision)["auprc_vs_markers"], "seconds_fast": round(sec, 1)})
    print(name, flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / ("fast_vs_exact_v2" + ("_new" if os.environ.get("ONLY_NEW") else "") + ".csv"), index=False)
print("EXIT 0")
