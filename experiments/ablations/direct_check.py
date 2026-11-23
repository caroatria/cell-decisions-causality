"""Direct-effect removal (3 Nov): the fate classifier reads the clamped gene itself, so part of every gene's score
is a direct (observational) effect; the K=0 ablation shows that part alone scores 0.32. Two removals:
  perfate_sub   per-fate shift minus the K=0 shift (clamp + read without rollout)
  perfate_mask  clamped gene reset to its unclamped rollout value before the classifier reads the cell
Same 28 held-out synthetic decisions as eval_v2.py / ablation_v2.py. Writes results/direct_check.csv."""
import glob, warnings
from pathlib import Path
import pandas as pd, anndata as ad
warnings.filterwarnings("ignore")
from driverbench.evaluate import contexts, metrics
from driverbench.method_timing import score_timing_cf_v2
from driverbench.paths import ROOT, SYNTH
rows = []
for f in sorted(glob.glob(str(SYNTH / "*.h5ad"))):
    name = Path(f).stem; model, rep = name.split("_rep"); rep = int(rep)
    if model in ("BF", "TF") and rep <= 2:
        continue
    a = ad.read_h5ad(f)
    for ctx in contexts(a, seed=0):
        if name == "HSC_rep2" and ctx.decision == "D2":
            continue
        for m in ("perfate", "perfate_sub", "perfate_mask"):
            rows.append({"dataset": name, "model": model, "decision": ctx.decision, "mode": m,
                         **metrics(score_timing_cf_v2(a, ctx, score_mode=m), a.var, ctx.decision)})
    print(name, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "direct_check.csv", index=False)
print("EXIT 0")
