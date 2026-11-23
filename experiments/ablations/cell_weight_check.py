"""Cell weighting of the intervention effect (7 Nov 2026), on the 44 held-out decisions.
Variants of probe_interventional.score_interventional_probe(cell_weight=...), everything else = version 2:
  fate_open    per-fate weight P_f(1-P_f) of the undecided cells
  fate_region  per-fate region: half of the non-terminal cells with the highest P_f(1-P_f)
  pseudotime   Gaussian weight on pseudotime, centred on the middle of the decision region
Uniform weighting = version 2 (results/v2_heldout.csv). Writes results/cell_weight_check.csv."""
import glob, warnings
from pathlib import Path
import pandas as pd, anndata as ad
warnings.filterwarnings("ignore")
from driverbench.evaluate import contexts, metrics
from driverbench.probe_interventional import score_interventional_probe
from driverbench.method_timing import timing_mask
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
        for w in ("fate_open", "fate_region", "pseudotime"):
            s = score_interventional_probe(a, ctx, allowed_fn=timing_mask(0.0, 0.2), K=3, score_mode="perfate", cell_weight=w)
            rows.append({"dataset": name, "model": model, "decision": ctx.decision, "variant": w,
                         **metrics(s, a.var, ctx.decision)})
    print(name, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "cell_weight_check.csv", index=False)
print("EXIT 0")
