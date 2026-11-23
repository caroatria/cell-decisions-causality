"""Further ablations of version 2 on the same 28 held-out decisions as ablation_v2.py (4 Nov 2026).
  no_timing   forward model without the timing rule (every gene may regulate every gene)
  K1, K10     rollout length (v2: K=3; K=0 and K=5 are in ablation_v2.csv / v2_heldout.csv)
  frac0.1, frac0.3   onset threshold (v2: 0.2)
Writes results/ablation_v2_extra.csv."""
import os, glob, warnings
from pathlib import Path
import pandas as pd, anndata as ad
warnings.filterwarnings("ignore")
from driverbench.evaluate import contexts, metrics
from driverbench.method_timing import score_timing_cf_v2, timing_mask
from driverbench.probe_interventional import score_interventional_probe
from driverbench.paths import ROOT, SYNTH
VARIANTS = {
    "no_timing": lambda a, c: score_interventional_probe(a, c, allowed_fn=None, K=3, score_mode="perfate"),
    "K1": lambda a, c: score_timing_cf_v2(a, c, K=1),
    "K10": lambda a, c: score_timing_cf_v2(a, c, K=10),
    "frac0.1": lambda a, c: score_interventional_probe(a, c, allowed_fn=timing_mask(0.0, 0.1), K=3, score_mode="perfate"),
    "frac0.3": lambda a, c: score_interventional_probe(a, c, allowed_fn=timing_mask(0.0, 0.3), K=3, score_mode="perfate"),
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
        for v, fn in VARIANTS.items():
            rows.append({"dataset": name, "model": model, "decision": ctx.decision, "variant": v, **metrics(fn(a, ctx), a.var, ctx.decision)})
    print(name, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / ("ablation_v2_extra" + ("_new" if os.environ.get("ONLY_NEW") else "") + ".csv"), index=False)
print("EXIT 0")
