"""Dev-only sweep of the onset threshold (frac) and edge tolerance for OURS_timing_cf (BF+TF reps 0-2)."""
import glob, itertools, warnings, sys
import anndata as ad, pandas as pd
warnings.filterwarnings("ignore")
from driverbench.evaluate import contexts, metrics
from driverbench.probe_interventional import score_interventional_probe
from driverbench.method_timing import timing_mask
from driverbench.paths import SYNTH, RESULTS
rows = []
files = [f for f in sorted(glob.glob(str(SYNTH / "BF_rep*.h5ad"))) + sorted(glob.glob(str(SYNTH / "TF_rep*.h5ad"))) if int(f[-6]) <= 2]
for frac, tol in itertools.product([0.05, 0.1, 0.2, 0.3], [0.0, 1.0]):
    for f in files:
        a = ad.read_h5ad(f); ctx = next(contexts(a, 0))
        s = score_interventional_probe(a, ctx, allowed_fn=timing_mask(tol, frac))
        m = metrics(s, a.var, "D1")
        rows.append({"frac": frac, "tol": tol, "dataset": f.split("/")[-1][:-5], **m})
    d = pd.DataFrame(rows); print(d[(d.frac == frac) & (d.tol == tol)].groupby(d.dataset.str[:2]).auprc_vs_markers.mean().round(3).to_dict(), frac, tol, flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "dev_sweep_timing.csv", index=False)
