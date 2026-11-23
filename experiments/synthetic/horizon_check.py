"""How does the number of fast-forward steps K affect (a) collapse of undecided cells into one fate and
(b) driver recovery? Synthetic dev networks (BF, TF reps 0-2) + Paul15 (TF variant C).
Usage: python experiments/synthetic/horizon_check.py benchmark | paul15 <path>"""
import sys, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
warnings.filterwarnings("ignore")
from driverbench.probe_interventional import score_interventional_probe
from driverbench.method_timing import timing_mask
from driverbench.score_modes import modes_from_shifts
from driverbench.paths import ROOT, SYNTH
KS = (1, 3, 5, 10, 15)

def benchmark():
    from driverbench.evaluate import contexts, metrics
    rows = []
    files = [f for f in sorted(glob.glob(str(SYNTH / "*.h5ad"))) if Path(f).stem.split("_")[0] in ("BF", "TF") and int(f[-6]) <= 2]
    for f in files:
        a = ad.read_h5ad(f); ctx = next(contexts(a, 0))
        for K in KS:
            _, sh = score_interventional_probe(a, ctx, allowed_fn=timing_mask(0.0, 0.2), K=K, return_shifts=True)
            for m, s in modes_from_shifts(sh, ctx.fates, np.ones(a.n_vars, bool)).items():
                rows.append({"dataset": Path(f).stem, "K": K, "mode": m, "auprc_vs_markers": metrics(s, a.var, ctx.decision)["auprc_vs_markers"],
                             "minority_prob": min(sh.attrs["baseline_fate_prob"].values())})
        cur = pd.DataFrame(rows); cur = cur[cur.dataset == Path(f).stem]
        print(Path(f).stem, cur.pivot_table(index="K", columns="mode", values="auprc_vs_markers").round(2).to_dict(), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "horizon_benchmark.csv", index=False)

def paul15(path):
    from driverbench import realdata as P15
    from paul15_diagnostics import tf_list
    a = P15.prepare(path); tfs = tf_list(list(a.var_names)); rows = []
    for decision in P15.LINEAGES:
        sub, ctx, fates = P15.context_for(a, decision)
        for K in KS:
            _, sh = score_interventional_probe(sub, ctx, allowed_fn=timing_mask(0.0, 0.2), K=K, n_jobs=2, candidates=tfs, regulators=tfs, return_shifts=True)
            for m, s in modes_from_shifts(sh, ctx.fates, sub.var_names.isin(tfs)).items():
                r = {"decision": decision, "K": K, "mode": m, **P15.evaluate(s.reindex(tfs), decision),
                     "baseline": {k: round(v, 2) for k, v in sh.attrs["baseline_fate_prob"].items()}}
                rows.append(r)
                print(f"{decision} K={K:2d} {m:8s} base={r['baseline']} core@10={r['core_hits@10']}/{r['n_core']} all@20={r['all_hits@20']}/{r['n_all']} median core={r['core_median_rank']:.0f}  ranks: {r['ranks']}", flush=True)
            pd.DataFrame(rows).to_csv(ROOT / "results" / "horizon_paul15.csv", index=False)

if __name__ == "__main__":
    benchmark() if sys.argv[1] == "benchmark" else paul15(sys.argv[2])
