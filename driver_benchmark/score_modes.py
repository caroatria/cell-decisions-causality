"""Compare the three scoring rules of OURS (prob / logodds / perfate) from ONE run per decision.
Writes results/score_modes_benchmark.csv (synthetic + HSC, metrics vs ground truth) and
results/score_modes_paul15.csv (literature regulators, TF variants B and C).
Usage: python score_modes.py benchmark | paul15 <path/to/paul15.h5ad>
"""
import sys, glob, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
warnings.filterwarnings("ignore")
from probe_interventional import score_interventional_probe
from method_timing import timing_mask

ROOT = Path(__file__).resolve().parent
MODES = ("prob", "logodds", "perfate")


def modes_from_shifts(sh, fates, cand_mask):
    """Recompute the three scores from per-fate shifts (same rules as score_interventional_probe)."""
    P = sh[[f"prob_{f}" for f in fates]].values; L = sh[[f"logodds_{f}" for f in fates]].values
    out = {}
    for m in MODES:
        s = np.zeros(len(sh))
        if m == "prob": s[cand_mask] = np.maximum(np.nanmax(P[cand_mask], 1), 0)
        elif m == "logodds": s[cand_mask] = np.maximum(np.nanmax(L[cand_mask], 1), 0)
        else: s[cand_mask] = np.nanmax(pd.DataFrame(P[cand_mask]).rank(pct=True).values, 1)
        out[m] = pd.Series(s, index=sh.index)
    return out


def benchmark():
    from evaluate import contexts, metrics
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "*.h5ad"))):
        a = ad.read_h5ad(f)
        for ctx in contexts(a, seed=0):
            name = Path(f).stem
            if name == "HSC_rep2" and ctx.decision == "D2":   # undefined decision, excluded everywhere
                continue
            _, sh = score_interventional_probe(a, ctx, allowed_fn=timing_mask(0.0, 0.2), return_shifts=True)
            cand = np.ones(a.n_vars, bool)
            for m, s in modes_from_shifts(sh, ctx.fates, cand).items():
                rows.append({"dataset": name, "model": a.uns["model"], "decision": ctx.decision, "mode": m,
                             **metrics(s, a.var, ctx.decision),
                             "baseline_minority_prob": min(sh.attrs["baseline_fate_prob"].values())})
            print(name, ctx.decision, {r["mode"]: round(r["auprc_vs_markers"], 2) for r in rows[-3:]}, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "score_modes_benchmark.csv", index=False)


def paul15(path):
    import paul15_eval as P15
    from paul15_diagnostics import tf_list
    a = P15.prepare(path); tfs = tf_list(list(a.var_names)); rows = []
    for decision in P15.LINEAGES:
        sub, ctx, fates = P15.context_for(a, decision)
        for v, kw in [("B TF candidates", dict(candidates=tfs)), ("C TF candidates + TF regulators", dict(candidates=tfs, regulators=tfs))]:
            t0 = time.time()
            _, sh = score_interventional_probe(sub, ctx, allowed_fn=timing_mask(0.0, 0.2), n_jobs=2, return_shifts=True, **kw)
            cand = sub.var_names.isin(tfs)
            for m, s in modes_from_shifts(sh, ctx.fates, cand).items():
                s = s.reindex(tfs)                      # ranked among TFs, like the TF-restricted baselines
                r = {"decision": decision, "variant": v, "mode": m, **P15.evaluate(s, decision),
                     "top10": " ".join(s.sort_values(ascending=False).index[:10]),
                     "baseline_fate_prob": sh.attrs["baseline_fate_prob"], "seconds": round(time.time() - t0, 1)}
                rows.append(r)
                print(f"{decision} {v} {m:8s} core@10={r['core_hits@10']}/{r['n_core']} core@20={r['core_hits@20']} all@20={r['all_hits@20']}/{r['n_all']} "
                      f"median core={r['core_median_rank']:.0f}  base={ {k: round(x, 2) for k, x in sh.attrs['baseline_fate_prob'].items()} }\n   ranks: {r['ranks']}\n   top10: {r['top10']}", flush=True)
            pd.DataFrame(rows).to_csv(ROOT / "results" / "score_modes_paul15.csv", index=False)


if __name__ == "__main__":
    benchmark() if sys.argv[1] == "benchmark" else paul15(sys.argv[2])
