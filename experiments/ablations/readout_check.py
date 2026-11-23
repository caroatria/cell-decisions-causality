"""Fate-probability source ("readout") for the version-2 method.
dev:     choose on BF + TF replicates 0-2 (K=3, perfate)          -> results/readout_dev.csv
heldout: all other synthetic + HSC decisions                       -> results/readout_heldout.csv
paul15:  diagnostic only (TF-restricted, variant C)                -> results/readout_paul15.csv
Usage: python experiments/ablations/readout_check.py dev|heldout|paul15 [paul15.h5ad]"""
import os, glob, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
warnings.filterwarnings("ignore")
from driverbench.evaluate import contexts, metrics
from driverbench.method_timing import score_timing_cf_v2
from driverbench.paths import ROOT, SYNTH
READOUTS = ["logreg", "pca_logreg", "soft_logreg", "knn_absorb"]


def synthetic(which):
    rows = []
    for f in sorted(glob.glob(str(SYNTH / "*.h5ad"))):
        name = Path(f).stem; model, rep = name.split("_rep"); rep = int(rep)
        if os.environ.get("ONLY_NEW") and not (model in ("BF", "TF") and int(rep) >= 5):
            continue                                            # ONLY_NEW=1: extra replicates 5-12 only (5 Nov 2026)
        dev = model in ("BF", "TF") and rep <= 2
        if dev != (which == "dev"):
            continue
        a = ad.read_h5ad(f)
        for ctx in contexts(a, seed=0):
            if name == "HSC_rep2" and ctx.decision == "D2":
                continue
            for ro in READOUTS:
                s, sh = score_timing_cf_v2(a, ctx, readout=ro, return_shifts=True)
                rows.append({"dataset": name, "model": model, "decision": ctx.decision, "readout": ro,
                             "baseline": sh.attrs["baseline_fate_prob"], **metrics(s, a.var, ctx.decision)})
        print(name, {r["readout"]: round(r["auprc_vs_markers"], 2) for r in rows if r["dataset"] == name}, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results" / (f"readout_{which}" + ("_new" if os.environ.get("ONLY_NEW") else "") + ".csv"), index=False)


def paul15(path):
    from driverbench import realdata as P15
    from paul15_diagnostics import tf_list
    a = P15.prepare(path); tfs = tf_list(list(a.var_names)); rows = []
    for decision in P15.LINEAGES:
        sub, ctx, fates = P15.context_for(a, decision)
        for ro in READOUTS:
            s, sh = score_timing_cf_v2(sub, ctx, readout=ro, n_jobs=2, candidates=tfs, regulators=tfs, return_shifts=True)
            r = {"decision": decision, "readout": ro, **P15.evaluate(s.reindex(tfs), decision),
                 "baseline": {k: round(v, 2) for k, v in sh.attrs["baseline_fate_prob"].items()}}
            rows.append(r)
            print(f"{decision} {ro:12s} base={r['baseline']} core@10={r['core_hits@10']}/{r['n_core']} "
                  f"all@20={r['all_hits@20']}/{r['n_all']} median core={r['core_median_rank']:.0f} ranks: {r['ranks']}", flush=True)
            pd.DataFrame(rows).to_csv(ROOT / "results" / "readout_paul15.csv", index=False)


if __name__ == "__main__":
    synthetic(sys.argv[1]) if sys.argv[1] in ("dev", "heldout") else paul15(sys.argv[2])
    print("EXIT 0")
