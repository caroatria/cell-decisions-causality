"""Confidence intervals for the LARRY early-vs-late AUPRC (run after larry_eval.py with TAG=s0,s1,s2).
Two sources of uncertainty:
  1. reference genes: resample the early and the late gene sets with replacement (stratified), 2000 times;
     paired, so the difference between two methods uses the same resampled genes;
  2. cells: three independent 8,000-cell subsamples (seeds 0-2), each a full rerun of every method.
Writes results/larry_bootstrap.csv (per seed x method x gene space) and results/larry_bootstrap_paired.csv."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
ROOT = Path(__file__).resolve().parent
B, SEEDS = 2000, ["s0", "s1", "s2"]


def auprc(s, early, late, rng):
    y = np.r_[np.ones(len(early)), np.zeros(len(late))]
    v = s.reindex(list(early) + list(late)).values
    return average_precision_score(y, v + rng.random(len(v)) * 1e-9)


rows, paired = [], []
for tag in SEEDS:
    f = ROOT / "results" / f"larry_scores_{tag}.csv"
    if not f.exists():
        continue
    S = pd.read_csv(f, index_col=0)
    for extra in ("d", "c", "r"):  # d: direct-effect variants (3 Nov); c: CEFCON / CellOracle (3 Nov); r: CellRank (4 Nov)
        g = ROOT / "results" / f"larry_scores_{tag.replace('s', extra)}.csv"
        if g.exists():
            S = S.join(pd.read_csv(g, index_col=0).drop(columns=list(S.columns), errors="ignore"))
    R = json.load(open(ROOT / "results" / f"larry_refsets_{tag}.json"))
    for space, ref in R.items():
        pool, early, late = ref["pool"], np.array(ref["early"]), np.array(ref["late"])
        meths = [m for m in S.columns if not (m.endswith("_TF") and space == "all genes")]
        filled = {m: S[m].reindex(pool).fillna(S[m].min() - 1) for m in meths}
        rng = np.random.default_rng(0)
        draws = []
        for b in range(B):
            e = rng.choice(early, len(early)); l = rng.choice(late, len(late))
            r2 = np.random.default_rng(b)
            draws.append({m: auprc(filled[m], e, l, r2) for m in meths})
        D = pd.DataFrame(draws)
        r0 = np.random.default_rng(0)
        for m in meths:
            lo, hi = np.percentile(D[m], [2.5, 97.5])
            rows.append({"seed": tag, "ranked_among": space, "method": m, "auprc": auprc(filled[m], early, late, r0),
                         "ci_low": lo, "ci_high": hi, "chance": len(early) / (len(early) + len(late))})
        for ours in [m for m in meths if m.startswith("OURS")]:
            for other in [m for m in meths if not m.startswith("OURS")]:
                d = D[ours] - D[other]
                paired.append({"seed": tag, "ranked_among": space, "ours": ours, "other": other, "mean_diff": d.mean(),
                               "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                               "p_ours_not_better": (d <= 0).mean()})
out = pd.DataFrame(rows); out.to_csv(ROOT / "results" / "larry_bootstrap.csv", index=False)
pd.DataFrame(paired).to_csv(ROOT / "results" / "larry_bootstrap_paired.csv", index=False)
pd.set_option("display.width", 200)
for space in ("TFs", "all genes"):
    t = out[out.ranked_among == space].pivot_table(index="method", columns="seed", values="auprc")
    sc_ = [c for c in t.columns if c.startswith("s")]; t["mean"] = t[sc_].mean(axis=1); t["sd"] = t[sc_].std(axis=1)
    print(f"\n== {space}: AUPRC per cell subsample (gene-bootstrap 95% CI in larry_bootstrap.csv)"); print(t.sort_values("mean", ascending=False).round(3))
p = pd.DataFrame(paired)
print("\n== paired gene-bootstrap, ours minus other, averaged over seeds (CI = mean of per-seed CIs)")
print(p.groupby(["ranked_among", "ours", "other"])[["mean_diff", "ci_low", "ci_high", "p_ours_not_better"]].mean().round(3).to_string())
print("EXIT 0")
