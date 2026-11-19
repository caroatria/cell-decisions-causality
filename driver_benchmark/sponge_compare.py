"""Compare sponge experiment runs (8 Nov 2026). No ground truth exists, so each run is judged per fate on:
  effect      strongest effect of any TF: max dP (P(fate) shift) and max dL (log-odds shift), mean over subsamples
  standout    number of TFs whose effect is > 5 robust SDs (MAD) above the median TF, mean over subsamples
  stable      top-5 TFs (by the run's own per-fate ranking) found in every subsample, and mean pairwise Jaccard
  direction   share of the top-5 TFs that are higher in the fate than in stem cells (logFC > 0)
Usage: python sponge_compare.py TAG [TAG ...] [--method OURS_v2_TF]
       e.g. python sponge_compare.py exp_base exp_fateregion exp_logodds exp_k5 exp_k10 exp_tree
Reads results/<TAG>_perfate_<method>_<decision>_s<seed>.csv (all decisions and seeds found for a tag).
Writes results/sponge_compare.csv."""
import argparse, glob, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(); p.add_argument("tags", nargs="+"); p.add_argument("--method", default="OURS_v2_TF")
a_ = p.parse_args()

rows = []
for tag in a_.tags:
    files = sorted(glob.glob(str(ROOT / "results" / f"{tag}_perfate_{a_.method}_*_s*.csv")))
    if not files:
        print(f"!! no files for tag {tag}"); continue
    by_dec = {}
    for f in files:
        m = re.match(rf"{re.escape(tag)}_perfate_{re.escape(a_.method)}_(.+)_s(\d+)\.csv", Path(f).name)
        by_dec.setdefault(m.group(1), []).append(pd.read_csv(f, index_col=0))
    for dec, runs in by_dec.items():
        fates = [c[3:] for c in runs[0].columns if c.startswith("dP_") and c != "dP_best"]
        on = runs[0]["ranked_on"].iloc[0] if "ranked_on" in runs[0] else "dP"      # measure the run ranked on
        for fate in fates:
            if fate == "branch":                     # tree1's combined fate is not a cell type
                continue
            eff_p = [r[f"dP_{fate}"].max() for r in runs]
            eff_l = [r[f"dL_{fate}"].max() for r in runs] if f"dL_{fate}" in runs[0] else [np.nan]
            rank_col = f"pct_{fate}"
            meas = f"{on}_{fate}"
            stand = []
            for r in runs:
                x = r[meas].dropna(); mad = np.median(np.abs(x - x.median())) * 1.4826 + 1e-12
                stand.append(((x - x.median()) / mad > 5).sum())
            tops = [set(r[rank_col].sort_values(ascending=False).index[:5]) for r in runs]
            jac = np.mean([len(x & y) / len(x | y) for i, x in enumerate(tops) for y in tops[i + 1:]]) if len(tops) > 1 else np.nan
            lfc = f"logFC_{fate}_vs_stem"
            up = np.mean([(r.loc[list(t), lfc] > 0).mean() for r, t in zip(runs, tops)]) if lfc in runs[0] else np.nan
            rows.append({"run": tag, "ranked_on": on, "decision": dec, "fate": fate, "subsamples": len(runs),
                         "max_dP": np.mean(eff_p), "max_dL": np.mean(eff_l), "n_standout": np.mean(stand),
                         "top5_in_all": len(set.intersection(*tops)), "top5_jaccard": jac, "top5_up_in_fate": up,
                         "top5_first_subsample": " ".join(runs[0][rank_col].sort_values(ascending=False).index[:5])})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "sponge_compare.csv", index=False)
pd.set_option("display.width", 200)
for fate, g in out.groupby("fate", sort=False):
    print(f"\n=== {fate}")
    print(g.drop(columns=["fate", "top5_first_subsample"]).round(3).to_string(index=False))
print("\nwritten: results/sponge_compare.csv")
