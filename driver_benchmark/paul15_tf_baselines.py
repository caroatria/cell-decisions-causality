"""Fair comparison for the TF-candidate diagnostics: rank every baseline among transcription factors only."""
import sys, warnings
import pandas as pd
warnings.filterwarnings("ignore")
import paul15_eval as P
from paul15_diagnostics import tf_list
from pathlib import Path
ROOT = Path(__file__).resolve().parent
a = P.prepare(sys.argv[1]); tfs = tf_list(list(a.var_names)); rows = []
for decision in P.LINEAGES:
    sub, ctx, fates = P.context_for(a, decision)
    for name, fn in P.METHODS.items():
        if name.startswith("OURS"): continue
        s = fn(sub, ctx).reindex(tfs)          # same scores, ranked among TFs only
        r = {"decision": decision, "method": name, **P.evaluate(s, decision), "top10": " ".join(s.sort_values(ascending=False).index[:10])}
        rows.append(r)
        print(f"{decision} {name:32s} core@10={r['core_hits@10']}/{r['n_core']} core@20={r['core_hits@20']} all@20={r['all_hits@20']}/{r['n_all']} median core rank={r['core_median_rank']:.0f}  top10: {r['top10']}", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "paul15_tf_baselines.csv", index=False)
