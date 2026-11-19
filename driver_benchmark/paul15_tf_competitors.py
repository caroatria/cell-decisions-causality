"""Paul15, everything ranked among transcription factors (as for Nestorowa): our v2 variants, the published
competitors (cached by paul15_eval.py) and the baselines. Output: results/paul15_tf_competitors.csv"""
import sys, warnings, time
import pandas as pd
warnings.filterwarnings("ignore")
import paul15_eval as P
from paul15_diagnostics import tf_list
from method_timing import score_timing_cf_v2
from probe_interventional import candidate_genes
from pathlib import Path
ROOT = Path(__file__).resolve().parent
a = P.prepare(sys.argv[1]); tfs = tf_list(list(a.var_names)); rows = []
M = {
    "OURS_v2_TF": lambda s, c: score_timing_cf_v2(s, c, candidates=[g for g in tfs if g in s.var_names],
                                                  regulators=[g for g in tfs if g in s.var_names], max_cells=300, n_jobs=2),
    "OURS_v2_c300TF": lambda s, c: score_timing_cf_v2(s, c, candidates=sorted(set(candidate_genes(s, c, 300)) | set(tfs)),
                                                      max_cells=300, n_jobs=2),
    **{k: P.METHODS[k] for k in ["CEFCON", "CellOracle", "CellRank", "fate_corr", "absorption_corr_decision_region", "de_terminal",
                                 "rf_shap_fixed", "obs_counterfactual"]},
}
for decision in P.LINEAGES:
    sub, ctx, fates = P.context_for(a, decision)
    for name, fn in M.items():
        t0 = time.time(); s = fn(sub, ctx).reindex(tfs)
        r = {"decision": decision, "method": name, **P.evaluate(s, decision),
             "top10": " ".join(s.sort_values(ascending=False).index[:10]), "seconds": time.time() - t0}
        rows.append(r)
        print(f"{decision} {name:32s} core@20={r['core_hits@20']}/{r['n_core']} all@20={r['all_hits@20']}/{r['n_all']} "
              f"median core rank={r['core_median_rank']:.0f}  top10: {r['top10']}", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "paul15_tf_competitors.csv", index=False)
