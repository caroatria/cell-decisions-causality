"""Paul15 diagnostic: mean fate probability of undecided cells before (K=0) and after K=1,3 forward steps, per readout."""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import probe_interventional as PI, paul15_eval as P15
from paul15_diagnostics import tf_list
from method_timing import timing_mask
# instrument: capture P before rollout and after K=0..5
orig = PI.score_interventional_probe
a = P15.prepare(sys.argv[1]); tfs = tf_list(list(a.var_names))
for dec in P15.LINEAGES:
    sub, ctx, fates = P15.context_for(a, dec)
    cl = sub.obs["paul15_clusters"].astype(str).values
    for ro in ["logreg", "pca_logreg", "soft_logreg"]:
        out = []
        for K in (0, 1, 3):
            _, sh = PI.score_interventional_probe(sub, ctx, allowed_fn=timing_mask(0.0, 0.2), K=K, n_jobs=2,
                    candidates=tfs[:5], regulators=tfs, return_shifts=True, readout=ro, score_mode="perfate")
            out.append(f"K={K}:{ {k: round(v,2) for k,v in sh.attrs['baseline_fate_prob'].items()} }")
        print(dec, ro, *out, flush=True)
    term = ctx.terminal.values
    print(dec, "terminal counts", {f: int((term == f).sum()) for f in fates}, "non-terminal", int(np.sum(~np.isin(term, list(fates)))), flush=True)
