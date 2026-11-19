"""Paul15 diagnostics for OURS (timing-constrained interventional counterfactuals).
A  exact: every gene is intervened on, all undecided cells           (is the candidate filter the problem?)
B  TF candidates: only transcription factors are intervened on       (biologically plausible drivers only)
C  TF candidates + TF regulators: forward model may only use TFs as inputs (plus each gene itself)
TF list: pySCENIC mm_mgi_tfs.txt (1,721 mouse TFs), external/tfs/. Sfpi1 is matched as an alias of Spi1.
Usage: python paul15_diagnostics.py path/to/paul15.h5ad [A,B,C]
"""
import sys, time, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
import paul15_eval as P
from probe_interventional import score_interventional_probe
from method_timing import timing_mask

ROOT = Path(__file__).resolve().parent
ALIASES = {"Spi1": "Sfpi1"}


def tf_list(var_names):
    tfs = [l.strip() for l in open(ROOT / "external" / "tfs" / "mm_mgi_tfs.txt") if l.strip()]
    tfs = {ALIASES.get(t, t) for t in tfs}
    return [g for g in var_names if g in tfs]


def run(path, variants):
    a = P.prepare(path)
    tfs = tf_list(list(a.var_names))
    lit = {g for d in P.LITERATURE.values() for v in d.values() for g in v}
    print(f"{a.n_vars} genes, {len(tfs)} of them TFs; literature genes that are TFs: "
          f"{sorted(set(tfs) & lit)}; not TFs: {sorted((lit & set(a.var_names)) - set(tfs))}", flush=True)
    mask = timing_mask(0.0, 0.2)
    cfg = {"A": ("exact (all genes, all cells)", dict(candidates=None, max_cells=None)),
           "B": ("TF candidates", dict(candidates=tfs, max_cells=None)),
           "C": ("TF candidates + TF regulators", dict(candidates=tfs, regulators=tfs, max_cells=None))}
    rows = []
    for decision in P.LINEAGES:
        sub, ctx, fates = P.context_for(a, decision)
        for v in variants:
            name, kw = cfg[v]
            t0 = time.time()
            s = score_interventional_probe(sub, ctx, allowed_fn=mask, n_jobs=2, **kw)
            r = {"decision": decision, "variant": v, "name": name, **P.evaluate(s, decision),
                 "top10": " ".join(s.sort_values(ascending=False).index[:10]), "seconds": round(time.time() - t0, 1)}
            rows.append(r)
            print(f"{decision} {v} {name:32s} core@20={r['core_hits@20']}/{r['n_core']} all@50={r['all_hits@50']}/{r['n_all']} "
                  f"median core rank={r['core_median_rank']:.0f} ({r['seconds']}s)\n   ranks: {r['ranks']}\n   top10: {r['top10']}", flush=True)
            pd.DataFrame(rows).to_csv(ROOT / "results" / f"paul15_diagnostics_{''.join(variants)}.csv", index=False)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else ["B", "C", "A"])
