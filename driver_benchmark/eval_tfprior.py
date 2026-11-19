"""TF-prior setting on the synthetic benchmark (17 Nov 2026).

Every method gets the same prior information: the list of transcription factors (= every gene that regulates at least
one other gene in the simulated network: switch TFs, timer mediators, the layer-1 markers that activate layer-2
markers, progenitor and repressilator genes). No specific edges are given.
  - Ours (v2): TFs are the only regulators in the forward model and the only genes intervened on.
  - CellOracle / CEFCON: prior network = every TF -> every other gene (in place of the mouse scATAC / NicheNet priors).
  - CellRank 2: uses no prior network; its scores are simply ranked among TFs like everyone else's.
Every method is evaluated among TFs only (as on Nestorowa): drivers vs the markers / mediators that are TFs.
Decisions: the held-out set of the main benchmark without HSC (BF and TF replicates 3-12, CBF 0-4) = 30 decisions.
Output: results/tfprior_synthetic.csv (appended per decision, resumable).
"""
from __future__ import annotations
import glob, re, time, warnings
from pathlib import Path
import anndata as ad, pandas as pd
warnings.filterwarnings("ignore")
from evaluate import contexts, metrics
from method_timing import score_timing_cf_v2
from competitors import score_cellrank, score_celloracle, score_cefcon
from networks import MODELS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "tfprior_synthetic.csv"
COLS = ["dataset", "model", "decision", "n_tfs", "n_drivers", "n_negatives", "method", "auprc_all", "auprc_vs_markers",
        "auprc_vs_fast_targets", "prec_at_k", "best_driver_rank", "chance_all", "chance_vs_markers",
        "chance_vs_fast_targets", "seconds", "error"]


def tf_list(model_name: str, genes) -> list[str]:
    """Genes that appear as an input in some other gene's rule (or their own plus another's)."""
    m = MODELS[model_name]()
    regs = set()
    for tgt, rule in m.rules.items():
        regs |= {t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rule) if t in m.rules and t != tgt}
    return [g for g in genes if g in regs]


def main():
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT); done = set(zip(prev.dataset, prev.decision, prev.method))
    files = []
    for f in sorted(glob.glob(str(ROOT / "data" / "*.h5ad"))):
        model, rep = Path(f).stem.split("_rep"); rep = int(rep)
        if (model in ("BF", "TF") and rep >= 3) or model == "CBF":
            files.append((f, model))
    for f, model in files:
        name = Path(f).stem
        a = ad.read_h5ad(f)
        tfs = tf_list(model, list(a.var_names))
        methods = {
            "OURS_v2_TF": lambda a, ctx: score_timing_cf_v2(a, ctx, candidates=tfs, regulators=tfs, n_jobs=1),
            "CellRank": lambda a, ctx: score_cellrank(a, ctx, f"tfp_{name}_{ctx.decision}"),
            "CellOracle_TF": lambda a, ctx: score_celloracle(a, ctx, f"tfp_{name}_{ctx.decision}", prior_tfs=tfs),
            "CEFCON_TF": lambda a, ctx: score_cefcon(a, ctx, f"tfp_{name}_{ctx.decision}", prior_tfs=tfs),
        }
        for ctx in contexts(a, seed=0):
            rows = []
            var = a.var.loc[tfs]
            drv = ((var.role == "driver") & (var.decision == ctx.decision)).sum()
            neg = (var.role.isin(["marker", "mediator"]) & (var.decision == ctx.decision)).sum()
            for m, fn in methods.items():
                if (name, ctx.decision, m) in done:
                    continue
                t0 = time.time()
                try:
                    r = metrics(fn(a, ctx).reindex(tfs), var, ctx.decision)
                except Exception as e:
                    r = {"error": repr(e)[:200]}
                rows.append({"dataset": name, "model": model, "decision": ctx.decision, "n_tfs": len(tfs),
                             "n_drivers": int(drv), "n_negatives": int(neg), "method": m, **r,
                             "seconds": round(time.time() - t0, 1)})
                print(f"{name} {ctx.decision} {m}: {r.get('auprc_vs_markers', r.get('error'))}", flush=True)
            if rows:
                pd.DataFrame(rows).reindex(columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


if __name__ == "__main__":
    main()
