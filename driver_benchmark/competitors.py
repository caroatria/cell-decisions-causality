"""Published competitors CEFCON (Wang et al. 2023), CellOracle (Kamimoto et al. 2023) and CellRank 2 (Weiler et al. 2024) as benchmark scorers.

Each runs in its own Python 3.10 environment (setup_competitors.sh), so the scorer here
  1. exports exactly what every other method sees (same cells, genes, terminal labels, root cell) plus the shared
     derived inputs: DPT pseudotime from the root, out-of-sample fate probabilities and the decision region
     (baselines.fate_probabilities / decision_region), to cache/competitors/<tag>/input.h5ad;
  2. runs competitors/run_<method>.py in that environment;
  3. reads the per-gene scores back. Genes the method does not score get the lowest score.
Results are cached per tag: delete cache/competitors/<tag>/<method>_scores.csv to rerun.
Both methods need a prior network on mouse gene symbols, so they run on real data only, not on the synthetic benchmark.
"""
from __future__ import annotations
import json, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad

from baselines import DecisionContext, fate_probabilities, decision_region, _X
from probe_interventional import dpt

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools" if (ROOT / "tools").exists() else Path("/home/claude/tools")
CACHE = ROOT / "cache" / "competitors"
PY = {"cefcon": TOOLS / "envs" / "cefcon" / "bin" / "python", "celloracle": TOOLS / "envs" / "celloracle" / "bin" / "python",
      "cellrank": TOOLS / "envs" / "cellrank" / "bin" / "python"}


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    """Object-dtype strings: the competitors' older anndata cannot read nullable-string columns."""
    df = df.copy(); df.index = np.asarray(df.index.astype(str), dtype=object)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = np.asarray(df[c].astype(str), dtype=object)
    return df


def export(a: ad.AnnData, ctx: DecisionContext, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    f = out / "input.h5ad"
    if f.exists():
        return f
    sub = a[ctx.cells]
    X = _X(a, ctx.cells).astype(np.float32)
    obs = pd.DataFrame(index=sub.obs_names)
    term = ctx.terminal[ctx.cells]
    obs["fate"] = np.where(term.isna(), "progenitor", term.astype(str))
    root = int(np.argmin(sub.obs["sim_time"].values))
    pt, pcs = dpt(sub, root)
    obs["dpt"] = pt
    P = fate_probabilities(a, ctx)
    for j, fa in enumerate(ctx.fates):
        obs[f"P_{fa}"] = P[:, j]
    obs["region"] = decision_region(a, ctx).astype(int)
    b = ad.AnnData(X=X, obs=_plain(obs), var=_plain(pd.DataFrame(index=sub.var_names)))
    b.obsm["X_pca2"] = np.asarray(pcs[:, :2], dtype=np.float32)
    b.uns["fates_json"] = json.dumps(list(ctx.fates))
    try:
        ad.settings.allow_write_nullable_strings = False
    except Exception:
        pass
    b.write_h5ad(f)
    return f


def _run(method: str, a, ctx, tag: str, cfg: dict) -> pd.Series:
    d = CACHE / tag
    res = d / f"{method}_scores.csv"
    if not res.exists():
        export(a, ctx, d)
        cfg = {"cefcon_root": str(TOOLS / "CEFCON"), **cfg}
        t0 = time.time()
        with open(d / f"{method}.log", "w") as log:
            subprocess.run([str(PY[method]), str(ROOT / "competitors" / f"run_{method}.py"), str(d), json.dumps(cfg)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"  [{method} {tag}: {time.time() - t0:.0f}s]", flush=True)
    s = pd.read_csv(res, index_col=0)["score"]
    s.index = s.index.astype(str)
    if method == "cefcon":                      # CEFCON upper-cases gene symbols; map back to the data's spelling
        up = {g.upper(): g for g in a.var_names}
        s.index = [up.get(g, g) for g in s.index]
    s = s.groupby(level=0).max().reindex(a.var_names)
    return s.fillna(s.min() - 1 if s.notna().any() else 0.0)


def score_cefcon(a, ctx, tag: str, **cfg) -> pd.Series:
    return _run("cefcon", a, ctx, tag, cfg)


def score_celloracle(a, ctx, tag: str, **cfg) -> pd.Series:
    return _run("celloracle", a, ctx, tag, cfg)


def score_cellrank(a, ctx, tag: str, **cfg) -> pd.Series:
    """CellRank 2 lineage drivers (PseudotimeKernel on the shared DPT; no velocity, no prior network)."""
    return _run("cellrank", a, ctx, tag, cfg)
