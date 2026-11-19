"""Timing-constrained interventional counterfactuals (framing B, version 1).

Same pipeline as probe_interventional.py, but the forward model may only use gene x to predict
gene y if x changes EARLIER along pseudotime than y (label-free onset), within a tolerance.
Rationale: downstream markers change after their drivers, so marker -> driver edges are blocked.
Known limitation (onset check, 22 Oct): low-threshold 'fast' targets reach their own relative
onset before the driver, so they are not blocked.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from probe_interventional import score_interventional_probe


def onset_times(X, pt, n_win=30, frac=0.3):
    """Label-free onset per gene: pseudotime at which the windowed variance first rises above
    `frac` of its maximum increase relative to the earliest window."""
    o = np.argsort(pt); Xs, ps = X[o], pt[o]
    w = max(len(X) // n_win, 10)
    starts = np.arange(0, len(X) - w, max(w // 2, 1))
    S = np.array([Xs[s:s + w].var(0) for s in starts]); S = S - S[0]
    tpos = np.array([ps[s:s + w].mean() for s in starts])
    on = np.full(X.shape[1], np.inf)
    for j in range(X.shape[1]):
        mx = S[:, j].max()
        if mx > 1e-9:
            on[j] = tpos[np.where(S[:, j] >= frac * mx)[0][0]]
    return on, (tpos[1] - tpos[0]) if len(tpos) > 1 else 0.0


def timing_mask(tol_windows=1.0, frac=0.3):
    def fn(X, pt):
        on, step = onset_times(X, pt, frac=frac)
        tol = tol_windows * step
        # allowed[y, x]: x may regulate y if x's onset is not later than y's (+ tolerance)
        return on[None, :] <= on[:, None] + tol
    return fn


# frac=0.2, tol=0 chosen by a dev-only sweep (BF+TF reps 0-2; results/dev_sweep_timing.csv).
# Sweep values: frac 0.05/0.1/0.2/0.3 -> BF 0.15/0.28/0.67/0.46, TF 0.49/0.71/0.86/0.94 (tol=0).
def score_timing_cf(adata, ctx, tol_windows=0.0, frac=0.2):
    return score_interventional_probe(adata, ctx, allowed_fn=timing_mask(tol_windows, frac))


def score_timing_cf_v2(adata, ctx, K=3, score_mode="perfate", **kw):
    """Version 2 (dev-chosen): 3 fast-forward steps, per-fate ranking. See README 'Version 2'."""
    return score_interventional_probe(adata, ctx, allowed_fn=timing_mask(0.0, 0.2), K=K, score_mode=score_mode, **kw)


def score_onset_only(adata, ctx):
    """Ablation: no forward model at all. Among fate-associated genes (|corr| with fate probability),
    prefer early onset. score = fate_corr * (1 - onset percentile)."""
    from baselines import score_fate_corr, _X
    from probe_interventional import dpt
    fc = score_fate_corr(adata, ctx)
    X = _X(adata, ctx.cells)
    pt, _ = dpt(adata[ctx.cells], int(np.argmin(adata.obs.loc[ctx.cells, "sim_time"].values)))
    on, _ = onset_times(X, pt)
    pct = pd.Series(on, index=adata.var_names).rank(pct=True)
    return fc * (1 - pct + 1e-3)


def score_timing_cf_fast(adata, ctx, candidates=300, max_cells=300, n_jobs=-1, **kw):
    """Same method with the speed-ups on: interventions only on the top-`candidates` genes by fate
    correlation (or a given gene list, e.g. TFs), at most `max_cells` undecided cells, parallel fits.
    Use this for real data (thousands of genes)."""
    return score_interventional_probe(adata, ctx, allowed_fn=timing_mask(kw.pop("tol_windows", 0.0), kw.pop("frac", 0.2)),
                                      candidates=candidates, max_cells=max_cells, n_jobs=n_jobs, **kw)


def score_timing_cf_stress(adata, ctx):
    """Stress test of the shortcuts on the small benchmark: only 30 candidate genes, 100 cells."""
    return score_timing_cf_fast(adata, ctx, candidates=30, max_cells=100, n_jobs=1)
