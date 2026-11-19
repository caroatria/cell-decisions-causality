"""Baseline driver-gene scorers.

All scorers share one signature:
    score(adata, ctx) -> pd.Series (index = genes, higher = more driver-like)

`ctx` is a DecisionContext holding only what a real analysis would have:
  * terminal labels (annotated terminal cell types; NaN elsewhere)
  * fate set for this decision
Anything using ground truth (true future fate) is marked ORACLE and is a reference, not a method.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import entropy, rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class DecisionContext:
    decision: str
    fates: list[str]
    terminal: pd.Series          # fate label for annotated terminal cells, NaN otherwise
    cells: np.ndarray            # boolean mask: cells belonging to this decision's lineage
    seed: int = 0

    # filled lazily
    _proba: np.ndarray | None = None
    _absorb: np.ndarray | None = None


def _X(adata, mask=None):
    X = adata.X if mask is None else adata.X[mask]
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float64)


def _auc_effect(x, groups, fates):
    """max over fates of |AUROC(one-vs-rest) - 0.5| * 2  (Mann-Whitney via ranks)."""
    r = rankdata(x)
    best = 0.0
    for f in fates:
        y = groups == f
        n1, n0 = y.sum(), (~y).sum()
        if n1 == 0 or n0 == 0:
            continue
        u = r[y].sum() - n1 * (n1 + 1) / 2
        best = max(best, abs(u / (n1 * n0) - 0.5) * 2)
    return best


# ---------------------------------------------------------------- fate probabilities
def fate_probabilities(adata, ctx: DecisionContext) -> np.ndarray:
    """Michal's design: classifier trained on annotated terminal cells only, applied to all cells
    of the lineage (out of sample for non-terminal cells). Returns (n_lineage_cells, n_fates)."""
    if ctx._proba is not None:
        return ctx._proba
    X = _X(adata, ctx.cells)
    term = ctx.terminal[ctx.cells].values
    tmask = pd.notna(term) & np.isin(term, ctx.fates)
    sc = StandardScaler().fit(X[tmask])
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tmask]), term[tmask])
    P = clf.predict_proba(sc.transform(X))
    order = [list(clf.classes_).index(f) for f in ctx.fates]
    ctx._proba = P[:, order]
    return ctx._proba


def decision_region(adata, ctx: DecisionContext, quantile: float = 0.5) -> np.ndarray:
    """Cells (within lineage, not annotated terminal) whose fate entropy is above the quantile."""
    P = fate_probabilities(adata, ctx)
    H = entropy(P.T)
    nonterm = pd.isna(ctx.terminal[ctx.cells].values)
    thr = np.quantile(H[nonterm], quantile)
    return nonterm & (H >= thr)


# ---------------------------------------------------------------- scorers
def score_random(adata, ctx):
    rng = np.random.default_rng(ctx.seed)
    return pd.Series(rng.random(adata.n_vars), index=adata.var_names)


def score_de_terminal(adata, ctx):
    """Differential expression between annotated terminal fates (marker-style)."""
    X = _X(adata, ctx.cells)
    term = ctx.terminal[ctx.cells].values
    m = pd.notna(term) & np.isin(term, ctx.fates)
    return pd.Series([_auc_effect(X[m, j], term[m], ctx.fates) for j in range(X.shape[1])],
                     index=adata.var_names)


def score_fate_corr(adata, ctx):
    """CellRank-`compute_lineage_drivers`-style: |corr(gene, fate probability)| over all lineage cells."""
    X = _X(adata, ctx.cells)
    P = fate_probabilities(adata, ctx)
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Pc = (P - P.mean(0)) / (P.std(0) + 1e-9)
    C = np.abs(Xc.T @ Pc) / X.shape[0]
    return pd.Series(C.max(1), index=adata.var_names)


def score_fate_corr_decision_region(adata, ctx):
    """Same, restricted to uncommitted (high-entropy) cells: an observational precedence score."""
    X = _X(adata, ctx.cells)
    P = fate_probabilities(adata, ctx)
    R = decision_region(adata, ctx)
    X, P = X[R], P[R]
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Pc = (P - P.mean(0)) / (P.std(0) + 1e-9)
    C = np.abs(Xc.T @ Pc) / X.shape[0]
    return pd.Series(C.max(1), index=adata.var_names)


def score_rf_shap(adata, ctx, n_estimators=200, max_depth=8):
    """BUNGEE.py, corrected: RF on terminal labels, SHAP on high-entropy cells,
    importance per fate = mean |SHAP[:, :, fate]| (the original indexed cells, not classes)."""
    import shap
    X = _X(adata, ctx.cells)
    term = ctx.terminal[ctx.cells].values
    m = pd.notna(term) & np.isin(term, ctx.fates)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                class_weight="balanced", random_state=ctx.seed, n_jobs=-1)
    rf.fit(X[m], term[m])
    R = decision_region(adata, ctx)
    sv = shap.TreeExplainer(rf).shap_values(X[R])
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv   # -> (cells, genes, classes)
    imp = np.abs(sv).mean(axis=0).max(axis=1)                   # max over fates
    return pd.Series(imp, index=adata.var_names)


def score_obs_counterfactual(adata, ctx, lam=0.05, steps=300, lr=0.1, tau=0.8):
    """Observational counterfactual (Wachter-style, L1-sparse, box-constrained) on the terminal
    classifier: for each decision-region cell, find a sparse change that makes the classifier
    predict another fate with prob >= tau. Score = mean |delta| per gene.
    NO propagation: this is the baseline that framing B argues will pick markers."""
    X = _X(adata, ctx.cells)
    term = ctx.terminal[ctx.cells].values
    m = pd.notna(term) & np.isin(term, ctx.fates)
    sc = StandardScaler().fit(X[m])
    Z = sc.transform(X)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(Z[m], term[m])
    W, b = clf.coef_, clf.intercept_
    if W.shape[0] == 1:                                    # binary -> two-row softmax form
        W, b = np.vstack([-W[0] / 2, W[0] / 2]), np.array([-b[0] / 2, b[0] / 2])
    lo, hi = Z.min(0), Z.max(0)
    R = decision_region(adata, ctx)
    Z0 = Z[R]
    cur = (Z0 @ W.T + b).argmax(1)
    acc = np.zeros(Z.shape[1]); n = 0
    for k in range(W.shape[0]):
        sel = cur != k
        if not sel.any():
            continue
        z0 = Z0[sel]; d = np.zeros_like(z0)
        for _ in range(steps):
            logits = (z0 + d) @ W.T + b
            p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            active = (p[:, k] < tau)[:, None]
            g = (p @ W - W[k]) * active                    # grad of CE(target k)
            d = d - lr * g
            d = np.sign(d) * np.maximum(np.abs(d) - lr * lam, 0)   # L1 prox
            d = np.clip(z0 + d, lo, hi) - z0               # stay within observed range
        acc += np.abs(d).sum(0); n += len(d)
    return pd.Series(acc / max(n, 1), index=adata.var_names)


def absorption_probabilities(adata, ctx: DecisionContext, k=15, n_pcs=20, back_weight=0.05):
    """CellRank-PseudotimeKernel-style fate probabilities without a gene-level classifier:
    kNN graph in PCA space, edges down-weighted when they go backwards in diffusion pseudotime,
    annotated terminal cells absorbing. Returns (n_lineage_cells, n_fates)."""
    import scanpy as sc
    from scipy.sparse.linalg import spsolve
    from scipy import sparse
    if getattr(ctx, "_absorb", None) is not None:
        return ctx._absorb
    b = adata[ctx.cells].copy()
    sc.pp.pca(b, n_comps=min(n_pcs, b.n_vars - 1))
    sc.pp.neighbors(b, n_neighbors=k)
    sc.tl.diffmap(b)
    b.uns["iroot"] = int(np.argmin(b.obs["sim_time"].values))   # annotated earliest cell
    sc.tl.dpt(b)
    pt = b.obs["dpt_pseudotime"].values
    C = b.obsp["connectivities"].tocoo()
    w = C.data * np.where(pt[C.col] >= pt[C.row], 1.0, back_weight)
    T = sparse.csr_matrix((w, (C.row, C.col)), shape=C.shape)
    T = sparse.diags(1 / np.maximum(np.asarray(T.sum(1)).ravel(), 1e-12)) @ T
    term = ctx.terminal[ctx.cells].values
    absorbing = pd.notna(term) & np.isin(term, ctx.fates)
    tr = np.where(~absorbing)[0]
    Q = T[tr][:, tr]
    Rm = np.column_stack([np.asarray(T[tr][:, np.where(term == f)[0]].sum(1)).ravel() for f in ctx.fates])
    A = (sparse.identity(len(tr)) - Q).tocsc()
    Bt = np.column_stack([spsolve(A, Rm[:, i]) for i in range(len(ctx.fates))])
    P = np.zeros((len(term), len(ctx.fates)))
    P[tr] = Bt
    for i, f in enumerate(ctx.fates):
        P[term == f, i] = 1.0
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    ctx._absorb = P
    return P


def score_absorption_corr_decision_region(adata, ctx, quantile=0.5):
    """|corr(gene, absorption probability)| among uncommitted (high-entropy) non-terminal cells."""
    X = _X(adata, ctx.cells)
    P = absorption_probabilities(adata, ctx)
    H = entropy(P.T + 1e-12)
    nonterm = pd.isna(ctx.terminal[ctx.cells].values)
    R = nonterm & (H >= np.quantile(H[nonterm], quantile))
    X, P = X[R], P[R]
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Pc = (P - P.mean(0)) / (P.std(0) + 1e-9)
    return pd.Series((np.abs(Xc.T @ Pc) / X.shape[0]).max(1), index=adata.var_names)


def score_oracle_early(adata, ctx):
    """ORACLE (uses true future fate, like lineage tracing): fate association among cells that
    have NOT yet committed. Upper reference for observational precedence; not a method."""
    X = _X(adata, ctx.cells)
    o = adata.obs.loc[ctx.cells]
    fate = o[f"fate_{ctx.decision}"].astype(object).values
    m = (~o[f"committed_{ctx.decision}"].values) & pd.notna(fate) & (o["sim_time"].values > 0.3)
    return pd.Series([_auc_effect(X[m, j], fate[m], ctx.fates) for j in range(X.shape[1])],
                     index=adata.var_names)


SCORERS = {
    "random": score_random,
    "de_terminal": score_de_terminal,
    "fate_corr": score_fate_corr,
    "fate_corr_decision_region": score_fate_corr_decision_region,
    "absorption_corr_decision_region": score_absorption_corr_decision_region,
    "rf_shap_fixed": score_rf_shap,
    "obs_counterfactual": score_obs_counterfactual,
    "ORACLE_early_true_fate": score_oracle_early,
}
