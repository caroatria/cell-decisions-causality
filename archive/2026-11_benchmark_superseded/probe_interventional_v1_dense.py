"""Quick feasibility probe for framing B (NOT the final method).

Question: in a forward model learned only from snapshot data + pseudotime, does clamping a
DRIVER in uncommitted cells shift the predicted future fate more than clamping a MARKER?

1. Pseudotime: diffusion pseudotime (DPT), root = earliest annotated cell.
2. Pairs: each cell -> mean of its k nearest PCA neighbours among cells slightly later in DPT.
3. Forward model: sparse linear (Lasso per gene) x(t+1) = W x(t) + b, fitted on non-terminal cells.
4. Fate readout: logistic regression on annotated terminal cells (same as baselines).
5. Intervention: for decision-region cells, clamp gene g at its typical level in fate f
   (90th pct among fate-f terminals), roll out K steps keeping it clamped, measure the change in
   P(f) vs. the unclamped rollout. Score(g) = max_f mean shift.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Lasso
from sklearn.neighbors import NearestNeighbors

from baselines import DecisionContext, decision_region, _X


def dpt(adata, root_idx):
    b = adata.copy()
    sc.pp.pca(b, n_comps=min(20, b.n_vars - 1))
    sc.pp.neighbors(b, n_neighbors=15)
    sc.tl.diffmap(b)
    b.uns["iroot"] = int(root_idx)
    sc.tl.dpt(b)
    return b.obs["dpt_pseudotime"].values, b.obsm["X_pca"]


def score_interventional_probe(adata, ctx: DecisionContext, k_nn=10, lag=0.05, alpha=0.01,
                               K=15, return_model=False, fit_on="nonterminal", regulators=None, allowed_fn=None):
    """fit_on: 'nonterminal' (all non-terminal cells) or 'region' (decision-region cells only).
    regulators: optional list of gene names allowed as inputs to the forward model (TF prior).
    allowed_fn: optional f(X, pt) -> bool matrix (target x regulator) restricting forward-model edges."""
    X = _X(adata, ctx.cells)
    obs = adata.obs.loc[ctx.cells]
    root = int(np.argmin(obs["sim_time"].values))           # = annotated earliest/stem cell
    pt, pcs = dpt(adata[ctx.cells], root)
    term = ctx.terminal[ctx.cells].values
    nonterm = pd.isna(term)

    # ---- pseudotime-coupled pairs
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    nn = NearestNeighbors(n_neighbors=min(80, len(X))).fit(pcs)
    _, idx = nn.kneighbors(pcs)
    src, tgt = [], []
    for i in range(len(X)):
        later = [j for j in idx[i] if pt[i] < pt[j] <= pt[i] + lag][:k_nn]
        if len(later) >= 3:
            src.append(i); tgt.append(Z[later].mean(0))
    src = np.array(src); tgt = np.array(tgt)
    keep = nonterm[src] if fit_on == "nonterminal" else decision_region(adata, ctx, quantile=0.3)[src]
    A_in, A_out = Z[src[keep]], tgt[keep]

    # ---- sparse linear forward model (residual form: dz = W z + b)
    W = np.zeros((Z.shape[1], Z.shape[1])); b = np.zeros(Z.shape[1])
    cols = np.arange(Z.shape[1]) if regulators is None else np.array([list(adata.var_names).index(r) for r in regulators])
    allowed = np.ones((Z.shape[1], Z.shape[1]), bool) if allowed_fn is None else allowed_fn(X, pt)
    np.fill_diagonal(allowed, True)
    for g in range(Z.shape[1]):
        cg = np.array([c for c in cols if allowed[g, c]])
        m = Lasso(alpha=alpha, max_iter=5000).fit(A_in[:, cg], A_out[:, g] - A_in[:, g])
        W[g, cg], b[g] = m.coef_, m.intercept_

    def rollout(z, clamp_idx=None, clamp_val=None):
        z = z.copy()
        for _ in range(K):
            if clamp_idx is not None:
                z[:, clamp_idx] = clamp_val
            z = z + z @ W.T + b
            z = np.clip(z, Z.min(0) - 1, Z.max(0) + 1)       # keep within plausible range
        if clamp_idx is not None:
            z[:, clamp_idx] = clamp_val
        return z

    # ---- fate readout trained on terminal cells
    from sklearn.linear_model import LogisticRegression
    tm = ~nonterm & np.isin(term, ctx.fates)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(Z[tm], term[tm])
    cls = list(clf.classes_)

    R = decision_region(adata, ctx)
    z0 = Z[R]
    base = clf.predict_proba(rollout(z0))
    scores = np.zeros(Z.shape[1])
    for g in range(Z.shape[1]):
        best = 0.0
        for f in ctx.fates:
            val = np.quantile(Z[tm & (term == f), g], 0.9)
            p = clf.predict_proba(rollout(z0, g, val))
            best = max(best, float((p[:, cls.index(f)] - base[:, cls.index(f)]).mean()))
        scores[g] = best
    s = pd.Series(scores, index=adata.var_names)
    return (s, W) if return_model else s


if __name__ == "__main__":
    import sys, anndata as ad
    from evaluate import contexts, metrics
    from baselines import SCORERS
    for f in sys.argv[1:]:
        a = ad.read_h5ad(f)
        for ctx in contexts(a, seed=0):
            s, W = score_interventional_probe(a, ctx, return_model=True)
            print(f, ctx.decision, {k: round(v, 3) for k, v in metrics(s, a.var, ctx.decision).items()})
            print("  top10:", list(s.sort_values(ascending=False).index[:10]))
            g = list(a.var_names)
            for tgt in ctx.fates:
                row = pd.Series(W[g.index(tgt)], index=g)
                print(f"  learned inputs to {tgt}:", row[row.abs() > 0.02].sort_values(key=abs, ascending=False).round(3).head(8).to_dict())
