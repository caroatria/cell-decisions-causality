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


def _fit_one(A_in, dA, cg, alpha):
    m = Lasso(alpha=alpha, max_iter=5000).fit(A_in[:, cg], dA)
    return m.coef_, m.intercept_


def candidate_genes(adata, ctx, n):
    """Top-n genes by |corr(expression, fate probability)| over the lineage (cheap pre-filter for
    which genes get an intervention; every gene stays in the forward model)."""
    from baselines import score_fate_corr
    return score_fate_corr(adata, ctx).sort_values(ascending=False).index[:n].tolist()


def score_interventional_probe(adata, ctx: DecisionContext, k_nn=10, lag=0.05, alpha=0.01,
                               K=15, return_model=False, fit_on="nonterminal", regulators=None, allowed_fn=None,
                               candidates=None, max_cells=None, batch=32, n_jobs=1, seed=0,
                               score_mode="prob", return_shifts=False, readout="logreg",
                               pseudotime="dpt", region_q=0.5):
    """fit_on: 'nonterminal' (all non-terminal cells) or 'region' (decision-region cells only).
    regulators: optional list of gene names allowed as inputs to the forward model (TF prior).
    allowed_fn: optional f(X, pt) -> bool matrix (target x regulator) restricting forward-model edges.
    Speed options (defaults = exact original behaviour):
      candidates: None = intervene on every gene; int n = only the top-n genes by fate correlation;
                  list of names = only those genes (e.g. TFs). Non-candidates get score 0.
      max_cells:  None = all decision-region cells; int = random subsample of that many.
      batch:      genes intervened on together in one sparse matrix product (no effect on results).
      n_jobs:     parallel Lasso fits (no effect on results).
    Scoring (how the per-fate shifts become one gene score):
      score_mode="prob"     max over fates of the mean increase in P(fate)   (original; default)
      score_mode="logodds"  max over fates of the mean increase in log-odds of the fate
                            (no ceiling effect when undecided cells already lean towards a fate)
      score_mode="perfate"  rank genes within each fate by their P(fate) increase, score = best
                            per-fate rank percentile (each fate contributes its top genes equally)
      return_shifts=True    also return a DataFrame of per-fate shifts (prob_<fate>, logodds_<fate>)
                            and the mean baseline fate probabilities of the undecided cells."""
    from scipy import sparse
    from joblib import Parallel, delayed
    X = _X(adata, ctx.cells)
    obs = adata.obs.loc[ctx.cells]
    root = int(np.argmin(obs["sim_time"].values))           # = annotated earliest/stem cell
    pt, pcs = dpt(adata[ctx.cells], root)
    # ablation: pseudotime source (default "dpt" = unchanged behaviour)
    if pseudotime == "true":                                 # simulator time (upper bound, synthetic only)
        pt = obs["sim_time"].values.astype(float); pt = (pt - pt.min()) / (pt.max() - pt.min())
    elif pseudotime == "pca_dist":                           # naive: Euclidean distance from root in PCA space
        d = np.linalg.norm(pcs - pcs[root], axis=1); pt = d / d.max()
    elif pseudotime == "dpt_root":                           # DPT from a perturbed root: random one of the root's 20 nearest cells
        nnr = NearestNeighbors(n_neighbors=min(21, len(pcs))).fit(pcs).kneighbors(pcs[[root]], return_distance=False)[0][1:]
        pt, _ = dpt(adata[ctx.cells], int(np.random.default_rng(seed).choice(nnr)))
    elif pseudotime != "dpt":
        raise ValueError(pseudotime)
    term = ctx.terminal[ctx.cells].values
    nonterm = pd.isna(term)
    G = X.shape[1]

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

    # ---- sparse linear forward model (residual form: dz = W z + b), one Lasso per gene
    cols = np.arange(G) if regulators is None else np.array([list(adata.var_names).index(r) for r in regulators])
    allowed = np.ones((G, G), bool) if allowed_fn is None else allowed_fn(X, pt)
    np.fill_diagonal(allowed, True)
    colset = set(cols.tolist())
    # each gene may always use itself (self-decay / self-activation), even if it is not in `regulators`
    cgs = [np.array(sorted({c for c in colset if allowed[g, c]} | {g})) for g in range(G)]
    fits = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(A_in, A_out[:, g] - A_in[:, g], cgs[g], alpha) for g in range(G))
    W = np.zeros((G, G)); b = np.zeros(G)
    for g, (coef, icpt) in enumerate(fits):
        W[g, cgs[g]], b[g] = coef, icpt
    # sparse arithmetic only pays off if W is really sparse; measured density is often ~10-15%,
    # where dense BLAS is faster, so choose automatically (identical results either way)
    use_sparse = (W != 0).mean() < 0.05
    Ws = sparse.csr_matrix(W) if use_sparse else None
    WT = W.T.copy()
    lo, hi = Z.min(0) - 1, Z.max(0) + 1

    def step(z2d):                     # z2d: (n, G) -> one forward step, = z + z @ W.T + b
        dz = (Ws @ z2d.T).T if use_sparse else z2d @ WT
        return np.clip(z2d + dz + b, lo, hi)

    def rollout(z):
        for _ in range(K):
            z = step(z)
        return z

    def rollout_clamped(z0, genes, vals):
        """Batch of interventions: copy z0 once per gene, clamp gene j to vals[j] in its own copy."""
        nb, R = len(genes), z0.shape[0]
        z = np.repeat(z0[None], nb, axis=0)                 # (nb, R, G)
        ar = np.arange(nb)
        for _ in range(K):
            z[ar, :, genes] = vals[:, None]
            z = step(z.reshape(nb * R, G)).reshape(nb, R, G)
        z[ar, :, genes] = vals[:, None]
        return z.reshape(nb * R, G)

    # ---- fate readout f(z) -> P(fate), columns in ctx.fates order
    #   logreg       logistic regression on all genes, terminal cells (original)
    #   pca_logreg   same on the top 20 PCs (fewer features -> less over-confident)
    #   soft_logreg  logistic regression on all genes, all lineage cells, soft labels = graph absorption probs
    #   knn_absorb   mean graph absorption probability of the 15 nearest observed cells (PCA space)
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    tm = ~nonterm & np.isin(term, ctx.fates)
    F = len(ctx.fates)
    if readout in ("logreg", "pca_logreg"):
        proj = (lambda z: z) if readout == "logreg" else PCA(min(20, G - 1), random_state=0).fit(Z).transform
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(proj(Z[tm]), term[tm])
        order = [list(clf.classes_).index(f) for f in ctx.fates]
        f_read = lambda z: clf.predict_proba(proj(z))[:, order]
    else:
        from baselines import absorption_probabilities
        Pabs = absorption_probabilities(adata, ctx)
        if readout == "soft_logreg":
            n = len(Z)
            clf = LogisticRegression(C=0.1, max_iter=2000).fit(
                np.tile(Z, (F, 1)), np.repeat(np.arange(F), n), sample_weight=Pabs.T.ravel() + 1e-6)
            f_read = lambda z: clf.predict_proba(z)
        elif readout == "knn_absorb":
            pca = PCA(min(20, G - 1), random_state=0).fit(Z)
            knn = NearestNeighbors(n_neighbors=15).fit(pca.transform(Z))
            f_read = lambda z: Pabs[knn.kneighbors(pca.transform(z), return_distance=False)].mean(1)
        else:
            raise ValueError(readout)
    cls = list(ctx.fates)

    if readout == "logreg":
        Ridx = np.where(decision_region(adata, ctx, quantile=region_q))[0]   # region_q=0: all non-terminal cells          # original: region from the baseline classifier
    else:                                                       # region from the same readout: top-half entropy
        Pobs = f_read(Z)
        H = -(Pobs * np.log(np.clip(Pobs, 1e-12, 1))).sum(1)
        Ridx = np.where(nonterm & (H >= np.quantile(H[nonterm], 0.5)))[0]
    if max_cells is not None and len(Ridx) > max_cells:
        Ridx = np.sort(np.random.default_rng(seed).choice(Ridx, max_cells, replace=False))
    z0 = Z[Ridx]; R = len(Ridx)
    base = f_read(rollout(z0.copy()))

    names = list(adata.var_names)
    if candidates is None:
        cand = np.arange(G)
    elif isinstance(candidates, (int, np.integer)):
        cand = np.array(sorted(names.index(g) for g in candidate_genes(adata, ctx, min(int(candidates), G))))
    else:
        cand = np.array(sorted(names.index(g) for g in candidates if g in names))
    eps = 1e-4
    logit = lambda q: np.log(np.clip(q, eps, 1 - eps)) - np.log(np.clip(1 - q, eps, 1 - eps))
    dprob = np.full((G, F), np.nan); dlogodds = np.full((G, F), np.nan)
    for k, f in enumerate(ctx.fates):
        fi = cls.index(f)
        clamp_level = np.quantile(Z[tm & (term == f)], 0.9, axis=0)   # gene's typical level in fate f
        for s0 in range(0, len(cand), batch):
            genes = cand[s0:s0 + batch]
            p = f_read(rollout_clamped(z0, genes, clamp_level[genes]))[:, fi].reshape(len(genes), R)
            dprob[genes, k] = (p - base[None, :, fi]).mean(1)
            dlogodds[genes, k] = (logit(p) - logit(base[None, :, fi])).mean(1)
    scores = np.zeros(G)
    if score_mode == "prob":
        scores[cand] = np.maximum(np.nanmax(dprob[cand], axis=1), 0.0)   # identical to the original rule
    elif score_mode == "logodds":
        scores[cand] = np.maximum(np.nanmax(dlogodds[cand], axis=1), 0.0)
    elif score_mode == "perfate":
        pct = pd.DataFrame(dprob[cand]).rank(pct=True).values            # per-fate rank percentile among candidates
        scores[cand] = np.nanmax(pct, axis=1)
    else:
        raise ValueError(score_mode)
    s = pd.Series(scores, index=adata.var_names)
    out = [s]
    if return_model:
        out.append(W)
    if return_shifts:
        sh = pd.DataFrame(np.hstack([dprob, dlogodds]), index=adata.var_names,
                          columns=[f"prob_{f}" for f in ctx.fates] + [f"logodds_{f}" for f in ctx.fates])
        sh.attrs["baseline_fate_prob"] = {f: float(base[:, cls.index(f)].mean()) for f in ctx.fates}
        out.append(sh)
    return out[0] if len(out) == 1 else tuple(out)


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
