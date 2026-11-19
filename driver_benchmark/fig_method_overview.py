"""Figure: how the method works, step by step, on one held-out synthetic decision (BF_rep5, bifurcation).
Every panel is computed with the method's own code and settings (version 2: onset threshold 0.2, K = 3,
per-fate ranking); the final ranking is checked against score_timing_cf_v2.
  a  cells in PCA space coloured by diffusion pseudotime from the root
  b  fate classifier output; the decision region = most uncertain half of the non-terminal cells
  c  gene onsets along pseudotime (drivers switch on early, most markers later)
  d  timing-constrained forward model: learned regulator -> target weights; forbidden edges grey
  e  effect of clamping each gene on P(fate) of the undecided cells: directly (K = 0) vs after the rollout (K = 3)
  f  ranking: fate correlation vs ours (drivers blue, markers red)
Outputs figures/fig_method_overview.{pdf,png} and figures/method_overview/panel_{a..f}.png (each panel separately)."""
from pathlib import Path
import warnings
import numpy as np, pandas as pd, anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")

from evaluate import contexts, metrics
from baselines import _X, decision_region, score_fate_corr
from probe_interventional import dpt
from method_timing import onset_times, score_timing_cf_v2, timing_mask
from probe_interventional import score_interventional_probe

ROOT = Path(__file__).resolve().parent
DATA = "BF_rep9"
LAG, KNN, ALPHA, FRAC, K = 0.05, 10, 0.01, 0.2, 3

a = ad.read_h5ad(ROOT / "data" / f"{DATA}.h5ad")
ctx = next(contexts(a, seed=0))
names = list(a.var_names); G = len(names)
role = a.var["role"].where(a.var["decision"].isin(["", ctx.decision]) | (a.var["role"] != "marker"), "other")
X = _X(a, ctx.cells); obs = a.obs.loc[ctx.cells]
root = int(np.argmin(obs["sim_time"].values))
pt, pcs = dpt(a[ctx.cells], root)
term = ctx.terminal[ctx.cells].values; nonterm = pd.isna(term)
fates = list(ctx.fates)

# ---- forward model (same code path as probe_interventional.score_interventional_probe)
mu, sd = X.mean(0), X.std(0) + 1e-6
Z = (X - mu) / sd
idx = NearestNeighbors(n_neighbors=min(80, len(X))).fit(pcs).kneighbors(pcs, return_distance=False)
src, tgt = [], []
for i in range(len(X)):
    later = [j for j in idx[i] if pt[i] < pt[j] <= pt[i] + LAG][:KNN]
    if len(later) >= 3:
        src.append(i); tgt.append(Z[later].mean(0))
src = np.array(src); tgt = np.array(tgt); keep = nonterm[src]
A_in, A_out = Z[src[keep]], tgt[keep]
onset, _ = onset_times(X, pt, frac=FRAC)
allowed = onset[None, :] <= onset[:, None]; np.fill_diagonal(allowed, True)
W = np.zeros((G, G)); b = np.zeros(G)
for g in range(G):
    cg = np.where(allowed[g])[0]
    m = Lasso(alpha=ALPHA, max_iter=5000).fit(A_in[:, cg], A_out[:, g] - A_in[:, g])
    W[g, cg], b[g] = m.coef_, m.intercept_
lo, hi = Z.min(0) - 1, Z.max(0) + 1
step = lambda z: np.clip(z + z @ W.T + b, lo, hi)

tm = ~nonterm & np.isin(term, fates)
clf = LogisticRegression(C=0.1, max_iter=2000).fit(Z[tm], term[tm])
order = [list(clf.classes_).index(f) for f in fates]
P = lambda z: clf.predict_proba(z)[:, order]
region = decision_region(a, ctx)
Ridx = np.where(region)[0]
Pall = P(Z)

# ---- check: our reimplementation gives the real method's ranking
ours = score_timing_cf_v2(a, ctx)
fc = score_fate_corr(a, ctx)
_, sh3 = score_interventional_probe(a, ctx, allowed_fn=timing_mask(0.0, FRAC), K=K, score_mode="perfate", return_shifts=True)
_, sh0 = score_interventional_probe(a, ctx, allowed_fn=timing_mask(0.0, FRAC), K=0, score_mode="perfate", return_shifts=True)
pc = [c for c in sh3.columns if c.startswith("prob_")]
eff3, eff0 = sh3[pc].max(1), sh0[pc].max(1)          # best per-fate mean increase in P(fate), as the method scores it
print("AUPRC ours", round(metrics(ours, a.var, ctx.decision)["auprc_vs_markers"], 3),
      "| fate corr", round(metrics(fc, a.var, ctx.decision)["auprc_vs_markers"], 3))


def traj(gene, fate):
    """Mean P(fate) of the decision-region cells over rollout steps 0..K, optionally with `gene` clamped."""
    fi = fates.index(fate); z = Z[Ridx].copy(); out = [P(z)[:, fi].mean()]
    lvl = None if gene is None else np.quantile(Z[tm & (term == fate)], 0.9, axis=0)[names.index(gene)]
    for _ in range(K):
        if gene is not None: z[:, names.index(gene)] = lvl
        z = step(z)
        if gene is not None: z[:, names.index(gene)] = lvl
        out.append(P(z)[:, fi].mean())
    return np.array(out)


drivers = [g for g in names if role[g] == "driver"]
markers = [g for g in names if role[g] == "marker"]
# example genes: the driver of fate 0, and the marker that fate correlation ranks highest (the "decoy")
fate0 = fates[0]
drv = max(drivers, key=lambda g: traj(g, fate0)[-1])
decoy = fc[markers].idxmax()
print("example driver", drv, "| top marker by fate correlation", decoy, "| fates", fates)

# ---------------------------------------------------------------- style (same system as the other paper figures)
BLUE, GREY, LIGHT = "#2a78d6", "#8a8a85", "#d6d5cf"          # drivers = blue (as "ours"), markers = grey, rest = light
FATE_C = {fates[0]: "#eb6834", fates[1]: "#1baf7a"}           # categorical slots 2 and 3 (pass the CVD check together)
INK, INK2, GRID_C, MID = "#0b0b0b", "#3d3d3a", "#ecebe6", "#f0efec"
STYLE = {"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": GREY, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
         "pdf.fonttype": 42, "ps.fonttype": 42}
plt.rcParams.update(STYLE)
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])  # palette blue ramp
DIV = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#ec9a99", MID, "#86b6ef", "#1c5cab"])       # red <-> blue, grey midpoint
FATE_CM = LinearSegmentedColormap.from_list("fate", [FATE_C[fates[1]], MID, FATE_C[fates[0]]])
gcol = lambda g: BLUE if role[g] == "driver" else GREY if role[g] == "marker" else LIGHT
DOT = dict(edgecolors="white", linewidths=0.4)

# every panel has the same printed size and the same plot area, so panels line up in any arrangement
PANEL_W, PANEL_H = 2.30, 2.15                 # inches; 3 panels fill a 7-inch text width (0.32\textwidth each)
L, R, B, T = 0.55, 0.10, 0.46, 0.50          # margins in inches (T leaves room for a two-line title)


def title(ax, letter, text):
    """Bold panel letter on the first line of a (two-line) left-aligned title."""
    ax.set_title(f"$\\bf{{{letter}}}$  " + text, loc="left", pad=4, linespacing=1.15)


def inset_colorbar(ax, mappable, label, ticks):
    cax = ax.inset_axes([0.60, -0.10, 0.38, 0.045])                 # below the plot, next to the x label
    cb = plt.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cax.text(-0.06, 0.5, label, transform=cax.transAxes, fontsize=6.5, color=INK2, ha="right", va="center")


def panel_a(ax):
    o = np.argsort(pt)
    s = ax.scatter(pcs[o, 0], pcs[o, 1], c=pt[o], cmap=SEQ, s=5, lw=0)
    for f in fates:
        m = term == f
        ax.scatter(pcs[m, 0], pcs[m, 1], s=8, facecolors="none", edgecolors=FATE_C[f], lw=0.6)
        c = pcs[m, :2].mean(0)
        ax.annotate(f"fate {f}", c, textcoords="offset points", xytext=(0, 18), fontsize=7, color=FATE_C[f],
                    ha="center", va="bottom", fontweight="bold")
    ax.scatter(*pcs[root, :2], marker="*", s=70, color=INK, edgecolors="white", lw=0.6, zorder=5)
    ax.annotate("root", pcs[root, :2], textcoords="offset points", xytext=(6, -2), fontsize=7, color=INK2, va="center")
    inset_colorbar(ax, s, "pseudotime", [0, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("PC 1", loc="left"); ax.set_ylabel("PC 2")
    title(ax, "a", "Order cells\nin time")


def panel_b(ax):
    p = Pall[:, 0]
    ax.scatter(pcs[~region, 0], pcs[~region, 1], c=p[~region], cmap=FATE_CM, vmin=0, vmax=1, s=5, lw=0, alpha=0.45)
    s = ax.scatter(pcs[region, 0], pcs[region, 1], c=p[region], cmap=FATE_CM, vmin=0, vmax=1, s=8,
                   edgecolors=INK, linewidths=0.35)
    inset_colorbar(ax, s, f"P(fate {fates[0]})", [0, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("PC 1", loc="left"); ax.set_ylabel("PC 2")
    title(ax, "b", "Predict fate, find\nundecided cells (outlined)")


def panel_c(ax):
    groups = [("drivers", drivers), ("markers", markers), ("other", [g for g in names if role[g] not in ("driver", "marker")])]
    rng = np.random.default_rng(0)
    for k, (lab, gs) in enumerate(groups):
        on = np.array([onset[names.index(g)] for g in gs]); on = on[np.isfinite(on)]
        c = [BLUE, GREY, LIGHT][k]
        ax.scatter(on, k + rng.uniform(-0.17, 0.17, len(on)), s=[24, 10, 8][k], color=c, zorder=3, **DOT)
        ax.vlines(np.median(on), k - 0.3, k + 0.3, color=[BLUE, GREY, GREY][k], lw=2, zorder=4)
    for g in drivers:
        ax.annotate(g, (onset[names.index(g)], 0), textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=7, color=BLUE)
    ax.set_yticks(range(3)); ax.set_yticklabels([g[0] for g in groups]); ax.set_ylim(2.6, -0.65)
    ax.set_xlabel("onset (pseudotime)"); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
    ax.xaxis.grid(True, color=GRID_C, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    title(ax, "c", "When does each\ngene switch on?")


def panel_d(ax):
    sel = sorted(drivers + markers, key=lambda g: onset[names.index(g)])
    ii = [names.index(g) for g in sel]
    M = W[np.ix_(ii, ii)].copy(); np.fill_diagonal(M, np.nan)
    forb = ~allowed[np.ix_(ii, ii)]
    vmax = np.nanmax(np.abs(M))
    cm = DIV.copy(); cm.set_bad("white")
    M = np.where(np.abs(M) < 1e-9, np.nan, M)                  # exact zeros (no edge) white
    ax.imshow(np.where(forb, np.nan, M), cmap=cm, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.imshow(np.where(forb, 1.0, np.nan), cmap=LinearSegmentedColormap.from_list("g", ["#b9b8b2", "#b9b8b2"]),
              aspect="auto", interpolation="nearest")
    n = len(sel); dk = [k for k, g in enumerate(sel) if role[g] == "driver"]
    lab = ", ".join(sel[k] for k in dk)
    ax.set_xticks([np.mean(dk)]); ax.set_xticklabels([lab], color=BLUE)
    ax.set_yticks([np.mean(dk)]); ax.set_yticklabels([lab], color=BLUE); ax.tick_params(length=2)
    ax.set_xlabel("regulator, by onset →"); ax.set_ylabel("target, by onset →")
    ax.text(n - 1, 0.5, "grey: forbidden\n(regulator switches\non later)\nblue +, red −", fontsize=6.5,
            color=INK2, ha="right", va="top")
    for s_ in ax.spines.values(): s_.set_visible(False)
    title(ax, "d", "Forward model with\nrestricted gene input")


def panel_e(ax):
    other = [g for g in names if role[g] not in ("driver", "marker")]
    for gs, c, sz, lab in [(other, LIGHT, 8, "other genes"), (markers, GREY, 12, "markers"), (drivers, BLUE, 28, "drivers")]:
        ax.scatter(eff0[gs], eff3[gs], s=sz, color=c, zorder=3, label=lab, **DOT)
    for g in drivers:
        ax.annotate(g, (eff0[g], eff3[g]), textcoords="offset points", xytext=(5, 0), va="center", fontsize=7, color=BLUE)
    lim = max(eff3.max(), eff0.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(lim * 0.98, lim * 0.62, "no gain from\npropagation", fontsize=6.5, color=GREY, ha="right", va="top")
    ax.set_xlim(min(0, eff0.min()) - 0.01, lim); ax.set_ylim(min(0, eff3.min()) - 0.01, lim)
    ax.set_xticks([0, 0.1, 0.2, 0.3]); ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("direct effect (K = 0)"); ax.set_ylabel(f"after rollout (K = {K})")
    ax.grid(color=GRID_C, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=6.5, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.3)
    title(ax, "e", "Intervene and\nroll forward")


def panel_f(ax, top=8):
    rk_fc = fc.rank(ascending=False, method="first"); rk_o = ours.rank(ascending=False, method="first")
    show = sorted(set(rk_fc[rk_fc <= top].index) | set(rk_o[rk_o <= top].index), key=lambda g: rk_fc[g])
    for g in show:
        y0, y1 = min(rk_fc[g], top + 1.4), min(rk_o[g], top + 1.4)
        c = gcol(g) if role[g] in ("driver", "marker") else LIGHT
        drv_ = role[g] == "driver"
        ax.plot([0, 1], [y0, y1], color=c, lw=2 if drv_ else 0.9, zorder=3 if drv_ else 2)
        ax.scatter([0, 1], [y0, y1], s=14 if drv_ else 9, color=c, zorder=4, **DOT)
        tc = BLUE if drv_ else INK2
        if rk_fc[g] <= top: ax.text(-0.07, y0, g, ha="right", va="center", fontsize=6.5, color=tc)
        if rk_o[g] <= top: ax.text(1.07, y1, g, ha="left", va="center", fontsize=6.5, color=tc)
    ax.set_ylim(top + 1.9, 0.3); ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["fate\ncorrelation", "ours"])
    ax.set_yticks([1, 4, 8]); ax.set_ylabel("rank"); ax.tick_params(axis="x", length=0)
    ax.text(0.5, top + 1.4, f"> {top}", ha="center", va="center", fontsize=6.5, color=GREY)
    ax.spines["bottom"].set_visible(False)
    title(ax, "f", "Rank genes by\neffect on fate")


PANELS = {"a": panel_a, "b": panel_b, "c": panel_c, "d": panel_d, "e": panel_e, "f": panel_f}
out = ROOT / "figures"; (out / "method_overview").mkdir(parents=True, exist_ok=True)


def axes_in_cell(fig, x0, y0):
    """Axes at the fixed position inside a PANEL_W x PANEL_H cell whose lower-left corner is (x0, y0) inches."""
    W_, H_ = fig.get_size_inches()
    return fig.add_axes([(x0 + L) / W_, (y0 + B) / H_, (PANEL_W - L - R) / W_, (PANEL_H - B - T) / H_])


# each panel separately (identical size and plot area), then all six as one figure with exactly the same geometry
for k, fn in PANELS.items():
    fig = plt.figure(figsize=(PANEL_W, PANEL_H)); fn(axes_in_cell(fig, 0, 0))
    fig.savefig(out / "method_overview" / f"panel_{k}.png", dpi=300)
    fig.savefig(out / "method_overview" / f"panel_{k}.pdf")
    plt.close(fig)
fig = plt.figure(figsize=(3 * PANEL_W, 2 * PANEL_H))
for i, fn in enumerate(PANELS.values()):
    fn(axes_in_cell(fig, (i % 3) * PANEL_W, (1 - i // 3) * PANEL_H))
fig.savefig(out / "fig_method_overview.pdf"); fig.savefig(out / "fig_method_overview.png", dpi=300)
plt.close(fig)
print("saved")
