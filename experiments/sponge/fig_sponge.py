"""Figures for the sponge case study, built from the outputs of sponge_run.py.

  a  UMAP coloured by role (stem / transition / terminal fates / other), arrows stem -> fates
  b  UMAP coloured by fate entropy; grey = outside the decision region the method intervenes on
  c  top-N candidate TFs x fates: probability shift dP (left) and expression vs stem (right)
  d  our driver score vs an association baseline (default fate_corr), percentiles among TFs;
     points high on ours but low on the baseline are candidate drivers an association method misses
     (needs sponge_run.py run with --methods OURS_v2_TF,fate_corr; skipped otherwise)

Usage: same data/cluster options as the sponge_run.py call that made the results, plus
  --method OURS_v2_TF --seed 0 --top 20 --names gene_names.csv --compare fate_corr [--out figures/suberites_domuncula/fig_<tag>_<method>]
--names: optional CSV/TSV (first column = gene ID, second = display name) or JSON {gene ID: "Mouse, HUMAN"};
         --name-part human|mouse|full picks which symbol to show (default human).
Writes <out>.png (300 dpi) and <out>.pdf, plus each panel separately as <out>_{a,b,c,d}.png."""
import argparse, warnings
import numpy as np, pandas as pd, scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import entropy
warnings.filterwarnings("ignore")
import sponge_run as R
from driverbench.baselines import fate_probabilities, decision_region
from driverbench.paths import ROOT

CLUSTER_NAMES = {"0": "Intermediary", "1": "Pinacocytes", "3": "Sclerocytes 1", "4": "Sclerocytes 2",
                 "8": "Archeocytes", "10": "Choanocytes", "6": "Metabolocytes"}
FATE_LABEL = {"choano": "Choanocytes", "sclero": "Sclerocytes", "digest": "Metabolocytes", "skin": "Pinacocytes"}
# categorical slots in fixed order (validated: scripts/validate_palette.js, light mode); one colour per entity
FATE_COLOR = {"choano": "#2a78d6", "sclero": "#eb6834", "digest": "#1baf7a", "skin": "#eda100"}
STEM_COLOR, TRANS_COLOR, OTHER_COLOR = "#6250d6", "#e87ba4", "#d6d5cf"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list("bluegrayred", ["#1c5cab", "#86b6ef", "#f0efec", "#ec835a", "#c63a3a"])
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "sans-serif"})


def fig_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="OURS_v2_TF"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=20); p.add_argument("--names", default=None)
    p.add_argument("--name-part", default="human", choices=["human", "mouse", "full"])
    p.add_argument("--compare", default="fate_corr"); p.add_argument("--out", default=None)   # default: figures/suberites_domuncula/fig_<tag>_<method>
    return p.parse_known_args()[0]


def gene_names(path, part="human"):
    """CSV/TSV (gene ID, name) or JSON {gene ID: name}. JSON values like "Prkaca, PRKACA" (mouse, human):
    part="human" keeps the last symbol, "mouse" the first, "full" the whole string."""
    if not path:
        return {}
    if str(path).endswith(".json"):
        import json
        raw = json.load(open(path))
    else:
        t = pd.read_csv(path, sep=None, engine="python")
        raw = dict(zip(t.iloc[:, 0].astype(str), t.iloc[:, 1].astype(str)))
    out = {}
    for g, v in raw.items():
        parts = [p.strip() for p in str(v).split(",") if p.strip() and p.strip().lower() not in ("nan", "none", "error", "na")]
        if parts:
            out[str(g)] = ", ".join(parts) if part == "full" else (parts[-1] if part == "human" else parts[0])
    return out


def roles(b, a_, fates):
    cl = b.obs[a_.cluster_key].values
    groups = {"choano": R.ints(a_.choano), "sclero": R.ints(a_.sclero), "digest": R.ints(a_.digest), "skin": R.ints(a_.skin)}
    role = np.full(b.n_obs, "other", dtype=object)
    role[np.isin(cl, R.ints(a_.transition))] = "transition"
    role[np.isin(cl, R.ints(a_.stem))] = "stem"
    for f in fates:
        role[np.isin(cl, groups[f])] = f
    return role, cl


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel_a(ax, U, role, cl, fates):
    col = {"stem": STEM_COLOR, "transition": TRANS_COLOR, "other": OTHER_COLOR, **FATE_COLOR}
    order = ["other", "transition", *fates, "stem"]                     # stem on top
    for r in order:
        m = role == r
        ax.scatter(U[m, 0], U[m, 1], s=1.5, c=col[r], lw=0, rasterized=True)
    for c in pd.unique(cl):                                             # direct labels (no colour-only identity)
        m = cl == c
        if m.sum() < 10:
            continue
        x, y = np.median(U[m], 0)
        ax.text(x, y, CLUSTER_NAMES.get(str(c), f"Cluster {c}"), fontsize=5.5, ha="center", va="center",
                color=INK if str(c) in CLUSTER_NAMES else MUTED,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    s0 = np.median(U[role == "stem"], 0)
    for f in fates:
        t = np.median(U[role == f], 0)
        ax.annotate("", xy=s0 + 0.8 * (t - s0), xytext=s0 + 0.15 * (t - s0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0))
    clean(ax); ax.set_title("a  Cell types and fate decision", loc="left")


def panel_b(ax, U, P, region):
    H = entropy(P.T) / np.log(P.shape[1])
    ax.scatter(U[~region, 0], U[~region, 1], s=1.5, c=OTHER_COLOR, lw=0, rasterized=True)
    sca = ax.scatter(U[region, 0], U[region, 1], s=2, c=H[region], cmap=SEQ, vmin=0, vmax=1, lw=0, rasterized=True)
    cb = plt.colorbar(sca, ax=ax, fraction=0.05, pad=0.02, shrink=0.6, orientation="horizontal", location="bottom")
    cb.set_label("fate entropy (normalised)", fontsize=6); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5)
    clean(ax); ax.set_title("b  Decision region (undecided cells)", loc="left")


def panel_c(ax_dp, ax_fc, tab, fates, names, top):
    t = tab.head(top).copy()
    t["_o"] = [fates.index(f) for f in t.best_fate]
    t = t.sort_values(["_o", "rank"])
    dp = t[[f"dP_{f}" for f in fates]].values
    fc = t[[f"logFC_{f}_vs_stem" for f in fates]].values
    labels = [names.get(g, g) for g in t.index]
    lim = np.nanmax(np.abs(dp)) or 1
    im1 = ax_dp.imshow(dp, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    lim2 = np.nanmax(np.abs(fc)) or 1
    im2 = ax_fc.imshow(fc, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim2, lim2), aspect="auto")
    for ax, title in ((ax_dp, "shift in P(fate)"), (ax_fc, "log expr. vs stem")):
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(fates))); ax.set_xticklabels([FATE_LABEL[f] for f in fates], rotation=45, ha="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(fates)), minor=True); ax.set_yticks(np.arange(-.5, len(t)), minor=True)
        ax.grid(which="minor", color="white", lw=1)                       # 1px surface gap between cells
        ax.tick_params(which="minor", length=0)
    ax_dp.set_yticks(range(len(t))); ax_dp.set_yticklabels(labels, fontsize=5.5)
    for i, f in enumerate(t.best_fate):                                 # best fate: marker + bold column
        ax_dp.add_patch(plt.Rectangle((fates.index(f) - .5, i - .5), 1, 1, fill=False, ec=INK, lw=0.7))
        ax_dp.scatter(-0.9, i, s=10, c=FATE_COLOR[f], clip_on=False, lw=0)
    ax_fc.set_yticks([])
    for ax, im, lab in ((ax_dp, im1, "ΔP(fate) after clamping"), (ax_fc, im2, "log expr. vs stem")):
        cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02, orientation="horizontal", location="bottom")
        cb.set_label(lab, fontsize=6); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5)
    ax_dp.annotate(f"c  Top {top} candidate drivers", xy=(0, 1), xycoords="axes fraction", xytext=(-40, 62),
                   textcoords="offset points", fontsize=8, va="bottom")


def panel_d(ax, ours, other, tab, names, top, compare):
    tfs = tab.index
    x = other.reindex(tfs).rank(pct=True); y = ours.reindex(tfs).rank(pct=True)
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=MUTED, zorder=0)
    topg = tab.index[:top]
    rest = ~tfs.isin(topg)
    ax.scatter(x[rest], y[rest], s=8, c=OTHER_COLOR, lw=0, rasterized=True)
    for f, g in tab.loc[topg].groupby("best_fate"):
        ax.scatter(x[g.index], y[g.index], s=14, c=FATE_COLOR[f], ec="white", lw=0.6,
                   label=FATE_LABEL[f], zorder=3)
    gap = (y[topg] - x[topg]).sort_values(ascending=False).head(6)       # label only the most off-diagonal
    for g in gap.index:
        ax.annotate(names.get(g, g), (x[g], y[g]), xytext=(4, 0), textcoords="offset points",
                    fontsize=5.5, va="center", color=INK)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{compare} (percentile among scored genes)"); ax.set_ylabel("ours (percentile among scored genes)")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.5, loc="lower right", handletextpad=0.2)
    ax.set_title("d  Driver score vs association", loc="left")


def main():
    a_, f_ = R.args_(), fig_args()
    fates = R.ints(a_.fates)
    if a_.pairwise:
        raise SystemExit("fig_sponge.py draws the multi-fate decision; run it without --pairwise")
    dec = "stem_to_" + "_".join(fates)
    res = R.ROOT / "results"
    tab = pd.read_csv(res / f"{a_.tag}_perfate_{f_.method}_{dec}_s{f_.seed}.csv", index_col=0)
    scores = pd.read_csv(res / f"{a_.tag}_scores_{dec}_s{f_.seed}.csv", index_col=0)
    names = gene_names(f_.names, f_.name_part)

    a, _ = R.load(a_)
    b, ctx = R.context(a, a_, fates, f_.seed)
    if "X_umap" not in b.obsm:
        sc.pp.neighbors(b, n_neighbors=15); sc.tl.umap(b, random_state=0)
    U = b.obsm["X_umap"]
    role, cl = roles(b, a_, fates)
    P = fate_probabilities(b, ctx)
    region = decision_region(b, ctx)

    have_d = f_.compare in scores.columns
    if not have_d:
        print(f"panel d skipped: no {f_.compare!r} column in the scores file "
              f"(rerun sponge_run.py with --methods {f_.method},{f_.compare})")
    fig = plt.figure(figsize=(7.2, 5.6 if have_d else 3.0))
    gs = fig.add_gridspec(2 if have_d else 1, 4, width_ratios=[1.25, 1.25, 0.7, 0.7], wspace=0.35, hspace=0.45)
    panel_a(fig.add_subplot(gs[0, 0]), U, role, cl, fates)
    panel_b(fig.add_subplot(gs[0, 1]), U, P, region)
    ax_dp = fig.add_subplot(gs[:, 2] if have_d else gs[0, 2]); ax_fc = fig.add_subplot(gs[:, 3] if have_d else gs[0, 3])
    panel_c(ax_dp, ax_fc, tab, fates, names, f_.top)
    if have_d:
        panel_d(fig.add_subplot(gs[1, 0:2]), scores[f_.method], scores[f_.compare], tab, names, f_.top, f_.compare)
    out = R.ROOT / (f_.out or f"figures/suberites_domuncula/fig_{a_.tag}_{f_.method}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"wrote {out}.png / .pdf and panels {out}_[abcd].png")

    # each panel on its own, for rearranging in the paper
    for key, w, h in (("a", 3, 2.8), ("b", 3.2, 2.8), ("c", 2.6, 0.18 * f_.top + 1.2)):
        fg = plt.figure(figsize=(w, h))
        if key == "a":
            panel_a(fg.add_subplot(), U, role, cl, fates)
        elif key == "b":
            panel_b(fg.add_subplot(), U, P, region)
        else:
            g2 = fg.add_gridspec(1, 2, wspace=0.1); panel_c(fg.add_subplot(g2[0]), fg.add_subplot(g2[1]), tab, fates, names, f_.top)
        fg.savefig(f"{out}_{key}.png", dpi=300, bbox_inches="tight"); plt.close(fg)
    if have_d:
        fg, ax = plt.subplots(figsize=(3.2, 3))
        panel_d(ax, scores[f_.method], scores[f_.compare], tab, names, f_.top, f_.compare)
        fg.savefig(f"{out}_d.png", dpi=300, bbox_inches="tight"); plt.close(fg)


if __name__ == "__main__":
    main()
