"""Figure: driver recovery vs depth of nested lineage trees (17 Nov 2026). Reads results/tree_depth.csv.

One panel per split probability p. x = network depth (number of nested splits on the deepest path),
y = AUPRC drivers vs markers. One line per method: mean over the 10 replicate networks (each network = mean over
its decisions), shaded band = 95% bootstrap confidence interval of the mean.
Second figure (fig_tree_level): the same, but every decision placed at its own level in the tree.
Writes figures/fig_tree_depth.{png,pdf} and figures/fig_tree_level.{png,pdf}.
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from driverbench.paths import ROOT
FIG = ROOT / "figures"
METHODS = [("OURS_v2", "Ours", "#2a78d6", "o"), ("CellRank", "CellRank 2", "#eb6834", "s"),
           ("rf_shap_fixed", "Random forest + SHAP", "#1baf7a", "D"), ("de_terminal", "DE (terminal states)", "#eda100", "^"),
           ("fate_corr", "Fate correlation", "#6f6e69", "v")]
INK2, GRID_C, AXIS_C = "#3d3d3a", "#ecebe6", "#8a8a85"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": AXIS_C, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def boot_ci(x, rng, n=2000):
    x = np.asarray(x)
    if len(x) < 2:
        return np.nan, np.nan
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def panel(ax, d, row_col, rows, rng, show_ylabel, xlabel):
    """One line per method: mean over replicates at each depth, shaded band = 95% bootstrap CI of that mean."""
    for key, lab, col, mar in METHODS:
        g = d[d.method == key].groupby(row_col)["auprc_vs_markers"]
        m = g.mean().reindex(rows)
        ci = np.array([boot_ci(d[(d.method == key) & (d[row_col] == r)]["auprc_vs_markers"].dropna(), rng) for r in rows])
        ax.fill_between(rows, ci[:, 0], ci[:, 1], color=col, alpha=0.15, lw=0, zorder=1)
        ax.plot(rows, m, color=col, lw=1.6, marker=mar, markersize=4.5, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ch = d["chance_vs_markers"].mean()
    ax.axhline(ch, color=AXIS_C, ls=(0, (3, 2)), lw=0.8, zorder=0)
    ax.set_xticks(rows); ax.set_xlim(min(rows) - 0.15, max(rows) + 0.15)
    ax.set_ylim(0, 1.02); ax.set_xlabel(xlabel)
    ax.yaxis.grid(True, color=GRID_C, lw=0.6); ax.set_axisbelow(True)
    if show_ylabel:
        ax.set_ylabel("AUPRC, drivers vs markers")
    return ch


def make(df, row_col, ylabel, name, per_network):
    rng = np.random.default_rng(0)
    if per_network:   # one value per network = mean over its decisions
        df = df.groupby(["p", "dataset", "network_depth", "method"], as_index=False)[["auprc_vs_markers", "chance_vs_markers"]].mean()
    rows = sorted(df[row_col].unique())
    ps = sorted(df.p.unique(), reverse=True)
    fig, axes = plt.subplots(1, len(ps), figsize=(3.4 * len(ps), 3.0), sharey=True, squeeze=False)
    for j, p in enumerate(ps):
        d = df[df.p == p]
        panel(axes[0, j], d, row_col, rows, rng, j == 0, ylabel)
        n_net = d.dataset.nunique()
        axes[0, j].set_title(f"split probability p = {p:g}  ({n_net} networks)", fontsize=8, color=INK2, loc="left")
    handles = [plt.Line2D([], [], color=c, marker=m, lw=1.4, markersize=5, label=l) for _, l, c, m in METHODS]
    handles.append(plt.Line2D([], [], color=AXIS_C, ls=(0, (3, 2)), lw=0.8, label="Chance"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.02),
               fontsize=7, handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "tree_depth.csv")
    if "error" in df:
        bad = df[df.error.notna()]
        if len(bad):
            print(f"{len(bad)} failed rows (excluded):", bad.groupby("method").size().to_dict())
        df = df[df.error.isna()]
    make(df, "network_depth", "Network depth (nested splits)", "fig_tree_depth", per_network=True)
    make(df, "decision_level", "Level of the decision in the tree", "fig_tree_level", per_network=False)
    net = df.groupby(["p", "dataset", "network_depth", "method"]).auprc_vs_markers.mean().reset_index()
    print(net.pivot_table(index=["p", "network_depth"], columns="method", values="auprc_vs_markers").round(2))
    print(df.pivot_table(index=["p", "decision_level"], columns="method", values="auprc_vs_markers", aggfunc="count"))
