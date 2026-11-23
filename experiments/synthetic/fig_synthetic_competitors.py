"""Figure: driver-vs-marker AUPRC per synthetic scenario, ours vs the competitors that run on simulated networks.
Same 28 held-out decisions as eval_v2.py (BF/TF replicates 3-4, all CBF and HSC decisions).
Outputs figures/fig_synthetic_competitors.{pdf,png}."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from driverbench.paths import ROOT
v = pd.read_csv(ROOT / "results/v2_heldout.csv"); v = v[(v.K == 3) & (v["mode"] == "perfate")].assign(method="ours")
keys = set(zip(v.dataset, v.decision))
b = pd.concat([pd.read_csv(ROOT / "results/baseline_metrics.csv"), pd.read_csv(ROOT / "results/hsc_metrics.csv"),
               pd.read_csv(ROOT / "results/cellrank_synthetic.csv").assign(method="CellRank")])
b = b[[(d, k) in keys for d, k in zip(b.dataset, b.decision)]]
df = pd.concat([v, b])[["dataset", "model", "decision", "method", "auprc_vs_markers"]]

METHODS = [  # (key, label, colour, marker) - fixed categorical order, shape as secondary encoding
    ("ours", "Ours", "#2a78d6", "o"),
    ("CellRank", "CellRank 2", "#eb6834", "s"),
    ("rf_shap_fixed", "Random forest + SHAP", "#1baf7a", "D"),
    ("de_terminal", "DE (terminal states)", "#eda100", "^"),
]
SCEN = [("BF", "Bifurcation"), ("TF", "Trifurcation"), ("CBF", "Nested\nbifurcation"),
        ("HSC", "Haematopoietic\nnetwork"), ("ALL", "All decisions")]

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.0, 2.6))
rng = np.random.default_rng(0)
width = 0.19
for i, (sk, sl) in enumerate(SCEN):
    sub = df if sk == "ALL" else df[df.model == sk]
    n = sub[sub.method == "ours"].shape[0]
    orc = sub[sub.method == "ORACLE_early_true_fate"].auprc_vs_markers.mean()
    ax.hlines(orc, i - 2.2 * width, i + 2.2 * width, colors="#8a8a85", linestyles=(0, (3, 2)), lw=1, zorder=1)
    for j, (mk, ml, col, mar) in enumerate(METHODS):
        y = sub[sub.method == mk].auprc_vs_markers.values
        x = i + (j - 1.5) * width
        ax.bar(x, y.mean(), width * 0.86, color=col, alpha=0.28, edgecolor="none", zorder=2)
        ax.hlines(y.mean(), x - width * 0.43, x + width * 0.43, colors=col, lw=2, zorder=3)
        ax.scatter(x + rng.uniform(-width * 0.25, width * 0.25, len(y)), y, s=11, marker=mar, color=col,
                   edgecolors="white", linewidths=0.4, zorder=4)
    if sk == "ALL":
        ax.axvline(i - 0.5, color="#d6d5cf", lw=0.8, zorder=0)
    SCEN[i] = (sk, f"{sl}\n(n={n})")
ax.set_xticks(range(len(SCEN))); ax.set_xticklabels([s[1] for s in SCEN])
ax.set_ylim(0, 1.05); ax.set_ylabel("AUPRC, drivers vs markers")
ax.yaxis.grid(True, color="#ecebe6", lw=0.6); ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
handles = [plt.Line2D([], [], color=c, marker=m, lw=2, markersize=5, label=l) for _, l, c, m in METHODS]
handles.append(plt.Line2D([], [], color="#8a8a85", ls=(0, (3, 2)), lw=1, label="Oracle (true future fate)"))
ax.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
          handlelength=1.8, columnspacing=1.2, fontsize=7.5)
fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "fig_synthetic_competitors.pdf", bbox_inches="tight")
fig.savefig(out / "fig_synthetic_competitors.png", dpi=300, bbox_inches="tight")
print(df.groupby(["model", "method"]).auprc_vs_markers.agg(["mean", "size"]).round(2).loc[:, :].to_string())
