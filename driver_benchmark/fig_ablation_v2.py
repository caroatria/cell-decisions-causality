"""Figure: ablations of version 2 on the 28 held-out synthetic decisions (driver-vs-marker AUPRC).
One panel per design choice; the chosen setting in blue, alternatives in grey; points = individual decisions;
* = paired Wilcoxon p < 0.05 against the chosen setting. Sources: results/ablation_v2.csv, ablation_v2_extra.csv,
v2_heldout.csv (K=5, scoring rule), readout_heldout.csv (fate predictor).
Outputs figures/fig_ablation_v2_panels.{pdf,png} and results/ablation_v2_summary.csv."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
KEY = ["dataset", "decision"]
a = pd.read_csv(R / "ablation_v2.csv")
x = pd.read_csv(R / "ablation_v2_extra.csv")
h = pd.read_csv(R / "v2_heldout.csv")
r = pd.read_csv(R / "readout_heldout.csv")

def col(df, **q):
    m = np.ones(len(df), bool)
    for k, v in q.items():
        m &= (df[k] == v).values
    return df[m].set_index(KEY)["auprc_vs_markers"]

ref = col(a, variant="v2")
idx = ref.index
root = pd.concat([col(a, variant=f"pt_dpt_root_s{s}") for s in range(3)], axis=1).mean(axis=1)
PANELS = [  # (title, [(label, series, is_reference)])
    ("Timing rule", [("with", ref, True), ("without", col(x, variant="no_timing"), False)]),
    ("Rollout steps $K$", [("0", col(a, variant="K0"), False), ("1", col(x, variant="K1"), False), ("3", ref, True),
                           ("5", col(h, K=5, mode="perfate"), False), ("10", col(x, variant="K10"), False)]),
    ("Onset threshold", [("0.1", col(x, variant="frac0.1"), False), ("0.2", ref, True), ("0.3", col(x, variant="frac0.3"), False)]),
    ("Pseudotime", [("DPT", ref, True), ("shifted\nroot", root, False), ("PCA\ndist.", col(a, variant="pt_pca_dist"), False),
                    ("true\ntime", col(a, variant="pt_true"), False)]),
    ("Cells intervened on", [("undecided", ref, True), ("all", col(a, variant="region_all"), False)]),
    ("Scoring rule", [("per-fate\nrank", ref, True), ("max\nshift", col(h, K=3, mode="prob"), False)]),
    ("Fate predictor", [("LR", ref, True), ("PCA\n+ LR", col(r, readout="pca_logreg"), False),
                        ("soft\nlabels", col(r, readout="soft_logreg"), False), ("graph", col(r, readout="knn_absorb"), False)]),
]

BLUE, GREY = "#2a78d6", "#8a8a85"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
ROWS = [PANELS[:4], PANELS[4:]]
units = [sum(len(p[1]) + 0.7 for p in row) for row in ROWS]
fig = plt.figure(figsize=(7.0, 4.6))
gs = fig.add_gridspec(2, 1, hspace=0.62)
axes, filler = [], None
for k, row in enumerate(ROWS):
    ratios = [len(p[1]) + 0.7 for p in row] + ([max(units) - units[k]] if units[k] < max(units) else [])
    sub = gs[k].subgridspec(1, len(ratios), width_ratios=ratios, wspace=0.14)
    for j in range(len(row)):
        axes.append(fig.add_subplot(sub[0, j]))
    if len(ratios) > len(row):
        filler = fig.add_subplot(sub[0, -1]); filler.axis("off")
rng = np.random.default_rng(0)
rows = []
for ax, (title, items) in zip(axes, PANELS):
    ax.axhline(ref.mean(), color=BLUE, ls=(0, (3, 2)), lw=0.8, zorder=1)
    for i, (lab, s, isref) in enumerate(items):
        y = s.reindex(idx)
        assert y.notna().all(), (title, lab)
        c = BLUE if isref else GREY
        ax.bar(i, y.mean(), 0.72, color=c, alpha=0.25, edgecolor="none", zorder=2)
        ax.hlines(y.mean(), i - 0.36, i + 0.36, colors=c, lw=2, zorder=3)
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(y)), y, s=7, color=c, edgecolors="white", linewidths=0.3, zorder=4)
        p = np.nan if isref else wilcoxon(ref, y).pvalue
        if not isref and p < 0.05:
            ax.text(i, 1.06, "*", ha="center", va="bottom", fontsize=10, color="#3d3d3a")
        rows.append({"parameter": title.replace("$", ""), "setting": lab.replace("\n", " "), "chosen": isref,
                     "mean_auprc": round(y.mean(), 3), "wins_chosen": int((ref > y).sum()) if not isref else None,
                     "losses_chosen": int((ref < y).sum()) if not isref else None, "wilcoxon_p": p})
    ax.set_xticks(range(len(items))); ax.set_xticklabels([it[0] for it in items], fontsize=7)
    ax.set_title(title, fontsize=8, pad=14)
    ax.set_xlim(-0.6, len(items) - 0.4)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color="#ecebe6", lw=0.6); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if ax in (axes[0], axes[len(ROWS[0])]):
        ax.set_ylabel("AUPRC, drivers vs markers")
    else:
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0, labelleft=False)
handles = [plt.Line2D([], [], color=BLUE, lw=2, marker="o", markersize=4, label="chosen setting"),
           plt.Line2D([], [], color=GREY, lw=2, marker="o", markersize=4, label="alternative"),
           plt.Line2D([], [], color="#3d3d3a", lw=0, marker="$*$", markersize=6, label="paired Wilcoxon $p<0.05$ vs chosen")]
filler.legend(handles=handles, loc="center", frameon=False, fontsize=7.5)
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "fig_ablation_v2_panels.pdf", bbox_inches="tight")
fig.savefig(out / "fig_ablation_v2_panels.png", dpi=300, bbox_inches="tight")
S = pd.DataFrame(rows); S.to_csv(R / "ablation_v2_summary.csv", index=False)
print(S.to_string())
