"""Presentation figures for the driver-gene benchmark (static PNG + PDF, light theme).
fig1_benchmark_design  - what drivers vs markers mean in the simulated networks
fig2_method_pipeline   - our method (version 1), step by step
fig3_method_comparison - all methods, synthetic vs curated HSC benchmark
fig4_per_decision      - ours vs best existing method vs oracle, per decision (nested marked)
Data: figures/all_scores_long.csv (built from results/*.csv).
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OURS, CELLRANK, GREY, ORACLE = "#2a78d6", "#eb6834", "#a3a29c", "#52514e"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
OUT = "figures/"
d = pd.read_csv("figures/all_scores_long.csv")
NAMES = {"ORACLE_early_true_fate": "Oracle (knows future fates)", "OURS_timing_cf": "Ours (version 1)",
         "fate_corr": "CellRank-style fate correlation", "obs_counterfactual": "Plain counterfactuals",
         "de_terminal": "Differential expression", "rf_shap_fixed": "SHAP / BUNGEE (fixed)",
         "absorption_corr_decision_region": "Graph fate corr., undecided cells",
         "fate_corr_decision_region": "Fate corr., undecided cells",
         "PROBE_interventional_linear": "Ours without timing rule", "random": "Random"}


def save(fig, name):
    fig.savefig(OUT + name + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold" if bold else None)


def arrow(ax, a, b, color=INK2, style="-|>", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, color=color, lw=lw, ls=ls,
                                 shrinkA=4, shrinkB=4))


# ---------------------------------------------------------------- fig 1: benchmark design
def fig1():
    fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.7); ax.axis("off")
    box(ax, 1.0, 2.3, 1.5, 0.8, "Stem\nprogram", "#f0efec", GREY)
    box(ax, 3.6, 3.3, 1.3, 0.75, "Driver A", "#dce9f8", OURS, bold=True)
    box(ax, 3.6, 1.3, 1.3, 0.75, "Driver B", "#dce9f8", OURS, bold=True)
    arrow(ax, (1.75, 2.5), (2.95, 3.2)); arrow(ax, (1.75, 2.1), (2.95, 1.4))
    ax.add_patch(FancyArrowPatch((3.6, 2.92), (3.6, 1.68), arrowstyle="]-[,widthA=0.6,widthB=0.6",
                                 color=CELLRANK, lw=2.2, shrinkA=0, shrinkB=0))
    ax.text(3.8, 2.3, "repress each other\n(the fate switch)", color=CELLRANK, fontsize=9, va="center")
    for yy, lab in [(3.3, "A"), (1.3, "B")]:
        for k, dy in enumerate([0.45, 0.0, -0.45]):
            box(ax, 6.2, yy + dy, 1.2, 0.34, f"marker {lab}{k + 1}", "#f0efec", GREY, tc=INK2, fs=9)
            arrow(ax, (4.25, yy), (5.6, yy + dy), color=GREY, lw=1.2)
        ax.text(6.95, yy, "…", fontsize=14, color=INK2, va="center")
        box(ax, 8.6, yy, 1.6, 0.75, f"Fate {lab}", "#ffffff", INK2, bold=True)
        arrow(ax, (7.2, yy), (7.8, yy), color=GREY, lw=1.2, ls="--")
    ax.text(3.6, 4.2, "DRIVERS\ndecide the fate", ha="center", va="center", color=OURS, fontweight="bold")
    ax.text(6.3, 4.2, "MARKERS\nfollow, cause nothing", ha="center", va="center", color=INK2, fontweight="bold")
    ax.text(5, 0.25, "Simulated with BoolODE. Every gene's role is known. Each cell also has its true future fate. "
            "+ nuisance genes, noise.", ha="center", color=INK2, fontsize=9.5)
    ax.set_title("The benchmark: can a method rank drivers above markers?", loc="left", fontsize=14,
                 fontweight="bold", color=INK, pad=6)
    save(fig, "fig1_benchmark_design")


# ---------------------------------------------------------------- fig 2: pipeline
def fig2():
    steps = [("1", "Order cells\nin time", "diffusion\npseudotime"),
             ("2", "Learn to\nrecognise fates", "classifier on\nterminal cells"),
             ("3", "Find undecided\ncells", "high fate\nentropy"),
             ("4", "When does each\ngene start?", "onset along\npseudotime"),
             ("5", "Learn how cells\nmove forward", "sparse model +\nTIMING RULE"),
             ("6", "Intervene &\nfast-forward", "fix gene X,\nroll out 15 steps"),
             ("7", "Score & rank", "shift in fate\nprobability")]
    fig, ax = plt.subplots(figsize=(13.6, 3.9)); ax.set_xlim(0, 15.2); ax.set_ylim(0, 3.9); ax.axis("off")
    xs = np.linspace(1.0, 14.2, len(steps))
    for i, (n, t, s) in enumerate(steps):
        hl = n in ("5", "6")
        box(ax, xs[i], 2.35, 1.78, 1.2, t, "#dce9f8" if hl else "#f0efec", OURS if hl else GREY,
            fs=9.5, bold=hl)
        ax.text(xs[i], 3.15, n, ha="center", va="center", color=OURS if hl else INK2, fontweight="bold")
        ax.text(xs[i], 1.32, s, ha="center", va="center", color=INK2, fontsize=9)
        if i < len(steps) - 1:
            arrow(ax, (xs[i] + 0.9, 2.35), (xs[i + 1] - 0.9, 2.35))
    ax.text(0.2, 0.42, "Timing rule: gene X may only predict gene Y if X starts changing no later than Y "
            "→ blocks backwards 'marker → driver' links.", color=OURS, fontsize=10)
    ax.text(0.2, 0.05, "Question asked per gene: if I force this gene on in an undecided cell, does it end up "
            "in a different fate?  Drivers: yes.  Markers: no.", color=INK2, fontsize=10)
    ax.set_title("Our method (version 1): interventional counterfactuals with a timing constraint",
                 loc="left", fontsize=14, fontweight="bold", color=INK)
    save(fig, "fig2_method_pipeline")


# ---------------------------------------------------------------- fig 3: method comparison
def fig3():
    syn = d[d.model.isin(["BF", "TF", "CBF"])].groupby(["method", "setting"]).auprc_vs_markers.mean().groupby("method").mean()
    hsc = d[d.model == "HSC"].groupby(["method", "setting"]).auprc_vs_markers.mean().groupby("method").mean()
    chance_syn = d[d.model.isin(["BF", "TF", "CBF"])].groupby("setting").chance_vs_markers.mean().mean()
    chance_hsc = d[d.model == "HSC"].chance_vs_markers.mean()
    order = (syn + hsc).drop("ORACLE_early_true_fate").sort_values().index.tolist() + ["ORACLE_early_true_fate"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, ser, title, ch in [(axes[0], syn, "Synthetic networks\n(3 networks, 4 decisions, 5 replicates)", chance_syn),
                               (axes[1], hsc, "Curated blood-stem-cell network\n(Krumsiek 2011; never used for tuning)", chance_hsc)]:
        y = np.arange(len(order))
        for i, m in enumerate(order):
            v = ser.get(m, np.nan)
            c = OURS if m == "OURS_timing_cf" else CELLRANK if m == "fate_corr" else GREY
            if m == "ORACLE_early_true_fate":
                ax.plot([v], [i], marker="D", ms=9, mfc="white", mec=ORACLE, mew=2, ls="none")
            else:
                ax.hlines(i, 0, v, color=c, lw=2.5 if c != GREY else 1.6)
                ax.plot([v], [i], "o", ms=9, color=c, mec=SURF, mew=1.5)
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=10,
                    color=INK if m in ("OURS_timing_cf", "fate_corr") else INK2,
                    fontweight="bold" if m == "OURS_timing_cf" else None)
        ax.axvspan(0, ch, color=GRID, alpha=0.7, lw=0); ax.text(ch / 2, -0.95, "chance", ha="center",
                                                                  fontsize=8.5, color=INK2)
        ax.set_xlim(0, 1.0); ax.set_ylim(-1.2, len(order) - 0.4)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("Driver-vs-marker AUPRC (higher = drivers ranked above markers; mean over decisions and replicates)",
                  fontsize=10.5, color=INK2)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([NAMES[m] for m in order])
    for lab in axes[0].get_yticklabels():
        if lab.get_text() == NAMES["OURS_timing_cf"]: lab.set_fontweight("bold"); lab.set_color(OURS)
        if lab.get_text() == NAMES["fate_corr"]: lab.set_color(CELLRANK)
    fig.suptitle("Our method ties the best existing method on synthetic data and leads on the curated network",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save(fig, "fig3_method_comparison")


# ---------------------------------------------------------------- fig 4: per decision
def fig4():
    rows = [("BF-D1", "Synthetic: single split", False), ("TF-D1", "Synthetic: three-way split", False),
            ("CBF-D1", "Synthetic: two-step, first decision*", True), ("CBF-D2", "Synthetic: two-step, second decision", True),
            ("HSC-D1", "Blood: MEP vs GMP", False), ("HSC-D2", "Blood: erythrocyte vs megakaryocyte", True),
            ("HSC-D3", "Blood: monocyte vs granulocyte", True)]
    g = d.groupby(["setting", "method"]).auprc_vs_markers.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for i, (s, lab, nested) in enumerate(rows[::-1]):
        if nested: ax.axhspan(i - 0.45, i + 0.45, color="#f3f2ee", lw=0)
        for m, c, off in [("fate_corr", CELLRANK, -0.13), ("OURS_timing_cf", OURS, 0.13)]:
            mu, sd = g.loc[(s, m)]
            ax.hlines(i + off, max(mu - sd, 0), min(mu + sd, 1), color=c, lw=1.4, alpha=0.6)
            ax.plot([mu], [i + off], "o", ms=9, color=c, mec=SURF, mew=1.5)
        mu = g.loc[(s, "ORACLE_early_true_fate"), "mean"]
        ax.plot([mu], [i], marker="D", ms=8, mfc="white", mec=ORACLE, mew=1.8, ls="none")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows[::-1]])
    ax.axhline(2.5, color=INK2, lw=0.8)
    ax.set_xlim(0, 1.02); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Driver-vs-marker AUPRC (dot = mean over 5 replicates, line = ± 1 SD)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=OURS, ms=9, label="Ours (version 1)"),
                       Line2D([], [], marker="o", ls="none", color=CELLRANK, ms=9, label="CellRank-style fate correlation"),
                       Line2D([], [], marker="D", ls="none", mfc="white", mec=ORACLE, mew=1.8, ms=8,
                              label="Oracle (knows future fates)"),
                       plt.Rectangle((0, 0), 1, 1, color="#f3f2ee", label="nested decision")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=9.5)
    ax.set_title("Our method wins clearly on 3 of 7 decisions; the existing method wins on simple splits",
                 loc="left", fontsize=14, fontweight="bold", color=INK, pad=34)
    ax.text(0, -0.17, "* held-out network (not used for tuning); the blood network was also never used for tuning.",
            transform=ax.transAxes, fontsize=9, color=INK2)
    save(fig, "fig4_per_decision")


fig1(); fig2(); fig3(); fig4()
print("done")
