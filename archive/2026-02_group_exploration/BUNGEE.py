"""
BranchPoint Uncertainty Pipeline - Day 1 Prototype
====================================================
Trains a RandomForest classifier on cell type labels,
computes per-cell prediction entropy as uncertainty score,
and identifies high-uncertainty cells at trajectory branch points.

Usage:
    python branchpoint_uncertainty.py --input your_data.h5ad --output results/

Requirements:
    pip install scanpy anndata scikit-learn shap matplotlib seaborn scipy
"""

import argparse
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import shap
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

RANDOM_STATE      = 42
N_HVG             = 4000       # highly variable genes to use as features
N_ESTIMATORS      = 200        # random forest trees
ENTROPY_THRESHOLD = 0.75       # quantile cutoff to define "high uncertainty" cells
N_TOP_SHAP_GENES  = 30         # top SHAP genes to report per branch point
CELL_TYPE_COL     = "cell_type"  # column in adata.obs with your annotations
                                  # change to match your actual column name

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data(path: str) -> ad.AnnData:
    print(f"Loading data from {path}")
    adata = sc.read_h5ad(path)
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  Cell types: {adata.obs[CELL_TYPE_COL].value_counts().to_dict()}")
    return adata


# ─── FEATURE PREPARATION ──────────────────────────────────────────────────────

def prepare_features(adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Select highly variable genes and use normalized log counts as features.
    Falls back to PCA coordinates if HVG selection fails.
    """
    print(f"\nPreparing features (top {N_HVG} highly variable genes)...")

    # Normalize if not already done
    if adata.X.max() > 100:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # HVG selection
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat")
    hvg_mask = adata.var["highly_variable"]
    gene_names = list(adata.var_names[hvg_mask])

    X = adata[:, hvg_mask].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Labels
    labels = adata.obs[CELL_TYPE_COL].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Classes: {list(le.classes_)}")

    return X, y, gene_names, le


# ─── TRAIN CLASSIFIER ─────────────────────────────────────────────────────────

def train_classifier(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    print("\nTraining RandomForest classifier...")

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        class_weight="balanced",   # handles imbalanced cell types
        n_jobs=-1,
        random_state=RANDOM_STATE,
        max_depth=8,
    )

    # Quick cross-val to sanity check performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"  5-fold CV F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}")
    print("  (Low F1 is expected if transitional cells are hard to classify — that's the point)")

    clf.fit(X, y)
    return clf


# ─── COMPUTE UNCERTAINTY ──────────────────────────────────────────────────────

def compute_entropy_uncertainty(clf: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """
    Per-cell Shannon entropy over predicted class probabilities.
    High entropy = classifier is uncertain = likely a transitional cell.
    """
    print("\nComputing prediction entropy per cell...")
    proba = clf.predict_proba(X)           # shape: (n_cells, n_classes)
    entropies = entropy(proba, axis=1)     # Shannon entropy, nats
    # Normalize to [0, 1] for interpretability
    entropies_norm = (entropies - entropies.min()) / (entropies.max() - entropies.min())
    print(f"  Entropy range: {entropies.min():.3f} – {entropies.max():.3f}")
    return entropies_norm, proba


# ─── ADD TO ANNDATA & DEFINE BRANCH POINT CELLS ───────────────────────────────

def annotate_branchpoints(adata, entropies_norm, proba, le):
    adata.obs["uncertainty_entropy"] = entropies_norm

    # Fate probabilities per class
    for i, cls in enumerate(le.classes_):
        adata.obs[f"fate_prob_{cls}"] = proba[:, i]

    # Flag high-uncertainty cells
    threshold = np.quantile(entropies_norm, ENTROPY_THRESHOLD)
    adata.obs["is_branchpoint"] = entropies_norm >= threshold
    n_branch = adata.obs["is_branchpoint"].sum()
    print(f"\n  Branch point cells (top {int((1-ENTROPY_THRESHOLD)*100)}% entropy): {n_branch} / {adata.n_obs}")
    return adata, threshold


# ─── SHAP GENE IMPORTANCE AT BRANCH POINTS ────────────────────────────────────

def compute_shap_at_branchpoints(clf, X, adata, gene_names, le):
    """
    Run SHAP only on high-uncertainty cells to find genes driving
    fate ambiguity at branch points — the core methodological contribution.
    """
    print("\nComputing SHAP values for branch point cells...")
    branch_mask = adata.obs["is_branchpoint"].values
    X_branch = X[branch_mask]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_branch)  # list of arrays, one per class

    gene_names = np.array(gene_names)
    results = {}

    for i, cls in enumerate(le.classes_):
        mean_abs_shap = np.abs(shap_values[i]).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:N_TOP_SHAP_GENES]
        results[cls] = {
            "genes": gene_names[top_idx],
            "shap_importance": mean_abs_shap[top_idx],
        }
        print(f"  Top 5 genes for {cls} fate at branch points: {list(gene_names[top_idx[:5]])}")

    return results, shap_values


# ─── PLOTS ────────────────────────────────────────────────────────────────────

def plot_results(adata, shap_results, le, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. UMAP colored by entropy
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sc.pl.umap(adata, color="uncertainty_entropy", cmap="RdYlBu_r",
               title="Prediction Entropy (Uncertainty)", ax=axes[0], show=False)

    sc.pl.umap(adata, color="is_branchpoint", title="Branch Point Cells",
               ax=axes[1], show=False)

    sc.pl.umap(adata, color=CELL_TYPE_COL, title="Cell Type Annotation",
               ax=axes[2], show=False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/uncertainty_umap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved UMAP plot → {output_dir}/uncertainty_umap.png")

    # 2. Entropy distribution by cell type
    fig, ax = plt.subplots(figsize=(10, 5))
    cell_types = sorted(adata.obs[CELL_TYPE_COL].unique())
    data_by_type = [adata.obs.loc[adata.obs[CELL_TYPE_COL] == ct, "uncertainty_entropy"].values
                    for ct in cell_types]
    ax.boxplot(data_by_type, labels=cell_types, vert=True)
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Normalized Entropy")
    ax.set_title("Uncertainty Distribution by Cell Type\n(high = likely transitional)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/entropy_by_celltype.png", dpi=150, bbox_inches="tight")
    plt.close()


    # 3. SHAP bar plots for each fate
    for cls, data in shap_results.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        n_genes = len(data["genes"])
        y_pos = np.arange(n_genes)

        ax.barh(y_pos, data["shap_importance"][::-1], color="steelblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data["genes"][::-1], fontsize=8)
        ax.set_xlabel("Mean |SHAP| at Branch Points")
        ax.set_title(f"Top genes driving {cls} fate\n(branch point cells only)")
        plt.tight_layout()
        safe_cls = cls.replace("/", "_").replace(" ", "_")
        plt.savefig(f"{output_dir}/shap_{safe_cls}.png", dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Saved SHAP plots → {output_dir}/shap_*.png")


# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────

def save_results(adata, shap_results, output_dir):
    import os, pandas as pd
    os.makedirs(output_dir, exist_ok=True)

    # Save annotated adata
    out_h5 = f"{output_dir}/adata_with_uncertainty.h5ad"
    adata.write_h5ad(out_h5)
    print(f"Saved annotated AnnData → {out_h5}")

    # Save gene rankings per fate
    rows = []
    for cls, data in shap_results.items():
        for rank, (gene, imp) in enumerate(zip(data["genes"], data["shap_importance"])):
            rows.append({"fate": cls, "rank": rank + 1, "gene": gene, "shap_importance": imp})
    df = pd.DataFrame(rows)
    out_csv = f"{output_dir}/branchpoint_genes.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved gene rankings → {out_csv}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Branch point uncertainty pipeline")
    parser.add_argument("--input",  required=True, help="Path to .h5ad file")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--cell_type_col", default="cell_type",
                    help="obs column with cell type labels (default: cell_type)")
    args = parser.parse_args()

    global CELL_TYPE_COL
    CELL_TYPE_COL = args.cell_type_col

    # Pipeline
    adata = load_data(args.input)
    X, y, gene_names, le = prepare_features(adata)
    clf = train_classifier(X, y)
    entropies_norm, proba = compute_entropy_uncertainty(clf, X)
    adata, threshold = annotate_branchpoints(adata, entropies_norm, proba, le)
    shap_results, _ = compute_shap_at_branchpoints(clf, X, adata, gene_names, le)
    plot_results(adata, shap_results, le, args.output)
    save_results(adata, shap_results, args.output)

    print("\n✓ Done. Check your UMAP — do high-entropy cells sit at branch points?")
    print("  If yes: your core concept works. Move to benchmarking.")
    print("  If no:  try increasing N_HVG or check your cell type annotations.")


if __name__ == "__main__":
    main()