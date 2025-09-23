import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import scanpy as sc
    import numpy as np
    import sklearn
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import copy
    return copy, np, pd, plt, sc, sklearn, sns


@app.cell
def _(sc):
    data = sc.read_h5ad("data/suberites_domuncula_cytotrace_SUB4.h5ad")
    return (data,)


@app.cell
def _():
    terminals = ["2", "4", "6", "8", "12"]
    transitions = ["1", "5"]
    return terminals, transitions


@app.cell
def _(data, np):
    X = data.obsm["X_pca"]
    y = data.obs[["clusters"]].to_numpy()[:, 0]
    y = np.array(['2' if ele == '3' else ele for ele in y])
    return X, y


@app.cell
def _(X, np, terminals, transitions, y):
    transitions_mask = np.logical_or.reduce([y == t for t in transitions]) 
    X_transitions = X[transitions_mask]
    y_transitions = y[transitions_mask]

    terminals_mask = np.logical_or.reduce([y == t for t in terminals]) 
    X_terminals = X[terminals_mask]
    y_terminals = y[terminals_mask]
    return (
        X_terminals,
        X_transitions,
        terminals_mask,
        transitions_mask,
        y_terminals,
    )


@app.cell
def _(X_transitions):
    X_transitions.shape
    return


@app.cell
def _(X_terminals, np, y_terminals):
    rng = np.random.default_rng(seed=42)
    order = rng.permutation(X_terminals.shape[0])
    X_terminals_shuffled = X_terminals[order]
    y_terminals_shuffled = y_terminals[order]
    return X_terminals_shuffled, order, y_terminals_shuffled


@app.cell
def _(sklearn):
    classifiers = [
        (
    
            sklearn.ensemble.RandomForestClassifier(n_estimators=10000),
            "RF"
        ),
        # (
        #     sklearn.neural_network.MLPClassifier(),
        #     "MLP"
        # )
    ]
    return (classifiers,)


@app.cell
def _(X_terminals_shuffled, X_transitions, classifiers, y_terminals_shuffled):
    results = dict()
    class_order = dict()
    for classifier, name in classifiers:
        classifier.fit(X_terminals_shuffled, y_terminals_shuffled)
        res = classifier.predict_proba(X_transitions)
        print(res)

        results[name] = res
        class_order = classifier.classes_
    return classifier, res


@app.cell
def _(data, order, sns):
    sns.scatterplot(x=data.obsm["X_umap"][:, 0], y=data.obsm["X_umap"][order][:, 1])
    return


@app.cell
def _(data):
    data.obsm["X_umap"][:, 0]
    return


@app.cell
def _(data, order, terminals_mask, transitions_mask):
    X_umap_transitions = data.obsm["X_umap"][transitions_mask]

    X_umap_terminals = data.obsm["X_umap"][terminals_mask]
    X_umap_terminals_shuffled = X_umap_terminals[order]
    return X_umap_terminals_shuffled, X_umap_transitions


@app.cell
def _(X_umap_terminals_shuffled, sns):
    sns.scatterplot(x = X_umap_terminals_shuffled[:, 0], y=X_umap_terminals_shuffled[:, 1])
    return


@app.cell
def _(copy):
    colors = {
        '2': [31/255, 119/255, 180/255],
        '3': [140/255, 86/255, 75/255],
        '4': [255/255, 127/255, 14/255],
        '6': [44/255, 160/255, 44/255],
        '8': [214/255, 39/255, 40/255],
        '12': [148/255, 103/255, 189/255]
    }



    def probability_to_alpha(probabiltiy):
        return ((probabiltiy - 1/5) / (1 - 1/5))


    def get_color(colors, label, proba):
        color_copy = copy.copy(colors[label])
        color_copy.append(probability_to_alpha(proba))
        return color_copy
    return colors, get_color


@app.cell
def _(colors, np, y_terminals_shuffled):
    terminals_colors = np.array([colors[ele] for ele in y_terminals_shuffled])
    return (terminals_colors,)


@app.cell
def _(classifiers, np, res):
    transitions_predicted_idx = np.argmax(res, axis=1)
    transitions_predicted_label = classifiers[0][0].classes_[transitions_predicted_idx]
    transitions_predicted_proba = res[np.arange(res.shape[0]), transitions_predicted_idx]
    return transitions_predicted_label, transitions_predicted_proba


@app.cell
def _(
    colors,
    get_color,
    transitions_predicted_label,
    transitions_predicted_proba,
):
    transitions_colors = [get_color(colors, label, proba) for label, proba in zip(transitions_predicted_label, transitions_predicted_proba)]
    return (transitions_colors,)


@app.cell
def _(
    X_umap_terminals_shuffled,
    X_umap_transitions,
    plt,
    terminals_colors,
    transitions_colors,
):
    plt.figure(figsize=(20, 10))
    plt.scatter(
        x=X_umap_terminals_shuffled[:, 0],
        y=X_umap_terminals_shuffled[:, 1],
        c=terminals_colors
    )
    plt.scatter(
        x=X_umap_transitions[:, 0],
        y=X_umap_transitions[:, 1],
        c=transitions_colors
    )
    return


@app.cell
def _(classifier):
    classifier.feature_importances_
    return


@app.cell
def _(classifiers, data):
    most_important_genes = classifiers[0][0].feature_importances_ @ data.varm["PCs"].T
    return (most_important_genes,)


@app.cell
def _(most_important_genes, np):
    n_best = np.argsort(most_important_genes)[-10:]
    return (n_best,)


@app.cell
def _(pd):
    to_human_df = pd.read_excel("data/sd_v4_hs_orthology_250818.xlsx")
    to_human_df
    return (to_human_df,)


@app.cell
def _(pd):
    def get_human_gene(sponge_gene, to_human_df):
        gene_name = to_human_df[to_human_df["Sponge gene"] == sponge_gene + ".t1"]["Gene Symbol"].iloc[0]
        if pd.isna(gene_name):
            gene_name = ""

        return gene_name
    return (get_human_gene,)


@app.cell
def _(data, get_human_gene, n_best, to_human_df):
    n_best_names = [data.var.iloc[ele]["gene_ids"] for ele in n_best]
    human_genes = [(get_human_gene(ele, to_human_df), ele) for ele in n_best_names]
    human_genes
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
