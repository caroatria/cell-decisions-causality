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
    return mo, np, pd, sc, sklearn


@app.cell
def _(sc):
    data = sc.read_h5ad("data/suberites_domuncula_cytotrace_SUB4.h5ad")
    return (data,)


@app.cell
def _(data):
    data.var
    return


@app.cell
def _(data):
    data.obs
    return


@app.cell
def _(data):
    data.obs.col
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    0. ???
    1. transition
    2. skeletal (terminal)
    3. skeletal (terminal)
    4. skin (terminal)
    5. stem cell
    6. digestion (6 OR 8 terminal)
    7. ???
    8. digestion (6 OR 8 terminal)
    9. ???
    10. ???
    11. ???
    12. ??? (terminal)

    """
    )
    return


@app.cell
def _(data, sc):
    sc.pl.umap(data, color="clusters")
    return


@app.cell
def _(data, np):
    X = data.obsm["X_pca"]
    y = data.obs[["clusters"]].to_numpy()[:, 0]
    y = np.array(['2' if ele == '3' else ele for ele in y])
    return X, y


@app.cell
def _(X, np, y):
    n_cluster = 15
    average_clusters = {str(i): np.mean(X[y==str(i), :], axis=0) for i in range(n_cluster) if i != 3}
    return (average_clusters,)


@app.cell
def _(X):
    X[0, :].shape
    return


@app.cell
def _(average_clusters):
    average_clusters
    return


@app.cell
def _(X, data, sklearn, y):
    silhouette = sklearn.metrics.silhouette_samples(X, y)
    data_output = data.copy()
    data_output.obs["silhouette"] = silhouette
    return (data_output,)


@app.cell
def _(data_output, sc):
    sc.pl.umap(data_output, color="silhouette", vmax=0.7)
    return


@app.cell
def _():
    return


@app.cell
def _(np, y_merged):
    np.unique(y_merged)
    return


@app.cell
def _():
    return


@app.cell
def _(np):
    def euclidean(a, b):
        return np.sqrt(np.sum((a - b) ** 2))
    return (euclidean,)


@app.cell
def _(euclidean, np):
    def silhouette_adapted(X, y, centers):
        unique_y = np.unique(y)
        output = np.zeros_like(y, dtype=float)
    
        for label in unique_y:
            X_label = X[y==label, :]
            center_label = centers[label]
    
            closest_cluster = None
            closest_cluster_dist = 1000
            for other_label in unique_y:
                if other_label == label:
                    continue
    
                distance = euclidean(center_label, centers[other_label])
            
                if closest_cluster is None:
                    closest_cluster = other_label
                    closest_cluster_dist = distance
                    continue
    
                if closest_cluster_dist > distance:
                    closest_cluster = other_label
                    closest_cluster_dist = distance
                    continue        
    
            output_label = []
            for sample in X_label:
                distance_inside = euclidean(sample, center_label)
    
                distance_closest = np.mean([
                    euclidean(sample, other_sample) for other_sample in X[y==closest_cluster, :]
                ])
            
                silhouette = (distance_closest - distance_inside) / np.max([distance_closest, distance_inside])
                output_label.append(silhouette)

            output[y==label] = output_label
    
        return output
    return (silhouette_adapted,)


@app.cell
def _(np, y):
    y_merged = np.array(['5' if ele == '1' else ele for ele in y])
    return (y_merged,)


@app.cell
def _(X, average_clusters, data, silhouette_adapted, y_merged):
    silhouette_merged = silhouette_adapted(X, y_merged, average_clusters)
    data_output_merged = data.copy()
    data_output_merged.obs["silhouette"] = silhouette_merged
    return (data_output_merged,)


@app.cell
def _(data_output_merged, sc):
    sc.pl.umap(data_output_merged, color="silhouette")
    return


@app.cell
def _(mo):
    mo.md(r"""## Just distance to stem centroid""")
    return


@app.cell
def _(X, average_clusters, euclidean, np):
    distances = [np.log(abs(euclidean(sample, average_clusters["5"]))) for sample in X]
    return (distances,)


@app.cell
def _(data, distances):
    data_output_stem_centroid = data.copy()
    data_output_stem_centroid.obs["silhouette"] = distances
    return (data_output_stem_centroid,)


@app.cell
def _(data_output_stem_centroid, sc):
    sc.pl.umap(data_output_stem_centroid, color="silhouette")
    return


@app.cell
def _(mo):
    mo.md(r"""## Compute Gradient to Terminal cells""")
    return


@app.cell
def _(pd):
    to_human_df = pd.read_excel("data/sd_v4_hs_orthology_250818.xlsx")
    to_human_df
    return (to_human_df,)


@app.cell
def _(to_human_df):
    to_human_df[to_human_df["Sponge gene"] == "SUB4.g12.t1"]["Gene Symbol"].iloc[0]
    return


@app.cell
def _(pd):
    def get_human_gene(sponge_gene, to_human_df):
        gene_name = to_human_df[to_human_df["Sponge gene"] == sponge_gene + ".t1"]["Gene Symbol"].iloc[0]
        if pd.isna(gene_name):
            gene_name = ""

        return gene_name
    return (get_human_gene,)


@app.cell
def _(average_clusters, data):
    terminal_cells = ["2", "4", "6", "8", "12"]
    stem_cell = "5"
    pc_translation_matrix = data.varm["PCs"]

    gene_directions = dict()
    for tc in terminal_cells:
        pc_direction = average_clusters[tc] - average_clusters[stem_cell]
        gene_direction = pc_translation_matrix @ pc_direction
        # gene_direction[gene_direction < 1] = 0
        gene_directions[tc] = gene_direction
    return (gene_directions,)


@app.cell
def _():
    target_terminal = "8"
    return (target_terminal,)


@app.cell
def _(gene_directions, np, target_terminal):
    ten_strongest = np.flip(np.argsort(np.abs(gene_directions[target_terminal]))[-10:])
    values = gene_directions[target_terminal][ten_strongest]
    return ten_strongest, values


@app.cell
def _(data, get_human_gene, ten_strongest, to_human_df, values):
    ten_strongest_names = [data.var.iloc[ele]["gene_ids"] for ele in ten_strongest]
    human_genes = [get_human_gene(ele, to_human_df) for ele in ten_strongest_names]
    human_genes_value = list(zip(human_genes, values))
    human_genes_value
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
