"""Build data_real/nestorowa16.h5ad from the raw downloads (Nestorowa et al. 2016, GSE81682, Smart-seq2).
Inputs (folder given as argument): GSE81682_HTSeq_counts.txt.gz, all_cell_types.txt, cluster_ids.txt,
MRK_ENSEMBL.rpt (MGI Ensembl -> symbol). Keeps the 1,656 cells with a paper cluster id, Ensembl genes only,
counts of Ensembl IDs mapping to the same symbol are summed. X = raw counts.
Usage: python nestorowa_prepare.py path/to/nestorowa_folder"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from scipy import sparse
d = Path(sys.argv[1])
counts = pd.read_csv(d / "GSE81682_HTSeq_counts.txt.gz", sep="\t", index_col=0)
counts = counts.loc[counts.index.str.startswith("ENS")]
cl = pd.read_csv(d / "cluster_ids.txt", sep=" ", header=None, index_col=0, names=["cluster"])
ct = pd.read_csv(d / "all_cell_types.txt", sep="\t", index_col=0)
mgi = pd.read_csv(d / "MRK_ENSEMBL.rpt", sep="\t", header=None, usecols=[1, 5], names=["symbol", "ensembl"]).dropna()
e2s = dict(zip(mgi.ensembl, mgi.symbol))
cells = [c for c in cl.index if c in counts.columns]
counts = counts[cells]
sym = counts.index.map(lambda e: e2s.get(e.split(".")[0], e))
print(f"{counts.shape[0]} Ensembl genes, {np.mean([s != e for s, e in zip(sym, counts.index)]):.1%} mapped to symbols")
counts = counts.groupby(sym).sum()
counts = counts.loc[counts.sum(1) > 0]
broad = ct.loc[cells].filter(like="_broad")
gate = broad.idxmax(1).where(broad.sum(1) > 0, "none").str.replace("_broad", "")
obs = pd.DataFrame({"cluster": cl.loc[cells, "cluster"].values, "gate": gate.values}, index=cells)
a = ad.AnnData(sparse.csr_matrix(counts.T.values.astype(np.float32)), obs=obs, var=pd.DataFrame(index=counts.index))
out = Path(__file__).resolve().parent / "data_real" / "nestorowa16.h5ad"
a.write_h5ad(out)
print(a, "\n", pd.crosstab(a.obs.gate, a.obs.cluster), "\nwrote", out)
