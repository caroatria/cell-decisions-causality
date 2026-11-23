"""Build data/real/larry_invitro.h5ad from the LARRY in-vitro downloads (Weinreb et al. 2020, GSE140802).

Inputs (folder given as argument, default data/raw/LARRY), from the GEO / AllonKleinLab release:
  stateFate_inVitro_normed_counts.mtx.gz   cells x genes, library-size normalised counts (as released)
  stateFate_inVitro_gene_names.txt.gz      one gene symbol per line
  stateFate_inVitro_metadata.txt.gz        per-cell table (Library, Cell barcode, Time point, Starting population,
                                           Cell type annotation, Well, SPRING-x, SPRING-y)
  stateFate_inVitro_clone_matrix.mtx.gz    cells x clones, 1 = cell belongs to clone
Output: 130,887 cells x 25,289 genes; X = normed counts (sparse CSR), obs = metadata, obsm["clones"] = clone matrix.
Cells are named c0, c1, ... in file order. larry_eval.py re-normalises and log-transforms, so X is kept as released.

Usage: python experiments/real_data/larry_prepare.py [raw folder]
"""
import sys
from pathlib import Path
import anndata as ad, pandas as pd
from scipy import io, sparse
from driverbench.paths import RAW, REAL

d = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW / "LARRY"
X = sparse.csr_matrix(io.mmread(d / "stateFate_inVitro_normed_counts.mtx.gz")).astype("float32")
genes = pd.read_csv(d / "stateFate_inVitro_gene_names.txt.gz", header=None)[0].astype(str).values
obs = pd.read_csv(d / "stateFate_inVitro_metadata.txt.gz", sep="\t")
clones = sparse.csr_matrix(io.mmread(d / "stateFate_inVitro_clone_matrix.mtx.gz"))
assert X.shape == (len(obs), len(genes)) and clones.shape[0] == len(obs), (X.shape, len(obs), len(genes), clones.shape)
obs.index = [f"c{i}" for i in range(len(obs))]
for c in ("Library", "Cell barcode", "Starting population", "Cell type annotation"):
    obs[c] = obs[c].astype("category")
a = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=pd.Index(genes)))
a.var_names_make_unique()
a.obsm["clones"] = clones
REAL.mkdir(parents=True, exist_ok=True)
a.write_h5ad(REAL / "larry_invitro.h5ad")
print(a, "\nwrote", REAL / "larry_invitro.h5ad")
