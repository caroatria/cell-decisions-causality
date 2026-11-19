#!/usr/bin/env bash
# Install the published competitors CEFCON, CellOracle and CellRank 2, each in its own Python 3.10 environment
# (neither supports Python 3.13, which the benchmark uses). Tested on Linux, CPU only, 3 Nov 2026.
# Needs uv (https://docs.astral.sh/uv/) and git. Creates ./tools/ next to this file. About 6 GB on disk.
#
# Usage:  bash setup_competitors.sh
# Then:   tools/envs/cefcon/bin/python ...        tools/envs/celloracle/bin/python ...
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p tools && cd tools
uv python install 3.10

# ---------------------------------------------------------------- CEFCON (Wang et al. 2023)
# Not on PyPI: install from GitHub. Mouse/human prior networks ship with the repository (prior_data/*.zip).
[ -d CEFCON ] || git clone --depth 1 https://github.com/WPZgithub/CEFCON.git
(cd CEFCON/prior_data && unzip -o -q network_mouse.zip && unzip -o -q network_human.zip)
uv venv -p 3.10 envs/cefcon
# torch 1.13.1 from PyPI (download.pytorch.org may be blocked; the PyPI wheel is the CUDA build, but runs on CPU)
VIRTUAL_ENV=envs/cefcon uv pip install "torch==1.13.1" "torch-geometric==2.3.1" "scanpy>=1.9,<1.11" "networkx>=3.0" \
    "cvxpy>=1.2" "numpy<2" pandas scipy scikit-learn tqdm "pyscenic>=0.12" matplotlib matplotlib-venn seaborn biomart \
    nxviz pyscipopt
VIRTUAL_ENV=envs/cefcon uv pip install --no-deps -e CEFCON
# Notes:
# - Use the Python API (cf.data_preparation / cf.NetModel), not the `cefcon` command: the CLI's genes_DE check
#   compares rows with lineages and fails on its own example.
# - Pass float32 expression (float64 raises "mat1 and mat2 must have the same dtype").
# - The driver step defaults to the commercial Gurobi solver; use driver_regulators(solver="SCIP") (free).
#   With an academic Gurobi licence, `uv pip install gurobipy` and keep the default.

# ---------------------------------------------------------------- CellOracle (Kamimoto et al. 2023)
uv venv -p 3.10 envs/celloracle
VIRTUAL_ENV=envs/celloracle uv pip install "numpy<2" cython "setuptools<70" wheel
VIRTUAL_ENV=envs/celloracle uv pip install --no-build-isolation velocyto      # needs numpy at build time
VIRTUAL_ENV=envs/celloracle uv pip install celloracle==0.20.0 "numpy<2" "setuptools<70"   # gimmemotifs needs pkg_resources
# Notes:
# - Base GRNs download from raw.githubusercontent.com on first use:
#   co.data.load_mouse_scATAC_atlas_base_GRN()  or  co.data.load_mouse_promoter_base_GRN(version="mm10_gimmemotifsv5_fpr2")
# - CellOracle's anndata is older: h5ad files written by the main environment (nullable strings) cannot be read.
#   Write a copy with object-dtype obs/var columns and index first.

# ---------------------------------------------------------------- CellRank 2 (Weiler et al. 2024)
uv venv -p 3.11 envs/cellrank
VIRTUAL_ENV=envs/cellrank uv pip install "cellrank==2.0.6" "scanpy>=1.10" "scipy<1.14"
# Note: scipy>=1.14 removed gmres(tol=...), which CellRank 2.0.6 still passes ("unexpected keyword argument 'tol'").

echo "smoke tests:"
envs/cefcon/bin/python -c "import cefcon, torch, torch_geometric; print('CEFCON ok, torch', torch.__version__)"
envs/celloracle/bin/python -c "import celloracle as co; print('CellOracle ok', co.__version__)" 2>/dev/null | tail -1
envs/cellrank/bin/python -c "import cellrank as cr; print('CellRank ok', cr.__version__)" 2>/dev/null | tail -1
