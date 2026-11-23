"""Every folder the code reads from or writes to, in one place.

ROOT is the repository root (two levels above this file). Set DRIVERBENCH_ROOT to use another checkout,
DRIVERBENCH_TOOLS to point at competitor environments installed elsewhere (default ROOT/tools, made by
scripts/setup_competitors.sh)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("DRIVERBENCH_ROOT", Path(__file__).resolve().parents[2]))

DATA = ROOT / "data"
SYNTH = DATA / "synthetic"   # BoolODE benchmark networks (BF, TF, CBF, HSC replicates), made by driverbench.simulate
TREE = DATA / "tree"         # random nested-tree networks, made by experiments/tree_depth/simulate_tree.py
REAL = DATA / "real"         # prepared real datasets (nestorowa16.h5ad, larry_invitro.h5ad)
RAW = DATA / "raw"           # downloads we did not make (LARRY/, nestorowa/, suberites_domuncula/)

RESULTS = ROOT / "results"   # one CSV per experiment; the paper tables and figures are built from these
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
CACHE = ROOT / "cache"       # competitor inputs/outputs (safe to delete; recomputed)
SIMS = ROOT / "sims"         # raw BoolODE trajectories (large, safe to delete after snapshots are built)

EXTERNAL = ROOT / "external"
BOOLODE = EXTERNAL / "BoolODE"                      # git clone, pinned + patched by scripts/setup_boolode.sh
MOUSE_TFS = EXTERNAL / "tfs" / "mm_mgi_tfs.txt"     # pySCENIC mouse TF list (1,721 TFs)
COMPETITOR_RUNNERS = ROOT / "competitors"           # run_<tool>.py, executed inside each tool's own environment
TOOLS = Path(os.environ.get("DRIVERBENCH_TOOLS", ROOT / "tools"))
