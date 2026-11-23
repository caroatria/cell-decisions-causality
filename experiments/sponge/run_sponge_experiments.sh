#!/usr/bin/env bash
# Sponge experiments (8 Nov 2026): does the method find clearer, more stable drivers per fate when we
#   a) give each fate its own undecided cells       b) rank on log-odds instead of probability shifts
#   c) roll out longer (K = 5, 10)                    d) follow the lineage tree instead of one 4-fate decision?
# Every experiment uses the same 4 cell subsamples (80% of cells), so results are directly comparable.
# Afterwards: sponge_compare.py prints effect size, stability and direction per fate for every run.
#
# Run from the repository root in the driver-benchmark conda env:   bash run_sponge_experiments.sh
# Re-running skips experiments whose result files already exist (delete them to redo one).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue KMP_DUPLICATE_LIB_OK=TRUE   # avoids the macOS OpenMP crash

# ---------------------------------------------------------------- CHECK: same data and clusters as your earlier runs
DATA=data/raw/suberites_domuncula/suberites_domuncula_cytotrace_SUB4.h5ad
TFS=data/raw/suberites_domuncula/SUB4_TF_gene_ids.csv
CLUSTERS="--cluster-key clusters --stem 8 --transition 0 --choano 10 --sclero 3,4 --digest 6 --skin 1"   # from fig_sponge.py
COMMON="$DATA --tfs $TFS $CLUSTERS --seeds 0,1,2,3 --subsample 0.8 --methods OURS_v2_TF"
# ----------------------------------------------------------------------------------------------------------------

run () {   # run TAG [options]
  local tag=$1; shift
  if ls results/${tag}_perfate_OURS_v2_TF_*_s3.csv >/dev/null 2>&1; then echo "== $tag: done already, skipping"; return; fi
  echo "== $tag ($*)   started $(date +%H:%M)"
  if ! python experiments/sponge/sponge_run.py $COMMON --tag "$tag" "$@" > "logs/${tag}.log" 2>&1; then
    echo "!! $tag failed; last lines of logs/${tag}.log:"; tail -5 "logs/${tag}.log"; exit 1
  fi
  echo "   $tag finished $(date +%H:%M)"
}
mkdir -p logs
run exp_base                                     # the frozen method, as reference
run exp_fateregion   --cell-weight fate_region   # a
run exp_logodds      --score-mode perfate_logodds  # b
run exp_k5           --k 5                       # c
run exp_k10          --k 10                      # c
run exp_tree         --tree                      # d
python experiments/sponge/sponge_compare.py exp_base exp_fateregion exp_logodds exp_k5 exp_k10 exp_tree | tee logs/sponge_compare.txt
