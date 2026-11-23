#!/usr/bin/env bash
# CellRank 2 on every benchmark (needs its own environment: bash scripts/setup_competitors.sh).
# Paul15 is optional and not in the paper: set PAUL15=path/to/paul15_processed.h5ad to include it.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p logs
python -m driverbench.evaluate --methods CellRank --out results/cellrank_synthetic.csv > logs/cellrank_synthetic.log 2>&1; echo "syn EXIT $?"
METHODS=CellRank OUT=nestorowa_eval_cellrank.csv python experiments/real_data/nestorowa_eval.py > logs/cellrank_nestorowa.log 2>&1; echo "nes EXIT $?"
for s in 0 1 2; do SEED=$s TAG=r$s METHODS=CellRank python experiments/real_data/larry_eval.py > logs/cellrank_larry_s$s.log 2>&1; echo "larry $s EXIT $?"; done
if [[ -n "${PAUL15:-}" ]]; then
  METHODS=CellRank OUT=paul15_cellrank.csv python -m driverbench.realdata "$PAUL15" > logs/cellrank_paul15.log 2>&1; echo "p15 EXIT $?"
  python experiments/real_data/paul15_tf_competitors.py "$PAUL15" > logs/paul15_tf_competitors.log 2>&1; echo "p15tf EXIT $?"
fi
python experiments/real_data/larry_bootstrap.py > logs/larry_bootstrap.log 2>&1; echo "boot EXIT $?"
