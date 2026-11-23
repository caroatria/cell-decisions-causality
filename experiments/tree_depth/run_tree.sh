#!/usr/bin/env bash
# Depth experiment (17 Nov 2026): simulate nested-tree networks and evaluate all methods.
# Usage: bash experiments/tree_depth/run_tree.sh 0.8   (then 0.5). Re-running skips finished networks and decisions.
set -e
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue KMP_DUPLICATE_LIB_OK=TRUE
P=$1; TAG=T$(python3 -c "print(f'{round($P*10):02d}')")
mkdir -p logs
python experiments/tree_depth/simulate_tree.py --p "$P" --depths 1 2 3 4 --n 10 2>&1 | grep -v -i warn | tee -a "logs/tree_sim_$TAG.log"
python experiments/tree_depth/eval_tree.py --pattern "${TAG}_*.h5ad" 2>&1 | grep -v -i warn | tee -a "logs/tree_eval_$TAG.log"
echo "ALL DONE $TAG"
