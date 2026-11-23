#!/usr/bin/env bash
# Reproduce the main synthetic benchmark (simulation, baselines, our method, ablation inputs) and the figures notebook.
#   bash scripts/run_all.sh            full pipeline (simulation ~2-3 h on 2 cores, evaluation ~30 min)
#   bash scripts/run_all.sh --no-sim   reuse data/synthetic/*.h5ad, rerun methods + notebook
#   bash scripts/run_all.sh --figures  only re-execute the figures notebook from results/*.csv (seconds)
# Real data, sponge, tree-depth and TF-prior experiments have their own run scripts in experiments/*/ (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue KMP_DUPLICATE_LIB_OK=TRUE   # avoids the macOS OpenMP crash
PY=${PYTHON:-python}
MODE=${1:-all}
mkdir -p data/synthetic results figures logs

if [[ "$MODE" == "all" ]]; then
  bash scripts/setup_boolode.sh
  # simulate: 3 synthetic networks + curated HSC network, 5 replicates x 500 trajectories each
  # (replicates 5-12 of BF/TF: experiments/synthetic/run_extra_reps.sh)
  $PY -m driverbench.simulate --models BF TF CBF --reps 5 --n-traj 500 | tee logs/sim_synthetic.txt
  $PY -m driverbench.simulate --models HSC --reps 5 --n-traj 500     | tee logs/sim_hsc.txt
fi

if [[ "$MODE" == "all" || "$MODE" == "--no-sim" ]]; then
  BASE="random de_terminal fate_corr fate_corr_decision_region absorption_corr_decision_region rf_shap_fixed obs_counterfactual ORACLE_early_true_fate PROBE_interventional_linear"
  # baselines on the synthetic networks (BF, TF, CBF)
  $PY -m driverbench.evaluate --methods $BASE --pattern '*F_rep*.h5ad' --out results/baseline_metrics.csv
  # dev-only tuning of the onset threshold (BF + TF replicates 0-2)
  $PY experiments/synthetic/sweep_timing.py
  # version 1 of our method (kept for the paper's history)
  $PY -m driverbench.evaluate --methods OURS_timing_cf --pattern '*F_rep*.h5ad' --out results/ours_timing_all.csv
  # all methods on the curated HSC network (never used for tuning)
  $PY -m driverbench.evaluate --methods $BASE OURS_timing_cf --pattern 'HSC_rep*.h5ad' --out results/hsc_metrics.csv
  # speed-ups: strict fast version vs exact
  $PY -m driverbench.evaluate --methods OURS_timing_cf_stress --pattern '*F_rep*.h5ad' --out results/ours_stress_synthetic.csv
  $PY -m driverbench.evaluate --methods OURS_timing_cf_stress --pattern 'HSC_rep*.h5ad' --out results/ours_stress_hsc.csv
  # version 2 (K and scoring rule chosen on dev data) + held-out test
  $PY experiments/synthetic/horizon_check.py benchmark
  $PY -m driverbench.score_modes benchmark
  $PY experiments/synthetic/eval_v2.py
  $PY experiments/synthetic/stress_v2.py
  # ablations
  $PY experiments/ablations/ablation_v2.py
  $PY experiments/ablations/ablation_v2_extra.py
  $PY experiments/ablations/readout_check.py heldout
  # runtime scaling (~30 min)
  $PY experiments/synthetic/scaling_v2.py
fi

# tables + figures -> figures/
jupyter nbconvert --to notebook --execute --inplace notebooks/results_and_figures.ipynb
echo "done: results/ and figures/ updated"
