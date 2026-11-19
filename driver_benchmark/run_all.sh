#!/usr/bin/env bash
# Reproduce every result and figure of the driver-vs-marker benchmark.
#   bash run_all.sh            full pipeline (simulation ~2-3 h on 2 cores, evaluation ~30 min)
#   bash run_all.sh --no-sim   reuse data/*.h5ad, rerun methods + figures only
#   bash run_all.sh --figures  only re-execute the figures notebook from results/*.csv (seconds)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
MODE=${1:-all}
mkdir -p data results figures logs

if [[ "$MODE" == "all" ]]; then
  # 1. BoolODE, pinned commit + our 3 small patches (see README)
  if [[ ! -d external/BoolODE ]]; then
    git clone https://github.com/Murali-group/BoolODE external/BoolODE
    git -C external/BoolODE checkout "$(cat boolode_commit.txt)"
    git -C external/BoolODE apply ../../boolode_patches.diff
  fi
  # 2. simulate: 3 synthetic networks + curated HSC network, 5 replicates x 500 trajectories each
  $PY simulate.py --models BF TF CBF --reps 5 --n-traj 500 | tee logs/sim_synthetic.txt
  $PY simulate.py --models HSC --reps 5 --n-traj 500     | tee logs/sim_hsc.txt
fi

if [[ "$MODE" == "all" || "$MODE" == "--no-sim" ]]; then
  BASE="random de_terminal fate_corr fate_corr_decision_region absorption_corr_decision_region rf_shap_fixed obs_counterfactual ORACLE_early_true_fate PROBE_interventional_linear"
  # 3. baselines on the synthetic networks (BF, TF, CBF)
  $PY evaluate.py --methods $BASE --pattern '*F_rep*.h5ad' --out results/baseline_metrics.csv
  # 4. dev-only tuning of the onset threshold (BF + TF replicates 0-2)
  $PY sweep_timing.py
  # 5. our method on the synthetic networks (CBF = held-out test set)
  $PY evaluate.py --methods OURS_timing_cf --pattern '*F_rep*.h5ad' --out results/ours_timing_all.csv
  # 6. all methods on the curated HSC network (never used for tuning)
  $PY evaluate.py --methods $BASE OURS_timing_cf --pattern 'HSC_rep*.h5ad' --out results/hsc_metrics.csv
  # 7. speed-ups: strict fast version (30 genes, 100 cells) vs exact on every decision
  $PY evaluate.py --methods OURS_timing_cf_stress --pattern '*F_rep*.h5ad' --out results/ours_stress_synthetic.csv
  $PY evaluate.py --methods OURS_timing_cf_stress --pattern 'HSC_rep*.h5ad' --out results/ours_stress_hsc.csv
  # 7b. version 2 (K and scoring rule chosen on dev data) + held-out test
  $PY horizon_check.py benchmark
  $PY score_modes.py benchmark
  $PY eval_v2.py
  # 8. runtime scaling (exact up to 800 genes, fast up to 4000 genes; ~30 min)
  $PY scaling.py
fi

# 9. tables + figures 1-5 -> figures/
jupyter nbconvert --to notebook --execute --inplace notebooks/results_and_figures.ipynb
echo "done: results/ and figures/ updated"
