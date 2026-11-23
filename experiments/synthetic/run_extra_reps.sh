#!/usr/bin/env bash
# Evaluate every method and ablation on the extra held-out replicates BF/TF 5-12 and merge (5 Nov 2026).
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs
BASE="random de_terminal fate_corr fate_corr_decision_region absorption_corr_decision_region rf_shap_fixed obs_counterfactual ORACLE_early_true_fate CellRank"
python -m driverbench.evaluate --methods $BASE --pattern '[BT]F_rep[5-9].h5ad'  --out results/_new_a.csv > logs/extra_evaluate_a.log 2>&1
python -m driverbench.evaluate --methods $BASE --pattern '[BT]F_rep1[0-2].h5ad' --out results/_new_b.csv > logs/extra_evaluate_b.log 2>&1
python3 - <<'PY'
import pandas as pd
d = pd.concat([pd.read_csv("results/_new_a.csv"), pd.read_csv("results/_new_b.csv")])
d[d.method != "CellRank"].to_csv("results/baseline_metrics_new.csv", index=False)
d[d.method == "CellRank"].to_csv("results/cellrank_synthetic_new.csv", index=False)
print("baselines", d.groupby("method").size().to_dict())
PY
echo "baselines done"
ONLY_NEW=1 python experiments/synthetic/eval_v2.py > logs/extra_eval_v2.log 2>&1;               echo "v2 done"
ONLY_NEW=1 python experiments/ablations/ablation_v2.py > logs/extra_ablation_v2.log 2>&1;       echo "ablation done"
ONLY_NEW=1 python experiments/ablations/ablation_v2_extra.py > logs/extra_ablation_v2_extra.log 2>&1; echo "ablation extra done"
ONLY_NEW=1 python experiments/ablations/readout_check.py heldout > logs/extra_readout.log 2>&1; echo "readout done"
python experiments/synthetic/merge_extra_reps.py methods
ONLY_NEW=1 python experiments/synthetic/stress_v2.py > logs/extra_stress_v2.log 2>&1
python experiments/synthetic/merge_extra_reps.py stress
rm -f results/_new_a.csv results/_new_b.csv
echo "ALL DONE"
