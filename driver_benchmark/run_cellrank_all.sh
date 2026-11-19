#!/usr/bin/env bash
# CellRank 2 on every dataset (4 Nov 2026).
python3 evaluate.py --methods CellRank --out results/cellrank_synthetic.csv > logs/cellrank_synthetic.log 2>&1; echo "syn EXIT $?"
METHODS=CellRank OUT=nestorowa_eval_cellrank.csv python3 nestorowa_eval.py > logs/cellrank_nestorowa.log 2>&1; echo "nes EXIT $?"
for s in 0 1 2; do SEED=$s TAG=r$s METHODS=CellRank python3 larry_eval.py > logs/cellrank_larry_s$s.log 2>&1; echo "larry $s EXIT $?"; done
METHODS=CellRank OUT=results/paul15_cellrank.csv python3 paul15_eval.py /mnt/user-data/uploads/cell-decisions-causality/archive/2026-02_group_exploration/trajectory_analysis_output/paul15_processed.h5ad > logs/cellrank_paul15.log 2>&1; echo "p15 EXIT $?"
python3 paul15_tf_competitors.py /mnt/user-data/uploads/cell-decisions-causality/archive/2026-02_group_exploration/trajectory_analysis_output/paul15_processed.h5ad > logs/paul15_tf_competitors.log 2>&1; echo "p15tf EXIT $?"
python3 larry_bootstrap.py > logs/larry_bootstrap.log 2>&1; echo "boot EXIT $?"
