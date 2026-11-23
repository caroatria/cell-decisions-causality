#!/usr/bin/env bash
# Both split probabilities, then the figures (resumable):  bash experiments/tree_depth/run_tree_all.sh
cd "$(dirname "$0")"
bash run_tree.sh 0.8 && bash run_tree.sh 0.5 && python fig_tree_depth.py
