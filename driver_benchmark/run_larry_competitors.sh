#!/usr/bin/env bash
# LARRY competitors for one cell sample, in three steps so CellOracle never shares memory with the evaluation process:
# 1) CEFCON via larry_eval (also exports the shared input), 2) CellOracle standalone, 3) evaluation from the cache.
s=$1; d=cache/competitors/larry_seed$s
[ -f $d/cefcon_scores.csv ] || SEED=$s TAG=c$s METHODS=CEFCON python3 larry_eval.py > logs/larry_competitors_s$s.log 2>&1
[ -f $d/celloracle_scores.csv ] || tools_py=/home/claude/tools/envs/celloracle/bin/python && \
  [ -f $d/celloracle_scores.csv ] || /home/claude/tools/envs/celloracle/bin/python competitors/run_celloracle.py $d '{}' > $d/celloracle.log 2>&1
SEED=$s TAG=c$s METHODS=CEFCON,CellOracle python3 larry_eval.py > logs/larry_competitors_s$s.log 2>&1
echo "sample $s EXIT $?" >> logs/larry_competitors_s$s.log
