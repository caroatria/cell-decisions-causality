#!/usr/bin/env bash
# CEFCON + CellOracle on one LARRY subsample:  bash experiments/real_data/run_larry_competitors.sh <seed 0|1|2>
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue KMP_DUPLICATE_LIB_OK=TRUE
TOOLS=${DRIVERBENCH_TOOLS:-tools}
s=$1; d=cache/competitors/larry_seed$s
mkdir -p logs
[ -f $d/cefcon_scores.csv ] || SEED=$s TAG=c$s METHODS=CEFCON python experiments/real_data/larry_eval.py > logs/larry_competitors_s$s.log 2>&1
[ -f $d/celloracle_scores.csv ] || $TOOLS/envs/celloracle/bin/python competitors/run_celloracle.py $d '{}' > $d/celloracle.log 2>&1
SEED=$s TAG=c$s METHODS=CEFCON,CellOracle python experiments/real_data/larry_eval.py > logs/larry_competitors_s$s.log 2>&1
echo "sample $s EXIT $?" >> logs/larry_competitors_s$s.log
