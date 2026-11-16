#!/bin/bash
# Finalize all sessions (holdout evaluation), then build the summary tables.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${SCALPEL_WORK:-$ROOT/work}; RUNS=${SCALPEL_RUNS:-$ROOT/runs}; BUNDLE=${SCALPEL_BUNDLE:-$ROOT/bundle}
A=${SCALPEL_ARCHIVE:-$ROOT/../model-archive}; PUB=${SCALPEL_PUB_TEMP:-$WORK/pub_temp}; P=${PYTHON:-python3}
mkdir -p "$WORK"; export SCALPEL_RUNS=$RUNS SCALPEL_BUNDLE=$BUNDLE SCALPEL_ARCHIVE=$A SCALPEL_WORK=$WORK
cd $ROOT
export OMP_NUM_THREADS=2
for no in 0 2 3 4 13 5 8 12; do
  $P -m scalpel.cli call --run $RUNS/M$no --actor claude-fable-5-2 finalize '{"note":"final evaluation of the study; all eight sessions are finalized together"}' > $WORK/finalize_M$no.json 2>/dev/null
  echo "finalized $no $(date +%H:%M)" >> $WORK/finalize_all.log
done
cd $WORK && $P $ROOT/study/aggregate.py $RUNS $BUNDLE $ROOT/results > $WORK/aggregate.log 2>&1
echo "ALLFINAL $(date +%H:%M)" >> $WORK/finalize_all.log
