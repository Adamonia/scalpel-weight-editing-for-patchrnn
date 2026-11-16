#!/bin/bash
# Base stage outputs for each model, one process for each model.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${SCALPEL_WORK:-$ROOT/work}; RUNS=${SCALPEL_RUNS:-$ROOT/runs}; BUNDLE=${SCALPEL_BUNDLE:-$ROOT/bundle}
A=${SCALPEL_ARCHIVE:-$ROOT/../model-archive}; PUB=${SCALPEL_PUB_TEMP:-$WORK/pub_temp}; P=${PYTHON:-python3}
mkdir -p "$WORK"; export SCALPEL_RUNS=$RUNS SCALPEL_BUNDLE=$BUNDLE SCALPEL_ARCHIVE=$A SCALPEL_WORK=$WORK
cd $ROOT
for no in 3 4 13 5 8 12; do
  nice -n 10 $P $ROOT/experiments/tools/precompute_base.py $no 2>&1 | grep -v Warn
done
echo ALLDONE
