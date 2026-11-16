#!/bin/bash
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${SCALPEL_WORK:-$ROOT/work}; RUNS=${SCALPEL_RUNS:-$ROOT/runs}; BUNDLE=${SCALPEL_BUNDLE:-$ROOT/bundle}
A=${SCALPEL_ARCHIVE:-$ROOT/../model-archive}; PUB=${SCALPEL_PUB_TEMP:-$WORK/pub_temp}; P=${PYTHON:-python3}
mkdir -p "$WORK"; export SCALPEL_RUNS=$RUNS SCALPEL_BUNDLE=$BUNDLE SCALPEL_ARCHIVE=$A SCALPEL_WORK=$WORK
cd $ROOT
$P -m scalpel.cli init --run $RUNS/M6 --archive $A --bundle ${BUNDLE}_bert --model 6 2>&1 | grep -v -i warn
$P -m scalpel.cli init --run $RUNS/M7 --archive $A --bundle ${BUNDLE}_roberta --model 7 2>&1 | grep -v -i warn
echo INITDONE
