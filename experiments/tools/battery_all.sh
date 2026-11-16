#!/bin/bash
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${SCALPEL_WORK:-$ROOT/work}; RUNS=${SCALPEL_RUNS:-$ROOT/runs}; BUNDLE=${SCALPEL_BUNDLE:-$ROOT/bundle}
A=${SCALPEL_ARCHIVE:-$ROOT/../model-archive}; PUB=${SCALPEL_PUB_TEMP:-$WORK/pub_temp}; P=${PYTHON:-python3}
mkdir -p "$WORK"; export SCALPEL_RUNS=$RUNS SCALPEL_BUNDLE=$BUNDLE SCALPEL_ARCHIVE=$A SCALPEL_WORK=$WORK
cd $ROOT
export OMP_NUM_THREADS=1
for no in 4 13 5 8 12; do
  $P -m scalpel.cli init --run $RUNS/M$no --archive $A --bundle $BUNDLE --model $no > /dev/null 2>&1
  nice -n 5 $P $ROOT/study/battery.py $no ov,census,branch,untr,freq,units,dead,ablate1,bias 2>&1 | grep -v -i warn > $WORK/battery_M$no.log
  echo "done $no $(date +%H:%M)" >> $WORK/battery_all.log
done
