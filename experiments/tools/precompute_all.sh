#!/bin/bash
# Base stage outputs for each model, one process for each model.
cd /home/claude/work
for no in 3 4 13 5 8 12; do
  nice -n 10 /home/claude/venv/bin/python precompute_base.py $no 2>&1 | grep -v Warn
done
echo ALLDONE
