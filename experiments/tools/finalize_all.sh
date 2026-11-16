#!/bin/bash
# Finalize all sessions (holdout evaluation), then build the summary tables.
cd /home/claude/work/scalpel_project
export OMP_NUM_THREADS=2
for no in 0 2 3 4 13 5 8 12; do
  /home/claude/venv/bin/python -m scalpel.cli call --run /home/claude/work/runs/M$no --actor claude-fable-5-2 finalize '{"note":"final evaluation of the study; all eight sessions are finalized together"}' > /home/claude/work/finalize_M$no.json 2>/dev/null
  echo "finalized $no $(date +%H:%M)" >> /home/claude/work/finalize_all.log
done
cd /home/claude/work && /home/claude/venv/bin/python aggregate.py /home/claude/work/runs /home/claude/work/bundle /home/claude/work/results > /home/claude/work/aggregate.log 2>&1
echo "ALLFINAL $(date +%H:%M)" >> /home/claude/work/finalize_all.log
