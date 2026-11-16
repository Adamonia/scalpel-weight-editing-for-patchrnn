#!/bin/bash
cd /home/claude/work/scalpel_project
export OMP_NUM_THREADS=1
for no in 14 6 7; do
  nice -n 5 /home/claude/venv/bin/python /home/claude/work/battery.py $no ov,census,branch,untr,freq,units,dead,ablate1,bias 2>&1 | grep -v -i warn > /home/claude/work/battery_M$no.log
  echo "done $no $(date +%H:%M)" >> /home/claude/work/battery_new.log
done
