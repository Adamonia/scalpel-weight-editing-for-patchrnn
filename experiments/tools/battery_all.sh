#!/bin/bash
cd /home/claude/work/scalpel_project
export OMP_NUM_THREADS=1
for no in 4 13 5 8 12; do
  /home/claude/venv/bin/python -m scalpel.cli init --run /home/claude/work/runs/M$no --archive /mnt/user-data/uploads/patchrnn-archive/model-archive --bundle /home/claude/work/bundle --model $no > /dev/null 2>&1
  nice -n 5 /home/claude/venv/bin/python /home/claude/work/battery.py $no ov,census,branch,untr,freq,units,dead,ablate1,bias 2>&1 | grep -v -i warn > /home/claude/work/battery_M$no.log
  echo "done $no $(date +%H:%M)" >> /home/claude/work/battery_all.log
done
