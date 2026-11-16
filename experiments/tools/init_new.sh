#!/bin/bash
cd /home/claude/work/scalpel_project
A=/mnt/user-data/uploads/patchrnn-archive/model-archive
/home/claude/venv/bin/python -m scalpel.cli init --run /home/claude/work/runs/M6 --archive $A --bundle /home/claude/work/bundle_bert --model 6 2>&1 | grep -v -i warn
/home/claude/venv/bin/python -m scalpel.cli init --run /home/claude/work/runs/M7 --archive $A --bundle /home/claude/work/bundle_roberta --model 7 2>&1 | grep -v -i warn
echo INITDONE
