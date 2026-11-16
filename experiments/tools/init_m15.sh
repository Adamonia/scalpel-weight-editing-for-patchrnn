#!/bin/bash
cd /home/claude/work/scalpel_project
A=/mnt/user-data/uploads/patchrnn-archive/model-archive
T=/home/claude/work/tokenizers/roberta
export OMP_NUM_THREADS=1
/home/claude/venv/bin/python -m scalpel.textbundle /home/claude/ext/pub_temp /home/claude/work/bundle /home/claude/work/bundle_codebert $T/vocab.json $T/merges.txt 2>&1 | grep -v -i warn
echo BUNDLEDONE
/home/claude/venv/bin/python -m scalpel.cli init --run /home/claude/work/runs/M15 --archive $A --bundle /home/claude/work/bundle_codebert --model 15 2>&1 | grep -v -i warn
echo INITDONE
