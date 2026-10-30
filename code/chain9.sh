#!/bin/bash
cd /home/claude/work; R=/home/claude/results
for seed in 1 2; do for mode in A B; do [ -f $R/dl_${mode}_r${seed}.json ] || SEED=$seed DROPOUT=0.3 L2=0.0001 REL=1 TAG=_r$seed python3 s7_dl.py $mode > $R/dl_${mode}_r${seed}.log 2>&1; done; done
touch $R/chain9.done
