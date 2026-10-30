#!/bin/bash
cd /home/claude/work; R=/home/claude/results
python3 s4c_causal.py > $R/causal.log 2>&1
python3 s4d_standalone.py > $R/standalone.log 2>&1
for seed in 1 2 3; do for mode in A B; do SEED=$seed DROPOUT=0.3 L2=0.0001 REL=1 TAG=_r$seed python3 s7_dl.py $mode > $R/dl_${mode}_r$seed.log 2>&1; done; done
echo done > $R/chain6.done
