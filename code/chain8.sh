#!/bin/bash
cd /home/claude/work; R=/home/claude/results
[ -f $R/gb_synthetic.json ] || python3 s12_extras.py >> $R/extras.log 2>&1
[ -f $R/extras2_store.npz ] || python3 s12c_more_decisions.py >> $R/extras2.log 2>&1
[ -f $R/forecast_extras.json ] || python3 s12d_forecast_extras.py > $R/forecast_extras.log 2>&1
[ -f $R/aci.json ] || python3 s8e_aci.py > $R/aci.log 2>&1
for seed in 1 2 3; do for mode in A B; do [ -f $R/dl_${mode}_r${seed}.json ] || SEED=$seed DROPOUT=0.3 L2=0.0001 REL=1 TAG=_r$seed python3 s7_dl.py $mode > $R/dl_${mode}_r${seed}.log 2>&1; done; done
touch $R/chain8.done
