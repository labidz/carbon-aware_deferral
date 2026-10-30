#!/bin/bash
cd /home/claude/work
R=/home/claude/results
for v in no_real no_mix main_w60; do python3 s4b_walkforward.py $v > $R/wf_$v.log 2>&1; done
python3 s8c_forecast_eval.py > $R/forecast_eval.log 2>&1
python3 s9_decisions.py > $R/decisions.log 2>&1
python3 s9b_lsn.py > $R/lsn.log 2>&1
python3 s7_dl.py A > $R/dl_A.log 2>&1
python3 s7_dl.py B > $R/dl_B.log 2>&1
echo done > $R/master.done
