#!/bin/bash
cd /home/claude/work; R=/home/claude/results
python3 s8c_forecast_eval.py > $R/forecast_eval.log 2>&1
python3 s8d_dl_eval.py > $R/dl_eval.log 2>&1
python3 s11_overhead.py > $R/overhead.log 2>&1
python3 fig_c_forecast.py > $R/figc.log 2>&1
echo done > $R/chain5.done
