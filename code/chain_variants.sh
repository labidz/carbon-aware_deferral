#!/bin/bash
cd /home/claude/work
while pgrep -f "s4b_walkforward.py main$" > /dev/null; do sleep 5; done
for v in no_vint no_adapt base_only base_adapt relative main_w60 no_real no_mix; do
  python3 s4b_walkforward.py $v > /home/claude/results/wf_$v.log 2>&1
done
echo chain_done > /home/claude/results/chain.done
