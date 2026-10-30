#!/bin/bash
cd /home/claude/work
while [ ! -f /home/claude/results/master.done ]; do sleep 10; done
for v in base_vint base_real base_vint_path; do python3 s4b_walkforward.py $v > /home/claude/results/wf_$v.log 2>&1; done
echo done > /home/claude/results/chain2.done
