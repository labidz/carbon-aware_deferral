#!/bin/bash
cd /home/claude/work; R=/home/claude/results
while ! grep -q "^DONE" $R/lsn.log 2>/dev/null; do sleep 5; done
python3 s7_dl.py A > $R/dl_A.log 2>&1
python3 s7_dl.py B > $R/dl_B.log 2>&1
for v in base_vint base_real base_vint_path; do python3 s4b_walkforward.py $v > $R/wf_$v.log 2>&1; done
echo done > $R/chain3.done
