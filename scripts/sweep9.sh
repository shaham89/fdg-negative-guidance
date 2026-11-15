#!/bin/bash
# Which band removes a class?  "Not a cat" (k=3), 5k images per arm.  Each arm is saved as soon as it ends, and a re-run skips it, so a VM stop loses only the arm in progress.
cd "$(dirname "$0")/../src"
F='Warning|warn|^  [0-9]+/'
run() { echo "=== $(date +%H:%M:%S) $*"; python neg.py "$@" 2>&1 | grep -vE "$F" | tail -3; }
run base --k 3
for s in 2 1 4 0.5; do   # middle strengths first
  run arm --k 3 --a $s --b $s
  run arm --k 3 --a $s --b 0
  run arm --k 3 --a 0 --b $s
done
echo "=== $(date +%H:%M:%S) SWEEP9 DONE"
date > sweep9.done
