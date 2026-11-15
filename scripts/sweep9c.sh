#!/bin/bash
# "Not a car" (k=1): is the cat result specific to cats?  Same strengths as the windows of sweep9b.  The no-push row comes from neg_none.py, so base runs only the correct answer.  Waits for sweep9b.
cd "$(dirname "$0")/../src"
while [ ! -f sweep9b.done ]; do sleep 20; done     # chained after sweep9b.sh; delete this line to run it alone
F='Warning|warn|^  [0-9]+/'
run() { echo "=== $(date +%H:%M:%S) $*"; python neg.py "$@" 2>&1 | grep -vE "$F" | tail -3; }
run base --k 1
run arm --k 1 --a 1 --b 0
run arm --k 1 --a 0 --b 2
echo "=== $(date +%H:%M:%S) SWEEP9C DONE"
date > sweep9c.done
