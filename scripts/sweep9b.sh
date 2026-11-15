#!/bin/bash
# Part 2 of sweep9: "when".  Low band only and high band only, pushed only in the early or only in the late steps (split at sigma 0.8: steps 1-11 and 12-18).
# Strengths: low 1, high 2, the ones nearest the correct answer's cat rate (ResNet-56: low 1.2%, high 2.1%, correct answer 1.5%).  Waits for sweep9.
cd "$(dirname "$0")/../src"
while [ ! -f sweep9.done ]; do sleep 20; done      # chained after sweep9.sh; delete this line to run it alone
F='Warning|warn|^  [0-9]+/'
run() { echo "=== $(date +%H:%M:%S) $*"; python neg.py "$@" 2>&1 | grep -vE "$F" | tail -3; }
run arm --k 3 --a 1 --b 0 --lo 0.8
run arm --k 3 --a 1 --b 0 --hi 0.8
run arm --k 3 --a 0 --b 2 --lo 0.8
run arm --k 3 --a 0 --b 2 --hi 0.8
echo "=== $(date +%H:%M:%S) SWEEP9B DONE"
date > sweep9b.done
