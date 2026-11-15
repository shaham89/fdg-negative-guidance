#!/bin/bash
# Project 1 follow-up (after sweep2): does the guidance-interval family trace the frequency-split family, and do the two stack?
# 6 configs x ~15 min = ~1.5 h.
cd "$(dirname "$0")/../src"
run() { echo "=== $(date +%H:%M:%S) $*"; python gen.py "$@" 2>&1 | grep -vE 'Warning|^  [0-9]+/' | tail -4; }
# (a) interval frontier: same window as the matching point, other scales  (does w=6 reach the (1,6) split's IS 10.2?)
run interval --w 2 --lo 0.3 --hi 5
run interval --w 4 --lo 0.3 --hi 5
run interval --w 6 --lo 0.3 --hi 5
# (b) stacking test: the (1,3) split restricted to the interval  (same mechanism -> no gain over either alone)
run freqint --w_low 1 --w_high 3 --lo 0.3 --hi 5
# (c) window ablation at w=3
run interval --w 3 --lo 1 --hi 5
run interval --w 3 --lo 0.3 --hi 2
echo "=== $(date +%H:%M:%S) SWEEP5 DONE"
