#!/bin/bash
# Sweep 1: baseline CFG Pareto curve + FDG-style static frequency splits.  10k images each, fp16, 18 Heun steps.
cd "$(dirname "$0")/../src"
run() { echo "=== $(date +%H:%M:%S) $*"; python gen.py "$@" 2>&1 | grep -vE 'Warning|^  [0-9]+/' | tail -4; }
for w in 1 1.5 2 3 4 6; do run cfg --w $w; done
# static split at r_c = 0.125 cycles/px (DFT radius 4 = the two lowest bands are "low")
run freq --w_low 1 --w_high 3 --r_c 0.125      # FDG direction: low kept at 1, high boosted
run freq --w_low 1 --w_high 6 --r_c 0.125
run freq --w_low 3 --w_high 1 --r_c 0.125      # the proposal's direction: boost low, keep high at 1
run freq --w_low 2 --w_high 4 --r_c 0.125
run freq --w_low 1.5 --w_high 3 --r_c 0.125
echo "=== $(date +%H:%M:%S) SWEEP1 DONE"
