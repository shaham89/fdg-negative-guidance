#!/bin/bash
# Sweep 2 (after sweep1): the falsification test + the SNR-derived freq x time rule + interval baseline.
cd "$(dirname "$0")/../src"
run() { echo "=== $(date +%H:%M:%S) $*"; python gen.py "$@" 2>&1 | grep -vE 'Warning|^  [0-9]+/' | tail -4; }
# (a) equivalent time-only schedules of the static splits from sweep1
for spec in "1 3" "1 6" "3 1" "2 4" "1.5 3"; do
  set -- $spec
  python make_time_sched.py freq --w_low $1 --w_high $2 --r_c 0.125 > /dev/null
  run time --table out/time_from_freq_wl$1_wh$2_rc0.125.json
done
# (b) SNR-derived freq x time rules (mu=0 is the parameter-free default)
run ft --w_min 1 --w_max 3 --mode unresolved --mu 0
run ft --w_min 1 --w_max 3 --mode resolved --mu 0
run ft --w_min 1 --w_max 3 --mode bump --mu 0 --s 1.5
run ft --w_min 1 --w_max 6 --mode unresolved --mu 0
run ft --w_min 1 --w_max 3 --mode unresolved --mu -2
run ft --w_min 1 --w_max 3 --mode unresolved --mu 2
# (c) interval guidance baseline (Kynkaanniemi et al.), mid-sigma window
run interval --w 3 --lo 0.3 --hi 5
# (d) E1.8 housekeeping: clip fraction for w=1, 1.5 (lost when the metrics stage crashed on the first run)
run cfg --w 1 --n 1000 --bs 500 --no_metrics --tag clip
run cfg --w 1.5 --n 1000 --bs 500 --no_metrics --tag clip
echo "=== $(date +%H:%M:%S) SWEEP2 DONE"
