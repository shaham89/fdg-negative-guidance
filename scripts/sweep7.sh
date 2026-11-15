#!/bin/bash
# Overnight, after sweep6: which part of the guidance is real, and does only the real part help?
# Does not start an arm that would end after $DEADLINE.  Each arm is saved as soon as it ends, so a VM stop loses only the arm in progress.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$0")/../src"
while [ ! -f sweep6.done ] && pgrep -f sweep6.sh >/dev/null; do sleep 60; done   # wait while sweep6.sh is still running
# The original run used DEADLINE="2026-11-02 08:00", when the VM was due to stop.  Set DEADLINE to limit a new run the same way.
DEADLINE=$(date -d "${DEADLINE:-2100-01-01}" +%s)
F='Warning|warn|^  [0-9]+/'
# arm <minutes> <arm name> <gen.py args...>
arm() { min=$1; n=$2; shift 2
  if [ $(( $(date +%s) + min * 60 )) -gt $DEADLINE ]; then echo "=== $(date +%H:%M:%S) SKIP $n: would pass the deadline"; return; fi
  echo "=== $(date +%H:%M:%S) $n: $*"; python gen.py "$@" 2>&1 | grep -vE "$F" | tail -4
  python evalx.py --prc "$n" 2>&1 | grep -vE "$F" | tail -1; }
echo "=== $(date +%H:%M:%S) diagnostics along the (1,3) path"
python source_diag.py --w_high 3 2>&1 | grep -vE "$F"
python replicate_diag.py --w_high 3 2>&1 | grep -vE "$F"
# Wiener guidance: shrink guidance by how well two independent pairs agree on it.  No tuned window or cutoff in the full-band arms
arm 18 wiener_w3_high_rc0.125 wiener --w 3 --high_only
arm 18 wiener_w6_high_rc0.125 wiener --w 6 --high_only
arm 18 wiener_w3              wiener --w 3
arm 18 wiener_w6              wiener --w 6
# Guidance direction from the other pair: only signal shared by both pairs can help the sampling net
arm 28 pair_cross_w3_high_rc0.125 pair --w 3 --high_only
arm 28 pair_cross_w3              pair --w 3
# Class part against gap part of FDG's high band.  11 net calls per evaluation, so 4k samples; compare with the first 4k of the baselines.
# The two only mean something together: the first starts only if both fit
arm 100 src_wc3_wg1_high_rc0.125 src --w_cls 3 --w_gap 1 --high_only --n 4000
if [ -f out/samples_src_wc3_wg1_high_rc0.125.npz ]; then arm 50 src_wc1_wg3_high_rc0.125 src --w_cls 1 --w_gap 3 --high_only --n 4000; fi
echo "=== $(date +%H:%M:%S) baselines on their first 4k"
for n in cfg_w1 cfg_w2 cfg_w3 freq_wl1_wh3_rc0.125; do python evalx.py --prc --fid --first 4000 "$n" 2>&1 | grep -vE "$F" | tail -1; done
# Last, only if time is left
source "$HERE/sweep7b.sh"  # noise windows, then pair_mean if time is left    
echo "=== $(date +%H:%M:%S) SWEEP7 DONE"
date > sweep7.done
