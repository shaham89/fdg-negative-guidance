# Sourced by sweep7.sh in place of its last arm.  When along the noise path does the high band steer the class?
# Wrong-class direction at the full dose, one noise window at a time.  Cuts follow the agreement of the two network pairs: high for sigma >= 3, low below 0.8.
# 2k samples are enough for a class rate (standard error about 0.011), and FID would mean nothing there, so no metrics.
armq() { min=$1; shift
  if [ $(( $(date +%s) + min * 60 )) -gt $DEADLINE ]; then echo "=== $(date +%H:%M:%S) SKIP $*: would pass the deadline"; return; fi
  echo "=== $(date +%H:%M:%S) $*"; python gen.py "$@" --n 2000 --no_metrics 2>&1 | grep -vE "$F" | tail -1 | cut -c1-40; }
armq 7 highdir --dir wu --w_high 3.67 --lo 8 --hi 1000
armq 7 highdir --dir wu --w_high 3.67 --lo 3 --hi 8
armq 7 highdir --dir wu --w_high 3.67 --lo 0.8 --hi 3
armq 7 highdir --dir wu --w_high 3.67 --lo 0 --hi 0.8
echo "=== $(date +%H:%M:%S) wrong-class rate per window"
python hybrid_diag.py $(ls out/samples_highdir_wu_wh3.67_rc0.125_sig*.npz | sed 's#out/samples_##; s#\.npz##') 2>&1 | grep -vE "$F|Using cache|Downloading"
# The same split of FDG's own high band: which noise range gives its accuracy, its IS, and its FID cost?
arm 18 freqint_wl1_wh3_3-1000 freqint --w_low 1 --w_high 3 --lo 3 --hi 1000
arm 18 freqint_wl1_wh3_0-3    freqint --w_low 1 --w_high 3 --lo 0 --hi 3
# Only if time is left
arm 36 pair_mean_w3_high_rc0.125 pair --w 3 --pair_mode mean --high_only
