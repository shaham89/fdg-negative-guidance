#!/bin/bash
# "Two dials": does FDG really use the class, or is it a colour fix plus a sharpening filter?
# 15 arms x ~15 min + metrics = ~4-4.5 h.  Control arm doses come from tune_dose.py (n=256): each matches the final sharpness of the (1,3) split.
# Do not re-run calib_sharpen.py before this: the shaped sharpening dose was tuned against the current out/sharpen_table.json.
cd "$(dirname "$0")/../src"
run() { echo "=== $(date +%H:%M:%S) $*"; python gen.py "$@" 2>&1 | grep -vE 'Warning|^  [0-9]+/' | tail -4; }
# Part A: the low band at w=3.  (w_dc, w_lr, w_hi); the other 4 corners are in the baselines below
run bands --w_dc 1 --w_low 3 --w_high 3    # DC frozen
run bands --w_dc 3 --w_low 1 --w_high 3    # only DC guided in the low band
run bands --w_dc 1 --w_low 3 --w_high 1
run bands --w_dc 3 --w_low 1 --w_high 1
# Part B: the high band at w_low=1, at equal sharpness
run highdir --dir cw --w_high 1.966        # same net, wrong class as reference, no uncond net
run highdir --dir wu --w_high 3.67         # direction carries the wrong class
run sharpen --lam 0.0171                   # no class direction, one net, constant lambda
run sharpen --lam 0.182 --table out/sharpen_table.json   # same, lambda(sigma)
# Check: the paper's operator against the FFT mask
run pyramid --w_low 1 --w_high 3
# Baselines again: their sample files stayed on an older VM.  Same seeds, so this also shows how well old numbers reproduce
for w in 1 1.5 2 3; do run cfg --w $w; done
run freq --w_low 1 --w_high 3 --r_c 0.125
run freq --w_low 3 --w_high 1 --r_c 0.125
# Extra metrics, one process per arm so a missing file cannot stop the rest
echo "=== $(date +%H:%M:%S) extra metrics"
for n in cfg_w1 cfg_w1.5 cfg_w2 cfg_w3 freq_wl1_wh3_rc0.125 freq_wl3_wh1_rc0.125 interval_w2_0.3-5 interval_w3_0.3-5 interval_w3_0.3-2 interval_w3_1-5 freqint_wl1_wh3_0.3-5 \
         bands_wd1_wl3_wh3_rc0.125 bands_wd3_wl1_wh3_rc0.125 bands_wd1_wl3_wh1_rc0.125 bands_wd3_wl1_wh1_rc0.125 \
         highdir_cw_wh1.966_rc0.125 highdir_wu_wh3.67_rc0.125 sharpen_const0.0171_rc0.125 sharpen_shaped0.182_rc0.125 pyr_wl1_wh3; do
  python evalx.py --prc "$n" 2>&1 | grep -vE 'Warning|warn' | tail -1
done
echo "=== $(date +%H:%M:%S) SWEEP6 DONE"
date > sweep6.done
