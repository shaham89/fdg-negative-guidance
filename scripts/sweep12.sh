#!/bin/bash
# Follow-up questions (2026-11-13), most valuable first.  About 3.5 h on an A100.
#  1. post_diag: which band's fit knows the class, and when
#  2. pmap: push one band in one window, toward or away, on the D_u path; also the band transfer
#  3. the exact rule with p fitted per band, against the whole-image fit (5k rows from sweep11: (1,1) 4.2% cats, FID 7.02)
#  4. "c, not k": low band pushed only early with the mean colour left alone, high band at all steps
#  5. seeds 1, 2 of the exact rule (1,1), and the per-band fit at a higher strength
# After every step: out/neg without the sample files goes to ../small.tar, so results can be copied off the machine while it runs.
cd "$(dirname "$0")/../src"
F='Warning|warn|^  [0-9]+/'
snap() { tar -cf ../small.tar.tmp --exclude='samples_*' out/neg sweep12.log 2>/dev/null && mv ../small.tar.tmp ../small.tar; }
run() { echo "=== $(date +%H:%M:%S) neg.py $*"; python neg.py "$@" 2>&1 | grep -vE "$F" | tail -3; snap; }
echo "=== $(date +%H:%M:%S) post_diag"; python post_diag.py --n 2000 2>&1 | grep -vE "$F"; snap
echo "=== $(date +%H:%M:%S) pmap"; python pmap.py --n 4000 --s 2 2>&1 | grep -vE "$F"; snap
run base --k 3
for pf in band low high; do run exact --k 3 --a 1 --b 1 --pfit $pf; done
run posneg --c 5 --k 3 --w_low 2 --w_high 2 --low_lo 2.5 --dc 1
run posneg --c 5 --k 3 --w_low 2 --w_high 2 --low_lo 4 --dc 1
run posneg --c 5 --k 3 --w_low 3 --w_high 2 --low_lo 4 --dc 1
run posneg --c 3 --k 5 --w_low 2 --w_high 2 --low_lo 2.5 --dc 1
run posneg --c 9 --k 1 --w_low 2 --w_high 2 --low_lo 2.5 --dc 1
run posneg --c 9 --k 1 --w_low 2 --w_high 1
run posneg --c 9 --k 1 --w_low 2 --w_high 2
for sd in 1 2; do
  run base --k 3 --seed $sd
  run exact --k 3 --a 1 --b 1 --seed $sd
done
run exact --k 3 --a 1.5 --b 1.5 --pfit band
echo "=== $(date +%H:%M:%S) diag k=3"; python neg_diag.py --k 3 --near 5 2>&1 | grep -vE "$F"; snap
tar -cf ../sweep12_out.tar out/neg sweep12.log
echo "=== $(date +%H:%M:%S) SWEEP12 DONE"; snap
