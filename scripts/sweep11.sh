#!/bin/bash
# Colab batch 2: confirm the exact rule at 5k, combine the cheap tricks, "c, not k" on 3 more pairs, seeds for the exact rule.  About 45 runs, 3.6 h on an A100.
# Each run is saved as soon as it ends and a re-run skips it.  At the end: paired diagnostics, and one tar with rows, grids, diagnostics and samples, to copy off the machine.
cd "$(dirname "$0")/../src"
F='Warning|warn|^  [0-9]+/'
run() { echo "=== $(date +%H:%M:%S) $*"; python neg.py "$@" 2>&1 | grep -vE "$F" | tail -3; }
# A. The exact rule (strength p/(1-p) per image) at 5k, with fixed-push rows from the same machine
run base --k 3
for ab in "2 1" "1 1" "1 2" "3 1"; do run exact --k 3 --a ${ab% *} --b ${ab#* }; done
run arm --k 3 --a 1 --b 0
run arm --k 3 --a 0 --b 2
run arm --k 3 --a 0.5 --b 0
# C. Cheap tricks combined: low band with the mean colour frozen and early only; high band early only, stronger
run arm --k 3 --a 1 --b 0 --dc 0 --lo 4
run arm --k 3 --a 1 --b 0 --dc 0 --lo 2.5
run arm --k 3 --a 2 --b 0 --dc 0 --lo 4
run arm --k 3 --a 0 --b 4 --lo 2.5
run arm --k 3 --a 0 --b 6 --lo 2.5
run arm --k 3 --a 0 --b 3 --lo 1.5
run arm --k 3 --a 0.5 --b 2 --dc 0 --lo 1.5
# B. "c, not k" (class k in place of the unconditional net) on 3 more pairs: cat not dog, truck not car, horse not deer
for ck in "3 5" "9 1" "7 4"; do
  c=${ck% *}; k=${ck#* }
  run posneg --c $c --k $k --w_low 1 --w_high 1
  run posneg --c $c --k $k --cfg 2
  run posneg --c $c --k $k --w_low 1 --w_high 2
  run posneg --c $c --k $k --w_low 2 --w_high 1
  run posneg --c $c --k $k --w_low 2 --w_high 2
done
# A, seeds: the exact rule's best setting on two new noise sets
for sd in 1 2; do
  run base --k 3 --seed $sd
  run exact --k 3 --a 2 --b 1 --seed $sd
  run exact --k 3 --a 1 --b 1 --seed $sd
done
echo "=== diag k=3"; python neg_diag.py --k 3 --near 5 2>&1 | grep -vE "$F"
tar -cf ../sweep11_out.tar out/neg
echo "=== $(date +%H:%M:%S) SWEEP11 DONE"
