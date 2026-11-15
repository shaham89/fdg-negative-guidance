#!/bin/bash
# Restart of sweep11 after the runtime died (Colab Pro needs a connected client).  Done rows come in results.jsonl and are skipped.
# Order: most valuable first.  About 4.5 h on an A100.
cd "$(dirname "$0")/../src"
F='Warning|warn|^  [0-9]+/'
run() { echo "=== $(date +%H:%M:%S) $*"; python neg.py "$@" 2>&1 | grep -vE "$F" | tail -3; }
run base --k 3
# A. The exact rule at 5k.  Does splitting it by band help, or only pushing harder?  Scaled copies (1.5,1.5), (2,2) are the fair control.
for ab in "2 2" "2 1" "1 2" "1.5 1.5"; do run exact --k 3 --a ${ab% *} --b ${ab#* }; done
run arm --k 3 --a 1 --b 0
run arm --k 3 --a 0 --b 2
run arm --k 3 --a 0.5 --b 0
# B. "c, not k" on 3 more pairs: cat not dog, truck not car, horse not deer
for ck in "3 5" "9 1" "7 4"; do
  c=${ck% *}; k=${ck#* }
  run posneg --c $c --k $k --w_low 1 --w_high 1
  run posneg --c $c --k $k --cfg 2
  run posneg --c $c --k $k --w_low 1 --w_high 2
  run posneg --c $c --k $k --w_low 2 --w_high 1
  run posneg --c $c --k $k --w_low 2 --w_high 2
done
# C. Cheap tricks combined
run arm --k 3 --a 1 --b 0 --dc 0 --lo 4
run arm --k 3 --a 1 --b 0 --dc 0 --lo 2.5
run arm --k 3 --a 2 --b 0 --dc 0 --lo 4
run arm --k 3 --a 0 --b 4 --lo 2.5
run arm --k 3 --a 0 --b 6 --lo 2.5
run arm --k 3 --a 0 --b 3 --lo 1.5
run arm --k 3 --a 0.5 --b 2 --dc 0 --lo 1.5
# A, seeds
for sd in 1 2; do
  run base --k 3 --seed $sd
  run exact --k 3 --a 1 --b 1 --seed $sd
  run exact --k 3 --a 2 --b 1 --seed $sd
done
run exact --k 3 --a 3 --b 1
echo "=== diag k=3"; python neg_diag.py --k 3 --near 5 2>&1 | grep -vE "$F"
tar -cf ../sweep11_out.tar out/neg
echo "=== $(date +%H:%M:%S) SWEEP11 DONE"
