#!/bin/bash
# Colab batch, "which band removes a class": tricks, dog-not-cat, exact rule, timing, seeds, all classes.  About 105 runs, 7 h on an A100.
# Each run is saved as soon as it ends and a re-run skips it, so a dead runtime loses only the run in progress.  Order: new ideas first.
cd "$(dirname "$0")/../src"
F='Warning|warn|^  [0-9]+/'
run() { echo "=== $(date +%H:%M:%S) $*"; python neg.py "$@" 2>&1 | grep -vE "$F" | tail -3; }
run base --k 3
# 1. Tricks.  Mean colour frozen / alone; early-only high band at more strengths; push each frequency only while the noise hides it
for s in 1 0.5 2; do run arm --k 3 --a $s --b 0 --dc 0; done
run arm --k 3 --a 0 --b 0 --dc 1
for s in 1 3 4 0.5; do run arm --k 3 --a 0 --b $s --lo 0.8; done
for s in 1 0.5 2; do run ft --k 3 --push $s; done
run ft --k 3 --push 1 --mu 2
run ft --k 3 --push 1 --mode bump
run ft --k 3 --push 2 --mode bump
run arm --k 3 --a 0.5 --b 1 --lo 0.8
# 2. "A dog, not a cat": FDG with the cat class in place of the unconditional net
run posneg --c 5 --k 3 --w_low 1 --w_high 1
for w in 2 3; do
  run posneg --c 5 --k 3 --w_low $w --w_high $w
  run posneg --c 5 --k 3 --w_low 1 --w_high $w
  run posneg --c 5 --k 3 --w_low $w --w_high 1
done
run posneg --c 5 --k 3 --cfg 2
# 3. The exact rule (strength per image) and its band split, 2000 images, with matching fixed-strength rows
run base --k 3 --n 2000 --tag _n2k
run exact --k 3 --a 1 --b 1 --n 2000 --tag _n2k
run exact --k 3 --a 1 --b 1 --xmode mix --n 2000 --tag _n2k
run exact --k 3 --a 1 --b 0 --n 2000 --tag _n2k
run exact --k 3 --a 0 --b 1 --n 2000 --tag _n2k
run exact --k 3 --a 2 --b 1 --n 2000 --tag _n2k
run exact --k 3 --a 1 --b 2 --n 2000 --tag _n2k
run arm --k 3 --a 1 --b 0 --n 2000 --tag _n2k
run arm --k 3 --a 0 --b 2 --n 2000 --tag _n2k
# 4. Push direction from the second (VE) pair
run pair --k 3 --a 1 --b 0
run pair --k 3 --a 0 --b 2
run pair --k 3 --a 0 --b 2 --lo 0.8
# 5. Finer timing: where does "early" end?  Noise levels: ... 5.3, 3.3, 1.9, 1.1, 0.585, 0.296, 0.14 ...
for lo in 2.5 1.5 0.4 0.2; do run arm --k 3 --a 0 --b 2 --lo $lo; done
run arm --k 3 --a 0 --b 2 --lo 0.4 --hi 2.5
for lo in 10 4 2.5 1.5; do run arm --k 3 --a 1 --b 0 --lo $lo; done
# 6. Two more seeds for the main rows
for sd in 1 2; do
  run base --k 3 --seed $sd
  for ab in "0.5 0" "1 0" "0 1" "0 2" "0.5 0.5"; do run arm --k 3 --a ${ab% *} --b ${ab#* } --seed $sd; done
  run arm --k 3 --a 1 --b 0 --lo 0.8 --seed $sd
  run arm --k 3 --a 0 --b 2 --lo 0.8 --seed $sd
  run arm --k 3 --a 0 --b 2 --hi 0.8 --seed $sd
done
# 7. The other classes (cat = 3 is above; car = 1 was run on Azure, redone here so that all rows share one machine)
for k in 1 5 8 0 7 2 9 4 6; do
  python neg_none.py --k $k 2>&1 | grep -vE "$F" | tail -1
  run base --k $k
  run arm --k $k --a 1 --b 0
  run arm --k $k --a 0 --b 2
  run arm --k $k --a 0 --b 2 --lo 0.8
  run arm --k $k --a 0.5 --b 0
done
# 8. Paired diagnostics for every class (no sampling)
near=(8 9 0 5 7 3 2 4 0 1)
for k in 3 1 5 8 0 7 2 9 4 6; do echo "=== diag k=$k"; python neg_diag.py --k $k --near ${near[$k]} 2>&1 | grep -vE "$F"; done
echo "=== $(date +%H:%M:%S) SWEEP10 DONE"
date > sweep10.done
