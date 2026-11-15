#!/bin/bash
# Re-run of an earlier batch (sweep12b) that never started because of a bug in its wait loop.
# The exact rule with p calibrated per sigma, then the push map at strength 1 (push size per step, VGG-16 as a second reader).  About 1.4 h.
cd "$(dirname "$0")/../src"
F='Warning|warn|^  [0-9]+/'
snap() { tar -cf ../small.tar.tmp --exclude='samples_*' out/neg sweep12c.log 2>/dev/null && mv ../small.tar.tmp ../small.tar; }
run() { echo "=== $(date +%H:%M:%S) neg.py $*"; python neg.py "$@" 2>&1 | grep -vE "$F" | tail -3; snap; }
run exact --k 3 --a 1 --b 1 --pfit cal
run exact --k 3 --a 1.5 --b 1.5 --pfit cal
echo "=== $(date +%H:%M:%S) pmap s=1"; python pmap.py --n 4000 --s 1 2>&1 | grep -vE "$F"; snap
echo "=== $(date +%H:%M:%S) diag k=3"; python neg_diag.py --k 3 --near 5 2>&1 | grep -vE "$F"; snap
tar -cf ../sweep12c_out.tar out/neg sweep12c.log
echo "=== $(date +%H:%M:%S) SWEEP12C DONE"; snap