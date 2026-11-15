#!/bin/bash
# 200-image check of every new command before the long batch.  Rows are tagged _smoke and removed at the end.
cd "$(dirname "$0")/../src"; set -o pipefail
T="--n 200 --bs 200 --n_read 200 --tag _smoke"
for c in "base --k 3" "arm --k 3 --a 1 --b 0 --dc 0" "ft --k 3 --push 1" "ft --k 3 --push 1 --mode bump" "posneg --c 5 --k 3 --w_low 1 --w_high 3" "posneg --c 5 --k 3 --cfg 2" \
         "exact --k 3 --a 1 --b 1" "exact --k 3 --a 1 --b 1 --xmode mix" "pair --k 3 --a 0 --b 2 --lo 0.8" "arm --k 3 --a 0 --b 2 --seed 1"; do
  echo "=== $c"; python neg.py $c $T 2>&1 | grep -vE "Warning|warn|^  [0-9]+/" | tail -4 || echo "FAILED: $c"
done
python - <<'PY'
import json
for l in open('out/neg/results.jsonl'):
    r = json.loads(l)
    if 'gate' in r: print(r['name'], 'gate (sigma, p_k, strength):', [tuple(round(v, 3) for v in g) for g in r['gate'][::4]])
PY
grep -v _smoke out/neg/results.jsonl > out/neg/r.tmp; mv out/neg/r.tmp out/neg/results.jsonl; rm -f out/neg/*_smoke*
echo SMOKE DONE
