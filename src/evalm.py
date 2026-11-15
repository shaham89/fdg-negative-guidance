"""Evaluate already-generated sample files:  python evalm.py samples_cfg_w1.npz [more.npz ...]  (or 'missing' = all without a results row)."""
import os, sys, json, glob, numpy as np, torch
from common import *

done = {json.loads(l)['name'] for l in open(os.path.join(OUT, 'results.jsonl')) if '"fid"' in l} if os.path.exists(os.path.join(OUT, 'results.jsonl')) else set()
files = sys.argv[1:]
if files == ['missing']:
    files = [f for f in sorted(glob.glob(os.path.join(OUT, 'samples_*.npz'))) if os.path.basename(f)[8:-4] not in done and not f.endswith('smoke.npz')]
for f in files:
    f = f if os.path.isabs(f) else os.path.join(OUT, os.path.basename(f))
    name = os.path.basename(f)[8:-4]
    d = np.load(f); imgs = torch.tensor(d['images']).permute(0, 3, 1, 2).contiguous(); labs = torch.tensor(d['labels']).long()
    x = imgs.float() / 127.5 - 1
    res = dict(name=name, rule=name.split('_')[0], n=len(imgs), evaluated_from=os.path.basename(f))
    res.update(evaluate(imgs, labs))
    with open(os.path.join(OUT, 'results.jsonl'), 'a') as fh: fh.write(json.dumps(res) + '\n')
    print(f"{name:45s} FID={res['fid']:.3f}  IS={res['is']:.3f}  acc={res['class_acc']:.3f}")
