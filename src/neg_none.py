"""'No push' row for another class k.  D = D_u whatever the label, so the images are those of neg3_none: only the class being counted and the FID reference change.
  python neg_none.py --k 1     # row neg1_none in out/neg/results.jsonl, so that neg.py base skips this arm and runs only the correct answer
"""
import os, json, argparse, numpy as np, torch
from common import *

NEG = os.path.join(OUT, 'neg'); RES = os.path.join(NEG, 'results.jsonl')
p = argparse.ArgumentParser(); p.add_argument('--k', type=int, default=1); p.add_argument('--suffix', default='', help='_seed1 for seed 1, _n2k for a tagged run'); a = p.parse_args(); sf = a.suffix
rows = {json.loads(l)['name']: json.loads(l) for l in open(RES)}
imgs = torch.tensor(np.load(os.path.join(NEG, f'samples_neg3_none{sf}.npz'))['images']).permute(0, 3, 1, 2)
ref = [c for c in range(10) if c != a.k]
if f'neg{a.k}_none{sf}' in rows: raise SystemExit('already done')
res = dict(rows['neg3_none' + sf], name=f'neg{a.k}_none{sf}', k=a.k, same_images_as='neg3_none' + sf, **fid_is(imgs, cifar_train_uint8(ref), 'cifar10-train-' + ''.join(map(str, ref))))
with open(RES, 'a') as f: f.write(json.dumps(res) + '\n')
print(f"{res['name']}  FID={res['fid']:.2f} IS={res['is']:.2f}  class {a.k}: resnet {res['hist_resnet'][a.k]:.4f} vgg {res['hist_vgg'][a.k]:.4f} low {res['hist_low'][a.k]:.3f}")
