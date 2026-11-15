"""Free checks on saved 'not class k' samples, no sampling.  All arms share noise, so image i of an arm is paired with image i of the no-push arm.
Mean colour shift, class changes against no push (removal and collateral change), change per band, and low/high hybrids.
  python neg_diag.py --k 3    -> one line per arm, and out/neg/diag_k3.json"""
import os, json, glob, argparse, numpy as np, torch
from common import *

dev = 'cuda'; NEG = os.path.join(OUT, 'neg'); R = radial_freq(32, 32, dev)
p = argparse.ArgumentParser(); p.add_argument('--k', type=int, default=3); p.add_argument('--near', type=int, default=5, help='the similar class to watch'); a = p.parse_args(); k = a.k
load = lambda f: torch.tensor(np.load(f)['images']).permute(0, 3, 1, 2)
sign = lambda u: u.float().to(dev) / 127.5 - 1
low = lambda x: x - high_pass(x, R, 0.125, 0.02)
back = lambda x: ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
mu = torch.stack([cifar_train_uint8([c]).float().mean((0, 2, 3)) for c in range(10)]); mu_all = mu.mean(0); d = mu[k] - mu_all
shift = lambda m, m0: float(((m - m0) @ d) / (d @ d))              # in units of (class k mean colour - global mean colour)
import re
def base(sf, n, cache={}):
    """The no-push images with the same noise: same seed suffix (_seed1) or tag (_n2k), cut to the arm's length."""
    if sf not in cache:
        x0 = load(os.path.join(NEG, f'samples_neg3_none{sf}.npz')); cache[sf] = (x0, classify(x0, dev))
    x0, P0 = cache[sf]; x0, P0 = x0[:n], P0[:n]; s0 = sign(x0); l0 = low(s0)
    return P0, x0.float().mean((0, 2, 3)), l0, s0 - l0
print('arm | colour k (mix alone predicts) | cats kept | non-cats that changed class | dogs kept | rms change low, high | hybrid rate of k: Low(arm)+High(none), Low(none)+High(arm)')
rows = []
for f in sorted(glob.glob(os.path.join(NEG, f'samples_neg{k}_*.npz'))):
    nm = os.path.basename(f)[8:-4]
    if '_none' in nm or '_oracle' in nm: continue
    sf = ''.join(re.findall(r'(?:_seed\d+|_n2k)', nm[len(f'neg{k}'):]))
    if not os.path.exists(os.path.join(NEG, f'samples_neg3_none{sf}.npz')): print(nm, 'no matching no-push run'); continue
    x = load(f); P0, m0, l0, h0 = base(sf, len(x)); P = classify(x, dev); sx = sign(x); lx = low(sx); hx = sx - lx
    hist = torch.bincount(P, minlength=10).float() / len(P)
    H1, H2 = classify(back(lx + h0), dev), classify(back(l0 + hx), dev)
    r = dict(name=os.path.basename(f)[8:-4], colour=shift(x.float().mean((0, 2, 3)), m0), colour_mix=shift(hist @ mu, (torch.bincount(P0, minlength=10).float() / len(P0)) @ mu),
             kept=(P[P0 == k] == k).float().mean().item(), changed=(P[P0 != k] != P0[P0 != k]).float().mean().item(), near_kept=(P[P0 == a.near] == a.near).float().mean().item(),
             rms_low=(lx - l0).pow(2).mean().sqrt().item(), rms_high=(hx - h0).pow(2).mean().sqrt().item(), hyb_low=(H1 == k).float().mean().item(), hyb_high=(H2 == k).float().mean().item(),
             to=[round(v, 3) for v in (torch.bincount(P[P0 == k], minlength=10).float() / (P0 == k).sum()).tolist()])
    rows.append(r)
    print(f"{r['name']:32s} {r['colour']:+.2f} ({r['colour_mix']:+.2f}) | {r['kept']:.3f} | {r['changed']:.3f} | {r['near_kept']:.3f} | {r['rms_low']:.3f} {r['rms_high']:.3f} | {r['hyb_low']:.3f} {r['hyb_high']:.3f} | to " + ' '.join(f'{v:.2f}' for v in r['to']), flush=True)
json.dump(rows, open(os.path.join(NEG, f'diag_k{k}.json'), 'w'))
