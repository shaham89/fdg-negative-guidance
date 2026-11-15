"""Noise floor of the metrics from the saved samples: split each 10k set into its first and second 5k halves (both
class-balanced, different seeds) and report IS / FID on each half.  The half-vs-half spread is the scale below
which differences between arms are not meaningful.      python noise.py cfg_w2 freq_wl1_wh3_rc0.125 'time[freq_wl1_wh3_rc0.125]'"""
import os, sys, json, numpy as np, torch
from common import *

names = sys.argv[1:]
print(f"{'config':40s} {'IS half A':>9s} {'IS half B':>9s} {'|dIS|':>6s} {'FID A':>7s} {'FID B':>7s} {'|dFID|':>7s}")
out = {}
for name in names:
    d = np.load(os.path.join(OUT, f'samples_{name}.npz'))
    imgs = torch.tensor(d['images']).permute(0, 3, 1, 2).contiguous()
    h = imgs.shape[0] // 2; A, B = imgs[:h], imgs[h:]     # first/second half: both class-balanced (labels cycle with period 10; 5000 % 10 == 0).
    # NOT imgs[0::2]/imgs[1::2] -- even/odd indices alias with the label cycle and each half would hold only 5 classes.
    ma, mb = fid_is(A), fid_is(B)
    out[name] = dict(is_a=ma['is'], is_b=mb['is'], fid_a=ma['fid'], fid_b=mb['fid'])
    print(f"{name:40s} {ma['is']:9.3f} {mb['is']:9.3f} {abs(ma['is'] - mb['is']):6.3f} {ma['fid']:7.2f} {mb['fid']:7.2f} {abs(ma['fid'] - mb['fid']):7.2f}")
json.dump(out, open(os.path.join(OUT, 'noise.json'), 'w'), indent=1)
dis = [abs(v['is_a'] - v['is_b']) for v in out.values()]
print(f"\nmean |dIS| between 5k halves = {np.mean(dis):.3f}  ->  IS std of a 10k estimate ~ {np.mean(dis) / 2:.3f}  (halves differ by ~2 std of the mean of each)")
print("note: FID on 5k is biased upward relative to 10k; use the half-vs-half |dFID| only as a spread, not as a level.")
