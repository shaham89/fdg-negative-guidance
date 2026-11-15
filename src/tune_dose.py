"""Find the dose of each control arm that gives the same final sharpness as the (1, w_high) split, so the arms can be compared at equal sharpness.
Sharpness = rms of the final image's high band (r >= r_c).  dose 0 gives D = D_c for every arm, so we iterate dose *= (target - base) / (s - base).
About 1 min per arm.  Needs out/sharpen_table.json from calib_sharpen.py.     python tune_dose.py"""
import os, json, argparse, numpy as np, torch
from common import *

p = argparse.ArgumentParser()
p.add_argument('--w_high', type=float, default=3.0); p.add_argument('--r_c', type=float, default=0.125); p.add_argument('--n', type=int, default=256); p.add_argument('--iters', type=int, default=3)
a = p.parse_args()

dev = 'cuda'; net_c, net_u = load_nets(dev)
labels = torch.arange(a.n, device=dev) % 10; neg = wrong_labels(labels, 0)
high = (radial_freq(32, 32, dev) >= a.r_c).float()
def run(G, neg_=None):
    x = sample(net_c, net_u, labels, G, seed=0, device=dev, neg_labels=neg_)
    y = x.clamp(-1, 1); F = torch.fft.fft2(y, norm='ortho')
    return ((F.real ** 2 + F.imag ** 2).mean(dim=(0, 1)) * high).sum().sqrt().item(), clip_fraction(x)

base, c0 = run(ConstCFG(1)); target, c1 = run(FreqStatic(1, a.w_high, a.r_c))
print(f'w=1: sharp {base:.3f} clip {c0:.4f}   target ({1},{a.w_high:g}): sharp {target:.3f} (ratio {target / base:.3f}) clip {c1:.4f}', flush=True)
t = json.load(open(os.path.join(OUT, 'sharpen_table.json')))
arms = {'highdir_cw': (lambda d: HighDir('cw', 1 + d, a.r_c), neg, a.w_high - 1),
        'highdir_wu': (lambda d: HighDir('wu', 1 + d, a.r_c), neg, a.w_high - 1),
        'sharpen_const': (lambda d: Sharpen([1.0], [d], a.r_c), None, 0.1),
        'sharpen_shaped': (lambda d: Sharpen(t['sigmas'], [d * l for l in t['lambdas']], a.r_c), None, 0.2)}
out = dict(target_ratio=target / base)
for name, (make, neg_, d) in arms.items():
    for i in range(a.iters):
        s, c = run(make(d), neg_)
        print(f'  {name:15s} dose {d:7.4f}  sharp ratio {s / base:.3f}  clip {c:.4f}', flush=True)
        if i < a.iters - 1: d *= (target - base) / (s - base)
    out[name] = dict(dose=d, sharp_ratio=s / base, clip=c)
json.dump(out, open(os.path.join(OUT, 'doses.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
