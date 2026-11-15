"""Derive the *equivalent purely-temporal schedule* of a spectral guidance rule.

For a spectral weight W(f) applied to the CFG delta at noise level sigma, the best scalar approximation
(least squares:  argmin_w || W*Delta - w*Delta ||^2 ) is the power-weighted mean of W:
    w_eq(sigma) = sum_f W(f) P_delta(f, sigma) / sum_f P_delta(f, sigma)
where P_delta is the mean power of the CFG delta measured along a baseline trajectory (diag.py).
We also report the L2-norm-matched scalar  w_l2 = sqrt( sum W^2 P / sum P ).

If a temporal-only rule with w_eq(sigma) reproduces the spectral rule's FID/IS, the frequency axis is redundant.

Usage:  python make_time_sched.py freq --w_low 1 --w_high 3 --r_c 0.125
        python make_time_sched.py ft --w_min 1 --w_max 3 --mode unresolved
"""
import os, json, argparse, numpy as np, torch
from common import *

p = argparse.ArgumentParser()
p.add_argument('rule', choices=['freq', 'ft'])
p.add_argument('--w_low', type=float); p.add_argument('--w_high', type=float); p.add_argument('--r_c', type=float, default=0.125); p.add_argument('--tau', type=float, default=0.02)
p.add_argument('--w_min', type=float); p.add_argument('--w_max', type=float); p.add_argument('--mode', default='unresolved'); p.add_argument('--mu', type=float, default=0.0); p.add_argument('--s', type=float, default=1.0)
p.add_argument('--diag', default='diag_w1.npz', help='which trajectory the delta power was measured along')
p.add_argument('--match', default='ls', choices=['ls', 'l2'])
a = p.parse_args()

if a.rule == 'freq': G = FreqStatic(a.w_low, a.w_high, a.r_c, a.tau)
else: sr, sP = load_spectrum(); G = FreqTime(sr, sP, a.w_min, a.w_max, a.mode, a.mu, a.s)
d = np.load(os.path.join(OUT, a.diag)); sig, P = d['sigma'], d['P_delta']        # steps ; steps,H,W
r = radial_freq(32, 32, 'cpu')
ws, wl2 = [], []
for s_, P_ in zip(sig, P):
    W = G.weight(float(s_), r); W = W.numpy() if torch.is_tensor(W) else np.full(P_.shape, float(W))
    ws.append(float((W * P_).sum() / P_.sum())); wl2.append(float(np.sqrt((W ** 2 * P_).sum() / P_.sum())))
w_use = ws if a.match == 'ls' else wl2
name = f'time[{G.name}]' + ('' if a.match == 'ls' else '_l2')
out = dict(sigmas=sig.tolist(), ws=w_use, ws_ls=ws, ws_l2=wl2, name=name, source=G.name, diag=a.diag)
path = os.path.join(OUT, f'time_from_{G.name}{"" if a.match == "ls" else "_l2"}.json')
json.dump(out, open(path, 'w'))
print(path)
for s_, w1, w2 in zip(sig, ws, wl2): print(f'  sigma={s_:8.4f}   w_ls={w1:.3f}   w_l2={w2:.3f}')
