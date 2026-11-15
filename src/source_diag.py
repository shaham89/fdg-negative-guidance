"""Is the guidance signal class information, or disagreement between the two networks?
A perfect pair of networks gives D_u(x) = sum_k p(k|x) D_k(x).  So fit the best such mix (simplex_fit) and split:
    Delta = (D_c - mix) [class part] + (mix - D_u) [gap part].
Per sigma and band (DC, rest of low band, high band), along a chosen trajectory: energy of Delta and of both parts, plus
p(c) and the entropy of p (when is the class decided?), and how well each part lines up with the detail in D_c.
    python source_diag.py --w_high 1      # plain conditional trajectory
    python source_diag.py --w_high 3      # (1,3) split trajectory          -> out/source_diag_wh{w}.npz"""
import os, argparse, numpy as np, torch
from common import *

p_ = argparse.ArgumentParser()
p_.add_argument('--w_high', type=float, default=1.0); p_.add_argument('--r_c', type=float, default=0.125); p_.add_argument('--tau', type=float, default=0.02)
p_.add_argument('--n', type=int, default=256); p_.add_argument('--bs', type=int, default=128)
a = p_.parse_args()

dev = 'cuda'; net_c, net_u = load_nets(dev)
r = radial_freq(32, 32, dev); S = torch.sigmoid((a.r_c - r) / a.tau)
dc = torch.zeros_like(r); dc[0, 0] = 1
masks = {'dc': dc, 'low': S ** 2 * (1 - dc), 'high': (1 - S) ** 2 * (1 - dc)}
fft = lambda x: torch.fft.fft2(x.float(), norm='ortho')
abs2 = lambda F: F.real ** 2 + F.imag ** 2
def energy(F, m): return (m * abs2(F)).sum((1, 2, 3)) / 3                           # per image, mean over channels
def cos(F1, F2, m): return (m * (F1 * F2.conj()).real).sum((1, 2, 3)) / (((m * abs2(F1)).sum((1, 2, 3)) * (m * abs2(F2)).sum((1, 2, 3))).sqrt().clamp(min=1e-20))

rows = []
def hook(xx, Dc, Du, sig):
    Dk = class_preds(net_c, xx.to(torch.float32), sig.to(torch.float32))
    p, mix = simplex_fit(Dk, Du)
    Fd, Fc, Fg, Fx = fft(Dc - Du), fft(Dc - mix), fft(mix - Du), fft(Dc)
    row = dict(step=len(rec.get('sigma', [])), sigma=float(sig), pc=p[torch.arange(len(labels)), labels].mean().item(), H=(-(p * (p + 1e-12).log()).sum(1)).mean().item())
    for k, m in masks.items():
        row[f'Ed_{k}'] = energy(Fd, m).mean().item(); row[f'Ec_{k}'] = energy(Fc, m).mean().item(); row[f'Eg_{k}'] = energy(Fg, m).mean().item()
        row[f'cos_cg_{k}'] = cos(Fc, Fg, m).mean().item(); row[f'cos_cx_{k}'] = cos(Fc, Fx, m).mean().item(); row[f'cos_gx_{k}'] = cos(Fg, Fx, m).mean().item(); row[f'cos_dx_{k}'] = cos(Fd, Fx, m).mean().item()
    rows.append(row)

G = FreqStatic(1.0, a.w_high, a.r_c, a.tau)
for b in range(0, a.n, a.bs):
    labels = torch.arange(b, b + a.bs, device=dev) % 10; rec = {'_hook': hook}
    sample(net_c, net_u, labels, G, seed=1000 + b, device=dev, record=rec)

steps = sorted({row['step'] for row in rows}); keys = [k for k in rows[0] if k != 'step']
T = {k: np.array([np.mean([row[k] for row in rows if row['step'] == s]) for s in steps]) for k in keys}
np.savez(os.path.join(OUT, f'source_diag_wh{a.w_high:g}.npz'), **T)
print(f'trajectory (1,{a.w_high:g}), n={a.n}.  gap% = gap energy / Delta energy.  cos(.,x) = alignment with the band of D_c (the image detail).')
print(f"{'sigma':>8s} {'p(c)':>5s} {'H(p)':>5s} | {'Ed_dc':>8s} {'gap%dc':>7s} | {'Ed_low':>8s} {'gap%low':>7s} | {'Ed_high':>8s} {'cls%hi':>6s} {'gap%hi':>6s} {'cos c,g':>7s} | {'d,x':>6s} {'cls,x':>6s} {'gap,x':>6s}")
for i in range(len(steps)):
    t = {k: T[k][i] for k in T}; pct = lambda n, d: 100 * t[n] / max(t[d], 1e-30)
    print(f"{t['sigma']:8.3f} {t['pc']:5.2f} {t['H']:5.2f} | {t['Ed_dc']:8.4f} {pct('Eg_dc', 'Ed_dc'):7.1f} | {t['Ed_low']:8.4f} {pct('Eg_low', 'Ed_low'):7.1f} | {t['Ed_high']:8.4f} {pct('Ec_high', 'Ed_high'):6.1f} {pct('Eg_high', 'Ed_high'):6.1f} {t['cos_cg_high']:7.3f} | {t['cos_dx_high']:6.3f} {t['cos_cx_high']:6.3f} {t['cos_gx_high']:6.3f}")
