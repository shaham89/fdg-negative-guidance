"""Do two independently trained network pairs agree on the guidance direction?  EDM's VP pair against its VE pair, no mixture fit involved.
If Delta_i = signal + error_i with independent errors, then cos(Delta_vp, Delta_ve) estimates the share of Delta's energy that is signal.
Per sigma and band, along the VP (1, w_high) trajectory.  Printed next to the class share from source_diag.py, as an independent check of that split.
    python replicate_diag.py --w_high 1          -> out/replicate_wh{w}.npz
    python replicate_diag.py --n 1024 --table    also writes out/wiener_table.json (rho per sigma and band) for rule=wiener
    python replicate_diag.py --wrong             same, for the direction of a random wrong class c' (D_c' - D_u), still on the path of the true class -> out/replicate_wrong_wh{w}.npz"""
import os, json, argparse, numpy as np, torch
from common import *

p_ = argparse.ArgumentParser()
p_.add_argument('--w_high', type=float, default=1.0); p_.add_argument('--r_c', type=float, default=0.125); p_.add_argument('--tau', type=float, default=0.02)
p_.add_argument('--n', type=int, default=256); p_.add_argument('--bs', type=int, default=128); p_.add_argument('--table', action='store_true'); p_.add_argument('--wrong', action='store_true')
a = p_.parse_args()

dev = 'cuda'; net_c, net_u = load_nets(dev); net_c2, net_u2 = load_nets(dev, arch='ve')
r = radial_freq(32, 32, dev); S = torch.sigmoid((a.r_c - r) / a.tau)
dc = torch.zeros_like(r); dc[0, 0] = 1
masks = {'dc': dc, 'low': S ** 2 * (1 - dc), 'high': (1 - S) ** 2 * (1 - dc)}
bands = torch.cat([dc[None], band_masks(r) * (1 - dc)])                              # for the rho table: DC, then BAND_EDGES bands without DC
fft = lambda x: torch.fft.fft2(x.float(), norm='ortho')
abs2 = lambda F: F.real ** 2 + F.imag ** 2
def energy(F, m): return (m * abs2(F)).sum((1, 2, 3)) / 3
def cos(F1, F2, m): return (m * (F1 * F2.conj()).real).sum((1, 2, 3)) / (((m * abs2(F1)).sum((1, 2, 3)) * (m * abs2(F2)).sum((1, 2, 3))).sqrt().clamp(min=1e-20))

rows = []
def hook(xx, Dc, Du, sig):
    x32, sig32 = xx.to(torch.float32), sig.to(torch.float32)
    Dc2, Du2 = net_c2(x32, sig32, onehot).to(torch.float64), net_u2(x32, sig32, None).to(torch.float64)
    if a.wrong: Dc, Dc2 = net_c(x32, sig32, onehot_w).to(torch.float64), net_c2(x32, sig32, onehot_w).to(torch.float64)
    F1, F2, Fcc, Fuu = fft(Dc - Du), fft(Dc2 - Du2), fft(Dc - Dc2), fft(Du - Du2)
    row = dict(step=len(rec.get('sigma', [])), sigma=float(sig))
    for k, m in masks.items():
        row[f'cos_{k}'] = cos(F1, F2, m).mean().item(); row[f'E1_{k}'] = energy(F1, m).mean().item(); row[f'E2_{k}'] = energy(F2, m).mean().item()
        row[f'Ecc_{k}'] = energy(Fcc, m).mean().item(); row[f'Euu_{k}'] = energy(Fuu, m).mean().item()
    for b, m in enumerate(bands):
        row[f'cov_b{b}'] = (m * (F1 * F2.conj()).real).sum((1, 2, 3)).mean().item(); row[f'P1_b{b}'] = (m * abs2(F1)).sum((1, 2, 3)).mean().item(); row[f'P2_b{b}'] = (m * abs2(F2)).sum((1, 2, 3)).mean().item()
    rows.append(row)

G = FreqStatic(1.0, a.w_high, a.r_c, a.tau)
for b in range(0, a.n, a.bs):
    labels = torch.arange(b, b + a.bs, device=dev) % 10; onehot = torch.eye(10, device=dev)[labels]; rec = {'_hook': hook}
    onehot_w = torch.eye(10, device=dev)[wrong_labels(labels, 1000 + b)]
    sample(net_c, net_u, labels, G, seed=1000 + b, device=dev, record=rec)

steps = sorted({row['step'] for row in rows}); keys = [k for k in rows[0] if k != 'step']
T = {k: np.array([np.mean([row[k] for row in rows if row['step'] == s]) for s in steps]) for k in keys}
np.savez(os.path.join(OUT, f"replicate{'_wrong' if a.wrong else ''}_wh{a.w_high:g}.npz"), **T)
rho = np.stack([T[f'cov_b{b}'] / np.sqrt(T[f'P1_b{b}'] * T[f'P2_b{b}']).clip(min=1e-30) for b in range(len(bands))], axis=1)    # steps,bands: energy-weighted agreement
if a.table: json.dump(dict(sigmas=T['sigma'].tolist(), rho=rho.tolist(), bands=['DC'] + BAND_NAMES, w_high=a.w_high, n=a.n), open(os.path.join(OUT, 'wiener_table.json'), 'w'))
print('rho per band, sigma descending:  ' + '  '.join(['DC'] + BAND_NAMES))
for s_, row_ in zip(T['sigma'], rho): print(f'{s_:8.3f} ' + ' '.join(f'{v:6.3f}' for v in row_))
sp =os.path.join(OUT, f'source_diag_wh{a.w_high:g}.npz'); Q = np.load(sp) if os.path.exists(sp) else None
print(f'trajectory (1,{a.w_high:g}), n={a.n}.  cos = agreement of the two pairs on Delta.  fit% = class share from source_diag.  Ecc = energy of (D_c vp - D_c ve); Eg = gap energy from the fit.')
print(f"{'sigma':>8s} | {'cos dc':>7s} {'cos low':>7s} {'cos hi':>7s} | {'fit% dc':>7s} {'fit%low':>7s} {'fit%hi':>7s} | {'E1 hi':>8s} {'E2 hi':>8s} {'Ecc hi':>8s} {'Euu hi':>8s} {'Eg hi':>8s}")
for i in range(len(steps)):
    t = {k: T[k][i] for k in T}
    fit = [100 * Q[f'Ec_{b}'][i] / max(Q[f'Ed_{b}'][i], 1e-30) for b in masks] if Q is not None else [float('nan')] * 3
    eg = Q['Eg_high'][i] if Q is not None else float('nan')
    print(f"{t['sigma']:8.3f} | {t['cos_dc']:7.3f} {t['cos_low']:7.3f} {t['cos_high']:7.3f} | {fit[0]:7.1f} {fit[1]:7.1f} {fit[2]:7.1f} | {t['E1_high']:8.4f} {t['E2_high']:8.4f} {t['Ecc_high']:8.4f} {t['Euu_high']:8.4f} {eg:8.4f}")
