"""Short run along the (1, w_high) split trajectory.  About 1 min on the A10.
1) lambda(sigma) such that lambda * High(D_c) has the energy of (w_high - 1) * High(Delta)   -> out/sharpen_table.json
2) cosine between each band of Delta and the same band of (class mean - global mean)         -> out/class_align.npz
   Near 1 at high sigma means Delta there is class signal; well below 1 means it also holds other things (the gap between the two nets)."""
import os, json, argparse, numpy as np, torch
from common import *

p = argparse.ArgumentParser()
p.add_argument('--w_high', type=float, default=3.0); p.add_argument('--r_c', type=float, default=0.125); p.add_argument('--tau', type=float, default=0.02)
p.add_argument('--n', type=int, default=512); p.add_argument('--bs', type=int, default=256); p.add_argument('--dev', default='cuda')
a = p.parse_args()

dev = a.dev; net_c, net_u = load_nets(dev)
r = radial_freq(32, 32, dev); S = torch.sigmoid((a.r_c - r) / a.tau); H2 = (1 - S) ** 2
dc = torch.zeros_like(r); dc[0, 0] = 1
masks = {'dc': dc, 'low_rest': S ** 2 * (1 - dc), 'high': H2 * (1 - dc)}
fft = lambda x: torch.fft.fft2(x.float(), norm='ortho')
abs2 = lambda F: F.real ** 2 + F.imag ** 2
to_float = lambda x: x.float() / 127.5 - 1
mu = to_float(cifar_train_uint8()).mean(0)
Fm = torch.stack([fft(to_float(cifar_train_uint8([c])).mean(0) - mu) for c in range(10)]).to(dev)   # 10,C,H,W

rows = []
def hook(xx, Dc, Du, sig):
    Fd, Fc, Fk = fft(Dc - Du), fft(Dc), Fm[labels]
    row = dict(step=len(rec.get('sigma', [])), sigma=float(sig), Ehd=(H2 * abs2(Fd)).sum((1, 2, 3)).mean().item(), Ehc=(H2 * abs2(Fc)).sum((1, 2, 3)).mean().item())
    for k, m in masks.items():
        num = (m * (Fd * Fk.conj()).real).sum((1, 2, 3)); den = ((m * abs2(Fd)).sum((1, 2, 3)) * (m * abs2(Fk)).sum((1, 2, 3))).sqrt()
        row[k] = (num / den.clamp(min=1e-12)).mean().item()
    rows.append(row)

G = FreqStatic(1.0, a.w_high, a.r_c, a.tau)
for b in range(0, a.n, a.bs):
    labels = torch.arange(b, b + a.bs, device=dev) % 10; rec = {'_hook': hook}
    sample(net_c, net_u, labels, G, seed=1000 + b, device=dev, record=rec)

steps = sorted({row['step'] for row in rows})
avg = lambda k, s: float(np.mean([row[k] for row in rows if row['step'] == s]))
sig = [avg('sigma', s) for s in steps]
lam = [(a.w_high - 1) * float(np.sqrt(avg('Ehd', s) / max(avg('Ehc', s), 1e-12))) for s in steps]
json.dump(dict(sigmas=sig, lambdas=lam, w_high=a.w_high, r_c=a.r_c, n=a.n), open(os.path.join(OUT, 'sharpen_table.json'), 'w'))
np.savez(os.path.join(OUT, 'class_align.npz'), sigma=np.array(sig), lam=np.array(lam), **{k: np.array([avg(k, s) for s in steps]) for k in masks})
print(f"{'sigma':>9s} {'lambda':>7s} {'cos DC':>7s} {'cos low':>8s} {'cos high':>9s}")
for s, x, l in zip(steps, sig, lam): print(f"{x:9.3f} {l:7.3f} {avg('dc', s):7.3f} {avg('low_rest', s):8.3f} {avg('high', s):9.3f}")
