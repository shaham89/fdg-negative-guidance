"""Diagnostic map: energy of the CFG delta (D_c - D_u) per frequency band per timestep,
recorded along a baseline trajectory.  Produces out/diag_w{w}.npz and out/diag_w{w}.png."""
import os, argparse, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *

p = argparse.ArgumentParser()
p.add_argument('--w', type=float, default=1.0, help='guidance scale of the trajectory along which we measure')
p.add_argument('--n', type=int, default=2000)
p.add_argument('--bs', type=int, default=500)
p.add_argument('--steps', type=int, default=18)
p.add_argument('--uncond_mode', default='sep')
a = p.parse_args()

dev = 'cuda'; net_c, net_u = load_nets(dev)
acc = {}
for b in range(0, a.n, a.bs):
    labels = torch.arange(b, b + a.bs, device=dev) % 10
    rec = {}
    sample(net_c, net_u, labels, ConstCFG(a.w), num_steps=a.steps, seed=1000 + b, device=dev, uncond_mode=a.uncond_mode, record=rec)
    for k in ['E_delta', 'E_Du', 'E_x', 'P_delta']:
        acc.setdefault(k, []).append(np.stack(rec[k]))       # steps,B  (P_delta: steps,H,W)
    acc['sigma'] = np.array(rec['sigma'])
E = {k: np.mean(acc[k], axis=0) for k in ['E_delta', 'E_Du', 'E_x', 'P_delta']}
sig = acc['sigma']
np.savez(os.path.join(OUT, f'diag_w{a.w:g}.npz'), sigma=sig, **E, edges=np.array(BAND_EDGES))

# relative quantities
frac = E['E_delta'] / E['E_delta'].sum(1, keepdims=True)              # where (in freq) the guidance energy sits at each step
rel = E['E_delta'] / E['E_Du']                                        # delta energy relative to the unconditional prediction's energy in that band
sr, sP = load_spectrum(); mids = [(lo + hi) / 2 for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
sig_star = np.sqrt(np.interp(np.clip(mids, 0, 0.7), sr, sP))       # SNR=1 crossover sigma per band (at band centre)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
def hm(a_, M, title, log=False):
    im = a_.imshow(np.log10(M.T) if log else M.T, aspect='auto', origin='lower', cmap='magma')
    a_.set_yticks(range(len(BAND_NAMES))); a_.set_yticklabels(BAND_NAMES); a_.set_ylabel('DFT radius band')
    a_.set_xticks(range(0, len(sig), 2)); a_.set_xticklabels([f'{s:.2g}' for s in sig[::2]], rotation=60); a_.set_xlabel(r'$\sigma_t$  (sampling goes left→right)')
    a_.set_title(title); plt.colorbar(im, ax=a_)
    # SNR=1 crossover: for each band, the step index where sigma crosses sigma*
    xs = [np.interp(np.log(ss), np.log(sig[::-1]), np.arange(len(sig))[::-1]) for ss in sig_star]
    a_.plot(xs, range(len(BAND_NAMES)), 'c.-', lw=1, label='SNR=1 crossover'); a_.legend(loc='upper right', fontsize=7)
hm(ax[0], np.log10(E['E_delta']), r'$\log_{10}$ energy of $(D_c-D_u)$')
hm(ax[1], frac, r'fraction of guidance energy per band (cols sum to 1)')
hm(ax[2], np.log10(rel), r'$\log_{10}$ [ energy($D_c-D_u$) / energy($D_u$) ]')
plt.suptitle(f'Where does the CFG delta live?  trajectory w={a.w:g}, {a.steps} Heun steps, n={a.n}')
plt.tight_layout(); plt.savefig(os.path.join(OUT, f'diag_w{a.w:g}.png'), dpi=130)
print('E_delta (steps x bands):\n', np.array2string(E['E_delta'], precision=4, suppress_small=True))
print('fraction:\n', np.array2string(frac, precision=3))
