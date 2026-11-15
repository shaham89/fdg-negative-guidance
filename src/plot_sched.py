"""Figure: the equivalent time-only schedule w_eq(sigma) of each static frequency split, and the SNR-derived
freq x time weight maps.  Reads out/time_from_*.json and out/cifar_spectrum.npz."""
import os, glob, json, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for f in sorted(glob.glob(os.path.join(OUT, 'time_from_freq_*.json'))):
    d = json.load(open(f))
    if f.endswith('_l2.json'): continue
    ax[0].semilogx(d['sigmas'], d['ws_ls'], 'o-', ms=3, label=d['source'].replace('freq_', '').replace('_rc0.125', ''))
ax[0].invert_xaxis(); ax[0].set_xlabel(r'$\sigma$  (sampling →)'); ax[0].set_ylabel(r'$w_{eq}(\sigma)$'); ax[0].grid(alpha=.3)
ax[0].set_title('Static frequency splits ≡ temporal ramps\n(least-squares scalar equivalent along the w=1 trajectory)'); ax[0].legend(fontsize=8)

sr, sP = load_spectrum(); r = radial_freq(32, 32, 'cpu')
class _N: sigma_min, sigma_max = 0.002, 80.0
_N.round_sigma = staticmethod(lambda t: t)
sig = edm_sigmas(_N, 18, device='cpu').numpy()[:-1]
prof_r = np.linspace(0.005, 0.7, 120)
for k, (mode, ttl) in enumerate([('unresolved', "SNR rule, mode 'unresolved': guide bands still noise-dominated"), ('resolved', "mode 'resolved': guide bands already fixed")]):
    G = FreqTime(sr, sP, 1.0, 3.0, mode)
    M = np.zeros((len(sig), len(prof_r)))
    for i, s in enumerate(sig):
        # evaluate the weight as a function of radius using a fake radius grid
        rr = torch.tensor(prof_r, dtype=torch.float32)[None, :]
        G._logP = None
        M[i] = G.weight(float(s), rr).numpy()[0]
    im = ax[k + 1].imshow(M.T, aspect='auto', origin='lower', cmap='viridis', extent=[0, len(sig), prof_r[0], prof_r[-1]], vmin=1, vmax=3)
    ax[k + 1].set_xticks(np.arange(len(sig)) + .5); ax[k + 1].set_xticklabels([f'{s:.2g}' for s in sig], rotation=60, fontsize=7)
    ax[k + 1].set_xlabel(r'$\sigma_t$'); ax[k + 1].set_ylabel('|f| (cycles/px)'); ax[k + 1].set_title(ttl + r'   $w(f,\sigma)$, $w_{min}$=1, $w_{max}$=3', fontsize=9)
    plt.colorbar(im, ax=ax[k + 1])
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'schedules.png'), dpi=130); print('ok')
