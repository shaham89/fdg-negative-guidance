"""Estimate the radially-averaged per-coefficient power spectrum P(|f|) of CIFAR-10 (pixels in [-1,1], ortho FFT).
Used for the per-frequency SNR:  SNR_t(f) = P(f) / sigma_t^2.   Also plots the SNR=1 crossover sigma per band."""
import os, numpy as np, torch, torchvision
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import OUT, radial_freq, radial_profile, edm_sigmas, BAND_EDGES

os.makedirs(OUT, exist_ok=True)
ds = torchvision.datasets.CIFAR10(os.path.join(OUT, 'data'), train=True, download=True)
x = torch.tensor(ds.data[:20000]).permute(0, 3, 1, 2).float().cuda() / 127.5 - 1  # N,3,32,32
F = torch.fft.fft2(x, norm='ortho')
P = (F.real ** 2 + F.imag ** 2).mean(dim=(0, 1))   # H,W  mean power per coefficient (incl. DC)
r = radial_freq(32, 32, 'cuda')
prof, centers = radial_profile(P, r, nbins=64)
keep = prof > 0                       # a 32x32 grid has few distinct radii -> drop empty bins so interpolation is sane
prof, centers = prof[keep].cpu().numpy(), centers[keep].cpu().numpy()
# smooth in log-log (few coefficients per radius bin -> noisy); keep DC + first bins exact, smooth the rest with a 5-bin running mean
logP = np.log(prof); sm = logP.copy()
for i in range(2, len(logP)): lo, hi = max(0, i - 2), min(len(logP), i + 3); sm[i] = logP[lo:hi].mean()
np.savez(os.path.join(OUT, 'cifar_spectrum.npz'), r=centers, P=np.exp(sm), P_raw=prof, P2d=P.cpu().numpy())
prof = torch.tensor(np.exp(sm)); centers = torch.tensor(centers)

# crossover: sigma at which SNR = 1 for each radius  ->  sigma* = sqrt(P)
sig_star = torch.sqrt(prof).cpu().numpy(); c = centers.cpu().numpy()
class _N: sigma_min, sigma_max = 0.002, 80.0
_N.round_sigma = staticmethod(lambda t: t)
sig = edm_sigmas(_N, 18, device='cpu').numpy()[:-1]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].loglog(c, prof.cpu().numpy()); ax[0].set_xlabel('|f| (cycles/px)'); ax[0].set_ylabel('per-coef power P(f)'); ax[0].set_title('CIFAR-10 radial power spectrum')
ax[1].semilogy(c, sig_star, label=r'$\sigma^*(f)=\sqrt{P(f)}$  (SNR=1)')
for s in sig: ax[1].axhline(s, color='gray', lw=0.4)
for e in BAND_EDGES[1:-1]: ax[1].axvline(e, color='C3', lw=0.5, ls='--')
ax[1].set_xlabel('|f| (cycles/px)'); ax[1].set_ylabel(r'$\sigma$'); ax[1].set_title('When is each frequency resolved? (gray = 18 Heun steps)'); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'spectrum.png'), dpi=130)
print('sigma* per band edge:', {f'{e:.3f}': float(np.interp(e, c, sig_star)) for e in BAND_EDGES[1:-1]})
print('EDM sigmas:', np.round(sig, 3))
