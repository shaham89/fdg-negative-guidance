"""CIFAR-10 as a 10-class Gaussian mixture with exact denoisers: does it reproduce "which band removes a class"?  CPU only.
Each class k: x = mu_k + U z, z with independent Fourier coefficients of variance P_k (per colour-PCA channel and frequency),
so D_k(x; sigma) = mu_k + U iFFT( P_k / (P_k + sigma^2) FFT(U^T (x - mu_k)) ), and p(k|x) is exact.
  python gmm_toy.py <cifar-10-python.tar.gz> [--n 2000] [--k 3]   (the saved runs: results/gaussian_model)
Reports per arm: share read as k (exact posterior), Frechet distance to exact "not k" samples, and paired class change against no push."""
import sys, math, tarfile, pickle, argparse, numpy as np, torch

p = argparse.ArgumentParser(); p.add_argument('tar'); p.add_argument('--n', type=int, default=2000); p.add_argument('--k', type=int, default=3)
p.add_argument('--steps', type=int, default=18); p.add_argument('--only', default='', help='regex: run only matching arms (no push always runs)'); a = p.parse_args()
torch.set_grad_enabled(False); dt = torch.float64; K = 10; k = a.k

# ---- fit the mixture to CIFAR-10 train
xs, ys = [], []
with tarfile.open(a.tar) as t:
    for i in range(1, 6):
        d = pickle.load(t.extractfile(f'cifar-10-batches-py/data_batch_{i}'), encoding='latin1')
        xs.append(d['data']); ys += d['labels']
X = torch.tensor(np.concatenate(xs), dtype=dt).reshape(-1, 3, 32, 32) / 127.5 - 1; Y = torch.tensor(ys)
U = torch.linalg.eigh(torch.cov(X.permute(1, 0, 2, 3).reshape(3, -1)))[1]            # colour PCA, shared by all classes
toP = lambda x: torch.einsum('cd,ndhw->nchw', U.T, x); fromP = lambda z: torch.einsum('cd,ndhw->nchw', U, z)
mu = torch.stack([X[Y == c].mean(0) for c in range(K)])
P = torch.stack([(torch.fft.fft2(toP(X[Y == c] - mu[c]), norm='ortho').abs() ** 2).mean(0) for c in range(K)]).clamp(min=1e-6)   # K,3,32,32
fy = torch.fft.fftfreq(32, dtype=dt); r = torch.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
lowmask = torch.sigmoid((0.125 - r) / 0.02)                                           # the project's soft mask, r_c = 0.125
print('energy share of class-mean differences in the low band:', float(((torch.fft.fft2(mu - mu.mean(0), norm='ortho').abs() ** 2) * lowmask).sum()
      / (torch.fft.fft2(mu - mu.mean(0), norm='ortho').abs() ** 2).sum()))

def denoise_all(x, s):
    """D_k for all classes (N,K,3,32,32) and log N(x; mu_k, C_k + s^2) (N,K)."""
    R =(x[:, None] - mu[None]).reshape(-1, 3, 32, 32)
    Z = torch.fft.fft2(toP(R), norm='ortho').reshape(len(x), K, 3, 32, 32)
    V = P[None] + s * s
    logl = -0.5 * ((Z.abs() ** 2) / V).sum((2, 3, 4)) - 0.5 * torch.log(V).sum((2, 3, 4))
    Dk = mu[None] + fromP(torch.fft.ifft2(P[None] / V * Z, norm='ortho').real.reshape(-1, 3, 32, 32)).reshape(len(x), K, 3, 32, 32)
    return Dk, logl

def band(v, W):   # apply a spectral weight W (32,32) to v (N,3,32,32)
    return torch.fft.ifft2(torch.fft.fft2(v, norm='ortho') * W, norm='ortho').real

def guided(rule):
    def D(x, s):
        Dk, logl = denoise_all(x, s); p = torch.softmax(logl, 1); Du = (p[:, :, None, None, None] * Dk).sum(1); d = Dk[:, k] - Du
        kind, A, B, lo, hi = rule
        if not (lo <= s <= hi) or kind == 'none': return Du
        W = A * lowmask + B * (1 - lowmask)
        if kind == 'neg': return Du - band(d, W)
        if kind == 'exact':
            g = (p[:, k] / (1 - p[:, k]).clamp(min=1e-9)).clamp(max=20)[:, None, None, None]
            return Du - g * band(d, W)
    return D

def sample(rule, z0):
    t = (80 ** (1 / 7) + torch.arange(a.steps, dtype=dt) / (a.steps - 1) * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7; t = torch.cat([t, torch.zeros(1, dtype=dt)])
    D = guided(rule); x = z0 * t[0]
    for i in range(a.steps):
        dc = (x - D(x, float(t[i]))) / t[i]; xn = x + (t[i + 1] - t[i]) * dc
        if i < a.steps - 1: xn = x + (t[i + 1] - t[i]) * 0.5 * (dc + (xn - D(xn, float(t[i + 1]))) / t[i + 1])
        x = xn
    return x

def classify(x): return denoise_all(x, 1e-3)[1].argmax(1)

def fd(A, B):
    """Frechet distance between two sample sets (N,3,32,32), exact in pixel space through the N x N cross matrix."""
    A = A.reshape(len(A), -1); B = B.reshape(len(B), -1); ma, mb = A.mean(0), B.mean(0); Ac, Bc = A - ma, B - mb
    tr = (Ac ** 2).sum() / len(A) + (Bc ** 2).sum() / len(B)
    return float(((ma - mb) ** 2).sum() + tr - 2 * torch.linalg.svdvals(Bc @ Ac.T / math.sqrt(len(A) * len(B))).sum())

g = torch.Generator().manual_seed(0); n = a.n
z0 = torch.randn(n, 3, 32, 32, generator=g, dtype=dt)
def exact_notk(m, seed):
    gg = torch.Generator().manual_seed(seed); others = torch.tensor([c for c in range(K) if c != k])
    cls = others[torch.randint(0, K - 1, (m,), generator=gg)]
    e = torch.fft.ifft2(torch.sqrt(P[cls]) * torch.fft.fft2(torch.randn(m, 3, 32, 32, generator=gg, dtype=dt), norm='ortho'), norm='ortho').real
    return mu[cls] + fromP(e)
ref = exact_notk(n, 1); ref2 = exact_notk(n, 2)
print(f'FD noise floor (two exact "not k" sets): {fd(ref, ref2):.2f}   |  reading k on exact not-k samples: {(classify(ref) == k).double().mean():.4f}')
x0 = sample(('none', 0, 0, 0, 1e9), z0); c0 = classify(x0)
print(f"{'arm':28s} {'k left':>7s} {'FD':>7s} {'kept':>6s} {'changed':>8s} {'near kept':>9s}")
near = 5
arms = [('no push', ('none', 0, 0, 0, 1e9))] + [(f'low {s:g}', ('neg', s, 0, 0, 1e9)) for s in (0.5, 1, 2)] + [(f'high {s:g}', ('neg', 0, s, 0, 1e9)) for s in (0.5, 1, 2, 4)] \
     + [(f'both {s:g}', ('neg', s, s, 0, 1e9)) for s in (0.5, 1)] + [('high 2, sigma>=0.8', ('neg', 0, 2, 0.8, 1e9)), ('high 2, sigma<0.8', ('neg', 0, 2, 0, 0.8)),
        ('low 1, sigma>=0.8', ('neg', 1, 0, 0.8, 1e9))] + [(f'exact ({A:g},{B:g})', ('exact', A, B, 0, 1e9)) for A, B in [(1, 1), (2, 1), (1, 2), (2, 2), (1.5, 1.5), (1, 0), (0, 1), (0.5, 0.5)]]
import re
for name, rule in arms:
    if a.only and rule[0] != 'none' and not re.search(a.only, name): continue
    x = x0 if rule[0] == 'none' else sample(rule, z0); c = classify(x)
    keep = (c[c0 == k] == k).double().mean(); ch = (c[c0 != k] != c0[c0 != k]).double().mean(); nk = (c[c0 == near] == near).double().mean()
    print(f'{name:28s} {100 * (c == k).double().mean():6.2f}% {fd(x, ref):7.2f} {keep:6.2f} {ch:8.2f} {nk:9.2f}', flush=True)
