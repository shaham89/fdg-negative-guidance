"""The push map of pmap.py on the 10-Gaussian model of CIFAR-10 (gmm_toy.py), where every denoiser and posterior is exact.  CPU only.
Push one band in one window of two steps, away from the class the image becomes or toward another class, on the D_u path.  Same windows,
strength and reads as pmap.py (flip, any change, rms change of the final low and high band), so the real nets can be compared with a model that has
only means and spectra.  Also: how much each band alone tells about the final class at each step (exact band-only posteriors p(k | Low x), p(k | High x)).
  python gmm_pmap.py <cifar-10-python.tar.gz> [--n 1000] [--s 2] [--out gmm_pmap.jsonl]   (the saved run: results/gaussian_model)"""
import math, tarfile, pickle, argparse, json, time, numpy as np, torch

ap = argparse.ArgumentParser(); ap.add_argument('tar'); ap.add_argument('--n', type=int, default=1000); ap.add_argument('--s', type=float, default=2.0)
ap.add_argument('--out', default='gmm_pmap.jsonl'); a = ap.parse_args()
torch.set_grad_enabled(False); dt = torch.float64; K = 10; NS = 18

xs, ys = [], []
with tarfile.open(a.tar) as t:
    for i in range(1, 6):
        d = pickle.load(t.extractfile(f'cifar-10-batches-py/data_batch_{i}'), encoding='latin1'); xs.append(d['data']); ys += d['labels']
X = torch.tensor(np.concatenate(xs), dtype=dt).reshape(-1, 3, 32, 32) / 127.5 - 1; Y = torch.tensor(ys)
U = torch.linalg.eigh(torch.cov(X.permute(1, 0, 2, 3).reshape(3, -1)))[1]
toP = lambda x: torch.einsum('cd,ndhw->nchw', U.T, x); fromP = lambda z: torch.einsum('cd,ndhw->nchw', U, z)
mu = torch.stack([X[Y == c].mean(0) for c in range(K)])
P = torch.stack([(torch.fft.fft2(toP(X[Y == c] - mu[c]), norm='ortho').abs() ** 2).mean(0) for c in range(K)]).clamp(min=1e-6)
del X
fy = torch.fft.fftfreq(32, dtype=dt); r = torch.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
m = torch.sigmoid((0.125 - r) / 0.02); BANDS = {'low': m, 'high': 1 - m}; hard_low = (r < 0.125).to(dt)

def denoise_all(x, s):
    """D_k for all classes (N,K,3,32,32) and the log-likelihood terms per frequency summed over the low and the high band (N,K) each."""
    Z = torch.fft.fft2(toP((x[:, None] - mu[None]).reshape(-1, 3, 32, 32)), norm='ortho').reshape(len(x), K, 3, 32, 32)
    V = P[None] + s * s; term = -0.5 * (Z.abs() ** 2) / V - 0.5 * torch.log(V)
    ll_low = (term * hard_low).sum((2, 3, 4)); ll_high = (term * (1 - hard_low)).sum((2, 3, 4))
    Dk = mu[None] + fromP(torch.fft.ifft2(P[None] / V * Z, norm='ortho').real.reshape(-1, 3, 32, 32)).reshape(len(x), K, 3, 32, 32)
    return Dk, ll_low, ll_high

def band(v, W): return torch.fft.ifft2(torch.fft.fft2(v, norm='ortho') * W, norm='ortho').real
tt = (80 ** (1 / 7) + torch.arange(NS, dtype=dt) / (NS - 1) * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7; tt = torch.cat([tt, torch.zeros(1, dtype=dt)])

def step(x, i, push, rec=None):
    tc, tn = float(tt[i]), float(tt[i + 1])
    def D(xx, s, rec=None):
        Dk, ll, lh = denoise_all(xx, s); p = torch.softmax(ll + lh, 1); Du = (p[:, :, None, None, None] * Dk).sum(1)
        if rec is not None: rec.append((s, p, torch.softmax(ll, 1), torch.softmax(lh, 1)))
        if push is None: return Du
        k, W = push; return Du + band(Dk[torch.arange(len(xx)), k] - Du, W)
    d = (x - D(x, tc, rec)) / tc; xn = x + (tn - tc) * d
    if i < NS - 1: xn = x + (tn - tc) * 0.5 * (d + (xn - D(xn, tn)) / tn)
    return xn

def classify(x): Dk, ll, lh = denoise_all(x, 1e-3); return (ll + lh).argmax(1)

t0 = time.time(); n = a.n
x = torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(0), dtype=dt) * tt[0]; states, rec = [], []
for i in range(NS): states.append(x); x = step(x, i, None, rec)
X0 = x; c0 = classify(X0); tgt = (c0 + 1 + torch.arange(n) % 9) % 10      # no clamp: the exact classifier is very sensitive to clipped pixels
print(f'no push: {n} images, {time.time() - t0:.0f} s, class shares ' + ' '.join(f'{v:.3f}' for v in (torch.bincount(c0, minlength=10).double() / n).tolist()), flush=True)
print('How much each band alone tells about the final class (exact posteriors): accuracy of the top class and mean p of the final class, full / low only / high only')
info = []
for s, p, pl, ph in rec:
    acc = [float((q.argmax(1) == c0).double().mean()) for q in (p, pl, ph)]; pf = [float(q[torch.arange(n), c0].mean()) for q in (p, pl, ph)]
    info.append(dict(sigma=s, acc=acc, p_final=pf)); print(f'  sigma {s:8.3f}: acc {acc[0]:.2f}/{acc[1]:.2f}/{acc[2]:.2f}   p_final {pf[0]:.2f}/{pf[1]:.2f}/{pf[2]:.2f}', flush=True)
rms = lambda d, W: band(d, W).pow(2).mean((1, 2, 3)).sqrt().mean().item()
rows = []
for bname, W0 in BANDS.items():
    for dr in ['away', 'toward']:
        for i0, i1 in [(2 * j, 2 * j + 2) for j in range(NS // 2)] + [(0, NS)]:
            t1 = time.time(); k = c0 if dr == 'away' else tgt; W = (-a.s if dr == 'away' else a.s) * W0; x = states[i0]
            for i in range(i0, NS): x = step(x, i, (k, W) if i < i1 else None)
            Xa = x; c = classify(Xa); d = Xa - X0; hit = (c != c0) if dr == 'away' else (c == tgt)
            row = dict(band=bname, dir=dr, i0=i0, i1=i1, sig_hi=float(tt[i0]), sig_lo=float(tt[i1 - 1]), s=a.s, n=n, flip=float(hit.double().mean()),
                       changed=float((c != c0).double().mean()), other=float(((c != c0) & (c != k)).double().mean()), rms_low=rms(d, BANDS['low']), rms_high=rms(d, BANDS['high']))
            rows.append(row)
            print(f"{bname:4s} {dr:6s} steps {i0:2d}-{i1 - 1:2d} sigma {row['sig_hi']:7.3f}-{row['sig_lo']:6.3f}: flip {row['flip']:.3f} changed {row['changed']:.3f} other {row['other']:.3f}"
                  f" | rms low {row['rms_low']:.4f} high {row['rms_high']:.4f}  {time.time() - t1:.0f} s", flush=True)
json.dump(dict(info=info, rows=rows), open(a.out.replace('.jsonl', '.json'), 'w'), indent=1)
print(f'done {(time.time() - t0) / 60:.1f} min')
