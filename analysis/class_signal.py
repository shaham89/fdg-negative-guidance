"""At sigma=80 the ideal CFG delta is (class mean - global mean), because x says almost nothing about x0.
Compare that with the delta measured from the two EDM nets (results/two_dials/diag_w1.npz).
Reads the CIFAR tarball with a restricted unpickler (no arbitrary code). Usage: python class_signal.py <cifar-10-python.tar.gz> ../results/two_dials"""
import io, os, pickle, sys, tarfile
import numpy as np

tar_path, OUT = sys.argv[1], sys.argv[2]

class Safe(pickle.Unpickler):
    ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'),
               ('numpy._core.multiarray', '_reconstruct')}
    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked {module}.{name}')

X, Y = [], []
with tarfile.open(tar_path) as t:
    for m in t.getmembers():
        if 'data_batch' in m.name:
            d = Safe(io.BytesIO(t.extractfile(m).read()), encoding='bytes').load()
            X.append(d[b'data']); Y += list(d[b'labels'])
X = np.concatenate(X).reshape(-1, 3, 32, 32).astype(np.float32) / 127.5 - 1
Y = np.array(Y)
print('train', X.shape)

def radial(H=32, W=32):
    fy, fx = np.fft.fftfreq(H), np.fft.fftfreq(W)
    return np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
r = radial()
S = 1 / (1 + np.exp(-(0.125 - r) / 0.02))   # FreqStatic soft mask, rc=.125

mu = X.mean(0)
Pcm = np.zeros((32, 32))
for c in range(10):
    F = np.fft.fft2(X[Y == c].mean(0) - mu, norm='ortho')
    Pcm += (np.abs(F) ** 2).mean(0) / 10           # mean over channels, then classes -> same convention as P_delta

for name in ['diag_w1.npz', 'diag_w3.npz']:
    d = np.load(os.path.join(OUT, name)); sig, P = d['sigma'], d['P_delta']
    print(f'\n== {name}: measured delta vs ideal class-mean delta, sigma={sig[0]:g}')
    P0 = P[0]
    def row(lbl, mask):
        m, c = (mask * P0).sum(), (mask * Pcm).sum()
        print(f'  {lbl:14s} measured {m:8.4f}  class-mean {c:8.4f}  ratio measured/class-mean {m / max(c, 1e-12):7.3f}')
    row('all', np.ones_like(r))
    dc = np.zeros_like(r); dc[0, 0] = 1
    row('DC', dc)
    row('low (no DC)', S ** 2 * (1 - dc))
    row('high', (1 - S) ** 2)
    for lo, hi in [(0.0, 0.0625), (0.0625, 0.125), (0.125, 0.1875), (0.1875, 0.25), (0.25, 0.375), (0.375, 1.0)]:
        row(f'r {lo:.3f}-{hi:.3f}', ((r >= lo) & (r < hi)).astype(float))

# per-image mean colour: how large is the class shift compared with the spread inside a class?
dc_img = X.mean(axis=(2, 3))                        # N,3
print('\n== per-image mean colour (pixels in [-1,1])')
print('  global std per channel', np.round(dc_img.std(0), 3))
within, shifts = [], []
for c in range(10):
    v = dc_img[Y == c]
    within.append(v.std(0)); shifts.append(v.mean(0) - dc_img.mean(0))
within, shifts = np.array(within), np.array(shifts)
print('  mean within-class std per channel', np.round(within.mean(0), 3))
print('  rms class shift per channel      ', np.round(np.sqrt((shifts ** 2).mean(0)), 3))
print('  rms class shift over all channels', round(float(np.sqrt((shifts ** 2).mean())), 4), ' (compare DC rms of measured delta at sigma=80: 0.0708)')
names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for c in range(10):
    print(f'  {names[c]:11s} shift RGB {np.round(shifts[c], 3)}   in within-class stds {np.round(shifts[c] / within[c], 2)}')
print('  if w=3 moved the mean colour by (w-1)*shift, typical move in within-class stds:', round(float(2 * np.sqrt(((shifts / within) ** 2).mean())), 3))

# how much is class information per band? between-class vs within-class power
Fall = np.fft.fft2(X[::5], norm='ortho'); yy = Y[::5]
Pw = np.zeros((32, 32))
for c in range(10):
    f = Fall[yy == c]; Pw += (np.abs(f - f.mean(0)) ** 2).mean(axis=(0, 1)) / 10
print('\n== between-class / within-class power per band (class-mean info only)')
for lo, hi in [(0.0, 1e-9), (1e-9, 0.0625), (0.0625, 0.125), (0.125, 0.1875), (0.1875, 0.25), (0.25, 0.375), (0.375, 1.0)]:
    m = (r >= lo) & (r < hi) if lo > 0 else (r <= hi)
    print(f'  r {lo:.3f}-{hi:.3f}: between {Pcm[m].sum():9.4f}  within {Pw[m].sum():10.2f}  ratio {Pcm[m].sum() / Pw[m].sum():.5f}')
