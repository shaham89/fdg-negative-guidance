"""Fit one temperature per noise level for the class posterior p(k|x) of the exact rule (used by neg.py exact --pfit cal).
Reads out/neg/post_diag.npz (from post_diag.py: p fitted on the whole image at every step of the no-push path, and the final
ResNet-56 class of each image).  For each step it picks T on a log grid that minimises the log loss of softmax(log(p + 1e-3) / T)
against the final class.  T > 1 softens an over-confident p, T < 1 sharpens an under-confident one.
Note: post_diag.py uses the same noise (seed 0) as the 5k "not a cat" runs, so a calibrated run on seed 0 is slightly optimistic.
  python fit_cal.py [path/to/post_diag.npz]      -> out/neg/post_cal.json   (the saved table: results/neg_sweep12/post_cal.json)"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEG = os.path.join(HERE, 'out', 'neg')
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(NEG, 'post_diag.npz')
d = np.load(src)
P, final, sigma = d['p_full'].astype(np.float64), d['final'], d['sigma']      # P: images x steps x 10
grid = np.geomspace(0.3, 20, 120)


def log_loss(L, T):
    z = L / T
    z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return -lp[np.arange(len(final)), final].mean()


T = []
for i in range(P.shape[1]):
    L = np.log(P[:, i] + 1e-3)
    T.append(float(grid[int(np.argmin([log_loss(L, t) for t in grid]))]))
    print(f'sigma {sigma[i]:8.3f}  T {T[-1]:6.3f}')
os.makedirs(NEG, exist_ok=True)
json.dump(dict(sigma=[float(s) for s in sigma], T=T,
               note=f'temperature per sigma minimizing the log loss of softmax(log(p_full + 1e-3)/T) against the final ResNet-56 class, no push, {len(final)} images (post_diag.npz)'),
          open(os.path.join(NEG, 'post_cal.json'), 'w'), indent=1)
print('wrote', os.path.join(NEG, 'post_cal.json'))
