"""Free pre-test of the mean-colour idea, from the saved 64-image grids.
All grids share seeds and labels (image i has class i%10), so arm-vs-w1 differences are paired.
Question: does guidance move the per-image mean colour along (class mean colour - global mean colour)?
Fit: mean_colour(arm) - mean_colour(w=1) = k * class_shift. CFG with no decay would give k = w-1.
Usage: python colour_shift.py <cifar-10-python.tar.gz> ../results/two_dials/figures"""
import io, os, pickle, sys, tarfile
import numpy as np
from PIL import Image

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
dc = X.mean(axis=(2, 3))
shift = np.stack([dc[Y == c].mean(0) - dc.mean(0) for c in range(10)])     # 10,3
within = np.stack([dc[Y == c].std(0) for c in range(10)])

def load_grid(name):
    a = np.asarray(Image.open(os.path.join(OUT, name)).convert('RGB')).astype(np.float32)
    assert a.shape[:2] == (274, 274), a.shape                               # 8*(32+2)+2 with torchvision padding=2
    imgs = [a[2 + 34 * r:34 + 34 * r, 2 + 34 * c:34 + 34 * c] for r in range(8) for c in range(8)]
    return (np.stack(imgs) - 127.5) / 127.5                                 # 64,32,32,3 in [-1,1]

arms = ['cfg_w1.5', 'cfg_w2', 'cfg_w3', 'cfg_w4', 'cfg_w6',
        'freq_wl1_wh3_rc0.125', 'freq_wl1_wh6_rc0.125', 'freq_wl3_wh1_rc0.125', 'freq_wl2_wh4_rc0.125', 'freq_wl1.5_wh3_rc0.125']
base = load_grid('grid_cfg_w1.png'); labs = np.arange(64) % 10
base_dc = base.mean(axis=(1, 2)); s = shift[labs]                           # 64,3
print('per-image colour spread of the base arm (std over images, per channel):', np.round(base_dc.std(0), 3))
print(f"{'arm':28s} {'k':>6s} {'+-2se':>6s} {'R2':>6s}  {'mean |dcolour|':>14s}  {'img std ratio':>13s}  {'sharpness ratio':>19s}")
rng = np.random.default_rng(0)
for arm in arms:
    g = load_grid(f'grid_{arm}.png'); g_dc = g.mean(axis=(1, 2)); d = (g_dc - base_dc)     # paired
    k = (d * s).sum() / (s * s).sum(); res = d - k * s
    boots = []
    for _ in range(2000):
        i = rng.integers(0, 64, 64); boots.append((d[i] * s[i]).sum() / (s[i] * s[i]).sum())
    r2 = 1 - (res ** 2).sum() / (d ** 2).sum()
    hf = lambda x: np.sqrt(((x[:, 1:] - x[:, :-1]) ** 2).mean())            # rms of vertical pixel differences = crude sharpness
    print(f"{arm:28s} {k:6.2f} {2 * np.std(boots):6.2f} {r2:6.2f}  {np.abs(d).mean():14.4f}  {g_dc.std() / base_dc.std():13.3f}  {hf(g) / hf(base):19.3f}")

print('\nper class, cfg_w3 minus w1 (RGB mean colour shift) vs (w-1)*class shift = 2*shift:')
g3 = load_grid('grid_cfg_w3.png').mean(axis=(1, 2)) - base_dc
f13 = load_grid('grid_freq_wl1_wh3_rc0.125.png').mean(axis=(1, 2)) - base_dc
f31 = load_grid('grid_freq_wl3_wh1_rc0.125.png').mean(axis=(1, 2)) - base_dc
names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for c in range(10):
    m = labs == c
    print(f'  {names[c]:11s} n={m.sum()}  2*shift {np.round(2 * shift[c], 2)}  cfg3 {np.round(g3[m].mean(0), 2)}  (1,3) {np.round(f13[m].mean(0), 2)}  (3,1) {np.round(f31[m].mean(0), 2)}')
