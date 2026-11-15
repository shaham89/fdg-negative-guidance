"""When does each band decide the class?  (The push-response map and the band transfer.)
On the unguided path (D_u), push one band during one window of two sampler steps, toward a class or away from it:
  away:   k = the class the image becomes with no push,  D = D_u - s Band(D_k - D_u) inside the window, D_u elsewhere
  toward: k = another class (fixed per image),           D = D_u + s Band(D_k - D_u)
Both directions use the same base path, so toward and away are compared fairly (sweep7b pushed toward a class on the D_c path).
Reads, against no push (paired by noise): the flip rate (away: share no longer class k; toward: share that became k), any class change,
the low-band reader on the first n_read images, and the rms change of the final image in the low and the high band
(band transfer: does a push in one band change the other band of the final image?).
Each arm starts from the stored no-push state at its window, so it only runs the remaining steps.  The 9 windows split the 18 steps.
  python pmap.py --n 4000 --s 2         # 9 windows x 2 bands x 2 directions + 4 all-step arms, about 40 min on an A100
Output: out/neg/pmap_s{s}.jsonl, one line per arm (written as each arm ends); out/neg/pmap_imgs_s{s}.npz holds the first 64 images of each arm."""
import os, json, argparse, time, numpy as np, torch
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=4000); ap.add_argument('--bs', type=int, default=500); ap.add_argument('--s', type=float, default=2.0)
ap.add_argument('--n_read', type=int, default=1000); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tag', default=''); a = ap.parse_args(); dev = 'cuda'
NEG = os.path.join(OUT, 'neg'); os.makedirs(NEG, exist_ok=True); OUTF = os.path.join(NEG, f'pmap_s{a.s:g}{a.tag}.jsonl')
torch.set_grad_enabled(False)
net_c, net_u = load_nets(dev); t = edm_sigmas(net_c, 18, device=dev); NS = len(t) - 1
R = radial_freq(32, 32, dev); m = torch.sigmoid((0.125 - R) / 0.02); BANDS = {'low': m, 'high': 1 - m}
eye = torch.eye(10, device=dev)
clf = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True, verbose=False, trust_repo=True).to(dev).eval()
MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.2470, 0.2435, 0.2616], device=dev).view(1, 3, 1, 1)
def resnet(u8): return torch.cat([clf((u8[i:i + 1000].to(dev).float() / 255 - MEAN) / STD).argmax(1).cpu() for i in range(0, len(u8), 1000)])
clf2 = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_vgg16_bn', pretrained=True, verbose=False, trust_repo=True).to(dev).eval()
def vgg(u8): return torch.cat([clf2((u8[i:i + 1000].to(dev).float() / 255 - MEAN) / STD).argmax(1).cpu() for i in range(0, len(u8), 1000)])

_reader, READER = None, os.path.join(NEG, 'reader.json')
def low_read(imgs, sig, noises=2, bs=200):
    """Same as neg.py's reader: keep only Low(x), add noise, score each class by how well the VE net's prediction fits in the low band."""
    global _reader
    if _reader is None: _reader = load_nets(dev, arch='ve')[0]
    g = torch.Generator(device=dev).manual_seed(0); sig32 = torch.tensor(sig, dtype=torch.float32, device=dev)
    low = lambda x: spec_apply(x, m); score = torch.zeros(len(imgs), 10, dtype=torch.float64)
    for _ in range(noises):
        for i in range(0, len(imgs), bs):
            xl = low(imgs[i:i + bs].to(dev).float() / 127.5 - 1); xt = xl + sig * torch.randn(xl.shape, generator=g, device=dev)
            Dk = class_preds(_reader, xt, sig32)
            score[i:i + bs] -= (low(Dk.reshape(-1, 3, 32, 32)).reshape(Dk.shape) - xl[:, None]).pow(2).sum((2, 3, 4)).cpu()
    return score.argmax(1)
rsig = json.load(open(READER))['sigma'] if os.path.exists(READER) else None

def step(x, i, push):
    """One EDM Heun step, as heun_step.  push: None (D_u) or (onehot of k, spectral weight W with sign and strength)."""
    tc, tn = t[i], t[i + 1]
    def D(xx, s):
        x32, s32 = xx.float(), s.float(); Du = net_u(x32, s32, None).double()
        if push is None: return Du
        return Du + spec_apply(net_c(x32, s32, push[0]).double() - Du, push[1])
    d = (x - D(x, tc)) / tc; xn = x + (tn - tc) * d
    if i < NS - 1: xn = x + (tn - tc) * 0.5 * (d + (xn - D(xn, tn)) / tn)
    return xn

t0 = time.time(); states, X0 = [], []
for lo in range(0, a.n, a.bs):
    x = init_noise(min(lo + a.bs, a.n) - lo, lo + 100000 * a.seed, dev) * t[0]; st = []
    for i in range(NS): st.append(x); x = step(x, i, None)
    states.append(st); X0.append(x.float().clamp(-1, 1))
X0 = torch.cat(X0); U0 = to_uint8(X0).cpu(); c0 = resnet(U0); n = len(c0)
tgt = (c0 + 1 + torch.arange(n) % 9) % 10                                  # toward: one of the 9 other classes, spread evenly
L0 = low_read(U0[:a.n_read], rsig) if rsig else None
print(f'no push: {n} images in {(time.time() - t0) / 60:.1f} min; class shares ' + ' '.join(f'{v:.3f}' for v in (torch.bincount(c0, minlength=10).float() / n).tolist()), flush=True)
rms = lambda d, W: spec_apply(d, W).pow(2).mean((1, 2, 3)).sqrt().mean().item()
v0 = vgg(U0)

# Push size on the no-push path: rms of Band(D_k - D_u) at each step, for k = own class (away) and k = target (toward).  The push is s times this.
SIZEF = os.path.join(NEG, f'pmap_size{a.tag}.json')
if not os.path.exists(SIZEF):
    size = []
    for i in range(NS):
        acc = {}
        for b, lo in enumerate(range(0, n, a.bs)):
            x = states[b][i]; x32, s32 = x.float(), t[i].float(); Du = net_u(x32, s32, None).double()
            for d, kk in [('away', c0), ('toward', tgt)]:
                Dk = net_c(x32, s32, eye[kk[lo:lo + a.bs].to(dev)]).double()
                for bn, W in BANDS.items(): acc[f'{bn}_{d}'] = acc.get(f'{bn}_{d}', 0.0) + spec_apply(Dk - Du, W).pow(2).mean((1, 2, 3)).sqrt().sum().item()
        size.append(dict(step=i, sigma=float(t[i]), **{key: v / n for key, v in acc.items()}))
        print(f"push size at sigma {float(t[i]):7.3f}: low away {size[-1]['low_away']:.4f} toward {size[-1]['low_toward']:.4f} | high away {size[-1]['high_away']:.4f} toward {size[-1]['high_toward']:.4f}", flush=True)
    json.dump(size, open(SIZEF, 'w'), indent=1)

wins = [(2 * j, 2 * j + 2) for j in range(NS // 2)] + [(0, NS)]
arms = [(b, d, i0, i1) for b in BANDS for d in ['away', 'toward'] for i0, i1 in wins]
done = set()
if os.path.exists(OUTF): done = {(r['band'], r['dir'], r['i0'], r['i1']) for r in map(json.loads, open(OUTF))}
grids = {'none': U0[:64].numpy()}
for band, dr, i0, i1 in arms:
    if (band, dr, i0, i1) in done: continue
    t1 = time.time(); k = c0 if dr == 'away' else tgt; W = (-a.s if dr == 'away' else a.s) * BANDS[band]; outs = []
    for b, lo in enumerate(range(0, n, a.bs)):
        x = states[b][i0]; oh = eye[k[lo:lo + a.bs].to(dev)]
        for i in range(i0, NS): x = step(x, i, (oh, W) if i < i1 else None)
        outs.append(x.float().clamp(-1, 1))
    X = torch.cat(outs); U = to_uint8(X).cpu(); c = resnet(U); v = vgg(U); dX = X - X0
    hit = (c != c0) if dr == 'away' else (c == tgt)
    row = dict(band=band, dir=dr, i0=i0, i1=i1, sig_hi=float(t[i0]), sig_lo=float(t[i1 - 1]), s=a.s, n=n, seed=a.seed,
               flip=float(hit.float().mean()), changed=float((c != c0).float().mean()),
               other=float(((c != c0) & (c != k)).float().mean()),                      # toward: changed, but not to the target
               flip_by_class=[float(hit[c0 == j].float().mean()) for j in range(10)],   # by the no-push class
               rms_low=rms(dX, BANDS['low']), rms_high=rms(dX, BANDS['high']), n_blank=int((U.float().std((1, 2, 3)) < 2).sum()),
               flip_vgg=float(((v != v0) if dr == 'away' else (v == tgt)).float().mean()), changed_vgg=float((v != v0).float().mean()))   # VGG-16 against its own no-push class
    if L0 is not None:
        L = low_read(U[:a.n_read], rsig); kk = k[:a.n_read]
        row.update(low_k_base=float((L0 == kk).float().mean()), low_k=float((L == kk).float().mean()), low_changed=float((L != L0).float().mean()))
    row['time'] = time.time() - t1
    with open(OUTF, 'a') as f: f.write(json.dumps(row) + '\n')
    grids[f'{band}_{dr}_{i0}-{i1}'] = U[:64].numpy()
    print(f"{band:4s} {dr:6s} steps {i0:2d}-{i1 - 1:2d} sigma {row['sig_hi']:7.3f}-{row['sig_lo']:6.3f}: flip {row['flip']:.3f} (vgg {row['flip_vgg']:.3f}) changed {row['changed']:.3f} other {row['other']:.3f}"
          f" | rms low {row['rms_low']:.4f} high {row['rms_high']:.4f}" + (f" | low reader k {row['low_k_base']:.3f} -> {row['low_k']:.3f}" if L0 is not None else '') + f"  {row['time']:.0f} s", flush=True)
np.savez_compressed(os.path.join(NEG, f'pmap_imgs_s{a.s:g}{a.tag}.npz'), **grids)
print(f'PMAP DONE {(time.time() - t0) / 60:.1f} min', flush=True)
