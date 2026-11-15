"""Which band removes a class?  'Not class k':  D = Du - a Low(Dk - Du) - b High(Dk - Du).  FDG's split applied to negative guidance.
Each arm is read by two classifiers, by a low-band reader (see low_read), and by FID against real images without class k.

  python neg.py reader                     # check the low-band reader on real test images and pick its sigma, about 5 min
  python neg.py base --k 3                 # no push (D_u), and the correct answer: labels from the 9 other classes
  python neg.py arm --k 3 --a 1 --b 0      # --lo, --hi: push only for sigma in [lo, hi], D_u elsewhere; --dc 0 leaves the mean colour alone
  python neg.py ft --k 3 --push 1          # push each frequency only while the noise still hides it (FreqTime, --mode, --mu, --s)
  python neg.py exact --k 3 --a 1 --b 1 --n 2000    # the exact rule, strength p/(1-p) per image; --xmode mix; 11 network calls per evaluation
  python neg.py pair --k 3 --a 0 --b 2     # push direction from the VE pair
  python neg.py posneg --c 5 --k 3 --w_low 1 --w_high 3   # "a dog, not a cat": FDG with class k in place of the unconditional net; --cfg W = plain CFG with D_u
  --seed 1 gives a new noise set (names end in _seed1); seed 0 is the original.
Results -> out/neg/results.jsonl ; samples -> out/neg/samples_*.npz ; an arm with a saved row is skipped.
"""
import os, json, argparse, time, numpy as np, torch
from common import *

NEG = os.path.join(OUT, 'neg'); os.makedirs(NEG, exist_ok=True)
RES, READER = os.path.join(NEG, 'results.jsonl'), os.path.join(NEG, 'reader.json')
p = argparse.ArgumentParser()
p.add_argument('cmd', choices=['reader', 'base', 'arm', 'ft', 'exact', 'pair', 'posneg'])
p.add_argument('--k', type=int, default=3); p.add_argument('--a', type=float, default=0); p.add_argument('--b', type=float, default=0)
p.add_argument('--lo', type=float, default=0.0); p.add_argument('--hi', type=float, default=float('inf'))
p.add_argument('--sigmas', type=float, nargs='+', default=[0.6, 1.2, 2, 3.25, 5, 8])
p.add_argument('--n', type=int, default=5000); p.add_argument('--bs', type=int, default=500); p.add_argument('--n_read', type=int, default=2000); p.add_argument('--tag', default='')
p.add_argument('--seed', type=int, default=0); p.add_argument('--dc', type=float); p.add_argument('--push', type=float, default=1.0); p.add_argument('--mode', default='unresolved')
p.add_argument('--mu', type=float, default=0.0); p.add_argument('--s', type=float, default=1.0); p.add_argument('--xmode', default='ratio'); p.add_argument('--cap', type=float, default=20.0)
p.add_argument('--c', type=int, default=5); p.add_argument('--w_low', type=float, default=1.0); p.add_argument('--w_high', type=float, default=1.0); p.add_argument('--cfg', type=float)
p.add_argument('--pfit', default='full', choices=['full', 'low', 'high', 'band', 'cal'], help='exact: fit p(k|x) on the whole image, one band, or each band for its own push; cal: whole image with the temperatures of out/neg/post_cal.json')
p.add_argument('--low_lo', type=float, default=0.0, help='posneg: low band pushed only for sigma >= low_lo; --dc sets the mean-colour weight (1 = no push)')
a = p.parse_args(); dev = 'cuda'
R = radial_freq(32, 32, dev)
low = lambda x: x - high_pass(x, R, 0.125, 0.02)
others = lambda k: [c for c in range(10) if c != k]

_reader = None
@torch.no_grad()
def low_read(imgs, sig, noises=2, bs=200):
    """Which class fits the coarse view?  Keep only Low(x), add noise, and score each class by how well its prediction fits Low(x), in the low band only.
    The high band of x never reaches the net, so this reader cannot use it.  It uses the VE net, which did not make the images.  imgs: uint8 N,3,32,32.  Returns predictions (N,)."""
    global _reader
    if _reader is None: _reader = load_nets(dev, arch='ve')[0]
    g = torch.Generator(device=dev).manual_seed(0); sig32 = torch.tensor(sig, dtype=torch.float32, device=dev)
    score = torch.zeros(len(imgs), 10, dtype=torch.float64)
    for _ in range(noises):
        for i in range(0, len(imgs), bs):
            xl = low(imgs[i:i + bs].to(dev).float() / 127.5 - 1)
            xt = xl + sig * torch.randn(xl.shape, generator=g, device=dev)
            Dk = class_preds(_reader, xt, sig32)                                          # same noise for all classes
            score[i:i + bs] -= (low(Dk.reshape(-1, 3, 32, 32)).reshape(Dk.shape) - xl[:, None]).pow(2).sum((2, 3, 4)).cpu()
    return score.argmax(1)

@torch.no_grad()
def clf_preds(cn, imgs):
    clf = torch.hub.load('chenyaofo/pytorch-cifar-models', cn, pretrained=True, verbose=False, trust_repo=True).to(dev).eval()
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.2470, 0.2435, 0.2616], device=dev).view(1, 3, 1, 1)
    return torch.cat([clf((imgs[i:i + 1000].to(dev).float() / 255 - mean) / std).argmax(1).cpu() for i in range(0, len(imgs), 1000)])

hist = lambda pred: (torch.bincount(pred, minlength=10).float() / len(pred)).tolist()

def run(name, labels_fn, G, k, neg_fn=None, ref=None):
    name += (f'_seed{a.seed}' if a.seed else '') + a.tag
    if os.path.exists(RES) and name in [json.loads(l)['name'] for l in open(RES)]: print(name, 'already done'); return
    net_c, net_u = load_nets(dev); t0 = time.time(); xs, clipf = [], []
    for lo in range(0, a.n, a.bs):
        hi = min(lo + a.bs, a.n)
        x = sample(net_c, net_u, labels_fn(lo, hi), G, seed=lo + 100000 * a.seed, device=dev, neg_labels=neg_fn(lo, hi) if neg_fn else None); clipf.append(clip_fraction(x)); xs.append(to_uint8(x).cpu())
    imgs = torch.cat(xs); gen_time = time.time() - t0
    np.savez_compressed(os.path.join(NEG, f'samples_{name}.npz'), images=imgs.permute(0, 2, 3, 1).numpy())
    import torchvision; torchvision.utils.save_image(imgs[:64].float() / 255, os.path.join(NEG, f'grid_{name}.png'), nrow=8)
    res = dict(name=name, k=k, cmd=a.cmd, rule=G.name, seed=a.seed, a=a.a, b=a.b, lo=a.lo, hi=a.hi if a.hi != float('inf') else None, n=a.n, n_blank=int((imgs.float().std((1, 2, 3)) < 2).sum()), clip_frac=float(np.mean(clipf)), gen_time=gen_time,
               hist_resnet=hist(clf_preds('cifar10_resnet56', imgs)), hist_vgg=hist(clf_preds('cifar10_vgg16_bn', imgs)))
    if os.path.exists(READER): res.update(hist_low=hist(low_read(imgs[:a.n_read], json.load(open(READER))['sigma'])), n_read=min(a.n_read, a.n))
    if hasattr(G, 'log'): res['gate'] = G.log[:35]            # first batch: sigma, mean p(k|x), mean strength
    ref_classes = ref or others(k); res.update(fid_is(imgs, cifar_train_uint8(ref_classes), 'cifar10-train-' + ''.join(map(str, ref_classes))))
    res['total_time'] = time.time() - t0
    with open(RES, 'a') as f: f.write(json.dumps(res) + '\n')
    print(f"{name:40s} FID={res['fid']:.2f} IS={res['is']:.2f} k: resnet {res['hist_resnet'][k]:.4f} vgg {res['hist_vgg'][k]:.4f} low {res.get('hist_low', [float('nan')] * 10)[k]:.3f}  mix=" + ' '.join(f'{h:.2f}' for h in res['hist_resnet']) + f"  {res['total_time'] / 60:.1f} min", flush=True)

if a.cmd == 'reader':     # real test images, 100 per class: the reader must beat chance (0.10) and beat ResNet-56 on the same blurred images
    import torchvision
    ds = torchvision.datasets.CIFAR10(os.path.join(OUT, 'data'), train=False, download=False)
    x = torch.tensor(ds.data).permute(0, 3, 1, 2).contiguous(); y = torch.tensor(ds.targets)
    idx = torch.cat([(y == c).nonzero()[:100, 0] for c in range(10)]); x, y = x[idx], y[idx]
    xl = ((low(x.to(dev).float() / 127.5 - 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
    show = lambda P: f'acc {(P == y).float().mean().item():.3f}  reads cat: real cats {(P[y == 3] == 3).float().mean().item():.3f}, others {(P[y != 3] == 3).float().mean().item():.3f}'
    print('ResNet-56 on Low(x): ', show(clf_preds('cifar10_resnet56', xl)), flush=True)
    print('ResNet-56 on x:      ', show(clf_preds('cifar10_resnet56', x)), flush=True)
    rows = []
    for s in a.sigmas:
        P = low_read(x, s); rows.append(dict(sigma=s, acc=(P == y).float().mean().item(), cat=(P[y == 3] == 3).float().mean().item(), other=(P[y != 3] == 3).float().mean().item()))
        print(f'reader sigma={s:<5g}', show(P), flush=True)
    best = max(rows, key=lambda d: d['acc']); json.dump(best, open(READER, 'w')); print('chosen', best)
elif a.cmd == 'base':
    k = a.k; oth = torch.tensor(others(k), device=dev)
    run(f'neg{k}_none', lambda lo, hi: torch.full((hi - lo,), k, device=dev), NegBand(0, 0), k)
    run(f'neg{k}_oracle', lambda lo, hi: oth[torch.arange(lo, hi, device=dev) % 9], ConstCFG(1.0), k)
elif a.cmd == 'posneg':   # all images ask for class c; FID against real images of class c
    c, k = a.c, a.k; G = ConstCFG(a.cfg) if a.cfg is not None else PosNeg(a.w_low, a.w_high, low_lo=a.low_lo, dc=a.dc)
    run(f'pn{c}not{k}_' + G.name, lambda lo, hi: torch.full((hi - lo,), c, device=dev), G, k, None if a.cfg is not None else (lambda lo, hi: torch.full((hi - lo,), k, device=dev)), [c])
else:
    k = a.k
    if a.cmd == 'arm': G = NegBand(a.a, a.b, a.lo, a.hi, dc=a.dc); name = G.name[4:]
    elif a.cmd == 'ft': sr, sP = load_spectrum(); G = FreqTime(sr, sP, 0.0, -a.push, a.mode, a.mu, a.s); name = G.name
    elif a.cmd == 'exact': G = NegExact(k, a.a, a.b, a.xmode, a.cap, pfit=a.pfit, cal=json.load(open(os.path.join(NEG, 'post_cal.json'))) if a.pfit == 'cal' else None); name = G.name[3:]
    elif a.cmd == 'pair': G = NegPair(*load_nets(dev, arch='ve'), a.a, a.b, a.lo, a.hi); name = G.name[3:]
    run(f'neg{k}_' + name, lambda lo, hi: torch.full((hi - lo,), k, device=dev), G, k)
