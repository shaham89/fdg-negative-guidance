"""Extra metrics from saved samples, with no sampling.  One JSON line per arm -> out/extra.jsonl
  colour_k    : mean colour moved by k * (class mean colour - global mean colour), paired against --base (same seeds)
  colour_std  : spread of per-image mean colour inside a class, relative to real images
  sharp       : rms of the high band (r >= r_c), relative to real images
  var_ratio   : per band, variance inside a class relative to real images (direct measure of diversity per band)
  mean_err    : per band, squared error of the class-mean image against real images
  precision, recall : torch-fidelity VGG16 k=3, only with --prc
  fid, is, class_acc : only with --fid.  --first N uses the first N samples, so a 10k arm can be compared with a shorter arm on the same seeds
    python evalx.py --prc cfg_w1 cfg_w3 freq_wl1_wh3_rc0.125
    python evalx.py --prc --fid --first 5000 cfg_w1 cfg_w3"""
import os, json, argparse, numpy as np, torch
from common import *

p = argparse.ArgumentParser()
p.add_argument('names', nargs='+'); p.add_argument('--base', default='cfg_w1'); p.add_argument('--prc', action='store_true')
p.add_argument('--r_c', type=float, default=0.125); p.add_argument('--dev', default='cuda'); p.add_argument('--first', type=int); p.add_argument('--fid', action='store_true')
a = p.parse_args()
dev = a.dev

r = radial_freq(32, 32, dev)
dc = torch.zeros_like(r); dc[0, 0] = 1
masks = torch.cat([dc[None], band_masks(r) * (1 - dc)])         # B,H,W: DC is its own band
BANDS = ['DC'] + BAND_NAMES
high = (r >= a.r_c).float()
to_float = lambda x: x.float().to(dev) / 127.5 - 1
abs2 = lambda F: F.real ** 2 + F.imag ** 2

def class_stats(x):
    """x: float images (N,3,32,32) of one class.  Per-band within-class power (B,), class-mean spectrum, mean colour, colour std, high-band rms."""
    F = torch.fft.fft2(x, norm='ortho'); m = F.mean(0)
    P = abs2(F - m).mean(dim=(0, 1))                            # H,W
    colour = x.mean(dim=(2, 3))
    return dict(var=(masks * P).sum(dim=(1, 2)), mean=m, colour=colour.mean(0), colour_std=colour.std(0), high=(high * abs2(F).mean(dim=(0, 1))).sum().sqrt())

real = [class_stats(to_float(cifar_train_uint8([c]))) for c in range(10)]
glob_colour = torch.stack([s['colour'] for s in real]).mean(0)
shift = torch.stack([s['colour'] for s in real]) - glob_colour  # 10,3

def load(name):
    d = np.load(os.path.join(OUT, f'samples_{name}.npz'))
    return torch.tensor(d['images']).permute(0, 3, 1, 2).contiguous(), torch.tensor(d['labels']).long().to(dev)

base_imgs, base_labs = load(a.base)
base_colour = to_float(base_imgs).mean(dim=(2, 3))
for name in a.names:
    imgs, labs = load(name)
    if a.first: imgs, labs = imgs[:a.first], labs[:a.first]
    x = to_float(imgs)
    d = x.mean(dim=(2, 3)) - base_colour[:len(x)]; s = shift[labs]
    k = (d * s).sum() / (s * s).sum(); r2 = 1 - ((d - k * s) ** 2).sum() / (d ** 2).sum().clamp(min=1e-12)
    st = [class_stats(x[labs == c]) for c in range(10)]
    res = dict(name=name, base=a.base, n=len(imgs), colour_k=k.item(), colour_r2=r2.item(), bands=BANDS,
               colour_std=torch.stack([g['colour_std'] / q['colour_std'] for g, q in zip(st, real)]).mean().item(),
               sharp=torch.stack([g['high'] / q['high'] for g, q in zip(st, real)]).mean().item(),
               var_ratio=torch.stack([g['var'] / q['var'] for g, q in zip(st, real)]).mean(0).tolist(),
               mean_err=torch.stack([(masks * abs2(g['mean'] - q['mean']).mean(0)).sum(dim=(1, 2)) for g, q in zip(st, real)]).mean(0).tolist())
    if a.fid: res.update(evaluate(imgs, labs.cpu(), dev))
    if a.prc:
        import torch_fidelity
        m = torch_fidelity.calculate_metrics(input1=TensorDS(imgs), input2='cifar10-train', cuda=True, prc=True, verbose=False, cache_root=os.path.join(OUT, 'fid_cache'), datasets_root=os.path.join(OUT, 'data'))
        res.update(precision=m['precision'], recall=m['recall'])
    with open(os.path.join(OUT, 'extra.jsonl'), 'a') as f: f.write(json.dumps(res) + '\n')
    print(f"{name:42s} k={res['colour_k']:5.2f} colour_std={res['colour_std']:.3f} sharp={res['sharp']:.3f} var_ratio={np.round(res['var_ratio'], 2).tolist()}" + (f" P={res['precision']:.3f} R={res['recall']:.3f}" if a.prc else ''))
