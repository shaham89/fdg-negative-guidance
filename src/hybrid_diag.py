"""Where does the class end up in the final image?  Hybrids of saved samples, with no sampling.
All arms share noise, so one image can take its low band from an arm and its high band from w=1, or the other way round.
Per arm and classifier: rate of the target class c, and of the wrong class c' (the one the wrong-class arms were steered by, same c' for all arms), on
  whole = the arm's image,  low = Low(arm) + High(w=1),  high = Low(w=1) + High(arm).
Also the rms change of each band against w=1, in the [-1, 1] pixel scale.
    python hybrid_diag.py cfg_w3 freq_wl1_wh3_rc0.125 highdir_wu_wh3.67_rc0.125     -> out/hybrid.jsonl, out/hybrid_<arm>.png"""
import os, json, argparse, numpy as np, torch
from common import *

p = argparse.ArgumentParser()
p.add_argument('names', nargs='+'); p.add_argument('--base', default='cfg_w1'); p.add_argument('--r_c', type=float, default=0.125); p.add_argument('--tau', type=float, default=0.02)
p.add_argument('--bs', type=int, default=500); p.add_argument('--clf', nargs='+', default=['cifar10_resnet56', 'cifar10_vgg16_bn'])
a = p.parse_args()
dev = 'cuda'

r = radial_freq(32, 32, dev); S = torch.sigmoid((a.r_c - r) / a.tau)
low = lambda x: torch.fft.ifft2(torch.fft.fft2(x, norm='ortho') * S, norm='ortho').real
to_u8 = lambda x: ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
def load(name):
    d = np.load(os.path.join(OUT, f'samples_{name}.npz'))
    return torch.tensor(d['images']).permute(0, 3, 1, 2).contiguous().float().to(dev) / 127.5 - 1, torch.tensor(d['labels']).long().to(dev)

clfs = {c: torch.hub.load('chenyaofo/pytorch-cifar-models', c, pretrained=True, verbose=False, trust_repo=True).to(dev).eval() for c in a.clf}
mean = torch.tensor([0.4914, 0.4822, 0.4465], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.2470, 0.2435, 0.2616], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def predict(clf, x):
    u = to_u8(x)                                                                    # the same 8-bit images a saved sample would be
    return torch.cat([clf((u[i:i + 1000].float() / 255 - mean) / std).argmax(1) for i in range(0, len(u), 1000)])

x0, labs = load(a.base); L0 = low(x0); H0 = x0 - L0
wrong = torch.cat([wrong_labels(torch.arange(b, b + a.bs, device=dev) % 10, b) for b in range(0, len(x0), a.bs)])      # the c' of seed 0, as in gen.py
assert (wrong != labs).all()
rms = lambda d: d.pow(2).mean().sqrt().item()
print(f"{'arm':34s} {'rmsL':>6s} {'rmsH':>6s} {'classifier':>18s} | target class c: whole  low  high | wrong class c': whole  low  high")
for name in a.names:
    x, y = load(name); n = len(x); assert (y == labs[:n]).all()                     # a shorter arm is paired with the first n of the base
    L = low(x); H = x - L
    res = dict(name=name, base=a.base, n=n, rms_low=rms(L - L0[:n]), rms_high=rms(H - H0[:n]))
    for cn, clf in clfs.items():
        P = dict(whole=predict(clf, x), low=predict(clf, L + H0[:n]), high=predict(clf, L0[:n] + H))
        res[cn] = {k: dict(c=(v == labs[:n]).float().mean().item(), w=(v == wrong[:n]).float().mean().item()) for k, v in P.items()}
        t = res[cn]
        print(f"{name:34s} {res['rms_low']:6.3f} {res['rms_high']:6.3f} {cn[8:]:>18s} | {t['whole']['c']:21.3f} {t['low']['c']:.3f} {t['high']['c']:.3f} | {t['whole']['w']:21.3f} {t['low']['w']:.3f} {t['high']['w']:.3f}")
    with open(os.path.join(OUT, 'hybrid.jsonl'), 'a') as f: f.write(json.dumps(res) + '\n')
    import torchvision
    rows = torch.cat([x0[:16], x[:16], L[:16] + H0[:16], L0[:16] + H[:16]])          # rows: w=1, arm, Low(arm)+High(w=1), Low(w=1)+High(arm)
    torchvision.utils.save_image(torch.nn.functional.interpolate((rows + 1) / 2, scale_factor=4, mode='nearest').clamp(0, 1), os.path.join(OUT, f'hybrid_{name}.png'), nrow=16)
print('first 16 labels', labs[:16].tolist()); print('first 16 wrong ', wrong[:16].tolist())
