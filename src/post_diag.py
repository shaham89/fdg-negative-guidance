"""Which band knows the class, and when?  (A posterior per band.)
No push: sample with D_u, and at every step fit p(class | x) from the predictions of all 10 classes (simplex_fit), on the whole image and on each band
alone.  With perfect nets every fit gives the same p, so the bands differ only through network error.  Then compare with the class the image ends as
(ResNet-56).  Checks:
  - mean p_k over all images should equal the share of class k at the end (a true posterior averages to the class shares),
  - images that end as k should get a high p_k (AUC), and the final class should get a high p as the noise falls.
The images are the first n of neg3_none (same noise, same D_u path).
  python post_diag.py --n 2000            # 11 network calls per evaluation, about 7 min on an A100
Output: out/neg/post_diag.json (per step and fit) and out/neg/post_diag.npz (every p, the final classes)."""
import os, json, argparse, time, numpy as np, torch
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=2000); ap.add_argument('--bs', type=int, default=500); ap.add_argument('--k', type=int, default=3); ap.add_argument('--tag', default='')
a = ap.parse_args(); dev = 'cuda'; NEG = os.path.join(OUT, 'neg'); os.makedirs(NEG, exist_ok=True)
FITS = ['full', 'low', 'high']

class Probe(Guidance):
    """Returns D_u (no push) and records, per evaluation, p and the share of D_u the class mixture cannot reach, for each fit."""
    needs_all = True; name = 'probe'
    def __init__(self): self.rec = []
    def __call__(self, Dc, Du, sigma, r, Dn=None):
        m = torch.sigmoid((0.125 - r) / 0.02); out = {}
        for q, W in [('full', None), ('low', m), ('high', 1 - m)]:
            Dq, Uq = (Dn, Du) if W is None else (spec_apply(Dn, W), spec_apply(Du, W))
            p, mix = simplex_fit(Dq, Uq)
            res = (mix - Uq).pow(2).flatten(1).sum(1) / (Uq - Dq.mean(1)).pow(2).flatten(1).sum(1).clamp(min=1e-30)
            out[q] = (p.float().cpu(), res.float().cpu())
        self.rec.append((float(sigma), out))
        return Du

@torch.no_grad()
def resnet(imgs):
    clf = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True, verbose=False, trust_repo=True).to(dev).eval()
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.2470, 0.2435, 0.2616], device=dev).view(1, 3, 1, 1)
    return torch.cat([clf((imgs[i:i + 1000].to(dev).float() / 255 - mean) / std).argmax(1).cpu() for i in range(0, len(imgs), 1000)])

def auc(score, y):
    """Area under the ROC curve of score for the positives y (rank form)."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    rk = torch.empty(len(score)); rk[score.argsort()] = torch.arange(1, len(score) + 1, dtype=torch.float32)
    return float((rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

net_c, net_u = load_nets(dev); G = Probe(); t0 = time.time()
P = {q: [] for q in FITS}; RES = {q: [] for q in FITS}; xs = []
for lo in range(0, a.n, a.bs):
    hi = min(lo + a.bs, a.n); G.rec = []
    x = sample(net_c, net_u, torch.full((hi - lo,), a.k, device=dev), G, seed=lo, device=dev); xs.append(to_uint8(x).cpu())
    pred = G.rec[0::2]                                   # evaluation order: predictor at t_i, corrector at t_{i+1}, ...; the even ones are the predictors
    sig = [s for s, _ in pred]
    for q in FITS:
        P[q].append(torch.stack([d[q][0] for _, d in pred], 1)); RES[q].append(torch.stack([d[q][1] for _, d in pred], 1))
P = {q: torch.cat(v) for q, v in P.items()}; RES = {q: torch.cat(v) for q, v in RES.items()}     # N, S, K and N, S
imgs = torch.cat(xs); c = resnet(imgs); k = a.k; share = torch.bincount(c, minlength=10).float() / len(c)
print(f'{a.n} images, {(time.time() - t0) / 60:.1f} min.  Final class shares (ResNet-56): ' + ' '.join(f'{v:.3f}' for v in share.tolist()), flush=True)

rows = []
for s, sg in enumerate(sig):
    row = dict(step=s, sigma=sg)
    for q in FITS:
        p = P[q][:, s]
        row[q] = dict(acc=float((p.argmax(1) == c).float().mean()),                  # the fit's top class is the final class
                      p_final=float(p[torch.arange(len(c)), c].mean()),              # mean p of the final class
                      calib_gap=float((p.mean(0) - share).abs().sum() / 2),          # total variation between mean p and the final class shares
                      pk_mean=float(p[:, k].mean()), pk_in=float(p[c == k, k].mean()), pk_out=float(p[c != k, k].mean()),
                      auc_k=auc(p[:, k], c == k), resid=float(RES[q][:, s].mean()))
    row['tv_low_high'] = float((P['low'][:, s] - P['high'][:, s]).abs().sum(1).mean() / 2)   # do the two bands agree on p?
    rows.append(row)
json.dump(dict(n=a.n, k=k, share=share.tolist(), rows=rows), open(os.path.join(NEG, f'post_diag{a.tag}.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(NEG, f'post_diag{a.tag}.npz'), sigma=np.array(sig), final=c.numpy(), **{f'p_{q}': P[q].numpy() for q in FITS}, **{f'res_{q}': RES[q].numpy() for q in FITS})

print(f'share of class {k} at the end: {share[k]:.3f}.  Per step: accuracy of the top class | mean p of the final class | mean p_k (should be {share[k]:.3f}) | p_k for final k / others | AUC for k | unexplained share of D_u')
print(f"{'sigma':>7s} | {'acc full/low/high':>17s} | {'p_final':>14s} | {'mean p_k':>14s} | {'p_k in/out full':>15s} {'low':>11s} {'high':>11s} | {'AUC k':>14s} | {'resid':>14s} | tv")
for rw in rows:
    f = lambda key, fmt='{:.2f}': '/'.join(fmt.format(rw[q][key]) for q in FITS)
    io = lambda q: f"{rw[q]['pk_in']:.2f}/{rw[q]['pk_out']:.3f}"
    print(f"{rw['sigma']:7.3f} | {f('acc'):>17s} | {f('p_final'):>14s} | {f('pk_mean', '{:.3f}'):>14s} | {io('full'):>15s} {io('low'):>11s} {io('high'):>11s} | {f('auc_k'):>14s} | {f('resid'):>14s} | {rw['tv_low_high']:.2f}", flush=True)
