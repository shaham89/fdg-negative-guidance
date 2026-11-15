"""Checks on the saved two-dials results (results/two_dials): DC share, low/high crossover, frontier gains, metric noise.  No GPU, numpy only.
  python check_saved_data.py"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'two_dials')

def radial(H=32, W=32):
    fy, fx = np.fft.fftfreq(H), np.fft.fftfreq(W)
    return np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

r = radial()
sig_mask = lambda rc, tau=0.02: 1 / (1 + np.exp(-(rc - r) / tau))   # amplitude of low band, as in FreqStatic

for w in ['1', '3']:
    d = np.load(os.path.join(OUT, f'diag_w{w}.npz'))
    print(f'\n===== trajectory w={w}; keys={list(d.keys())} P_delta={d["P_delta"].shape}')
    sig, P = d['sigma'], d['P_delta']
    print('sigma   |Delta|^2   DC%   low-nonDC%  high%  (rc=.125 soft mask, same as FreqStatic) | DC share of low band')
    S = sig_mask(0.125)
    for s, p in zip(sig, P):
        tot = p.sum(); dc = p[0, 0]
        low = (S ** 2 * p).sum(); high = ((1 - S) ** 2 * p).sum()
        print(f'{s:8.3f} {tot:9.3f}  {100*dc/tot:5.1f}   {100*(low-S[0,0]**2*dc)/tot:5.1f}    {100*high/tot:5.1f}   | {100*S[0,0]**2*dc/low:5.1f}')
    print('crossover sigma where |Low(Delta)| = |High(Delta)|, by cutoff (linear interp in log sigma):')
    for rc in [0.0625, 0.125, 0.1875, 0.25]:
        S = sig_mask(rc)
        ratio = np.array([np.sqrt((S ** 2 * p).sum() / ((1 - S) ** 2 * p).sum()) for p in P])
        # first index (going down in sigma) where high exceeds low
        idx = np.where(ratio < 1)[0]
        if len(idx) == 0:
            print(f'  rc={rc}: no crossover'); continue
        i = idx[0]
        if i == 0:
            print(f'  rc={rc}: high already bigger at sigma max'); continue
        a, b = np.log(ratio[i - 1]), np.log(ratio[i])
        t = a / (a - b)
        sx = np.exp(np.log(sig[i - 1]) + t * (np.log(sig[i]) - np.log(sig[i - 1])))
        print(f'  rc={rc}: sigma*={sx:.3f}  (later steps stay high-dominated: {bool((ratio[i:] < 1).all())})')

# ----- frontier gains and noise
rows = {}
for l in open(os.path.join(OUT, 'results.jsonl')):
    r_ = json.loads(l)
    if 'fid' in r_ and not r_['name'].endswith('smoke'):
        rows[r_['name']] = r_
cfg = sorted([(float(n[5:]), v['fid'], v['is'], v['class_acc']) for n, v in rows.items() if n.startswith('cfg_w')])
print('\n===== CFG frontier (w, FID, IS, acc)')
for c in cfg: print('  ', c)
fid_c = np.array([c[1] for c in cfg]); is_c = np.array([c[2] for c in cfg])
# frontier part used: w=1..2 where IS rises with FID, then w>2 where it falls.  Interpolate IS at equal FID.
print('\n===== IS gain over CFG at the same FID (interp on the CFG polyline), and class acc')
for n, v in sorted(rows.items(), key=lambda kv: kv[1]['fid']):
    if n.startswith('cfg_w'): continue
    print(f"  {n:42s} FID {v['fid']:6.2f}  IS {v['is']:6.3f}  dIS {v['is'] - np.interp(v['fid'], fid_c, is_c):+.3f}  acc {v['class_acc']:.4f}  clip {v.get('clip_frac', float('nan')):.4f}")

noise = json.load(open(os.path.join(OUT, 'noise.json')))
dis = np.array([abs(v['is_a'] - v['is_b']) for v in noise.values()])
dfd = np.array([abs(v['fid_a'] - v['fid_b']) for v in noise.values()])
print('\n===== noise from 5k half splits (n=%d configs)' % len(dis))
print('  |dIS|', np.round(dis, 3), 'mean', round(dis.mean(), 3))
print('  |dFID|', np.round(dfd, 3), 'mean', round(dfd.mean(), 3))
# E|A-B| = 2 s/sqrt(pi) for iid normal A,B with std s
s5_is, s5_fid = dis.mean() * np.sqrt(np.pi) / 2, dfd.mean() * np.sqrt(np.pi) / 2
print(f'  std of a 5k estimate: IS {s5_is:.3f}, FID {s5_fid:.3f}')
print(f'  std of a 10k estimate if std ~ 1/sqrt(N): IS {s5_is/np.sqrt(2):.3f}, FID {s5_fid/np.sqrt(2):.3f}')
print(f'  std of a difference of two independent 10k arms: IS {s5_is:.3f}, FID {s5_fid:.3f}')
print(f'  noise.py formula (mean/2): IS {dis.mean()/2:.3f}')

# ----- how much guidance does the high band get early on, in absolute terms?
d = np.load(os.path.join(OUT, 'diag_w3.npz')); S = sig_mask(0.125)
print('\n===== |High(Delta)| rms per pixel vs sigma (trajectory w=3); compare with sigma')
for s, p in zip(d['sigma'], d['P_delta']):
    print(f'  sigma {s:8.3f}  rms High(Delta) {np.sqrt(((1-S)**2*p).sum()/(32*32)):.4f}   rms Low(Delta) {np.sqrt((S**2*p).sum()/(32*32)):.4f}   DC rms {np.sqrt(p[0,0]/(32*32)):.4f}')

# ----- data spectrum: DC share of the data itself
sp = np.load(os.path.join(OUT, 'cifar_spectrum.npz'))
print('\n===== spectrum keys', list(sp.keys()))
if 'P2d' in sp: print('  data DC power share', float(sp['P2d'][0, 0] / sp['P2d'].sum()))
