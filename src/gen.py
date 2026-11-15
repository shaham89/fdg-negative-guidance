"""Generate N images under a guidance rule and evaluate: FID + IS (torch-fidelity vs CIFAR-10 train),
class-accuracy of a pretrained CIFAR-10 ResNet-56 on the conditioning label, and the clip fraction.
Results appended as one JSON line to out/results.jsonl.

Examples:
  python gen.py cfg --w 3
  python gen.py freq --w_low 1 --w_high 3 --r_c 0.125
  python gen.py interval --w 3 --lo 0.5 --hi 5
  python gen.py ft --w_min 1 --w_max 3 --mode unresolved --mu 0 --s 1
  python gen.py time --table out/time_from_freq_wl1_wh3_rc0.125.json
  python gen.py bands --w_dc 1 --w_low 3 --w_high 3          # own weight for the mean colour
  python gen.py highdir --dir cw --w_high 3                  # high band only, other guidance direction
  python gen.py sharpen --lam 0.1                            # constant lambda, no class direction, one network
  python gen.py sharpen --lam 0.2 --table out/sharpen_table.json   # lambda(sigma) from calib_sharpen.py, scaled
  python gen.py pyramid --w_low 1 --w_high 3                 # the paper's Laplacian pyramid (needs kornia)
  python gen.py src --w_cls 3 --w_gap 1 --n 5000             # dial only the class part of Delta; 11 net calls per evaluation
  python gen.py pair --w 3 --high_only                       # guidance direction from EDM's second, independent network pair
  python gen.py wiener --w 6 --high_only                     # shrink guidance by its measured agreement; table from replicate_diag.py
"""
import os, sys, json, time, argparse, numpy as np, torch
from common import *

p = argparse.ArgumentParser()
p.add_argument('rule', choices=['cfg', 'freq', 'interval', 'ft', 'time', 'freqint', 'bands', 'highdir', 'sharpen', 'pyramid', 'src', 'pair', 'wiener'])
p.add_argument('--w_cls', type=float); p.add_argument('--w_gap', type=float); p.add_argument('--high_only', action='store_true'); p.add_argument('--pair_mode', default='cross', choices=['cross', 'mean'])
p.add_argument('--w_dc', type=float); p.add_argument('--dir', default='cu', choices=['cu', 'cw', 'wu'])
p.add_argument('--lam', type=float, help='rule=sharpen: constant lambda, or a scale on the --table lambdas')
p.add_argument('--seed', type=int, default=0, help='0 = the original noise; other values give a new, class-balanced noise set')
p.add_argument('--w', type=float, default=1.0); p.add_argument('--w_low', type=float); p.add_argument('--w_high', type=float)
p.add_argument('--r_c', type=float, default=0.125); p.add_argument('--tau', type=float, default=0.02)
p.add_argument('--lo', type=float); p.add_argument('--hi', type=float)
p.add_argument('--w_min', type=float); p.add_argument('--w_max', type=float); p.add_argument('--mode', default='unresolved')
p.add_argument('--mu', type=float, default=0.0); p.add_argument('--s', type=float, default=1.0)
p.add_argument('--table', help='json with {"sigmas":[...],"ws":[...],"name":...} for rule=time')
p.add_argument('--n', type=int, default=10000); p.add_argument('--bs', type=int, default=500)
p.add_argument('--steps', type=int, default=18); p.add_argument('--uncond_mode', default='sep')
p.add_argument('--rho', type=float, default=7.0); p.add_argument('--sched', help='json with {"sigmas": [...]} (Project 2 schedules)')
p.add_argument('--delta_cache', type=float, default=None, help='sigma threshold above which corrector reuses the predictor delta (inf = always)')
p.add_argument('--tag', default=''); p.add_argument('--save_png', type=int, default=64)
p.add_argument('--no_metrics', action='store_true')
a = p.parse_args()

if a.rule == 'cfg': G = ConstCFG(a.w)
elif a.rule == 'freq': G = FreqStatic(a.w_low, a.w_high, a.r_c, a.tau)
elif a.rule == 'interval': G = Interval(a.w, a.lo, a.hi)
elif a.rule == 'freqint': G = FreqInterval(a.w_low, a.w_high, a.lo, a.hi, a.r_c, a.tau)
elif a.rule == 'ft': sr, sP = load_spectrum(); G = FreqTime(sr, sP, a.w_min, a.w_max, a.mode, a.mu, a.s)
elif a.rule == 'time': t = json.load(open(a.table)); G = TimeSched(t['sigmas'], t['ws'], name=t.get('name', 'time'))
elif a.rule == 'bands': G = FreqBands(a.w_dc, a.w_low, a.w_high, a.r_c, a.tau)
elif a.rule == 'highdir': G = HighDir(a.dir, a.w_high, a.r_c, a.tau, a.lo if a.lo is not None else 0.0, a.hi if a.hi is not None else float('inf'))
elif a.rule == 'sharpen' and a.table: t = json.load(open(a.table)); G = Sharpen(t['sigmas'], [a.lam * l for l in t['lambdas']], a.r_c, a.tau, name=f'sharpen_shaped{a.lam:g}')
elif a.rule == 'sharpen': G = Sharpen([1.0], [a.lam], a.r_c, a.tau, name=f'sharpen_const{a.lam:g}')
elif a.rule == 'pyramid': G = Pyramid(a.w_low, a.w_high)
elif a.rule == 'src': G = SourceSplit(a.w_cls, a.w_gap, a.high_only, a.r_c, a.tau)
elif a.rule == 'pair': G = CrossPair(*load_nets('cuda', arch='ve'), a.w, a.pair_mode, a.high_only, a.r_c, a.tau)
elif a.rule == 'wiener': G = WienerGuidance(json.load(open(a.table or os.path.join(OUT, 'wiener_table.json'))), a.w, a.high_only, a.r_c, a.tau)
sched = json.load(open(a.sched))['sigmas'] if a.sched else None
name = G.name + (f'_{a.uncond_mode}' if a.uncond_mode != 'sep' else '') + (f'_s{a.steps}' if a.steps != 18 else '') \
     + (f'_rho{a.rho:g}' if a.rho != 7 else '') + (f'_{os.path.basename(a.sched)[6:-5]}' if a.sched else '') + (f'_dc{a.delta_cache:g}' if a.delta_cache is not None else '') + (f'_seed{a.seed}' if a.seed else '') + (f'_{a.tag}' if a.tag else '')
os.makedirs(OUT, exist_ok=True)

dev = 'cuda'; net_c, net_u = load_nets(dev)
t0 = time.time(); imgs, labs, clipf = [], [], []
for b in range(0, a.n, a.bs):
    labels = torch.arange(b, b + a.bs, device=dev) % 10        # balanced classes
    seed = a.seed * 100000 + b; neg = wrong_labels(labels, seed) if a.rule == 'highdir' and a.dir != 'cu' else None
    x = sample(net_c, net_u, labels, G, num_steps=a.steps, seed=seed, device=dev, uncond_mode=a.uncond_mode, rho=a.rho, sigmas=sched, delta_cache=a.delta_cache, neg_labels=neg)
    clipf.append(clip_fraction(x)); imgs.append(to_uint8(x).cpu()); labs.append(labels.cpu())
    print(f'  {b + a.bs}/{a.n}  {time.time() - t0:.0f}s', flush=True)
imgs = torch.cat(imgs); labs = torch.cat(labs); gen_time = time.time() - t0
np.savez_compressed(os.path.join(OUT, f'samples_{name}.npz'), images=imgs.permute(0, 2, 3, 1).numpy(), labels=labs.numpy())
if a.save_png:
    import torchvision; torchvision.utils.save_image(imgs[:a.save_png].float() / 255, os.path.join(OUT, f'grid_{name}.png'), nrow=8)
res = dict(name=name, rule=a.rule, args={k: v for k, v in vars(a).items() if v is not None}, n=a.n, clip_frac=float(np.mean(clipf)), gen_time=gen_time)
if not a.no_metrics:
    res.update(evaluate(imgs, labs, dev))
with open(os.path.join(OUT, 'results.jsonl'), 'a') as f: f.write(json.dumps(res) + '\n')
print(json.dumps(res, indent=1))
