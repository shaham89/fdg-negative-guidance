"""Pareto plots from out/results.jsonl: FID vs IS, FID vs class accuracy, clip fraction.  Runs on CPU."""
import sys, json, collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
path = sys.argv[1] if len(sys.argv) > 1 else 'out/results.jsonl'
outpng = sys.argv[2] if len(sys.argv) > 2 else 'pareto.png'
rows = [json.loads(l) for l in open(path) if l.strip()]
rows = [r for r in rows if 'fid' in r and not r['name'].endswith('smoke')]
# merge rows by name (a later row for the same name wins); rows from evalm.py lack clip_frac -> take it from the gen row
byname = {}
for r in rows:
    prev = byname.get(r['name'], {}); prev.update(r); byname[r['name']] = prev
rows = list(byname.values())
for r in rows: r.setdefault('clip_frac', float('nan'))
def family(r):
    n = r['name']
    if n.startswith('cfg_'): return 'CFG (constant w)'
    if n.startswith('freq_'): return 'static freq split'
    if n.startswith('time['): return 'equivalent time-only'
    if n.startswith('ft_'): return 'freq x time (SNR)'
    if n.startswith('interval'): return 'interval'
    return 'other'
fam = collections.defaultdict(list)
for r in rows: fam[family(r)].append(r)
mk = {'CFG (constant w)': 'o-', 'static freq split': 's', 'equivalent time-only': 'x', 'freq x time (SNR)': '^', 'interval': 'd', 'other': '.'}
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for f, rs in fam.items():
    if f == 'CFG (constant w)': rs = sorted(rs, key=lambda r: float(r['name'][5:]))   # draw the frontier in order of w, not IS
    ax[0].plot([r['is'] for r in rs], [r['fid'] for r in rs], mk[f], label=f, ms=7)
    ax[1].plot([r['class_acc'] for r in rs], [r['fid'] for r in rs], mk[f], label=f, ms=7)
    ax[2].plot([r['is'] for r in rs], [r['clip_frac'] for r in rs], mk[f], label=f, ms=7)
    for r in rs:
        lab = r['name'].replace('cfg_w', 'w=').replace('_rc0.125', '')
        ax[0].annotate(lab, (r['is'], r['fid']), fontsize=6, xytext=(3, 3), textcoords='offset points')
        ax[1].annotate(lab, (r['class_acc'], r['fid']), fontsize=6, xytext=(3, 3), textcoords='offset points')
ax[0].set_xlabel('Inception Score ↑'); ax[0].set_ylabel('FID ↓'); ax[0].set_title('quality/diversity vs alignment')
ax[1].set_xlabel('class accuracy (ResNet-56) ↑'); ax[1].set_ylabel('FID ↓'); ax[1].set_title('FID vs condition alignment')
ax[2].set_xlabel('Inception Score ↑'); ax[2].set_ylabel('clip fraction (pixels outside [-1,1])'); ax[2].set_title('saturation')
for a_ in ax: a_.grid(alpha=0.3); a_.legend(fontsize=8)
plt.tight_layout(); plt.savefig(outpng, dpi=130); print('wrote', outpng)
print(f"{'name':48s} {'FID':>7s} {'IS':>6s} {'acc':>6s} {'clip':>7s}")
for r in sorted(rows, key=lambda r: r['fid']): print(f"{r['name']:48s} {r['fid']:7.3f} {r['is']:6.3f} {r['class_acc']:6.3f} {r['clip_frac']:7.4f}")
