"""Push-response map at two strengths, and the size of the push per step (trained nets only).
Top: share of images whose class flips after a 2-step push in one band (strength 2 solid, strength 1 dashed).
Bottom: rms of Band(D_k - D_u) on the no-push path, for k = own class (away) and k = another class (toward); the push is s times this.
  python plot_pmap_s.py [pmap_s2.jsonl] [pmap_s1.jsonl] [pmap_size.json] [out.png]      defaults: the saved files in results/neg_sweep12 and results/neg_sweep12c"""
import os, sys, json, math
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

arg = lambda i, d: sys.argv[i] if len(sys.argv) > i else d
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
f2, f1 = arg(1, os.path.join(RES, 'neg_sweep12', 'pmap_s2.jsonl')), arg(2, os.path.join(RES, 'neg_sweep12c', 'pmap_s1.jsonl'))
fsize, out = arg(3, os.path.join(RES, 'neg_sweep12c', 'pmap_size.json')), arg(4, os.path.join(RES, 'neg_sweep12c', 'pmap_strength.png'))
R = {2: [json.loads(l) for l in open(f2)], 1: [json.loads(l) for l in open(f1)]}; size = json.load(open(fsize))
COL = {'away': '#2a78d6', 'toward': '#eb6834'}; INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

def series(rows, band, d):
    r = sorted([x for x in rows if x['band'] == band and x['dir'] == d and x['i1'] - x['i0'] == 2], key=lambda x: -x['sig_hi'])
    return [math.sqrt(x['sig_hi'] * x['sig_lo']) for x in r], [x['flip'] for x in r]

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2})
fig, axs = plt.subplots(2, 2, figsize=(10, 7.2), facecolor=SURF, gridspec_kw=dict(height_ratios=[1.4, 1]))
for j, band in enumerate(['low', 'high']):
    for ax in axs[:, j]:
        ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xscale('log'); ax.set_xlim(120, 0.0015)
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax = axs[0, j]
    for d in ['away', 'toward']:
        for s, ls in [(2, '-'), (1, '--')]:
            x, y = series(R[s], band, d)
            ax.plot(x, y, ls=ls, marker='o', ms=6 if s == 2 else 5, lw=2, color=COL[d], mec=SURF, mew=1.5, label=f'{d}, strength {s}')
            if s == 2:
                i = max(range(len(y)), key=lambda q: y[q]); left = d == 'away'
                ax.annotate(f'peak σ≈{x[i]:.2g}', (x[i], y[i]), xytext=(-8 if left else 8, 6), textcoords='offset points', ha='right' if left else 'left', fontsize=8.5, color=INK)
    ax.set_ylim(0, 0.9); ax.set_title(f'{band.capitalize()} band pushed for two steps', color=INK, fontsize=11, loc='left')
    if j == 0: ax.set_ylabel('share of images whose class flips')
    ax = axs[1, j]
    xs = [r['sigma'] for r in size]
    for d in ['away', 'toward']:
        ax.plot(xs, [r[f'{band}_{d}'] for r in size], marker='o', ms=4, lw=2, color=COL[d], mec=SURF, mew=1)
    ax.set_ylim(0, None); ax.set_xlabel('noise level σ (sampling runs left to right)')
    if j == 0: ax.set_ylabel('size of one push step\n(rms of Band(D_k − D_u))')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle('Away works earlier than toward at both strengths, although the toward push is the larger one', x=0.01, ha='left', color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(out, dpi=150, facecolor=SURF); print('saved', out)
