"""Push-response map: share of images whose class flips after a 2-step push in one band, away from their own class or toward another one.
Real nets (pmap.py, ResNet-56 reads the class) against the exact Gaussian model (analysis/gmm_pmap.py, exact posterior reads the class).
  python plot_pmap.py [pmap_s2.jsonl] [gmm_pmap_n1000.json] [out.png]      defaults: the saved files in results/neg_sweep12 and results/gaussian_model"""
import os, sys, json, math
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
real_f = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, 'neg_sweep12', 'pmap_s2.jsonl')
toy_f = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RES, 'gaussian_model', 'gmm_pmap_n1000.json')
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(RES, 'neg_sweep12', 'pmap_flip.png')
real = [json.loads(l) for l in open(real_f)]; toy = json.load(open(toy_f))['rows']
COL = {'away': '#2a78d6', 'toward': '#eb6834'}; INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

def series(rows, band, d):
    r = sorted([x for x in rows if x['band'] == band and x['dir'] == d and x['i1'] - x['i0'] == 2], key=lambda x: -x['sig_hi'])
    return [math.sqrt(x['sig_hi'] * x['sig_lo']) for x in r], [x['flip'] for x in r]

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2})
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURF)
for ax, band in zip(axs, ['low', 'high']):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for d in ['away', 'toward']:
        for rows, ls, mk, lab in [(real, '-', 'o', 'trained nets'), (toy, '--', 's', 'Gaussian model')]:
            x, y = series(rows, band, d)
            ax.plot(x, y, ls=ls, marker=mk, ms=6, lw=2, color=COL[d], mec=SURF, mew=1.5, label=f'{d}, {lab}')
        x, y = series(real, band, d); i = max(range(len(y)), key=lambda j: y[j])
        left = d == 'away'
        ax.annotate(f'{d}, trained nets\npeak at σ≈{x[i]:.2g}', (x[i], y[i]), xytext=(-10 if left else 10, 14), textcoords='offset points',
                    ha='right' if left else 'left', fontsize=9, color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    ax.set_xscale('log'); ax.set_xlim(120, 0.0015); ax.set_ylim(0, 1)
    ax.set_title(f'{band.capitalize()} band pushed for two steps', color=INK, fontsize=11, loc='left')
    ax.set_xlabel('noise level σ of the push (sampling runs left to right)')
axs[0].set_ylabel('share of images whose class flips')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle('Pushing away from a class works earlier than pushing toward one (strength 2, D_u path)', x=0.01, ha='left', color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0.08, 1, 0.95)); fig.savefig(out, dpi=150, facecolor=SURF); print('saved', out)
