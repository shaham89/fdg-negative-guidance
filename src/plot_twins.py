"""Project 1 final figure: each static frequency split and its time-only twin, drawn over the constant-w CFG frontier,
plus the IS lift of every arm above the frontier at equal FID.   python plot_twins.py out/results.jsonl out/twins.png"""
import sys, json, re, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'out/results.jsonl'
outpng = sys.argv[2] if len(sys.argv) > 2 else 'out/twins.png'
rows = {}
for l in open(path):
    if not l.strip(): continue
    r = json.loads(l)
    if r['name'].endswith('smoke') or r['name'].endswith('clip'): continue
    prev = rows.get(r['name'], {}); prev.update(r); rows[r['name']] = prev
rows = {n: r for n, r in rows.items() if 'fid' in r}

cfg = sorted([r for n, r in rows.items() if n.startswith('cfg_w')], key=lambda r: float(r['name'][5:]))
cfg_fid = np.array([r['fid'] for r in cfg]); cfg_is = np.array([r['is'] for r in cfg])
def is_frontier(fid): return float(np.interp(fid, cfg_fid, cfg_is))      # CFG's IS at the same FID (FID is monotone in w)
def lift(r): return r['is'] - is_frontier(r['fid'])

splits = {n: r for n, r in rows.items() if n.startswith('freq_')}
twins = {n[len('time['):-1]: r for n, r in rows.items() if n.startswith('time[')}   # 'time[freq_wl1_wh3_rc0.125]' -> key 'freq_wl1_wh3_rc0.125'
snr = {n: r for n, r in rows.items() if n.startswith('ft_')}
interval = {n: r for n, r in rows.items() if n.startswith('interval')}
short = lambda n: n.replace('freq_wl', '(').replace('_wh', ',').replace('_rc0.125', ')').replace('ft_', '').replace('_wmin1_wmax', ' wmax=').replace('_mu', ' mu=').replace('_s1.5', ' s=1.5').replace('_s1', '')

fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={'width_ratios': [1.35, 1]})
a = ax[0]
a.plot(cfg_is, cfg_fid, 'o-', color='C0', ms=6, label='CFG, constant w', zorder=3)
for r in cfg: a.annotate(f"w={r['name'][5:]}", (r['is'], r['fid']), fontsize=7, xytext=(4, -9), textcoords='offset points', color='C0')
for n, s in splits.items():
    a.plot(s['is'], s['fid'], 's', color='C1', ms=8, zorder=4)
    a.annotate(short(n), (s['is'], s['fid']), fontsize=7, xytext=(5, 3), textcoords='offset points', color='C1')
    t = twins.get(n)
    if t:
        a.plot(t['is'], t['fid'], 'x', color='C3', ms=9, mew=2, zorder=4)
        a.annotate('', xy=(t['is'], t['fid']), xytext=(s['is'], s['fid']), arrowprops=dict(arrowstyle='->', color='gray', lw=1.1, shrinkA=5, shrinkB=5))
for n, r in snr.items():
    a.plot(r['is'], r['fid'], '^', color='C2', ms=7, zorder=4); a.annotate(short(n), (r['is'], r['fid']), fontsize=6, xytext=(5, -3), textcoords='offset points', color='C2')
for n, r in interval.items():
    a.plot(r['is'], r['fid'], 'D', color='C4', ms=10, mec='k', zorder=5); a.annotate(n.replace('interval_', 'interval '), (r['is'], r['fid']), fontsize=7, xytext=(6, -10), textcoords='offset points', color='C4')
a.plot([], [], 's', color='C1', label='static frequency split'); a.plot([], [], 'x', color='C3', mew=2, label='its time-only twin  (arrow: split → twin)')
a.plot([], [], '^', color='C2', label='SNR-derived w(f,σ)'); a.plot([], [], 'D', color='C4', label='guidance interval')
a.set_xlabel('Inception Score ↑'); a.set_ylabel('FID ↓'); a.set_ylim(top=min(20, a.get_ylim()[1])); a.grid(alpha=.3); a.legend(fontsize=8, loc='upper left')
a.set_title('Frequency splits vs time-only schedules over the constant-w CFG frontier\n(least-squares twins fall back onto the frontier; a mid-σ guidance interval matches the (1,3) split)', fontsize=10)

b = ax[1]
order = sorted(splits, key=lambda n: splits[n]['fid'])
x = np.arange(len(order)); w_ = 0.38
b.bar(x - w_ / 2, [lift(splits[n]) for n in order], w_, color='C1', label='frequency split')
b.bar(x + w_ / 2, [lift(twins[n]) if n in twins else np.nan for n in order], w_, color='C3', label='time-only twin')
for i, n in enumerate(order):
    if n in twins and lift(splits[n]) > 0.1:      # only where the split's own lift is clearly above the IS noise floor (~0.05)
        frac = max(0.0, lift(twins[n])) / lift(splits[n])
        b.text(i, max(lift(splits[n]), lift(twins[n])) + 0.01, f'{100 * frac:.0f}% temporal', ha='center', fontsize=7)
b.axhspan(-0.05, 0.05, color='gray', alpha=0.15, lw=0, label='≈ IS noise floor (10k samples)')
b.axhline(0, color='k', lw=.8); b.set_xticks(x); b.set_xticklabels([short(n) for n in order], fontsize=8)
b.set_ylabel('IS above the CFG frontier at equal FID'); b.set_title('How much of each split\'s gain does its least-squares temporal ramp explain?', fontsize=10); b.legend(fontsize=8); b.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig(outpng, dpi=140); print('wrote', outpng)

print(f"\n{'arm':44s} {'FID':>6s} {'IS':>6s} {'IS_cfg@FID':>10s} {'lift':>6s}")
for n, r in sorted(rows.items(), key=lambda kv: kv[1]['fid']):
    if n.startswith('cfg_'): continue
    print(f"{n:44s} {r['fid']:6.2f} {r['is']:6.2f} {is_frontier(r['fid']):10.2f} {lift(r):+6.2f}")
