"""CPU checks for the "Two dials" guidance rules, using fake networks.  python test_twodials.py   (Pyramid needs kornia, else it is skipped.)"""
import math, numpy as np, torch
from common import *

class FakeNet:
    """A smooth stand-in for an EDM denoiser: output depends on x, sigma and the label.  Counts calls."""
    sigma_min, sigma_max, label_dim = 0.002, 80.0, 10
    def __init__(self, scale): self.scale, self.calls = scale, 0
    def round_sigma(self, t): return t
    def __call__(self, x, sig, onehot):
        self.calls += 1
        lab = 0 if onehot is None else (onehot.argmax(1).float() - 4.5)[:, None, None, None] / 10
        return x / (1 + sig ** 2) + self.scale * torch.cos(x) * 0.01 + lab

torch.manual_seed(0)
r = radial_freq(32, 32, 'cpu')
Dc, Du, Dn = torch.randn(4, 3, 32, 32, dtype=torch.float64), torch.randn(4, 3, 32, 32, dtype=torch.float64), torch.randn(4, 3, 32, 32, dtype=torch.float64)
delta = Dc - Du
close = lambda a, b, tol=1e-5: (a - b).abs().max().item() < tol
sig = torch.tensor(1.0, dtype=torch.float64)

# Part A: the three bands add up exactly, and each corner means what it should
S = torch.sigmoid((0.125 - r) / 0.02)
DC = delta.mean(dim=(2, 3), keepdim=True); R = delta - DC
FR = torch.fft.fft2(R.float(), norm='ortho')
LowRest = torch.fft.ifft2(FR * S, norm='ortho').real.double(); High = R - LowRest
assert close(DC + LowRest + High, delta)
for wd, wl, wh in [(1, 3, 3), (3, 1, 3), (1, 3, 1), (3, 1, 1), (2, 5, 7)]:
    want = Dc + (wd - 1) * DC + (wl - 1) * LowRest + (wh - 1) * High
    assert close(FreqBands(wd, wl, wh)(Dc, Du, sig, r), want), (wd, wl, wh)
assert close(FreqBands(3, 3, 3)(Dc, Du, sig, r), ConstCFG(3)(Dc, Du, sig, r))
assert close(FreqBands(1, 1, 1)(Dc, Du, sig, r), Dc)
got = FreqBands(1, 3, 3)(Dc, Du, sig, r)
assert close(got.mean(dim=(2, 3)), (Du + 3 * delta).mean(dim=(2, 3)) - 2 * delta.mean(dim=(2, 3)))    # DC frozen: mean colour stays at the conditional one
print('part A ok; corner (1,1,3) vs FreqStatic(1,3) max diff', (FreqBands(1, 1, 3)(Dc, Du, sig, r) - FreqStatic(1, 3)(Dc, Du, sig, r)).abs().max().item())

# Part B: directions
assert close(HighDir('cu', 3)(Dc, Du, sig, r), FreqStatic(1, 3)(Dc, Du, sig, r))
assert close(HighDir('cw', 3)(Dc, None, sig, r, Dn=Dn), Dc + 2 * high_pass(Dc - Dn, r, 0.125, 0.02))
assert close(HighDir('wu', 3)(Dc, Du, sig, r, Dn=Dn), Dc + 2 * high_pass(Dn - Du, r, 0.125, 0.02))
hw = HighDir('wu', 3, lo=2.0, hi=5.0)                                                                 # noise window: guided inside, D_c outside
assert close(hw(Dc, Du, torch.tensor(3.0, dtype=torch.float64), r, Dn=Dn), HighDir('wu', 3)(Dc, Du, sig, r, Dn=Dn)) and close(hw(Dc, Du, sig, r, Dn=Dn), Dc)
assert HighDir('wu', 3).name == 'highdir_wu_wh3_rc0.125' and hw.name == 'highdir_wu_wh3_rc0.125_sig2-5'
s = Sharpen([0.1, 1.0, 10.0], [0.2, 0.5, 0.8])
assert close(s(Dc, None, sig, r), Dc + 0.5 * high_pass(Dc, r, 0.125, 0.02))
assert abs(s.t.weight(math.sqrt(10.0), r) - 0.65) < 1e-9                                              # interpolated in log sigma
assert high_pass(Dc, r, 0.125, 0.02).dtype == Dc.dtype
print('part B ok')

# Source split: the fit finds a known mixture, and the dials reduce to known rules
Dk = torch.randn(4, 10, 3, 32, 32, dtype=torch.float64)
p_true = proj_simplex(torch.randn(4, 10, dtype=torch.float64))
assert close(p_true.sum(1), torch.ones(4, dtype=torch.float64)) and (p_true >= 0).all()
mix_true = (p_true[:, :, None, None, None] * Dk).sum(1)
p_fit, mix_fit = simplex_fit(Dk, mix_true)
assert close(mix_fit, mix_true, 1e-3), (mix_fit - mix_true).abs().max()
far = Dk.mean(1) + 5 * torch.randn(4, 3, 32, 32, dtype=torch.float64)                                 # off the simplex: weights stay valid
p_far, _ = simplex_fit(Dk, far)
assert close(p_far.sum(1), torch.ones(4, dtype=torch.float64)) and (p_far >= 0).all()
Dc4 = Dk[:, 0]; Du4 = torch.randn(4, 3, 32, 32, dtype=torch.float64)
assert close(SourceSplit(3, 3)(Dc4, Du4, sig, r, Dn=Dk), ConstCFG(3)(Dc4, Du4, sig, r))
assert close(SourceSplit(1, 1)(Dc4, Du4, sig, r, Dn=Dk), Dc4)
assert close(SourceSplit(3, 3, high_only=True)(Dc4, Du4, sig, r, Dn=Dk), FreqStatic(1, 3)(Dc4, Du4, sig, r))
a_ = SourceSplit(3, 1)(Dc4, Du4, sig, r, Dn=Dk) - Dc4; b_ = SourceSplit(1, 3)(Dc4, Du4, sig, r, Dn=Dk) - Dc4
assert close(a_ + b_, 2 * (Dc4 - Du4))                                                                # the two parts add up to Delta
print('source split ok')

# Second network pair: direction from the other pair, or the mean of both
c2, u2 = FakeNet(1.5), FakeNet(2.5)
assert close(CrossPair(c2, u2, 3)(Dc, None, sig, r, Dn=Dn), Dc + 2 * Dn)
assert close(CrossPair(c2, u2, 3, 'mean')(Dc, Du, sig, r, Dn=Dn), Dc + (Dc - Du + Dn))
assert close(CrossPair(c2, u2, 3, high_only=True)(Dc, None, sig, r, Dn=Dn), Dc + 2 * high_pass(Dn, r, 0.125, 0.02))
c, u = FakeNet(1.0), FakeNet(2.0); c2, u2 = FakeNet(1.5), FakeNet(2.5)
assert torch.isfinite(sample(c, u, torch.arange(8) % 10, CrossPair(c2, u2, 3), num_steps=4, seed=0, device='cpu')).all() and (c.calls, u.calls, c2.calls, u2.calls) == (7, 0, 7, 7)
c, u = FakeNet(1.0), FakeNet(2.0); c2, u2 = FakeNet(1.5), FakeNet(2.5)
assert torch.isfinite(sample(c, u, torch.arange(8) % 10, CrossPair(c2, u2, 3, 'mean'), num_steps=4, seed=0, device='cpu')).all() and (c.calls, u.calls, c2.calls, u2.calls) == (7, 7, 7, 7)
print('second pair ok')

# Wiener guidance: rho = 1 gives CFG or the split, rho = 0 gives D_c, and a known rho gives the known weight
tab = lambda v: dict(sigmas=[0.01, 1.0, 100.0], rho=[[v] * 7] * 3)
assert close(WienerGuidance(tab(1.0), 3)(Dc, Du, sig, r), ConstCFG(3)(Dc, Du, sig, r))
assert close(WienerGuidance(tab(0.0), 3)(Dc, Du, sig, r), Dc)
assert close(WienerGuidance(tab(1.0), 3, high_only=True)(Dc, Du, sig, r), FreqStatic(1, 3)(Dc, Du, sig, r))
assert close(WienerGuidance(tab(0.5), 3)(Dc, Du, sig, r), ConstCFG(2)(Dc, Du, sig, r))
assert close(WienerGuidance(tab(2.0), 3)(Dc, Du, sig, r), ConstCFG(3)(Dc, Du, sig, r))                # rho is clipped to [0, 1]
t2 = dict(sigmas=[1.0, 100.0], rho=[[0.0] * 7, [1.0] * 7])                                            # interpolated in log sigma: halfway at sigma = 10
assert close(WienerGuidance(t2, 3)(Dc, Du, torch.tensor(10.0, dtype=torch.float64), r), ConstCFG(2)(Dc, Du, sig, r))
td = dict(sigmas=[1.0, 2.0], rho=[[1.0] + [0.0] * 6] * 2)                                             # only DC is trusted: mean colour moves, nothing else
gd = WienerGuidance(td, 3)(Dc, Du, sig, r)
assert close(gd, Dc + 2 * delta.mean(dim=(2, 3), keepdim=True))
assert torch.isfinite(sample(FakeNet(1.0), FakeNet(2.0), torch.arange(8) % 10, WienerGuidance(tab(0.5), 3), num_steps=4, seed=0, device='cpu')).all()
print('wiener ok')

# Negative guidance per band: equal strengths give plain negation, each dial acts on its own band, and outside the window the result is D_u
hp = high_pass(delta, r, 0.125, 0.02)
assert close(NegBand(2, 2)(Dc, Du, sig, r), ConstCFG(-2)(Dc, Du, sig, r))
assert close(NegBand(0, 2)(Dc, Du, sig, r), Du - 2 * hp) and close(NegBand(2, 0)(Dc, Du, sig, r), Du - 2 * (delta - hp))
assert close(NegBand(0, 0)(Dc, Du, sig, r), Du)
nw = NegBand(0, 2, lo=2.0, hi=5.0)
assert close(nw(Dc, Du, sig, r), Du) and close(nw(Dc, Du, torch.tensor(3.0, dtype=torch.float64), r), NegBand(0, 2)(Dc, Du, sig, r))
assert NegBand(0, 2).name == 'neg_a0_b2_rc0.125' and nw.name == 'neg_a0_b2_rc0.125_sig2-5'
assert torch.isfinite(sample(FakeNet(1.0), FakeNet(2.0), torch.full((8,), 3), nw, num_steps=4, seed=0, device='cpu')).all()
print('negative bands ok')

# New negative rules: mean colour frozen, exact rule, second pair, positive class with a negative class
nd = NegBand(1, 0, dc=0)(Dc, Du, sig, r)
assert close((nd - Du).mean(dim=(2, 3)), torch.zeros(4, 3, dtype=torch.float64)) and close(nd - (nd - Du).mean(dim=(2, 3), keepdim=True) * 0, nd)
assert close(NegBand(1, 0)(Dc, Du, sig, r) - nd, -delta.mean(dim=(2, 3), keepdim=True).expand_as(delta), 1e-2)       # the mask is 0.998 at DC
assert close(NegBand(0, 0, dc=1)(Dc, Du, sig, r), Du - delta.mean(dim=(2, 3), keepdim=True)) and NegBand(1, 0, dc=0, lo=0.8).name == 'neg_a1_b0_rc0.125_dc0_sig0.8-inf'
assert close(PosNeg(1, 1)(Dc, None, sig, r, Dn=Dn), Dc) and close(PosNeg(3, 3)(Dc, None, sig, r, Dn=Dn), Dn + 3 * (Dc - Dn))
assert close(PosNeg(1, 3)(Dc, None, sig, r, Dn=Dn), Dc + 2 * high_pass(Dc - Dn, r, 0.125, 0.02))
Dall = torch.randn(4, 10, 3, 32, 32, dtype=torch.float64); pt = torch.rand(4, 10, dtype=torch.float64); pt = pt / pt.sum(1, keepdim=True)
Dmix = (pt[:, :, None, None, None] * Dall).sum(1); Dk3 = Dall[:, 3]                  # a perfect pair: D_u is the mixture
want = ((pt[:, :, None, None, None] * Dall).sum(1) - pt[:, 3, None, None, None] * Dk3) / (1 - pt[:, 3, None, None, None])
ex = NegExact(3, cap=1e6)
assert close(ex(Dk3, Dmix, sig, r, Dn=Dall), want, 1e-4) and close(NegExact(3, mode='mix')(Dk3, Dmix, sig, r, Dn=Dall), want, 1e-4)
assert close(NegExact(3, 1, 0, cap=1e6)(Dk3, Dmix, sig, r, Dn=Dall) + NegExact(3, 0, 1, cap=1e6)(Dk3, Dmix, sig, r, Dn=Dall) - Dmix, want, 1e-4)
Dbad = Dall.clone(); Dbad[1, 2] = float('nan')                                     # one image with a non-finite prediction: no push for it, the rest unchanged
out = NegExact(3, cap=1e6)(Dk3, Dmix, sig, r, Dn=Dbad)
assert torch.isfinite(out).all() and close(out[0], want[0], 1e-4) and close(out[1], Dmix[1])
P = band_posteriors(Dall, Dmix, r)                                                      # a perfect pair: every band gives the same p
assert all(close(P[q], pt, 1e-4) for q in P), {q: (P[q] - pt).abs().max().item() for q in P}
for q in ['low', 'high', 'band']:
    assert close(NegExact(3, cap=1e6, pfit=q)(Dk3, Dmix, sig, r, Dn=Dall), want, 1e-4), q
assert close(NegExact(3, 2, 1, cap=1e6, pfit='band')(Dk3, Dmix, sig, r, Dn=Dall), NegExact(3, 2, 1, cap=1e6)(Dk3, Dmix, sig, r, Dn=Dall), 1e-4)
assert torch.isfinite(NegExact(3, cap=1e6, pfit='band')(Dk3, Dmix, sig, r, Dn=Dbad)).all() and NegExact(3, pfit='band').name == 'negx_ratio_a1_b1_rc0.125_pband'
cal1 = dict(sigma=[80.0, 1.0, 0.002], T=[1.0, 1.0, 1.0])                               # T = 1: the calibrated rule is the plain one, up to the 1e-3 floor
assert close(NegExact(3, cap=1e6, pfit='cal', cal=cal1)(Dk3, Dmix, sig, r, Dn=Dall), want, 2e-2) and NegExact(3, pfit='cal', cal=cal1).name == 'negx_ratio_a1_b1_rc0.125_pcal'
hot = NegExact(3, cap=1e6, pfit='cal', cal=dict(sigma=[80.0, 0.002], T=[1e6, 1e6]))(Dk3, Dmix, sig, r, Dn=Dall)   # T huge: p uniform, g = 1/9
assert close(hot, Dmix - (Dk3 - Dmix) / 9, 1e-4)
pn = PosNeg(2, 3, low_lo=2.5, dc=1)                                                     # low band pushed only for sigma >= 2.5, mean colour never
assert close(pn(Dc, None, 1.0, r, Dn=Dn), PosNeg(1, 3, dc=1)(Dc, None, 1.0, r, Dn=Dn)) and pn.name == 'posneg_wl2_wh3_rc0.125_dc1_lowsig2.5'
hi = pn(Dc, None, 5.0, r, Dn=Dn); assert close((hi - Dc).mean(dim=(2, 3)), torch.zeros(4, 3, dtype=torch.float64))
assert close(hi, Dn + spec_apply(Dc - Dn, FreqBands(1, 2, 3).weight(5.0, r)))
k3 = torch.full((8,), 3)
for G, neg, want_calls in [(NegExact(3), None, (7, 7, 0)), (PosNeg(1, 3), torch.full((8,), 5), (14, 0, 0)), (NegPair(FakeNet(3.0), FakeNet(4.0), 0, 2), None, (7, 7, 7))]:
    c, u = FakeNet(1.0), FakeNet(2.0)
    x = sample(c, u, k3, G, num_steps=4, seed=0, device='cpu', neg_labels=neg)
    assert torch.isfinite(x).all() and (c.calls, u.calls, getattr(G, 'c2', FakeNet(0)).calls) == want_calls, (G.name, c.calls, u.calls)
npair = NegPair(FakeNet(1.0), FakeNet(2.0), 1, 2)                                     # with the same nets as the sampler it is NegBand
assert close(sample(FakeNet(1.0), FakeNet(2.0), k3, npair, num_steps=4, seed=0, device='cpu'), sample(FakeNet(1.0), FakeNet(2.0), k3, NegBand(1, 2), num_steps=4, seed=0, device='cpu'), 1e-4)
print('new negative rules ok')

# Wrong labels are never the true label, and are fixed per seed
labels = torch.arange(1000) % 10
w = wrong_labels(labels, 7)
assert (w != labels).all() and (w == wrong_labels(labels, 7)).all() and not (w == wrong_labels(labels, 8)).all() and set(w.tolist()) == set(range(10))

# Network calls per arm, through the real sampler: 4 steps = 7 evaluations of each branch in use
labels = torch.arange(8) % 10
for G, neg, want in [(ConstCFG(3), None, (7, 7)), (FreqBands(1, 3, 3), None, (7, 7)), (HighDir('cu', 3), None, (7, 7)), (Sharpen([1.0, 2.0], [0.5, 0.5]), None, (7, 0)),
                     (HighDir('cw', 3), wrong_labels(labels, 0), (14, 0)), (HighDir('wu', 3), wrong_labels(labels, 0), (14, 7)), (SourceSplit(3, 1), None, (7, 7))]:
    c, u = FakeNet(1.0), FakeNet(2.0)
    x = sample(c, u, labels, G, num_steps=4, seed=0, device='cpu', neg_labels=neg)
    assert torch.isfinite(x).all() and (c.calls, u.calls) == want, (G.name, c.calls, u.calls)
print('network calls ok')

# Same seed gives the same result, and the seed=0 noise is unchanged by the new flag (seed = a.seed * 100000 + b)
a1 = sample(FakeNet(1.0), FakeNet(2.0), labels, ConstCFG(3), num_steps=4, seed=0, device='cpu')
a2 = sample(FakeNet(1.0), FakeNet(2.0), labels, FreqBands(3, 3, 3), num_steps=4, seed=0, device='cpu')
assert close(a1, a2, 1e-3)
print('sampler ok')

try:
    import kornia
except ImportError:
    print('kornia not installed: Pyramid not tested')
else:
    assert close(Pyramid(1, 1)(Dc, Du, sig, r), Dc, 1e-4)
    assert close(Pyramid(3, 3)(Dc, Du, sig, r), Du + 3 * delta, 1e-4)
    assert torch.isfinite(sample(FakeNet(1.0), FakeNet(2.0), labels, Pyramid(1, 3), num_steps=4, seed=0, device='cpu')).all()
    # Where does the paper's low band cut off, compared with the FFT mask?  Gain of Low() on single cosines, on 32x32
    print('pyramid ok.  low-band gain by frequency (cycles/px):  pyramid vs FFT mask r_c=0.125')
    from kornia.geometry.transform import build_laplacian_pyramid, pyrup
    yy, xx = torch.meshgrid(torch.arange(32.), torch.arange(32.), indexing='ij')
    for k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]:
        c = torch.cos(2 * math.pi * k * xx / 32)[None, None]
        _, lo = build_laplacian_pyramid(c, 2); L = pyrup(lo)
        print(f'  f={k / 32:.3f}  pyramid {float((L * c).sum() / (c * c).sum()):6.3f}   mask {float(torch.sigmoid(torch.tensor((0.125 - k / 32) / 0.02))):6.3f}')
print('ALL OK')
