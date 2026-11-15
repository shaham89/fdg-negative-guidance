"""Shared utilities: EDM checkpoint loading, Heun sampler with pluggable guidance,
frequency-band tools, and the per-frequency SNR schedule.

Notation (EDM): the network is a denoiser D(x; sigma) ~ E[x0 | x_t], with x_t = x0 + sigma*n.
CFG on the denoiser:  D_g = D_u + w * (D_c - D_u).  (Affine in eps, so identical to CFG on eps.)
"""
import os, sys, pickle, math
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'edm'))   # makes torch_utils / dnnlib importable for the pickles
CKPT = os.path.join(ROOT, 'ckpt')
OUT = os.path.join(ROOT, 'out')

# ----------------------------------------------------------------------------- models
def load_nets(device, fp16=True, arch='vp'):
    """fp16=True runs the U-Nets in half precision (2.4x faster on A10; mean abs err 0.002 on [-1,1] pixels, see bench.py).
    arch='ve' loads EDM's second pair, trained independently with another architecture."""
    torch.backends.cudnn.benchmark = True
    nets = {}
    for k in ['cond', 'uncond']:
        with open(os.path.join(CKPT, f'edm-cifar10-32x32-{k}-{arch}.pkl'), 'rb') as f:
            nets[k] = pickle.load(f)['ema'].to(device).eval().requires_grad_(False)
        nets[k].use_fp16 = fp16
    return nets['cond'], nets['uncond']

def edm_sigmas(net, num_steps=18, sigma_min=0.002, sigma_max=80.0, rho=7.0, device='cuda'):
    sigma_min = max(sigma_min, net.sigma_min); sigma_max = min(sigma_max, net.sigma_max)
    i = torch.arange(num_steps, dtype=torch.float64, device=device)
    t = (sigma_max ** (1 / rho) + i / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([net.round_sigma(t), torch.zeros_like(t[:1])])  # length num_steps+1, last is 0

# ----------------------------------------------------------------------------- frequency tools
def radial_freq(H=32, W=32, device='cuda'):
    """|f| in cycles/pixel for each DFT coefficient (unshifted layout). Range [0, ~0.707]."""
    fy = torch.fft.fftfreq(H, device=device); fx = torch.fft.fftfreq(W, device=device)
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

# Band edges in cycles/pixel. For 32px: DFT radius bins ~ [0-2, 2-4, 4-6, 6-8, 8-12, 12-Nyq+].
BAND_EDGES = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.375, 1.0]
BAND_NAMES = ['0-2', '2-4', '4-6', '6-8', '8-12', '12+']

def band_masks(r, edges=BAND_EDGES):
    return torch.stack([((r >= lo) & (r < hi)).float() for lo, hi in zip(edges[:-1], edges[1:])])  # B,H,W

def band_energy(x, masks):
    """Per-band energy of x (N,C,H,W), averaged over N and C.  Parseval: sum over bands = mean ||x||^2 per image-channel."""
    F = torch.fft.fft2(x.float(), norm='ortho')
    P = (F.real ** 2 + F.imag ** 2)                     # N,C,H,W
    return torch.einsum('nchw,bhw->b', P, masks) / (x.shape[0] * x.shape[1])

def radial_profile(P, r, nbins=64):
    """Radially bin a per-coefficient power map P (H,W) -> (bins, centers)."""
    rb = torch.clamp((r / 0.7072 * nbins).long(), max=nbins - 1)
    num = torch.zeros(nbins, device=P.device).index_add_(0, rb.flatten(), P.flatten())
    cnt = torch.zeros(nbins, device=P.device).index_add_(0, rb.flatten(), torch.ones_like(P.flatten()))
    centers = (torch.arange(nbins, device=P.device) + 0.5) / nbins * 0.7072
    return num / cnt.clamp(min=1), centers

# ----------------------------------------------------------------------------- guidance rules
class Guidance:
    """Base: returns guided denoiser output.  Subclasses override weight(sigma, r) -> (H,W) tensor or scalar."""
    name = 'base'
    needs_u = True               # False: make_denoiser skips the unconditional net and passes Du=None
    needs_all = False            # True: make_denoiser passes the predictions of all classes (B,K,C,H,W) as Dn
    def weight(self, sigma, r):  # spectral weight map W(f) at noise level sigma
        raise NotImplementedError
    def __call__(self, Dc, Du, sigma, r, Dn=None):
        W = self.weight(float(sigma), r)
        delta = Dc - Du
        if not torch.is_tensor(W) or W.ndim == 0:
            return Du + W * delta
        Fd = torch.fft.fft2(delta.float(), norm='ortho')
        return Du + torch.fft.ifft2(Fd * W, norm='ortho').real.to(Dc.dtype)

class ConstCFG(Guidance):
    def __init__(self, w): self.w = w; self.name = f'cfg_w{w:g}'
    def weight(self, sigma, r): return self.w

class ComposedNeg(Guidance):
    """Project 3: positive class c with a negated class k:   D = Du + w_pos (Dc - Du) - w_neg (Dn - Du).
    w_pos=0 gives pure negation ('anything but k'); w_neg=0 gives plain CFG."""
    def __init__(self, w_pos, w_neg): self.wp, self.wn = w_pos, w_neg; self.name = f'neg_wp{w_pos:g}_wn{w_neg:g}'
    def __call__(self, Dc, Du, sigma, r, Dn=None):
        return Du + self.wp * (Dc - Du) - self.wn * (Dn - Du)

class FreqStatic(Guidance):
    """FDG/FreSca-style: w_low below cutoff r_c (cycles/px), w_high above, soft edge tau."""
    def __init__(self, w_low, w_high, r_c=0.125, tau=0.02):
        self.wl, self.wh, self.rc, self.tau = w_low, w_high, r_c, tau
        self.name = f'freq_wl{w_low:g}_wh{w_high:g}_rc{r_c:g}'
    def weight(self, sigma, r):
        s = torch.sigmoid((self.rc - r) / self.tau)   # 1 at low freq, 0 at high
        return self.wh + (self.wl - self.wh) * s

class TimeSched(Guidance):
    """Purely temporal: scalar w as a function of sigma, given by a table (sigmas, ws), interpolated in log sigma."""
    def __init__(self, sigmas, ws, name='time'):
        self.ls = np.log(np.asarray(sigmas, dtype=np.float64)); self.ws = np.asarray(ws, dtype=np.float64); self.name = name
        o = np.argsort(self.ls); self.ls, self.ws = self.ls[o], self.ws[o]
    def weight(self, sigma, r):
        return float(np.interp(math.log(max(sigma, 1e-8)), self.ls, self.ws))

class Interval(Guidance):
    """Kynkaanniemi et al.: guidance only for sigma in [lo, hi]."""
    def __init__(self, w, lo, hi): self.w, self.lo, self.hi = w, lo, hi; self.name = f'interval_w{w:g}_{lo:g}-{hi:g}'
    def weight(self, sigma, r): return self.w if (self.lo <= sigma <= self.hi) else 1.0

class FreqInterval(Guidance):
    """Stacking test: the static frequency split applied only inside a sigma interval (no guidance outside).
    If the split and the interval exploit the same mechanism, this should not beat either alone."""
    def __init__(self, w_low, w_high, lo, hi, r_c=0.125, tau=0.02):
        self.f = FreqStatic(w_low, w_high, r_c, tau); self.lo, self.hi = lo, hi
        self.name = f'freqint_wl{w_low:g}_wh{w_high:g}_{lo:g}-{hi:g}'
    def weight(self, sigma, r): return self.f.weight(sigma, r) if (self.lo <= sigma <= self.hi) else 1.0

class NegBand(Guidance):
    """Negative guidance per band, with labels = k:  D = Du - a Low(Dk - Du) - b High(Dk - Du).  a = b gives ConstCFG(-a).
    lo, hi: push only for sigma in [lo, hi], and D_u outside (not D_k, which FreqInterval's weight of 1 would give).
    dc: own strength for the mean colour (the DC term), 0 = leave the mean colour alone."""
    def __init__(self, a, b, lo=0.0, hi=float('inf'), r_c=0.125, tau=0.02, dc=None):
        self.f = FreqStatic(-a, -b, r_c, tau); self.lo, self.hi, self.dc = lo, hi, dc
        self.name = f'neg_a{a:g}_b{b:g}_rc{r_c:g}' + (f'_dc{dc:g}' if dc is not None else '') + (f'_sig{lo:g}-{hi:g}' if (lo, hi) != (0.0, float('inf')) else '')
    def weight(self, sigma, r):
        if not (self.lo <= sigma <= self.hi): return 0.0
        W = self.f.weight(sigma, r)
        if self.dc is not None: W = W.clone(); W[0, 0] = -self.dc
        return W

def spec_apply(delta, W):
    """Multiply delta by a spectral weight W (scalar or H,W map)."""
    if not torch.is_tensor(W) or W.ndim == 0: return W * delta
    return torch.fft.ifft2(torch.fft.fft2(delta.float(), norm='ortho') * W, norm='ortho').real.to(delta.dtype)

def band_posteriors(Dn, Du, r, r_c=0.125, tau=0.02, which=('full', 'low', 'high')):
    """p(class | x) from simplex_fit, on the whole image ('full') or on one band only ('low', 'high').  A perfect pair gives the same p in every band.
    Dn: B,K,C,H,W predictions of all classes.  Returns {name: p (B,K)}."""
    m = torch.sigmoid((r_c - r) / tau); out = {}
    if 'full' in which: out['full'] = simplex_fit(Dn, Du)[0]
    if 'low' in which: out['low'] = simplex_fit(spec_apply(Dn, m), spec_apply(Du, m))[0]
    if 'high' in which: out['high'] = simplex_fit(spec_apply(Dn, 1 - m), spec_apply(Du, 1 - m))[0]
    return out

class NegExact(Guidance):
    """The exact 'not k' rule, with labels = k:  D = Du - g(x) [a Low + b High](Dk - Du),  g = p(k|x) / (1 - p(k|x)), capped at cap.
    p comes from simplex_fit on the predictions of all classes (11 network calls per evaluation).  a = b = 1 is the exact rule (DNG eq. 10).
    mode 'mix': target = sum_{j != k} p_j D_j / (1 - p_k), and D = Du + [a Low + b High](target - Du).  With perfect nets both modes agree at a = b = 1.
    pfit: where p is fitted.  'full' (whole image), 'low' or 'high' (one band, used for both pushes),
    'band' (each band is pushed with its own p: g_low from the low-band fit, g_high from the high-band fit).
    Every band estimates the same p (one number per image), so with perfect nets all choices give the exact rule; they differ in how network errors enter."""
    needs_all = True
    def __init__(self, k, a=1.0, b=1.0, mode='ratio', cap=20.0, r_c=0.125, tau=0.02, pfit='full', cal=None):
        self.k, self.f, self.mode, self.cap, self.pfit, self.rc, self.tau, self.a, self.b = k, FreqStatic(a, b, r_c, tau), mode, cap, pfit, r_c, tau, a, b; self.log = []
        if pfit == 'cal':     # temperature per sigma, fitted on the no-push path against the final class (post_cal.json): p ~ (p_fit + 1e-3)^(1/T)
            o = np.argsort(cal['sigma']); self.cal_ls, self.cal_T = np.log(np.asarray(cal['sigma'])[o]), np.asarray(cal['T'])[o]
        self.name = f'negx_{mode}_a{a:g}_b{b:g}_rc{r_c:g}' + (f'_cap{cap:g}' if cap != 20 else '') + (f'_p{pfit}' if pfit != 'full' else '')
    def _g(self, p, ok):
        pk = (p[:, self.k] * ok).clamp(max=self.cap / (1 + self.cap)); return pk, pk / (1 - pk)
    def __call__(self, Dc, Du, sigma, r, Dn=None):
        ok = torch.isfinite(Dn).flatten(1).all(1) & torch.isfinite(Du).flatten(1).all(1)          # an image that blew up gets no push, and no crash
        Dn, Du, Dc = Dn.nan_to_num(0, 0, 0), Du.nan_to_num(0, 0, 0), Dc.nan_to_num(0, 0, 0)
        if self.pfit in ('low', 'high', 'band'):
            P = band_posteriors(Dn, Du, r, self.rc, self.tau, ('low', 'high'))
            if self.pfit == 'band':
                (pl, gl), (ph, gh) = self._g(P['low'], ok), self._g(P['high'], ok); m = torch.sigmoid((self.rc - r) / self.tau)
                self.log.append((float(sigma), float(pl.mean()), float(ph.mean()), float(gl.mean()), float(gh.mean())))
                d = Dc - Du; v = lambda t: t[:, None, None, None]
                return Du - v(gl) * spec_apply(d, self.a * m) - v(gh) * spec_apply(d, self.b * (1 - m))
            p = P[self.pfit]
        else:
            p = simplex_fit(Dn, Du)[0]
            if self.pfit == 'cal': p = torch.softmax(torch.log(p + 1e-3) / float(np.interp(math.log(float(sigma)), self.cal_ls, self.cal_T)), 1)
        p = p * ok[:, None]; pk, g = self._g(p, torch.ones_like(ok, dtype=p.dtype))
        self.log.append((float(sigma), float(pk.mean()), float(g.mean())))
        W = self.f.weight(float(sigma), r)
        if self.mode == 'ratio': return Du - g[:, None, None, None] * spec_apply(Dc - Du, W)
        q = p.clone(); q[:, self.k] = 0; q = q / q.sum(1, keepdim=True).clamp(min=1e-6)
        return Du + spec_apply((q[:, :, None, None, None] * Dn).sum(1) - Du, W)

class NegPair(Guidance):
    """NegBand with the push direction from the second (VE) pair:  D = Du - a Low(Dk2 - Du2) - b High(Dk2 - Du2)."""
    extra_nfe = 2
    def __init__(self, net_c2, net_u2, a, b, lo=0.0, hi=float('inf'), r_c=0.125, tau=0.02):
        self.c2, self.u2, self.f, self.lo, self.hi = net_c2, net_u2, FreqStatic(-a, -b, r_c, tau), lo, hi
        self.name = f'negpair_a{a:g}_b{b:g}_rc{r_c:g}' + (f'_sig{lo:g}-{hi:g}' if (lo, hi) != (0.0, float('inf')) else '')
    def extra(self, x32, sig32, onehot):
        return (self.c2(x32, sig32, onehot) - self.u2(x32, sig32, None)).to(torch.float64)
    def __call__(self, Dc, Du, sigma, r, Dn=None):
        return Du + spec_apply(Dn, self.f.weight(float(sigma), r)) if (self.lo <= float(sigma) <= self.hi) else Du

class PosNeg(Guidance):
    """FDG with a negative prompt: class k takes the place of the unconditional net (as in diffusers).  Dn = D_k from neg_labels.
    D = Dk + w_low Low(Dc - Dk) + w_high High(Dc - Dk).  (1, 1) is plain D_c.
    low_lo: the low band gets w_low only for sigma >= low_lo, and 1 (no push) below.  dc: own weight for the mean colour (1 = no push on it)."""
    needs_u = False
    def __init__(self, w_low, w_high, r_c=0.125, tau=0.02, low_lo=0.0, dc=None):
        self.f, self.g, self.low_lo, self.dc = FreqStatic(w_low, w_high, r_c, tau), FreqStatic(1.0, w_high, r_c, tau), low_lo, dc
        self.name = f'posneg_wl{w_low:g}_wh{w_high:g}_rc{r_c:g}' + (f'_dc{dc:g}' if dc is not None else '') + (f'_lowsig{low_lo:g}' if low_lo else '')
    def __call__(self, Dc, Du, sigma, r, Dn=None):
        W = (self.f if float(sigma) >= self.low_lo else self.g).weight(float(sigma), r)
        if self.dc is not None: W = W.clone(); W[0, 0] = self.dc
        return Dn + spec_apply(Dc - Dn, W)

def high_pass(x, r, r_c, tau):
    """High(x) with the FreqStatic soft mask: x minus its low band (cutoff r_c cycles/px)."""
    Fx = torch.fft.fft2(x.float(), norm='ortho')
    return torch.fft.ifft2(Fx * (1 - torch.sigmoid((r_c - r) / tau)), norm='ortho').real.to(x.dtype)

class FreqBands(Guidance):
    """Two dials, part A: own weight for the mean colour (DC), the rest of the low band, and the high band."""
    def __init__(self, w_dc, w_lr, w_hi, r_c=0.125, tau=0.02):
        self.wd, self.f = w_dc, FreqStatic(w_lr, w_hi, r_c, tau)
        self.name = f'bands_wd{w_dc:g}_wl{w_lr:g}_wh{w_hi:g}_rc{r_c:g}'
    def weight(self, sigma, r):
        W = self.f.weight(sigma, r).clone(); W[0, 0] = self.wd   # r[0,0] = 0 is the DC coefficient
        return W

class HighDir(Guidance):
    """Two dials, part B:  D = D_c + (w_high - 1) High(dir), with the low band left at the conditional prediction.
    dir 'cu': D_c - D_u (equals FreqStatic(1, w_high)) | 'cw': D_c - D_c' (c' a wrong class, same net, no uncond net) | 'wu': D_c' - D_u.
    lo, hi: guide only for sigma in [lo, hi]."""
    def __init__(self, direction, w_high, r_c=0.125, tau=0.02, lo=0.0, hi=float('inf')):
        self.dir, self.wh, self.rc, self.tau, self.lo, self.hi = direction, w_high, r_c, tau, lo, hi
        self.needs_u = direction != 'cw'
        self.name = f'highdir_{direction}_wh{w_high:g}_rc{r_c:g}' + (f'_sig{lo:g}-{hi:g}' if (lo, hi) != (0.0, float('inf')) else '')
    def __call__(self, Dc, Du, sigma, r, Dn=None):
        if not self.lo <= sigma <= self.hi: return Dc
        d = {'cu': lambda: Dc - Du, 'cw': lambda: Dc - Dn, 'wu': lambda: Dn - Du}[self.dir]()
        return Dc + (self.wh - 1) * high_pass(d, r, self.rc, self.tau)

class Sharpen(Guidance):
    """Two dials, part B control:  D = D_c + lambda(sigma) High(D_c).  No class direction, one network.
    Dose it by final sharpness (tune_dose.py), not by per-step energy: High(D_c) lines up with the image's own detail, so it accumulates."""
    needs_u = False
    def __init__(self, sigmas, lambdas, r_c=0.125, tau=0.02, name='sharpen'):
        self.t, self.rc, self.tau = TimeSched(sigmas, lambdas), r_c, tau
        self.name = f'{name}_rc{r_c:g}'
    def __call__(self, Dc, Du, sigma, r, Dn=None):
        return Dc + self.t.weight(float(sigma), r) * high_pass(Dc, r, self.rc, self.tau)

def class_preds(net, x32, sig32, chunk=1000):
    """D_k(x; sigma) for all K classes -> (B,K,C,H,W) float64."""
    B, K = x32.shape[0], net.label_dim
    xr = x32.repeat_interleave(K, 0); lab = torch.eye(K, device=x32.device).repeat(B, 1)
    out = torch.cat([net(xr[i:i + chunk], sig32, lab[i:i + chunk]) for i in range(0, B * K, chunk)])
    return out.to(torch.float64).reshape(B, K, *x32.shape[1:])

def proj_simplex(v):
    """Euclidean projection of each row of v (B,K) onto the probability simplex."""
    u = v.sort(dim=1, descending=True)[0]; css = u.cumsum(1) - 1
    rho = ((u - css / torch.arange(1, v.shape[1] + 1, device=v.device, dtype=v.dtype)) > 0).sum(1, keepdim=True)
    return (v - css.gather(1, rho - 1) / rho).clamp(min=0)

def simplex_fit(Dk, Du, iters=200):
    """p on the simplex minimizing || sum_k p_k D_k - D_u ||^2 by accelerated projected gradient.  Dk: B,K,C,H,W.  Returns p (B,K) and mix (B,C,H,W).
    A perfect pair of networks gives D_u = sum_k p(k|x) D_k exactly, so whatever mix cannot reach is disagreement between the networks."""
    B, K = Dk.shape[:2]; m = Dk.mean(1, keepdim=True)
    A = (Dk - m).reshape(B, K, -1); t = (Du - m[:, 0]).reshape(B, -1)
    G = A @ A.transpose(1, 2); b = (A @ t[:, :, None])[:, :, 0]
    try: L = torch.linalg.eigvalsh(G.cpu()).to(G.device)[:, -1:].clamp(min=1e-30)      # on the CPU: the GPU solver failed on near-singular G
    except RuntimeError: L = G.diagonal(dim1=1, dim2=2).sum(1, keepdim=True).clamp(min=1e-30)    # trace >= largest eigenvalue: slower steps, same answer
    p = y = torch.full((B, K), 1 / K, dtype=Dk.dtype, device=Dk.device); s = 1.0
    for _ in range(iters):
        p_new = proj_simplex(y - ((G @ y[:, :, None])[:, :, 0] - b) / L)
        s_new = (1 + math.sqrt(1 + 4 * s * s)) / 2; y = p_new + (s - 1) / s_new * (p_new - p); p, s = p_new, s_new
    return p, (p[:, :, None, None, None] * Dk).sum(1)

class SourceSplit(Guidance):
    """Two dials on the source of the guidance:  Delta = (D_c - mix) [class part] + (mix - D_u) [gap between the networks], mix from simplex_fit.
    D = D_c + (w_cls - 1) part(D_c - mix) + (w_gap - 1) part(mix - D_u),  part = identity, or High if high_only.
    (w, w) is CFG; (w, w) with high_only is the (1, w) split."""
    needs_all = True
    def __init__(self, w_cls, w_gap, high_only=False, r_c=0.125, tau=0.02):
        self.wc, self.wg, self.high, self.rc, self.tau = w_cls, w_gap, high_only, r_c, tau
        self.name = f'src_wc{w_cls:g}_wg{w_gap:g}' + (f'_high_rc{r_c:g}' if high_only else '')
    def __call__(self, Dc, Du, sigma, r, Dn=None):
        mix = simplex_fit(Dn, Du)[1]
        part = (lambda x: high_pass(x, r, self.rc, self.tau)) if self.high else (lambda x: x)
        return Dc + (self.wc - 1) * part(Dc - mix) + (self.wg - 1) * part(mix - Du)

class CrossPair(Guidance):
    """Guidance direction from a second, independently trained network pair.  Delta2 = D_c2 - D_u2.
    mode 'cross': D = D_c + (w - 1) part(Delta2)            only signal shared by both pairs can help the sampling net
    mode 'mean' : D = D_c + (w - 1) part((Delta + Delta2) / 2)   errors of the two pairs partly cancel, signal does not"""
    extra_nfe = 2
    def __init__(self, net_c2, net_u2, w, mode='cross', high_only=False, r_c=0.125, tau=0.02):
        self.c2, self.u2, self.w, self.mode, self.high, self.rc, self.tau = net_c2, net_u2, w, mode, high_only, r_c, tau
        self.needs_u = mode == 'mean'
        self.name = f'pair_{mode}_w{w:g}' + (f'_high_rc{r_c:g}' if high_only else '')
    def extra(self, x32, sig32, onehot):
        return (self.c2(x32, sig32, onehot) - self.u2(x32, sig32, None)).to(torch.float64)
    def __call__(self, Dc, Du, sigma, r, Dn=None):
        d = Dn if self.mode == 'cross' else 0.5 * (Dc - Du + Dn)
        return Dc + (self.w - 1) * (high_pass(d, r, self.rc, self.tau) if self.high else d)

class WienerGuidance(Guidance):
    """Guide only as far as the guidance signal can be trusted.  If Delta = signal + error, the best linear estimate of the signal is rho * Delta,
    rho = share of energy that is signal = agreement of two independent network pairs, per band and sigma (table from replicate_diag.py).
    W(f, sigma) = 1 + (w - 1) rho(f, sigma);  high_only keeps the low band at the conditional prediction.  rho = 1 gives CFG, or the (1, w) split."""
    def __init__(self, table, w, high_only=False, r_c=0.125, tau=0.02):
        self.ls = np.log(np.asarray(table['sigmas'], dtype=np.float64)); o = np.argsort(self.ls); self.ls = self.ls[o]
        self.rho = np.clip(np.asarray(table['rho'], dtype=np.float64), 0, 1)[o]     # steps,bands: DC, then BAND_EDGES bands without DC
        self.w, self.high, self.rc, self.tau = w, high_only, r_c, tau
        self.name = f'wiener_w{w:g}' + (f'_high_rc{r_c:g}' if high_only else '')
        self.m = None
    def weight(self, sigma, r):
        if self.m is None:
            dc = torch.zeros_like(r); dc[0, 0] = 1
            self.m = torch.cat([dc[None], band_masks(r) * (1 - dc)])
            self.h = (1 - torch.sigmoid((self.rc - r) / self.tau)) if self.high else torch.ones_like(r)
        ls = math.log(max(sigma, 1e-8))
        rho = torch.tensor([np.interp(ls, self.ls, self.rho[:, b]) for b in range(self.rho.shape[1])], dtype=r.dtype, device=r.device)
        return 1 + (self.w - 1) * self.h * torch.einsum('b,bhw->hw', rho, self.m)

class Pyramid(Guidance):
    """The paper's operator (its Algorithm 2): single-level Laplacian pyramid of the CFG delta, via kornia."""
    def __init__(self, w_low, w_high): self.wl, self.wh = w_low, w_high; self.name = f'pyr_wl{w_low:g}_wh{w_high:g}'
    def __call__(self, Dc, Du, sigma, r, Dn=None):
        from kornia.geometry.transform import build_laplacian_pyramid, pyrup
        high, low = build_laplacian_pyramid((Dc - Du).float(), 2)
        return Du + (self.wl * pyrup(low) + self.wh * high).to(Dc.dtype)

class FreqTime(Guidance):
    """w(f,t) = w_min + (w_max - w_min) * g(logSNR_t(f)),  logSNR = log P(f) - 2 log sigma.
    P(f): radially averaged per-coefficient power of the data (ortho FFT, pixel range [-1,1]).
    mode 'unresolved': g = sigmoid(-(logSNR - mu)/s)   -> guide bands not yet resolved (noise-dominated)
    mode 'resolved'  : g = sigmoid(+(logSNR - mu)/s)   -> guide bands already resolved
    mode 'bump'      : g = exp(-(logSNR - mu)^2 / 2 s^2) -> guide the band being decided right now
    """
    def __init__(self, spec_r, spec_P, w_min, w_max, mode='unresolved', mu=0.0, s=1.0):
        self.spec_r, self.spec_P = np.asarray(spec_r), np.asarray(spec_P)
        self.wmin, self.wmax, self.mode, self.mu, self.s = w_min, w_max, mode, mu, s
        self.name = f'ft_{mode}_wmin{w_min:g}_wmax{w_max:g}_mu{mu:g}_s{s:g}'
        self._logP = None
    def logsnr(self, sigma, r):
        if self._logP is None or self._logP.shape != r.shape:
            P = np.interp(r.cpu().numpy().ravel(), self.spec_r, self.spec_P).reshape(r.shape)
            self._logP = torch.tensor(np.log(P), device=r.device, dtype=torch.float32)
        return self._logP - 2.0 * math.log(sigma)
    def weight(self, sigma, r):
        z = (self.logsnr(sigma, r) - self.mu) / self.s
        if self.mode == 'unresolved': g = torch.sigmoid(-z)
        elif self.mode == 'resolved': g = torch.sigmoid(z)
        elif self.mode == 'bump': g = torch.exp(-0.5 * z ** 2)
        else: raise ValueError(self.mode)
        return self.wmin + (self.wmax - self.wmin) * g

# ----------------------------------------------------------------------------- sampler
def make_denoiser(net_c, net_u, onehot, guidance, r, uncond_mode='sep', onehot_neg=None, record=None, delta_cache=None):
    """Returns D(x, sigma, log=False, corrector=False) -> guided denoiser output (float64).  D.nfe counts network calls.
    delta_cache: None/False = off.  A float threshold: at Heun corrector evaluations with sigma > threshold, skip the
    unconditional net and reuse the predictor's (Dc - Du)  (3 instead of 4 NFE for those steps).  0 = always."""
    cache = {}; nfe = {'c': 0, 'u': 0, 'n': 0}
    thr = None if (delta_cache is None or delta_cache is False) else (float('inf') if delta_cache is True else float(delta_cache))
    def D(xx, sig, log=False, corrector=False):
        sig32 = sig.to(torch.float32); x32 = xx.to(torch.float32)
        if guidance.needs_all:
            Dall = class_preds(net_c, x32, sig32); Dc = Dall[torch.arange(x32.shape[0]), onehot.argmax(1)]; nfe['c'] += onehot.shape[1]
        else:
            Dall = None; Dc = net_c(x32, sig32, onehot).to(torch.float64); nfe['c'] += 1
        if not guidance.needs_u:
            Du = None
        elif thr is not None and corrector and 'delta' in cache and float(sig) > thr:
            Du = Dc - cache['delta']
        else:
            if uncond_mode == 'sep': Du = net_u(x32, sig32, None).to(torch.float64)
            else: Du = net_c(x32, sig32, torch.zeros_like(onehot)).to(torch.float64)
            nfe['u'] += 1; cache['delta'] = Dc - Du
        Dn = None
        if onehot_neg is not None: Dn = net_c(x32, sig32, onehot_neg).to(torch.float64); nfe['n'] += 1
        if hasattr(guidance, 'extra'): Dn = guidance.extra(x32, sig32, onehot); nfe['n'] += guidance.extra_nfe
        if log and record is not None and Du is not None:
            if '_hook' in record: record['_hook'](xx, Dc, Du, sig)
            m = record.setdefault('_masks', band_masks(r))
            record.setdefault('E_delta', []).append(band_energy(Dc - Du, m).cpu().numpy())
            Fd = torch.fft.fft2((Dc - Du).float(), norm='ortho')
            record.setdefault('P_delta', []).append((Fd.real ** 2 + Fd.imag ** 2).mean(dim=(0, 1)).cpu().numpy())  # H,W full power map
            record.setdefault('E_Du', []).append(band_energy(Du, m).cpu().numpy())
            record.setdefault('E_x', []).append(band_energy(xx, m).cpu().numpy())
            record.setdefault('sigma', []).append(float(sig))
        return guidance(Dc, Du, sig, r, Dn=Dall if guidance.needs_all else Dn)
    D.nfe = nfe
    return D

def heun_step(x, t_cur, t_next, D, last, log=False):
    """One EDM Heun step (Euler predictor + trapezoidal corrector; Euler only on the final step to sigma=0)."""
    d_cur = (x - D(x, t_cur, log=log)) / t_cur
    x_next = x + (t_next - t_cur) * d_cur
    if not last:
        d_prime = (x_next - D(x_next, t_next, corrector=True)) / t_next
        x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next

def init_noise(N, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn([N, 3, 32, 32], generator=g, device=device, dtype=torch.float64)

def wrong_labels(labels, seed):
    """A random wrong CIFAR-10 class for each sample (fixed for the whole trajectory): label + offset in 1..9."""
    g = torch.Generator(device=labels.device).manual_seed(seed + 987654321)
    return (labels + torch.randint(1, 10, labels.shape, generator=g, device=labels.device)) % 10

@torch.no_grad()
def sample(net_c, net_u, labels, guidance, num_steps=18, seed=0, device='cuda', uncond_mode='sep',
           record=None, sigmas=None, rho=7.0, neg_labels=None, delta_cache=None, stats=None):
    """EDM deterministic Heun sampler (S_churn=0) with guidance applied to the denoiser output.
    labels: (N,) int64 class indices.  uncond_mode: 'sep' (separate uncond net) | 'zero' (cond net, zero label).
    sigmas: optional explicit schedule (num_steps+1 values ending in 0) overriding (num_steps, rho).
    neg_labels: optional (N,) class indices evaluated as a third branch (Project 3).
    record: optional dict; per-step band energies of delta / D_u / x are appended (predictor evaluations only).
    Returns float images in [-1,1] (N,3,32,32) before clamping."""
    N = labels.shape[0]
    eye = torch.eye(net_c.label_dim, device=device)
    onehot = eye[labels]; onehot_neg = eye[neg_labels] if neg_labels is not None else None
    t_steps = edm_sigmas(net_c, num_steps, rho=rho, device=device) if sigmas is None else torch.as_tensor(sigmas, dtype=torch.float64, device=device)
    n_steps = len(t_steps) - 1
    r = radial_freq(32, 32, device)
    D = make_denoiser(net_c, net_u, onehot, guidance, r, uncond_mode, onehot_neg, record, delta_cache)
    x = init_noise(N, seed, device) * t_steps[0]
    for i in range(n_steps):
        x = heun_step(x, t_steps[i], t_steps[i + 1], D, last=(i == n_steps - 1), log=True)
    if stats is not None: stats.update(D.nfe)     # network calls for this batch: cond / uncond / neg
    return x.float()

def to_uint8(x):
    return (x * 127.5 + 128).clip(0, 255).to(torch.uint8)   # N,3,H,W

def clip_fraction(x):
    """Fraction of pixel values outside [-1,1] before clamping (the proposal's saturation proxy)."""
    return float(((x < -1) | (x > 1)).float().mean())

def load_spectrum():
    d = np.load(os.path.join(OUT, 'cifar_spectrum.npz'))
    return d['r'], d['P']

# ----------------------------------------------------------------------------- metrics
_clf = None
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class TensorDS(torch.utils.data.Dataset):
    def __init__(self, imgs): self.imgs = imgs        # uint8 (N,3,H,W)
    def __len__(self): return self.imgs.shape[0]
    def __getitem__(self, i): return self.imgs[i]

def cifar_train_uint8(classes=None):
    """CIFAR-10 train as uint8 (N,3,32,32), optionally restricted to a list of classes."""
    import torchvision
    ds = torchvision.datasets.CIFAR10(os.path.join(OUT, 'data'), train=True, download=False)
    x = torch.tensor(ds.data).permute(0, 3, 1, 2).contiguous(); y = torch.tensor(ds.targets)
    if classes is not None: x = x[torch.isin(y, torch.tensor(classes))]
    return x

@torch.no_grad()
def classify(imgs, device='cuda'):
    """Pretrained CIFAR-10 ResNet-56 predictions (N,) for uint8 (N,3,32,32)."""
    global _clf
    if _clf is None:
        _clf = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True, verbose=False, trust_repo=True).to(device).eval()
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1); std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
    preds = []
    for i in range(0, len(imgs), 1000):
        xb = (imgs[i:i + 1000].to(device).float() / 255 - mean) / std
        preds.append(_clf(xb).argmax(1).cpu())
    return torch.cat(preds)

def fid_is(imgs, ref=None, ref_name='cifar10-train'):
    """FID + IS of uint8 imgs vs a reference (default: CIFAR-10 train; or a uint8 tensor with a cache name)."""
    import torch_fidelity
    m = torch_fidelity.calculate_metrics(input1=TensorDS(imgs), input2=('cifar10-train' if ref is None else TensorDS(ref)), cuda=True, fid=True, isc=True,
                                         verbose=False, cache_root=os.path.join(OUT, 'fid_cache'), input2_cache_name=ref_name,
                                         datasets_root=os.path.join(OUT, 'data'))
    return {'fid': m['frechet_inception_distance'], 'is': m['inception_score_mean']}

def evaluate(imgs, labs, device='cuda'):
    """imgs: uint8 (N,3,32,32) tensor, labs: (N,) int64.  Returns dict(fid, is, class_acc)."""
    res = fid_is(imgs)
    res['class_acc'] = float((classify(imgs, device) == labs).float().mean())
    return res
