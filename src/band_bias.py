"""Which band does the classifier read?  Real CIFAR-10 images, no sampling: Low(image of class a) + High(image of class b), with a != b.
This calibrates class accuracy: it says how far "the classifier reads the class in the high band" is a property of the classifier, not of guidance.
    python band_bias.py   -> rate of a, rate of b, at several cutoffs"""
import numpy as np, torch
from common import *
dev = 'cuda'
imgs = torch.cat([cifar_train_uint8([c])[:1000] for c in range(10)]).float().to(dev) / 127.5 - 1; labs = torch.arange(10, device=dev).repeat_interleave(1000)
g = torch.Generator(device=dev).manual_seed(0); perm = torch.randperm(len(imgs), generator=g, device=dev); ok = labs != labs[perm]
xa, ya, xb, yb = imgs[ok], labs[ok], imgs[perm][ok], labs[perm][ok]
mean = torch.tensor([0.4914, 0.4822, 0.4465], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.2470, 0.2435, 0.2616], device=dev).view(1, 3, 1, 1)
r = radial_freq(32, 32, dev)
for cn in ['cifar10_resnet56', 'cifar10_vgg16_bn']:
    clf = torch.hub.load('chenyaofo/pytorch-cifar-models', cn, pretrained=True, verbose=False, trust_repo=True).to(dev).eval()
    @torch.no_grad()
    def pred(x):
        u = ((x + 1) * 127.5).round().clamp(0, 255) / 255
        return torch.cat([clf((u[i:i + 1000] - mean) / std).argmax(1) for i in range(0, len(u), 1000)])
    print(cn, 'whole images: rate of own class', round((pred(xa) == ya).float().mean().item(), 3))
    for r_c in [0.0625, 0.125, 0.1875, 0.25]:
        S = torch.sigmoid((r_c - r) / 0.02); low = lambda x: torch.fft.ifft2(torch.fft.fft2(x, norm='ortho') * S, norm='ortho').real
        P = pred(low(xa) + xb - low(xb)); Pl = pred(low(xa))
        print(f'  r_c={r_c:.4f}  Low(a)+High(b): reads a {(P == ya).float().mean().item():.3f}  reads b {(P == yb).float().mean().item():.3f}   |  Low(a) alone reads a {(Pl == ya).float().mean().item():.3f}')
