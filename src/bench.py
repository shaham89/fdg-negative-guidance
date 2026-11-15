"""Speed/fidelity check: fp32 vs fp16 network execution, and batch size."""
import time, torch
from common import *
torch.backends.cudnn.benchmark = True
dev = 'cuda'; net_c, net_u = load_nets(dev)
labels = torch.arange(250, device=dev) % 10
G = ConstCFG(3.0)

def run(tag):
    torch.cuda.synchronize(); t = time.time()
    x = sample(net_c, net_u, labels, G, seed=0, device=dev)
    torch.cuda.synchronize(); print(f'{tag}: {time.time() - t:.1f}s / 250 imgs'); return x

x32 = run('fp32')
for n in (net_c, net_u): n.use_fp16 = True
x16 = run('fp16')
x16b = run('fp16 (warm)')
err = (x32 - x16).abs()
print(f'fp16 vs fp32: max abs err {err.max():.4f}, mean {err.mean():.5f} (pixel range 2.0); uint8-identical frac {(to_uint8(x32)==to_uint8(x16)).float().mean():.3f}')
print(f'clip_frac fp32 {clip_fraction(x32):.4f}  fp16 {clip_fraction(x16):.4f}')
