#!/bin/bash
# One-time setup on a Linux machine with a CUDA GPU (also used on Google Colab).  About 5 min; it skips what is already done.
#   bash scripts/setup.sh                 run from anywhere, inside your Python environment
# It installs the Python packages, clones NVIDIA's EDM code (needed to unpickle the networks), downloads the four EDM CIFAR-10
# checkpoints (VP and VE pairs, about 900 MB), CIFAR-10 and the two classifiers, and copies the small tables the code reads.
# Everything goes into src/: src/edm, src/ckpt, src/out (none of them is tracked by git).
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO/src"
pip install -q -r "$REPO/requirements.txt"
[ -d edm ] || git clone -q --depth 1 https://github.com/NVlabs/edm.git
mkdir -p ckpt out/neg out/data
for k in cond-vp uncond-vp cond-ve uncond-ve; do
  f=ckpt/edm-cifar10-32x32-$k.pkl
  [ -s $f ] || wget -q -O $f https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-$k.pkl
done
# torchvision downloads CIFAR-10 from the Toronto server, which can be slow.  To use a copy you already have:
#   CIFAR_TAR=/path/to/cifar-10-python.tar.gz bash scripts/setup.sh
if [ -n "$CIFAR_TAR" ] && [ -s "$CIFAR_TAR" ] && [ ! -s out/data/cifar-10-python.tar.gz ]; then cp "$CIFAR_TAR" out/data/; fi
python - <<'PY'
import torch, torchvision
torchvision.datasets.CIFAR10('out/data', train=True, download=True)        # the code reads it later with download=False
for m in ['cifar10_resnet56', 'cifar10_vgg16_bn']:                         # the two classifiers, cached for later runs
    torch.hub.load('chenyaofo/pytorch-cifar-models', m, pretrained=True, verbose=False, trust_repo=True)
PY
# Small tables from earlier runs: CIFAR-10 power spectrum, Wiener trust table, time-only schedules, low-band reader, calibration
cp -n "$REPO"/results/two_dials/cifar_spectrum.npz "$REPO"/results/two_dials/wiener_table.json "$REPO"/results/two_dials/time_from_*.json out/
cp -n "$REPO"/results/neg/reader.json "$REPO"/results/neg_sweep12/post_cal.json out/neg/
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo SETUP DONE
