# Frequency-decoupled guidance, and which frequency band removes a class

Course project for "Generative Learning and Diffusion Models" (Technion, 2026), built around one paper:
**Frequency-Decoupled Guidance (FDG)**, Sadat et al. 2025, [arXiv:2506.19713](https://arxiv.org/abs/2506.19713).

Everything here is inference only. There is no training: all runs use NVIDIA's public EDM checkpoints for CIFAR-10
(32x32 images, a class-conditional and an unconditional network, sampled with EDM's 18-step Heun sampler).

The project has two parts:

1. **Two dials.** A check of the paper's method on this model: when FDG guides the low and the high frequencies with
   different weights, is it really guiding toward the class, or is it a colour fix plus a sharpening filter?
2. **Which band removes a class? (the main part)** FDG's split applied to *negative* guidance, that is, "make anything
   except class k". Which band should push away from the class, what does each band cost, and when in sampling should
   it push?

The report is in [`report/`](report/), the proposal in [`proposal/`](proposal/) and the slides in [`slides/`](slides/).

## Background in four lines

- A diffusion model here is a denoiser `D(x; sigma)`: from a noisy image `x = x0 + sigma * n` it guesses the clean image `x0`.
- Classifier-free guidance (CFG) runs the net with the class label (`D_c`) and without it (`D_u`) and uses
  `D = D_u + w (D_c - D_u)`. With `w > 1` the class shows more strongly, at a cost in variety.
- FDG splits the guidance term into a low-frequency part and a high-frequency part and gives each its own weight:
  `D = D_u + w_low Low(D_c - D_u) + w_high High(D_c - D_u)`. Here the split is a soft FFT mask with its edge at
  4 cycles per image (`r_c = 0.125`). The paper's Laplacian pyramid gives a similar result (FID 6.31 against 6.07 for the mask).
- Negative guidance flips the sign to push away from a class k. This project splits that push by band:
  `D = D_u - a Low(D_k - D_u) - b High(D_k - D_u)`, so `a` is the low-band strength and `b` the high-band strength.

## Main results

Unless a line says otherwise: one noise seed, FID and IS from torch-fidelity, class rates from a pretrained CIFAR-10
ResNet-56 (VGG-16 gives nearly the same rates). Part 1 uses 10,000 images per arm. Part 2 uses 5,000 images per arm,
and its FID is measured against real CIFAR-10 images that are *not* class k. The runs used two GPUs (an A10 first,
then an A100 on Colab), and the same arm scores about 0.1 FID higher on the A100, so numbers are compared within one
machine. All numbers below come from the files in [`results/`](results/) ([guide](results/README.md)).

### Part 1: FDG is more than a colour fix plus sharpening

| Arm (10k images, A10) | FID | IS | class accuracy |
|---|---|---|---|
| CFG w = 1 (no guidance beyond the conditional net) | 3.84 | 9.49 | 0.951 |
| CFG w = 1.5 | 4.87 | 9.79 | 0.985 |
| CFG w = 3 | 9.98 | 9.70 | 0.991 |
| FDG (w_low, w_high) = (1, 3) | 6.07 | 10.05 | 0.985 |
| FDG (3, 1) | 8.50 | 9.60 | 0.990 |
| CFG w = 3 only while sigma is in [0.3, 2] | 5.22 | 10.04 | 0.987 |

**Compare on a frontier, not at equal w.** At w = 3, FDG's FID is far better than CFG's (6.07 against 9.98). But
plain CFG at w = 1.5 reaches the same accuracy with a lower FID (4.87), while FDG keeps a higher IS (10.05 against 9.79).
A simple time window does better here (FID 5.22 at the same IS). Our reading of the paper's Table 1 and Fig. 5 (EDM2-S, values read
from the plot) is that its headline gain is also partly an equal-w effect. The time-window result goes against the
paper's Table 3, which used another model and other metrics.

**The low band's cost is mostly the coarse layout, not the mean colour.** We split the low band into the mean colour
(DC) and the rest (1 to 4 cycles per image: coarse layout and large colour blobs) and guided each part separately,
in all 8 combinations. Starting from w = 1, guiding only the mean colour at w = 3 costs +1.07 FID, only the coarse layout
+3.52, only the high band +2.23. Averaged over the 8 arms the costs are 0.86, 3.43 and 1.90, and the interactions are
small (the largest is -0.25). The shift of the mean colour itself comes almost entirely from the DC part.

**The high band is not just sharpening.** A sharpening filter that needs no class, dosed to give the same final
sharpness as FDG (1, 3), leaves the accuracy where w = 1 has it (0.953, FDG 0.985) and costs almost no FID (3.95).

**At 32x32 the high band alone can steer the class.** Guiding only the high band toward a random *wrong* class makes 69%
of the images read as that class (0.5% without the push). Note that the FDG paper says the low band is the main driver
of class alignment, and in our grid both bands raise accuracy ((3, 1): 0.990, (1, 3): 0.985). At 32x32 the "high band"
(above 4 cycles per image) still holds object outlines, so this may not carry over to larger images.

**A useful side result.** Taking the high-band guidance direction from EDM's second, independently trained network
pair (VE) lowers FDG's FID from 6.07 to 5.28 at the same IS (10.06), at the cost of a third network call.

### Part 2: which band removes a class

"Not a cat" (k = 3), 5,000 images, A10, seed 0. "Others changed" is the share of non-cat images (same noise, no push)
whose class changes; "dogs kept" is the share of no-push dogs that stay dogs (dog is the class nearest to cat).

| Arm | cats left | FID | others changed | dogs kept |
|---|---|---|---|---|
| No push (`D_u`) | 9.0% | 7.61 | - | - |
| Reference: conditional net with labels from the other 9 classes | 1.5% | 6.42 | - | - |
| Low band, a = 0.5 | 3.3% | 10.43 | 27% | 50% |
| Low band, a = 1 | 1.2% | 19.24 | 40% | 31% |
| High band, b = 1 | 3.9% | 12.97 | 13% | 78% |
| High band, b = 2 | 2.1% | 22.09 | 19% | 63% |
| Both bands, 0.5 | 1.9% | 16.18 | 29% | 47% |
| High band b = 2, early only (sigma >= 0.8) | 3.2% | 11.07 | 18% | 68% |
| Low band a = 1, late only (sigma < 0.8) | 7.8% | 7.39 | 2% | 96% |

The reference is not a target that a push can reach: it uses the conditional network and exactly the right class mix,
while every push starts from the separate unconditional network.

**With a fixed strength, the low band removes more cats for less FID.** At about the same cat rate, low 0.5 gives
3.3% at FID 10.43 and high 1 gives 3.9% at 12.97. Pushing both bands is about as good as the low band alone
(both 0.5: 1.9% at 16.18). Seeds 1 and 2 (A100) repeat the main rows; most move by less than 0.5 FID, at most by 0.9.

**But the low band changes many other images.** At similar removal (low 0.5 against high 1), 27% against 13% of
the non-cat images change class, and 50% against 78% of the dogs stay dogs. The low band moves an image to a different
picture and shifts its colours; the high band keeps the picture and changes details.

**Early pushes are cheap, late pushes are expensive.** The high band at b = 2 pushed only while sigma >= s
(A100, seed 0): s = 2.5 gives 6.3% cats at FID 7.67, s = 1.5 gives 4.6% at 8.58, s = 0.4 gives 2.8% at 15.62, and all
steps 2.4% at 22.44. The low band pushed only late does almost nothing (7.8% cats).

**The same holds for all ten classes** (A100, one seed per class, the nine classes other than cat): low 0.5 leaves
2.6-4.0% of class k at FID 9.0-11.1, high 2 leaves 2.4-5.1% at FID 20.8-23.7, and high 2 early-only leaves 3.1-5.5% at
FID 10.2-12.3. Leaving the mean colour alone in the low-band push helps a little (low 1: 1.4% at 16.01 against 1.2% at
19.11 on the same machine).

**A strength per image beats every fixed push.** With perfect networks, "not k" can be sampled exactly with
`D = D_u - p(k|x) / (1 - p(k|x)) (D_k - D_u)`. This is the rule of Dynamic Negative Guidance (Koulischer et al., 2025);
it is not ours. We estimate p(k|x) at every step by fitting `D_u` as a mixture of the ten class predictions `D_k`
(11 network calls per step). `(a, b)` below multiplies this strength in the low and the high band; only (1, 1) is the
exact rule, the others are scaled versions.

| A100, 5,000 images, seed 0 | cats left | FID |
|---|---|---|
| No push | 9.00% | 7.72 |
| Reference | 1.52% | 6.41 |
| Fixed low a = 1 / fixed high b = 2 | 1.20% / 2.40% | 19.11 / 22.44 |
| Exact rule (1, 1) | 4.24% | 7.02 |
| Scaled (1.5, 1.5) / (2, 2) | 2.60% / 1.80% | 7.16 / 7.58 |
| Split (2, 1) / (1, 2) | 2.28% / 2.96% | 7.40 / 6.95 |
| (1, 1) / (1.5, 1.5) with calibrated p | 3.02% / 1.66% | 6.93 / 7.07 |

The exact rule (1, 1) has a lower FID than no push on all three seeds (7.02, 6.81, 6.71 against 7.72, 7.77, 7.66). In the
five (1, 1) runs with paired data it changes only 9-11% of the other images (the fixed low 0.5 push: 27%). **Splitting it by band does not help**: (2, 1) and (1, 2) lie on the same
curve as the equal-weight arms, within noise, and fitting p on one band only changes nothing (4.0-4.4% cats, FID
6.98-7.04). The fitted p is badly calibrated (too confident at high noise, not confident enough at low noise), and one
temperature per noise level improves both numbers. That calibration result is one seed, and the temperatures were fitted
on the same noise, so it is optimistic.

**When does a push change the class? (a measurement, with caveats)** `pmap.py` pushes one band for only two sampler
steps, at strength 2, on 4,000 images that all start from the same unguided path, and then lets sampling finish.
"Away" pushes from the class the image would become, "toward" pushes to one other class.

| Peak of the response | away (any class change) | toward (reached the target) | toward (any class change) |
|---|---|---|---|
| low band | 0.77 at sigma 8.4-5.3 | 0.48 at sigma 3.3-1.9 | 0.73 at sigma 3.3-1.9 |
| high band | 0.35 at sigma 3.3-1.9 | 0.43 at sigma 1.1-0.59 | 0.67 at sigma 1.1-0.59 |

So pushing away peaks earlier than pushing toward, in both bands, even with the same measure. Three things keep this
from being a firm result: the toward push is larger than the away push late in sampling (`pmap_size.json`), at strength 1
the low-band toward curve is flat (0.15-0.16 between sigma 8.4 and 1.9), and the Gaussian model with exact denoisers
(`analysis/gmm_pmap.py`) does not clearly show the shift (low-band toward 0.13 against 0.15, within noise at n = 1000).
We read "an image leaves a class earlier than it can reach a new one" as a hypothesis.

**"A dog, not a cat".** Here FDG is used as published, with `D_cat` in place of `D_u` (FID against real dogs, one seed).
High band only, (w_low, w_high) = (1, 2): 0.5% cats at FID 15.21, against plain CFG toward dog at w = 2: 1.9% at 15.18.
For two other pairs (cat not dog, truck not car) there is no clear winner, and for truck-not-car only 2-6 cars in 5,000
remain in any arm, so only FID separates the arms.

**What did not help:** pushing each frequency only while the noise still hides it (`neg.py ft`), taking the push direction
from the VE network pair, and, in Part 1, shrinking the guidance by how well the two network pairs agree ("Wiener guidance" did not beat the
simple time window).

## Limitations

- Most rows are one noise seed. Seeds 1 and 2 exist only for the main "not a cat" rows and for the exact rule (1, 1).
- 5,000 images per arm in Part 2 and 10,000 in Part 1. From half splits (four arms only) the FID noise is roughly
  0.1, and IS noise about 0.1, at 10k images. Small IS differences are not meaningful.
- 32x32 CIFAR-10 only. The band edge at 4 cycles per image puts object outlines in the "high" band; this may differ
  at the paper's resolutions and in latent models.
- The classifiers read mostly the high band (on real images, ResNet-56 gets 14% of blurred images right). A low-band
  reader built from the VE network (59% on blurred images) is used as a second check, on 2,000 images per arm.
- FID for "not k" is against real images without class k, so removing k lowers FID by itself.
- Strength-4 arms contain many black images; their FIDs are not valid.
- The image samples (`.npz`) are not included. A few diagnostics shown in the slides have no saved result file (see the
  results guide).

## Setup

You need Linux and an NVIDIA GPU. One 10k-image arm takes about 15-20 minutes on an A10 including metrics, and a 5k arm about
10 minutes; an A100 is about 2.6 times faster.

```bash
git clone https://github.com/shaham89/fdg-negative-guidance.git
cd fdg-negative-guidance
# make a Python environment with a CUDA build of PyTorch first (see pytorch.org), then:
bash scripts/setup.sh
```

`scripts/setup.sh` installs `requirements.txt`, clones [NVlabs/edm](https://github.com/NVlabs/edm) into `src/edm`
(its `dnnlib` and `torch_utils` are needed to unpickle the networks), downloads the four EDM CIFAR-10 checkpoints
(`edm-cifar10-32x32-{cond,uncond}-{vp,ve}.pkl`) into `src/ckpt`, downloads CIFAR-10 and the two classifiers
([chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)), and copies the small tables the
code reads into `src/out`. On Google Colab, use [`scripts/colab.ipynb`](scripts/colab.ipynb).

## Running

All experiment scripts run from `src/` and write to `src/out/`.

```bash
cd src
python test_twodials.py                                  # CPU check of the guidance rules with fake networks (no GPU needed)
python gen.py cfg --w 3 --n 200 --bs 200 --tag smoke     # quick GPU check, under a minute

python gen.py cfg --w 3                                  # one CFG arm: 10k images, FID / IS / accuracy -> out/results.jsonl
python gen.py freq --w_low 1 --w_high 3 --r_c 0.125      # FDG with the FFT mask
python gen.py bands --w_dc 1 --w_low 3 --w_high 3        # own weights for the mean colour, the coarse layout and the high band
python gen.py pyramid --w_low 1 --w_high 3               # the paper's Laplacian pyramid (needs kornia)

python neg.py reader                                     # check the low-band reader and choose its noise level
python neg.py base --k 3                                 # "not a cat": no push, and the reference
python neg.py arm --k 3 --a 1 --b 0                      # low-band push; --lo 0.8 pushes only while sigma >= 0.8
python neg_diag.py --k 3 --near 5                        # paired comparison of every saved arm with no push
python neg.py exact --k 3 --a 1 --b 1                    # the exact rule with p fitted at every step
python post_diag.py --n 2000 && python fit_cal.py        # fitted p against the final class, then one temperature per sigma
python neg.py exact --k 3 --a 1 --b 1 --pfit cal         # the exact rule with calibrated p
python pmap.py --n 4000 --s 2                            # the push map
```

Each script explains its options at the top of the file. To repeat a whole experiment, run its sweep script from anywhere
(for example `bash scripts/sweep9.sh`); each script starts with a comment that says what it tests.

| Experiment | Script | Results |
|---|---|---|
| CFG baseline and fixed frequency splits | `scripts/sweep1.sh` | `results/two_dials` |
| Time-only versions of the splits, frequency-by-time rules | `scripts/sweep2.sh` | `results/two_dials` |
| Guidance interval frontier | `scripts/sweep5.sh` | `results/two_dials` |
| Two dials: the 2x2x2 grid, high-band controls, pyramid | `scripts/sweep6.sh` (doses from `src/tune_dose.py`) | `results/two_dials` |
| Which part of the guidance is real: network-pair agreement, Wiener guidance, VE direction, noise windows | `scripts/sweep7.sh` (sources `sweep7b.sh`) | `results/two_dials` |
| "Not a cat" and "not a car": fixed pushes and early/late | `scripts/sweep9.sh`, `sweep9b.sh`, `sweep9c.sh` | `results/neg` |
| Tricks, "a dog, not a cat", exact rule at 2k, timing, seeds, all ten classes | `scripts/sweep10.sh` (check first with `smoke10.sh`) | `results/neg_sweep10` |
| Exact rule at 5k, more "c, not k" pairs | `scripts/sweep11.sh`, `sweep11b.sh` | `results/neg_sweep11` |
| Posterior per band, push map, p per band, seeds | `scripts/sweep12.sh` | `results/neg_sweep12` |
| Calibrated p, push map at strength 1 | `src/fit_cal.py`, then `scripts/sweep12c.sh` | `results/neg_sweep12c` |
| The Gaussian model with exact denoisers (CPU) | `analysis/gmm_toy.py`, `analysis/gmm_pmap.py` | `results/gaussian_model` |

Charts: `src/plot_pareto.py` and `src/plot_twins.py` (Part 1), `src/plot_pmap.py` and `src/plot_pmap_s.py` (push map).
They run on CPU and read the saved files.

## Repository layout

```
src/                 all experiment code; run scripts from here
  common.py          sampler, guidance rules (one class per rule), band filters, metrics
  gen.py             Part 1: one guided arm, then FID / IS / accuracy
  neg.py             Part 2: "not class k" arms, the exact rule, "c, not k"
  neg_diag.py        paired comparison of each arm with the no-push images
  pmap.py            push map: one band, two steps, toward or away
  post_diag.py       how well the fitted p(k|x) predicts the final class; fit_cal.py makes the calibration table
  *_diag.py, ...     other diagnostics (network-pair agreement, class and gap parts, hybrids, sharpening dose)
  plot_*.py          charts
  test_twodials.py   CPU checks of the guidance rules with fake networks
scripts/             setup.sh, colab.ipynb and the sweep scripts that produced the results
analysis/            CPU-only scripts: the Gaussian model, and checks that read saved results
results/             saved result rows, diagnostics, logs, charts and image grids (see results/README.md)
slides/              the talk (PowerPoint)
proposal/            the project proposal (PDF)
report/              the final report (PDF and LaTeX)
```

## References

- S. Sadat et al. Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales (FDG). 2025. [arXiv:2506.19713](https://arxiv.org/abs/2506.19713)
- J. Ho and T. Salimans. Classifier-Free Diffusion Guidance. 2022. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- T. Karras et al. Elucidating the Design Space of Diffusion-Based Generative Models (EDM). NeurIPS 2022. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364), code: [NVlabs/edm](https://github.com/NVlabs/edm)
- F. Koulischer et al. Dynamic Negative Guidance of Diffusion Models (DNG). ICLR 2025. [arXiv:2410.14398](https://arxiv.org/abs/2410.14398)
- T. Kynkäänniemi et al. Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models. NeurIPS 2024. [arXiv:2404.07724](https://arxiv.org/abs/2404.07724)
- T. Karras et al. Guiding a Diffusion Model with a Bad Version of Itself (Autoguidance). NeurIPS 2024. [arXiv:2406.02507](https://arxiv.org/abs/2406.02507)
- Y. Ban et al. Understanding the Impact of Negative Prompts: When and How Do They Take Effect? 2024. [arXiv:2406.02965](https://arxiv.org/abs/2406.02965)
- A. C. Li et al. Your Diffusion Model is Secretly a Zero-Shot Classifier. ICCV 2023. [arXiv:2303.16203](https://arxiv.org/abs/2303.16203) (the low-band reader is a variant of this idea)
- Classifiers: [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models). Metrics: [torch-fidelity](https://github.com/toshas/torch-fidelity).

## License

The code is under the MIT license (see `LICENSE`). The EDM checkpoints and CIFAR-10 are downloaded from their sources and have their own terms.
