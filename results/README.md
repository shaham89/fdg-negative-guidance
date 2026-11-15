# Results

These are the saved outputs of every run that the main README talks about. The image sample files (`samples_*.npz`, about
14 MB for each 5,000-image arm) are not included. Everything here is small: result rows, paired diagnostics, logs, a few
charts and some 8x8 image grids.

Two GPUs were used. The first runs (folders `two_dials` and `neg`) ran on an NVIDIA A10 (an Azure VM). Later runs
(`neg_sweep10` to `neg_sweep12c`) ran on an A100 in Google Colab. The same arm with the same noise gives an FID about 0.1
higher on the A100 (no push for "not a cat": 7.61 on the A10, 7.72 on the A100), so compare rows from the same machine.

## Reading a row

`gen.py` writes one JSON line per arm to `results.jsonl`: `name`, `n` (number of images), `fid` and `is` (torch-fidelity
against CIFAR-10 train), `class_acc` (a pretrained ResNet-56 agrees with the requested label) and `clip_frac`. If a name
appears twice, the later row wins (`plot_pareto.py` does the same). Rows whose name ends in `smoke` are quick checks.

`neg.py` rows ("not class k") have `k`, the push strengths `a` (low band) and `b` (high band), the window `lo`, `hi` in
sigma, `hist_resnet` and `hist_vgg` (share of images each classifier puts in each class), `hist_low` (the same for the
low-band reader, on the first `n_read` images), and `fid`, `is` against real CIFAR-10 images that are not class `k`.
`n_blank` counts nearly constant images (newer rows only). For "c, not k" rows (`pn{c}not{k}_...`) the FID is against
real images of class `c`.

`diag_k{k}.json` (from `neg_diag.py`) compares each arm with the no-push images made from the same noise:
`kept` is the share of no-push class-k images that are still class k, `changed` the share of other images whose class
changed, `near_kept` the same as `kept` for the nearest class (dog for cat, truck for car), `colour` the shift of the
mean colour along the class direction, and `colour_mix` the part of that shift explained by the new class mix alone.

## Folders

| Folder | What | Machine | Made by |
|---|---|---|---|
| `two_dials/` | CFG, FDG and all "two dials" arms, 10,000 images each | A10 | `scripts/sweep1.sh`, `sweep2.sh`, `sweep5.sh`, `sweep6.sh`, `sweep7.sh` |
| `neg/` | "Not a cat" (k=3) and "not a car" (k=1), 5,000 images each, seed 0 | A10 | `sweep9.sh`, `sweep9b.sh`, `sweep9c.sh` |
| `neg_sweep10/` | tricks, "a dog, not a cat", exact rule at 2,000 images, timing, seeds 1-2, all ten classes | A100 | `sweep10.sh` |
| `neg_sweep11/` | exact rule at 5,000 images, more "c, not k" pairs | A100 | `sweep11.sh`, then `sweep11b.sh` |
| `neg_sweep12/` | posterior per band, push map at strength 2, p fitted per band, seeds 1-2 of the exact rule | A100 | `sweep12.sh` |
| `neg_sweep12c/` | exact rule with calibrated p, push map at strength 1, push size per step | A100 | `sweep12c.sh` |
| `gaussian_model/` | the same questions on a 10-class Gaussian model of CIFAR-10 with exact denoisers | CPU | `analysis/gmm_toy.py`, `analysis/gmm_pmap.py` |

### two_dials

- `results.jsonl`: FID, IS and accuracy of every arm. `extra.jsonl`: colour shift, sharpness, power per band and
  precision/recall (`evalx.py`); some names have more than one row (for example all 10k images and the first 4k: check `n`).
- `hybrid.jsonl`: class read on hybrids that take the low band from one arm and the high band from another
  (`hybrid_diag.py`), and the rms change of each band against w = 1.
- `diag_w1.npz`, `diag_w3.npz`: guidance energy per band and noise level along the w = 1 and w = 3 paths (`diag.py`).
- `replicate_wh3.npz`, `replicate_wrong_wh1.npz`: agreement of the two independent EDM network pairs on the guidance
  direction (`replicate_diag.py`), true class on the (1,3) path and wrong class on the w = 1 path.
  `source_diag_wh3.npz`: class part and gap part of the guidance (`source_diag.py`).
- `wiener_table.json`, `time_from_*.json`, `cifar_spectrum.npz`: small tables that the code reads (`scripts/setup.sh`
  copies them into `src/out/`). `noise.json`: FID and IS spread from half splits (`noise.py`).
- `cpu_checks_2026-11-01.txt`: output of the three scripts in `analysis/` that read these files.
- `logs/`: the console logs of the sweeps. `figures/`: charts (`pareto.png`, `twins.png`, `schedules.png`,
  `spectrum.png`, `diag_w*.png`), hybrid image sheets and 8x8 grids of the first 64 images of some arms.

Missing: `sharpen_table.json` (the lambda(sigma) table that the shaped sharpening control used) stayed on the VM.
`calib_sharpen.py` makes a new one, and the dose 0.182 may then need to be tuned again with `tune_dose.py`.
The true-class agreement curve on the w = 1 path (n = 1024) and the "decided at sigma*" values per frequency ring,
which the slides show, also have no saved file.

### neg

`results.jsonl`, `diag_k3.json`, `diag_k1.json`, `reader.json` (the noise level chosen for the low-band reader) and
the 8x8 grids of all arms. `logs/reader_check.log` is the check of the low-band reader on real test images
(`neg.py reader`). The strength-4 arms (`a4`, `b4`) contain many black images, which VGG-16 reads as cats, so their
FIDs are not valid.

### neg_sweep10

`results.jsonl` (120 rows) and `sweep10.log`, which also holds the paired diagnostics of all ten classes. In that log,
the paired numbers for seeds 1-2 and for the `ft` runs are wrong: a bug in `neg_diag.py` (fixed since) paired them with
the seed-0 images. FID, IS and class rates of all rows are fine. The run `exact --a 1 --b 2` crashed there (also fixed).

### neg_sweep11

`sweep11.sh` produced three rows before the Colab runtime stopped; `sweep11b.sh` restarted it and skipped done rows.
Two rows (`pn9not1_cfg_w2`, `pn9not1_posneg_wl1_wh2`) reached only `sweep11b.log`, not `results.jsonl`.
No grids or paired diagnostics were saved for this batch.

### neg_sweep12

`results.jsonl`, `diag_k3.json`, `pmap_s2.jsonl` (push map, one line per arm), `post_diag.json` and `post_diag.npz`
(the fitted p of every image at every step, and its final class), `post_cal.json` (one temperature per noise level,
remade exactly by `src/fit_cal.py` from `post_diag.npz`), `pmap_flip.png` (`src/plot_pmap.py`) and a few grids.

In `pmap_s2.jsonl`, `flip` means two different things: for `away` it is the share of images that end in any class other
than their no-push class, for `toward` it is the share that end in the chosen target class. `changed` (any class
change) is the same measure for both directions.

### neg_sweep12c

`results.jsonl` (the two calibrated-p arms), `diag_k3.json`, `pmap_s1.jsonl`, `pmap_size.json` (rms of the push
Band(D_k - D_u) per step, for away and toward), `pmap_strength.png` (`src/plot_pmap_s.py`) and the grids of the
two arms. The temperatures were fitted on the same noise (seed 0) that these arms use.

### gaussian_model

Console output of `gmm_toy.py` (n = 1000: fixed pushes in `gmm_toy_n1000.txt`; both bands, time windows and the exact rule
in `gmm_toy_n1000_b.txt`) and of `gmm_pmap.py` (the push map, n = 1000; the `.json` file has the rows that `plot_pmap.py` draws).
