# Dataset preparation

`dataset_xception/` as delivered had four defects that would have made the reported
metrics unreliable. This document records what they were, what was done, and how to
re-verify. The original directory is **never modified** — everything is rebuilt into
`dataset_clean/`.

## Pipeline

```bash
python prepare_dataset.py   # builds dataset_clean/ + reports/
python verify_dataset.py    # independent audit; exits non-zero on any failure
```

`prepare_dataset.py` is deterministic (`SEED = 1337`) — rerunning reproduces the split
exactly. Output files are hardlinked, so `dataset_clean/` costs almost no extra disk.

## The four defects and their fixes

### 1. Train/eval contamination — 198 images (7.7% of eval)

Images present in both train and the evaluation splits, including byte-identical pairs:
`train/real/tran_real (3179).jpg` was the same file as
`test/real/lfw_Pauline_Landers_0001_00568.jpg`. `test/real` was 10.4% contaminated.

**Fix.** Every image is fingerprinted (MD5 + 64-bit dHash) and near-duplicates are
merged into clusters — a dHash candidate pass at Hamming ≤ 10, each pair then confirmed
by pixel correlation > 0.95 so coincidental hash collisions are not merged. Clusters are
**atomic**: all members land in one split, so a duplicate cannot straddle a boundary.
9,705 images collapsed into 8,505 independent clusters.

### 2. valid and test drew from different generators

`valid/fake` was mostly face-swap video frames; `test/fake` was mostly GAN portraits.
Tuning against validation would have optimised for the wrong distribution.

**Fix.** Splits are stratified per `(class, source_group)`, so each split gets the same
generator mix. Residual valid↔test gap is now **≤ 0.2 pp** on every group.

### 3. Fakes were ~1.8× sharper than reals

Laplacian variance: real ≈ 280, fake ≈ 510. The model could learn *"sharp = fake"* — a
rule that scores well here and fails on any real high-resolution upload.

**Fix.** `augment.py` randomises frequency content of every training image regardless of
label: random downscale-upscale, Gaussian blur, and JPEG re-compression at quality
35–92. This drops the train-time gap from **1.76× to 1.11×**. Use `train_augment()` for
training and `eval_transform()` for evaluation.

### 4. No way to test generalisation to an unseen generator

Every generator family appeared in all three splits, so "does not overfit to one
generator" — the stated goal of the multi-source requirement — could not be measured.

**Fix.** The `gen_studio_batch` family (500 images from a single generator) is excluded
from train/valid/test entirely and paired with 500 held-out real images in
`holdout_unseen_gen/`. Verified absent from training data.

This holdout doubles as a cheat detector: its fakes are *less* sharp than its reals
(0.90×, inverted from the training data), so a model relying on the sharpness shortcut
scores near or below chance here while looking excellent on `test/`.

## Result

| split | real | fake | fake share |
|---|---|---|---|
| train | 2,715 | 3,347 | 55.2% |
| valid | 522 | 612 | 54.0% |
| test | 520 | 613 | 54.1% |
| holdout_unseen_gen | 500 | 500 | 50.0% |

9,329 images. `valid` and `test` keep one image per cluster, so repeated video frames of
one face cannot dominate the metrics; `train` keeps all members as natural augmentation.

Train is mildly fake-heavy. Do not discard data to balance it — pass class weights.

**Watch the class index.** Keras `image_dataset_from_directory` sorts class folders
alphabetically, so `fake` becomes **0** and `real` becomes **1** — the opposite of the
order they are written in throughout this document. The weights must follow Keras' order:

```python
# fake = 0, real = 1   (alphabetical, as Keras assigns them)
class_weight = {0: 0.9056, 1: 1.1164}

# assert it rather than trusting it:
ds = tf.keras.utils.image_dataset_from_directory("dataset_clean/train", ...)
assert ds.class_names == ["fake", "real"], ds.class_names
```

A model's output probability is therefore **P(real)**, not P(fake). Invert it before
displaying a "fake" confidence score, and point Grad-CAM at the correct logit.

Filenames are prefixed with provenance (`gan_stylegan_web__1E0CZZ96D7.jpg`) so any
metric can be broken down per generator. `reports/manifest.csv` maps every output file
to its original path, cluster id, and source group.

### Source groups

| group | class | what it is |
|---|---|---|
| `real_pool`, `real_celeba`, `real_lfw` | real | web photos, CelebA crops, LFW |
| `gan_stylegan_web`, `gan_stylegan_grid` | fake | StyleGAN-family portraits |
| `gen_stock_portrait` | fake | AI stock-style portraits (diffusion) |
| `swap_video_frames` | fake | face-swap deepfake video frames |
| `gan_manipulated` | fake | heavily artifacted / manipulated GAN |
| `gen_studio_batch` | fake | single generator batch — **holdout only** |

## Reporting caveat

Report `test/` **and** `holdout_unseen_gen/` metrics side by side. `test/` measures
accuracy on generators the model trained on; the holdout measures whether it generalises
to one it has never seen. A large gap between them is the honest finding, not a failure.