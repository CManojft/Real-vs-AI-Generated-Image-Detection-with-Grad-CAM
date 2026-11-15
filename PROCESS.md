# Build process

Ordered so that anything that can go wrong goes wrong **cheaply and early**.
Dataset prep is already done — see `DATASET.md`. This covers everything after it.

---

## Phase 0 — Decide where you train  (15 min)

No NVIDIA GPU on this machine, and TensorFlow >= 2.11 cannot use a GPU on native
Windows at all (it says so itself on import). Training is CPU-only here.

**Recommended: train on Colab, run the demo locally.**
Zip `dataset_clean/` (~292 MB), upload to Drive, train on a free T4, download the
`.keras` file. Inference for one image is fast on CPU, so the app stays local.

Keep the local TensorFlow install — you need it for the app and for Phase 1.

---

## Phase 1 — Prove the pipeline before training anything  (30 min)

**Do not skip this.** It is the difference between finding a bug in 20 minutes and
finding it after an overnight run.

1. **Assert the class order.** Keras sorts folders alphabetically → `fake=0, real=1`.
   ```python
   train_ds = tf.keras.utils.image_dataset_from_directory("dataset_clean/train", ...)
   assert train_ds.class_names == ["fake", "real"], train_ds.class_names
   ```
   Your sigmoid output is therefore **P(real)**. Invert it for a "fake" score.

2. **Look at one augmented batch.** Plot 16 images after `train_augment()`. If they
   are unrecognisable, augmentation is too strong; if they look untouched, it is not
   wired in.

3. **Overfit 100 images on purpose.** Take 50 real + 50 fake, no augmentation, and
   train until training accuracy hits ~100%. If it cannot memorise 100 images, the
   bug is in your code, not your hyperparameters. Then throw this model away.

---

## Phase 2 — Grad-CAM now, not last  (45 min)

Build it against the throwaway model from Phase 1, before real training starts.

Reasons: it is a hard requirement, so it must not be rushed at the end; and it is
the best debugging tool you have — a model exploiting the sharpness shortcut produces
heatmaps that wander off the face, which you can *see*.

- Target layer for Xception: **`block14_sepconv2_act`** (10x10x2048 at 299 input)
- Gradient of the output logit w.r.t. that layer -> channel weights -> weighted sum
  -> ReLU -> resize to 299x299 -> colormap -> blend at ~0.4 alpha
- Take gradients of the **pre-sigmoid logit**, not the sigmoid output — a saturated
  sigmoid gives near-zero gradients and a blank heatmap. Common failure.

Sanity check: a blank or uniform heatmap means something is wrong, always.

---

## Phase 3 — Stage 1: frozen base  (~10 epochs)

```
base = Xception(weights="imagenet", include_top=False, input_shape=(299,299,3))
base.trainable = False
head = GlobalAveragePooling2D -> Dropout(0.3) -> Dense(1)      # logits, no activation
```

- Optimiser Adam, **LR 1e-3**
- `BinaryCrossentropy(from_logits=True)`
- Batch 32 (16 if RAM complains)
- `class_weight = {0: 0.9056, 1: 1.1164}`   # fake=0, real=1
- Metrics: AUC, Precision, Recall — **monitor `val_auc`, not `val_accuracy`**
  (accuracy is misleading at 54/46 class balance)
- Callbacks: `EarlyStopping(monitor="val_auc", mode="max", patience=3,
  restore_best_weights=True)`, `ModelCheckpoint(save_best_only=True)`

Expect validation AUC to plateau high. Do not celebrate yet — Phase 5 is the real test.

---

## Phase 4 — Stage 2: fine-tune  (~10 epochs)

```
base.trainable = True
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False          # <-- critical
```

- Unfreeze roughly the last 30–40 layers (from `block12_` onward); leave early
  generic edge/texture filters frozen
- **LR 1e-5** — 100x lower than Stage 1. A higher LR here destroys the pretrained
  weights in one epoch and the run is unrecoverable
- **Keep BatchNorm layers frozen.** If they update on small batches they overwrite
  the ImageNet statistics and validation collapses while training looks fine. This is
  the single most common fine-tuning bug
- Recompile after changing `trainable` — otherwise the change is ignored

---

## Phase 5 — Evaluate honestly  (30 min)

Report **two** numbers side by side, never just the first:

| set | what it measures |
|---|---|
| `test/` | generators the model trained on |
| `holdout_unseen_gen/` | a generator it has **never** seen |

1. **Pick the threshold on `valid/`, not `test/`.** Maximise F1 or use Youden's J.
   Then apply that fixed threshold to test and holdout. Choosing it on test is
   leakage and inflates your F1.
2. Report accuracy, precision, recall, F1, AUC for both sets.
3. **Break the score down per generator** using the filename prefixes
   (`gan_stylegan_web__`, `swap_video_frames__`, ...). Discovering the model is 99%
   on StyleGAN and 70% on face-swaps is a finding worth writing up.
4. Plot the confusion matrix and ROC curve.

A gap between test and holdout is expected and is the honest result. A *large* gap
means it learned generator fingerprints rather than generic artifacts — report it
rather than hiding it.

**Cross-check with Grad-CAM.** If holdout accuracy is poor, look at those heatmaps.
Attention on backgrounds or image borders = the sharpness shortcut. Attention on
eyes, teeth, hairline, ear boundaries = genuine artifact detection.

---

## Phase 6 — The app  (2–3 hrs)

Streamlit is the fastest route to something demonstrable:

```
upload (jpg/jpeg/png)
  -> eval_transform + preprocess_input
  -> predict -> logit -> sigmoid -> P(real); fake_score = 1 - P(real)
  -> Grad-CAM on the same forward pass
  -> render original | heatmap overlay, side by side, with the confidence score
```

Requirements to honour:
- Accept **PNG too**, not just JPG — convert to RGB (PNGs carry an alpha channel
  that will otherwise give you a 4-channel tensor and a shape error)
- **Never render a label without its heatmap.** Generate both or show neither
- Show the confidence as a number, e.g. "Fake — 87.3% confidence"

---

## Phase 7 — Write-up

Cover, in this order: dataset provenance and the cleanup (`DATASET.md` is your
source), two-stage training curves, the metrics table for test **and** holdout,
per-generator breakdown, and Grad-CAM examples — include at least one failure case.
A report showing where the model breaks is stronger than one claiming it never does.

---

## Traps specific to this dataset

| trap | consequence |
|---|---|
| `fake=0, real=1`, not the reverse | class weights applied backwards, silently |
| Sigmoid output is P(real) | confidence displayed inverted |
| Gradients on sigmoid, not logit | blank Grad-CAM heatmaps |
| BatchNorm left trainable in Stage 2 | validation collapses, training looks fine |
| Threshold tuned on test | inflated F1 |
| Reporting `test/` alone | hides whether it generalises at all |
| Dropping `augment.py` to "train faster" | model learns "sharp = fake" and fails on real uploads |