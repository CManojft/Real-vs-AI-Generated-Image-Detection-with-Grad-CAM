# Real vs AI-Generated Image Detection with Grad-CAM

Upload an image, get a Real/Fake verdict with a confidence score, and a Grad-CAM
heatmap showing which regions drove the decision. No prediction is ever displayed
without its heatmap.

Xception (ImageNet) + transfer learning, trained on a de-duplicated
multi-generator face dataset.

## Results

| | test | unseen-generator holdout |
|---|---|---|
| Accuracy | 89.50% | 92.00% |
| Precision | 0.8706 | 0.9038 |
| Recall | 0.9058 | 0.9400 |
| F1 | 0.8878 | 0.9216 |
| AUC | 0.9596 | 0.9735 |

Read as a detector: **543/613 fakes caught (88.6%)**, 70 missed, 49 genuine photos
wrongly flagged (9.4% of reals).

The generalisation gap is **negative** (-0.0139) - the model scores higher on a
generator family it has never seen, which is evidence it learned generic artifacts
rather than one generator's fingerprint. Caveat: the held-out family is visually
homogeneous and therefore an easy unseen generator; this is evidence of no
fingerprint overfitting, not proof of universal generalisation.

Per-source accuracy on test spans a wide range - face-swap deepfakes on
low-quality video frames are hardest, fully-synthetic stock portraits easiest:

```
swap_video_frames    0.7967
gan_stylegan_web     0.8427
real_pool            0.8825
real_celeba          0.9424
gen_stock_portrait   0.9907
real_lfw             1.0000
```

## Quick start

```bash
pip install -r requirements.txt
python verify_model.py        # confirms the model reproduces its reported metrics
streamlit run app.py          # demo at localhost:8501
```

`app.py` needs `xception_final.keras` and `metrics.json` in this folder. The
threshold lives in `metrics.json`; without it the confidence display is calibrated
to the wrong decision point.

## Files

| file | purpose |
|---|---|
| `prepare_dataset.py` | rebuilds the raw data into a leak-free, stratified split |
| `verify_dataset.py` | independent audit of that output; non-zero exit on failure |
| `augment.py` | shortcut-suppressing augmentation (framework-agnostic, PIL) |
| `app.py` | Streamlit demo: upload, classify, Grad-CAM |
| `verify_model.py` | re-scores the local test set, compares to `metrics.json` |
| `colab/deepfake_detector.ipynb` | training notebook with outputs |
| `DATASET.md` | the four dataset defects found and how each was fixed |
| `PROCESS.md` | build process, phase by phase, with the traps |
| `reports/manifest.csv` | every image mapped to its origin, cluster, and source group |

## Reproducing

```bash
python prepare_dataset.py     # deterministic, SEED=1337
python verify_dataset.py      # must print ALL CHECKS PASSED
```

Then run `colab/deepfake_detector.ipynb` on a T4 GPU. Training is CPU-only on
native Windows - TensorFlow >= 2.11 cannot use a GPU there regardless of CUDA.

## Things a reader should know

**Class order.** Keras sorts folders alphabetically, so `fake = 0` and `real = 1`.
The model outputs **P(real)**. Invert it for a fake score.

**Threshold.** 0.5612, chosen on the validation split via Youden's J, then held
fixed for test and holdout. Choosing it on test would leak and inflate F1.

**The model is under-trained.** Both training stages hit their epoch caps while
`val_auc` was still improving; early stopping never fired. Roughly 15 more
fine-tuning epochs would likely raise these numbers.

**Grad-CAM is diagnostic, not decorative.** The two error types have opposite
signatures: missed fakes show heat on hair and background (the model never
inspected the face), while false alarms show heat locked tightly on facial
features (it inspected the face and mis-read genuine skin texture). Confidence is
also miscalibrated in a specific way - false alarms arrive at 90%+ confidence
while correct fake detections sit near 55%.