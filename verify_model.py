"""
verify_model.py — confirm the downloaded model reproduces its reported metrics.

Run after downloading xception_final.keras + metrics.json from Drive:
    python verify_model.py

Re-scores the LOCAL dataset_clean/test set on CPU and compares against
metrics.json. Catches a corrupted download, a stale metrics.json, or a
threshold that no longer matches the saved weights.
"""
import os, sys, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

MODEL_PATH, META_PATH, ROOT, SIZE = "xception_final.keras", "metrics.json", "dataset_clean", 299

for f in (MODEL_PATH, META_PATH):
    if not os.path.exists(f):
        sys.exit(f"missing {f} - download it from Google Drive into this folder")

meta = json.load(open(META_PATH))
TAU = float(meta["threshold"])
print(f"model      {os.path.getsize(MODEL_PATH)/1e6:.1f} MB")
print(f"threshold  {TAU:.6f}")
print(f"reported   test AUC {meta.get('test', {}).get('auc', float('nan')):.4f}\n")

model = keras.models.load_model(MODEL_PATH, compile=False)
assert model.output_shape[-1] == 1, model.output_shape
try:
    model.get_layer("block14_sepconv2_act")
except ValueError:
    sys.exit("block14_sepconv2_act missing - Grad-CAM will not work with this file")
print("loaded; Grad-CAM target layer present")

fake_dir, real_dir = f"{ROOT}/test/fake", f"{ROOT}/test/real"
fake = [f"{fake_dir}/{f}" for f in sorted(os.listdir(fake_dir))]
real = [f"{real_dir}/{f}" for f in sorted(os.listdir(real_dir))]
paths = fake + real
y = np.array([0] * len(fake) + [1] * len(real))

def read(p):
    img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3)
    return tf.cast(tf.image.resize(img, (SIZE, SIZE)), tf.float32) / 127.5 - 1.0

ds = tf.data.Dataset.from_tensor_slices(paths).map(read, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(16).prefetch(tf.data.AUTOTUNE)
print(f"scoring {len(paths)} local test images on CPU (a few minutes)...")
t = time.time()
prob = tf.sigmoid(model.predict(ds, verbose=1)).numpy().ravel()
print(f"done in {time.time()-t:.0f}s\n")

pred = (prob >= TAU).astype(int)
p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
acc, auc = accuracy_score(y, pred), roc_auc_score(y, prob)
cm = confusion_matrix(y, pred)

print(f"local  accuracy {acc:.4f} | precision {p:.4f} | recall {r:.4f} | F1 {f1:.4f} | AUC {auc:.4f}")
print(f"fakes caught   {cm[0,0]}/{len(fake)} = {cm[0,0]/len(fake):.1%}")
print(f"fakes missed   {cm[0,1]}")
print(f"reals flagged  {cm[1,0]}/{len(real)} = {cm[1,0]/len(real):.1%}")

exp = meta.get("test", {}).get("auc")
if exp is not None:
    d = abs(auc - exp)
    print(f"\nreported AUC {exp:.4f} vs local {auc:.4f}  (diff {d:.4f})")
    if d < 0.01:
        print("MATCH - model and metrics.json agree, app is safe to run")
    else:
        print("MISMATCH - metrics.json is stale, or the download is corrupt.")
        print("Re-run the evaluation cell in Colab, re-save, and download again.")