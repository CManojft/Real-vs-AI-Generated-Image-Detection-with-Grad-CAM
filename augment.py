"""
augment.py — shortcut-suppressing augmentation.

The audit found fake images are ~1.8x sharper than real ones (Laplacian variance).
A classifier can exploit that instead of learning generation artifacts, which
collapses the moment a user uploads a sharp real photo.

These transforms randomise the frequency content of EVERY image, regardless of
label, so sharpness stops being predictive while genuine artifacts survive.
Framework-agnostic: PIL in, PIL out. Wrap for tf.data or a torch Dataset.
"""
from __future__ import annotations
import io, random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

SIZE = 299


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _rescale(img: Image.Image, factor: float) -> Image.Image:
    """Downscale then upscale — destroys high-frequency detail like a real re-upload."""
    w, h = img.size
    small = img.resize((max(8, int(w * factor)), max(8, int(h * factor))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def train_augment(img: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Applied to training images only. Order mirrors a real upload pipeline."""
    r = rng or random
    img = img.convert("RGB")

    if r.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # --- frequency randomisation: the actual shortcut fix ---
    if r.random() < 0.55:
        img = _rescale(img, r.uniform(0.35, 0.85))
    if r.random() < 0.45:
        img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.4, 1.6)))
    if r.random() < 0.65:
        img = _jpeg(img, r.randint(35, 92))

    # --- mild photometric jitter: stops colour/exposure becoming a proxy ---
    if r.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(r.uniform(0.80, 1.20))
    if r.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(r.uniform(0.80, 1.20))
    if r.random() < 0.3:
        img = ImageEnhance.Color(img).enhance(r.uniform(0.7, 1.3))

    if r.random() < 0.3:
        a = np.asarray(img, dtype=np.float32)
        a += np.random.normal(0, r.uniform(1.5, 6.0), a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE), Image.BICUBIC)
    return img


def eval_transform(img: Image.Image) -> Image.Image:
    """No randomness at eval time — metrics must be reproducible."""
    img = img.convert("RGB")
    return img if img.size == (SIZE, SIZE) else img.resize((SIZE, SIZE), Image.BICUBIC)


def to_xception_input(img: Image.Image) -> np.ndarray:
    """Xception expects float32 scaled to [-1, 1]."""
    return np.asarray(img, dtype=np.float32) / 127.5 - 1.0