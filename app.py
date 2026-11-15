"""
Streamlit demo — real vs AI-generated face detector with Grad-CAM.

Run:  streamlit run app.py

Requires xception_final.keras and metrics.json in this folder (download from Drive).
Every prediction is rendered together with its Grad-CAM heatmap; the app never
shows a label without one.
"""
import io, os, json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.cm as cm

MODEL_PATH = "xception_final.keras"
META_PATH = "metrics.json"
SIZE = 299
LAST_CONV = "block14_sepconv2_act"

st.set_page_config(page_title="Real vs AI-Generated Detector", layout="wide")


@st.cache_resource
def load_everything():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model = keras.models.load_model(MODEL_PATH, compile=False)
    grad_model = keras.Model(model.input,
                             [model.get_layer(LAST_CONV).output, model.output])
    tau = 0.5
    if os.path.exists(META_PATH):
        tau = float(json.load(open(META_PATH))["threshold"])
    return model, grad_model, tau


def preprocess(pil_img):
    """PNG alpha and grayscale both collapse to 3-channel RGB here."""
    img = pil_img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)
    return img, (arr / 127.5 - 1.0)[None, ...]


def grad_cam(grad_model, arr, tau):
    """Explains the PREDICTED class: evidence for 'real' if real, else for 'fake'."""
    with tf.GradientTape() as tape:
        conv, logit = grad_model(arr)
        prob = float(tf.sigmoid(logit)[0, 0])
        target = logit[0, 0] if prob >= tau else -logit[0, 0]
    grads = tape.gradient(target, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm = tf.reduce_sum(conv[0] * weights, axis=-1)
    hm = tf.maximum(hm, 0) / (tf.reduce_max(hm) + 1e-8)
    return hm.numpy(), prob


def overlay(base_img, heatmap, alpha=0.4):
    big = np.array(tf.image.resize(heatmap[..., None], (SIZE, SIZE)))[..., 0]
    colour = cm.jet(big)[..., :3]
    base = np.asarray(base_img, dtype=np.float32) / 255.0
    blend = np.clip(base * (1 - alpha) + colour * alpha, 0, 1)
    return (blend * 255).astype(np.uint8)


def threshold_calibrated_confidence(prob, tau):
    """Maps the decision threshold to 50% so confidence never contradicts the label."""
    span = (1 - tau) if prob >= tau else tau
    return 0.5 + 0.5 * abs(prob - tau) / span


# ----------------------------------------------------------------- UI
def main():
    st.title("Real vs AI-Generated Image Detector")
    st.caption("Xception + transfer learning. Every prediction ships with a Grad-CAM "
               "heatmap showing which regions drove the decision.")

    model, grad_model, TAU = load_everything()
    if model is None:
        st.error(f"`{MODEL_PATH}` not found. Download it from Google Drive into this folder.")
        st.stop()

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a JPG, JPEG or PNG to begin.")
        st.stop()

    pil = Image.open(io.BytesIO(uploaded.read()))
    disp_img, arr = preprocess(pil)

    with st.spinner("Analysing..."):
        heatmap, prob = grad_cam(grad_model, arr, TAU)

    label = "REAL" if prob >= TAU else "AI-GENERATED (FAKE)"
    conf = threshold_calibrated_confidence(prob, TAU)
    p_fake = 1.0 - prob

    # verdict
    if prob >= TAU:
        st.success(f"### {label} — {conf * 100:.1f}% confidence")
    else:
        st.error(f"### {label} — {conf * 100:.1f}% confidence")

    # image + heatmap, always together
    c1, c2 = st.columns(2)
    c1.image(disp_img, caption="Original", use_container_width=True)
    c2.image(overlay(disp_img, heatmap), caption="Grad-CAM overlay — red = most influential",
             use_container_width=True)

    with st.expander("Details"):
        st.write(f"**P(real)** = {prob:.4f}  |  **P(fake)** = {p_fake:.4f}")
        st.write(f"**Decision threshold** = {TAU:.4f} (chosen on the validation split)")
        st.progress(float(np.clip(p_fake, 0, 1)), text=f"P(fake) = {p_fake:.1%}")
        if os.path.exists(META_PATH):
            m = json.load(open(META_PATH))
            st.write("**Model performance**")
            st.json({k: m[k] for k in ("test", "holdout_unseen_gen") if k in m})

    st.caption("Heat on facial features (eyes, mouth, skin texture) indicates the model "
               "examined the face. Heat on hair or background suggests it did not — treat "
               "those predictions with less confidence.")


if __name__ == "__main__":
    main()