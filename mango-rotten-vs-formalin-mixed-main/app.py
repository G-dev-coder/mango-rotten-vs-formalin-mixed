"""
app.py
------
Streamlit web application for the GET 324 mini-project:
"Rotten Mango vs Formalin-Treated Mango" image classifier.

Run locally:
    streamlit run app.py

Deploy for free:
    1. Push this repo to GitHub (include saved_models/*.keras via Git LFS
       or keep the model under 100MB so plain git works).
    2. Go to https://share.streamlit.io , sign in with GitHub, click
       "New app", select this repo/branch and set the main file to app.py.
    3. Streamlit Cloud installs requirements.txt automatically and gives
       you a public URL -- that URL is your Deliverable #3.
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

import config

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title=config.APP_TITLE, page_icon=config.APP_ICON,
                    layout="centered")


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(path=config.ACTIVE_MODEL_PATH):
    return tf.keras.models.load_model(path)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize + convert a PIL image into the (1, H, W, 3) float32 batch the
    model expects. NOTE: rescaling/preprocess_input is baked into the saved
    model itself (see model.py), so we only resize + add a batch dim here.
    """
    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
    arr = np.array(pil_image).astype("float32")
    return np.expand_dims(arr, axis=0)


def predict(model, pil_image: Image.Image):
    batch = preprocess_image(pil_image)
    prob_rotten = float(model.predict(batch, verbose=0)[0][0])
    # label index 0 = "formalin", label index 1 = "rotten"  (see config.CLASS_NAMES)
    prob_formalin = 1.0 - prob_rotten

    if prob_rotten >= 0.5:
        label, confidence = "rotten", prob_rotten
    else:
        label, confidence = "formalin", prob_formalin

    return label, confidence, prob_formalin, prob_rotten


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title(f"{config.APP_ICON} {config.APP_TITLE}")
st.write(
    "Upload a photo of a mango and this CNN model will predict whether it "
    "is a **naturally rotten mango** or a **mango suspected of formalin "
    "treatment**, based on visual skin/texture cues."
)

with st.sidebar:
    st.header("About")
    st.write(
        "GET 324 - Cloud Computing and AI Model Deployment for "
        "Engineering Applications\n\n"
        "**Mini-Project:** Binary image classification\n\n"
        "**Classes:**\n- 🥭 Formalin-treated mango\n- 🥭 Rotten mango"
    )
    st.divider()
    st.caption(
        "⚠️ Educational tool only. Do not use this app as the sole basis "
        "for real food-safety decisions -- confirm suspected formalin "
        "contamination with a proper laboratory test."
    )

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        "No trained model found yet at "
        f"`{config.ACTIVE_MODEL_PATH}`.\n\n"
        "Train one first with:\n\n"
        "```bash\npython train_model.py --arch transfer\n```\n\n"
        f"Details: {e}"
    )

uploaded_file = st.file_uploader(
    "Upload a mango image (JPG or PNG)", type=["jpg", "jpeg", "png"]
)

col1, col2 = st.columns(2)
use_camera = col1.toggle("Use camera instead", value=False)
camera_file = col2.camera_input("Take a photo") if use_camera else None

image_source = camera_file if camera_file is not None else uploaded_file

if image_source is not None and model_loaded:
    pil_image = Image.open(image_source)
    st.image(pil_image, caption="Input image", use_container_width=True)

    with st.spinner("Classifying..."):
        label, confidence, prob_formalin, prob_rotten = predict(model, pil_image)

    st.subheader("Result")
    if label == "formalin":
        st.warning(f"⚠️ **Predicted: Formalin-treated mango** "
                    f"({confidence * 100:.1f}% confidence)")
    else:
        st.success(f"🍂 **Predicted: Naturally rotten mango** "
                    f"({confidence * 100:.1f}% confidence)")

    st.write("**Class probabilities:**")
    st.progress(prob_formalin, text=f"Formalin-treated: {prob_formalin * 100:.1f}%")
    st.progress(prob_rotten, text=f"Rotten (natural): {prob_rotten * 100:.1f}%")

    st.caption(
        "Interpretation tip: formalin-treated fruit tends to look unnaturally "
        "firm, glossy, and free of mold/insect activity despite being "
        "over-ripe, while naturally rotten fruit shows soft, sunken, "
        "irregular dark patches and visible decay."
    )
elif image_source is None:
    st.info("Upload or capture an image above to get a prediction.")
