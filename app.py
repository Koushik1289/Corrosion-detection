import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("saved_model.h5")
    return model


def get_model_img_size(model):
    """
    Read input spatial size (H, W) from model.input_shape.
    Works for Sequential and Functional models.
    """
    input_shape = model.input_shape

    # Some models have a list of input shapes
    if isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)):
        input_shape = input_shape[0]

    # Expected something like (None, H, W, C)
    if len(input_shape) >= 4:
        return (int(input_shape[1]), int(input_shape[2]))
    else:
        # Fallback if something weird happens
        return (128, 128)


def preprocess_image(image: Image.Image, img_size) -> np.ndarray:
    """
    Convert a PIL image to a model-ready tensor:
    - RGB
    - resized to img_size
    - normalized to [0, 1]
    - add batch dimension
    """
    img = image.convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(model, image: Image.Image):
    img_size = get_model_img_size(model)
    x = preprocess_image(image, img_size)
    preds = model.predict(x)
    # Assuming binary sigmoid output: shape (1, 1)
    prob_corrosion = float(preds[0][0])
    return prob_corrosion, img_size


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="Corrosion Detection from Images",
    page_icon="🛠️",
    layout="centered",
)

st.title("🛠️ Automated Corrosion Detection")
st.markdown(
    """
Upload infrastructure or drone images and let the CNN model classify whether **corrosion** is present.
"""
)

model = load_model()
model_img_size = get_model_img_size(model)

st.success(
    f"Model loaded successfully from `saved_model.h5` ✅  \n"
    f"Expected input size: **{model_img_size[0]}×{model_img_size[1]}**"
)

uploaded_files = st.file_uploader(
    "Upload one or more images (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

CLASS_NAMES = ["NO CORROSION", "CORROSION"]

threshold = st.slider(
    "Decision threshold for corrosion",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="Predictions above this probability will be labeled as CORROSION."
)

if uploaded_files:
    for file in uploaded_files:
        st.divider()
        st.subheader(f"File: {file.name}")

        # Read image
        bytes_data = file.read()
        image = Image.open(io.BytesIO(bytes_data))

        # Show image
        st.image(image, caption="Uploaded image", use_container_width=True)

        # Run prediction
        with st.spinner("Running corrosion detection..."):
            prob_corrosion, img_size = predict_image(model, image)

        label_idx = 1 if prob_corrosion >= threshold else 0
        label = CLASS_NAMES[label_idx]

        st.markdown(
            f"**Prediction:** `{label}`  \n"
            f"**Corrosion probability:** `{prob_corrosion:.4f}`  \n"
            f"(Image was resized internally to **{img_size[0]}×{img_size[1]}**)"
        )

        st.progress(prob_corrosion)

        if label_idx == 1:
            st.warning("⚠️ Corrosion likely present. Consider further inspection.")
        else:
            st.info("✅ No significant corrosion detected (according to the model).")
else:
    st.info("👆 Upload an image to start corrosion detection.")
