import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io

# -----------------------------
# CONFIG
# -----------------------------
# If the model was trained on a specific input size, set it here.
# Common sizes: (150, 150), (180, 180), (224, 224)
IMG_SIZE = (150, 150)  # change if your model used a different size

CLASS_NAMES = ["NO CORROSION", "CORROSION"]  # 0 -> no, 1 -> yes


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("saved_model.h5")
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a model-ready tensor:
    - RGB
    - resized
    - normalized to [0, 1]
    - add batch dimension
    """
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(model, image: Image.Image):
    x = preprocess_image(image)
    preds = model.predict(x)
    # For a binary sigmoid model, preds is shape (1, 1)
    prob_corrosion = float(preds[0][0])
    return prob_corrosion


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
st.success("Model loaded successfully from `saved_model.h5` ✅")

uploaded_files = st.file_uploader(
    "Upload one or more images (JPG/PNG)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

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
            prob_corrosion = predict_image(model, image)

        label_idx = 1 if prob_corrosion >= threshold else 0
        label = CLASS_NAMES[label_idx]

        st.markdown(
            f"**Prediction:** `{label}`  \n"
            f"**Corrosion probability:** `{prob_corrosion:.4f}`"
        )

        st.progress(prob_corrosion)

        if label_idx == 1:
            st.warning("⚠️ Corrosion likely present. Consider further inspection.")
        else:
            st.info("✅ No significant corrosion detected (according to the model).")
else:
    st.info("👆 Upload an image to start corrosion detection.")
