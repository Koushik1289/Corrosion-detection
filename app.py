import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Imports for Gemini API
from google import genai
from google.genai import types
import os

# =========================
# MODEL LOADING (Keras/TensorFlow)
# =========================
@st.cache_resource
def load_model():
    # Ensure this file exists in your running directory
    try:
        model = tf.keras.models.load_model("saved_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading Keras model 'saved_model.h5'. Please ensure the file exists. Error: {e}")
        return None

def get_model_img_size(model):
    """
    Read input spatial size (H, W) from model.input_shape.
    Works for most Sequential / Functional models with image input.
    """
    if model is None:
        return (128, 128) # Default fallback

    input_shape = model.input_shape  # e.g. (None, H, W, C) or [(None, H, W, C)]

    # Some models have a list of input shapes
    if isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)):
        input_shape = input_shape[0]

    # Expecting something like (None, H, W, C)
    if len(input_shape) >= 4:
        h, w = int(input_shape[1]), int(input_shape[2])
        return (h, w)
    else:
        # Fallback
        return (128, 128)


# =========================
# PREPROCESSING & PREDICTION (Keras)
# =========================
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
    return preds, img_size


def interpret_predictions(raw_preds: np.ndarray, threshold: float = 0.5):
    """
    Interpret model predictions as corrosion / no corrosion.
    """
    CLASS_NAMES = ["NO CORROSION", "CORROSION"]

    preds = np.array(raw_preds)

    # Defaults
    prob_corrosion = 0.0
    class_idx = 0
    class_probs = [1.0, 0.0]

    # Case 1: binary sigmoid -> shape (1,1)
    if preds.ndim == 2 and preds.shape[1] == 1:
        prob_corrosion = float(preds[0, 0])
        class_probs = [1.0 - prob_corrosion, prob_corrosion]
        class_idx = 1 if prob_corrosion >= threshold else 0

    # Case 2: 2-class softmax -> shape (1,2)
    elif preds.ndim == 2 and preds.shape[1] == 2:
        class_probs = preds[0].tolist()  # [P(no), P(yes)]
        class_idx = int(np.argmax(class_probs))
        prob_corrosion = float(class_probs[1])  # treat index 1 as CORROSION

    else:
        # Unexpected shape – keep defaults
        pass

    label = CLASS_NAMES[class_idx]
    return label, prob_corrosion, class_probs, class_idx


# =========================
# GEMINI INTEGRATION (Advanced Analysis)
# =========================

@st.cache_resource
def get_gemini_client():
    """Initializes and returns the Gemini client."""
    if "gemini_api_key" not in st.secrets:
        # Check environment variable as fallback
        if "GEMINI_API_KEY" in os.environ:
            api_key = os.environ["GEMINI_API_KEY"]
        else:
            st.error(".")
            return None
    else:
        api_key = st.secrets["gemini_api_key"]

    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None

def get_gemini_analysis(client, image: Image.Image):
    """
    Uses Gemini to analyze the image and describe corrosion/cracks.
    Note: Gemini provides a text description, not bounding box coordinates.
    """
    if not client:
        return "**analysis skipped: API key missing.**"

    # Convert PIL Image to bytes buffer for API
    img_byte_arr = io.BytesIO()
    # Save the image as JPEG for better compatibility and smaller size
    image.save(img_byte_arr, format='JPEG') 
    image_bytes = img_byte_arr.getvalue()

    # Define the prompt for detailed analysis
    prompt = (
        "Analyze this image of infrastructure or a metal surface for defects. "
        "Provide a detailed description of any **corrosion, cracks, or structural defects** present. "
        "For any visible defects, estimate the **severity and location** (e.g., 'severe pitting corrosion visible on the lower left side'). "
        "Be concise. If no defects are visible, state that clearly."
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')]
        )
        return response.text
    except Exception as e:
        return f"API error during content generation: {e}"


# =========================
# STREAMLIT APP LAYOUT
# =========================
st.set_page_config(
    page_title="Corrosion & Defect Detection",
    page_icon="🛠️",
    layout="centered",
)

st.title("🛠️ Automated Corrosion & Defect Detection")
st.markdown(
    """
This application uses a two-pronged approach:
1.  A **Keras Classification Model** to quickly classify the image as 'Corrosion' or 'No Corrosion'.
2.  The **Gemini API** for a detailed, **text-based visual inspection** and localization of cracks and corrosion.
"""
)

# Load Keras Model and Gemini Client
model = load_model()
model_img_size = get_model_img_size(model)
gemini_client = get_gemini_client()

if model is not None:
    st.success(
        f"Keras Model loaded from `saved_model.h5` ✅  \n"
        f"Expected input image size: **{model_img_size[0]}×{model_img_size[1]}** (height × width)"
    )

st.divider()

uploaded_files = st.file_uploader(
    "Upload one or more images (JPG/PNG) for analysis",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# Threshold for Keras binary model
threshold = st.slider(
    "Decision threshold for Keras Classification (Sigmoid Models)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Used if the Keras model has a single sigmoid output. For softmax models, argmax is used instead.",
)

if uploaded_files:
    for file in uploaded_files:
        st.divider()
        st.header(f"🔍 Analysis for: {file.name}")

        # Read image
        bytes_data = file.read()
        image = Image.open(io.BytesIO(bytes_data))

        # Show original image
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown("---")


        # --- 1. Keras Classification ---
        st.subheader("1️⃣ Keras Model Classification")
        if model is not None:
            with st.spinner("Running Keras corrosion detection..."):
                raw_preds, img_size = predict_image(model, image)

            # Interpret predictions
            label, prob_corrosion, class_probs, class_idx = interpret_predictions(
                raw_preds, threshold=threshold
            )

            # Display results
            st.markdown(
                f"**Prediction:** `{label}`  \n"
                f"**P(CORROSION)** ≈ `{prob_corrosion:.4f}`  \n"
                f"**Class probabilities** `[NO CORROSION, CORROSION]` = `{[round(p, 4) for p in class_probs]}`"
            )

            st.progress(float(prob_corrosion))

            if class_idx == 1:
                st.warning("⚠️ **Keras Result:** Corrosion likely present.")
            else:
                st.info("✅ **Keras Result:** No significant corrosion detected.")

            # Optional debug info
            with st.expander("Show Keras Model output (debug)"):
                st.write("Raw predictions from model:", raw_preds)
        else:
             st.warning("Keras model not loaded. Skipping classification.")

        st.markdown("---")

        # --- 2. Gemini Advanced Analysis ---
        st.subheader("2️⃣Detailed Visual Analysis (Cracks & Corrosion)")
        with st.spinner("Running visual inspection..."):
            gemini_analysis = get_gemini_analysis(gemini_client, image)

        # Display Gemini results
        st.markdown(gemini_analysis)

        # Guidance on Bounding Boxes
