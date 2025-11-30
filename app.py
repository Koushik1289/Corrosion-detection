import io
import numpy as np
from PIL import Image
import streamlit as st
import os

# Imports for Gemini API
from google import genai
from google.genai import types

# =========================
# GEMINI INTEGRATION (Secure Client Setup)
# =========================

# The API key is securely loaded via st.secrets, removing any trace from the UI.
@st.cache_resource
def get_gemini_client():
    """Initializes and returns the Gemini client object securely."""
    # 1. Retrieve API Key from Streamlit Secrets (most secure method for deployment)
    api_key = st.secrets.get("gemini_api_key")
    
    if not api_key:
        st.error("🔑 API Key Error: Gemini API key not found in `st.secrets['gemini_api_key']`.")
        st.markdown("Please check your `.streamlit/secrets.toml` file.")
        return None
    
    # 2. Initialize Client Object (Correct method for the new google-genai SDK)
    try:
        # The Client constructor handles the API key
        client = genai.Client(api_key=api_key)
        return client # Return the client object
    except Exception as e:
        # The user will only see the general error, not the API key logic
        st.error(f"Error connecting to the analysis service. Please check the server configuration.")
        return None

def get_gemini_analysis(client, image: Image.Image):
    """
    Uses Gemini to analyze the image and return a detailed text analysis.
    """
    if not client:
        return "**Analysis failed: Service client is unavailable.**"

    # Convert PIL Image to bytes buffer for API
    img_byte_arr = io.BytesIO()
    # Save as JPEG for best compatibility and size
    image.save(img_byte_arr, format='JPEG') 
    image_bytes = img_byte_arr.getvalue()

    # --- Prompt for Detailed Text Analysis ---
    prompt = (
        "Analyze this image of infrastructure or a a metal surface. "
        "Provide a detailed textual description of any **corrosion, cracks, or structural defects** present. "
        "For any visible defects, describe their **severity, type, and precise location** within the image (e.g., 'severe pitting corrosion visible on the lower left beam', 'a vertical crack runs down the center-right of the metal'). "
        "Be concise and professional. If no defects are visible, state that clearly and conclude with 'Analysis complete: No defects detected.' "
        "Do NOT mention 'Gemini', 'AI', or 'model' in the output. Just act as an expert inspector."
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')]
        )
        return response.text
    except Exception as e:
        # Hide internal error details from the end-user
        return "An unexpected error occurred during the image analysis. Please try again."

# =========================
# STREAMLIT APP LAYOUT (No Mention of Gemini/API Key)
# =========================
st.set_page_config(
    page_title="Infrastructure Defect Inspection",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Automated Infrastructure Inspection")
st.markdown(
    """
This application performs a detailed visual inspection of structural surfaces 
to identify and report **corrosion, cracks, and other physical defects**. 
Upload an image and receive an expert analysis immediately.
"""
)

# Initialize the hidden client
client = get_gemini_client()

st.divider()

uploaded_files = st.file_uploader(
    "Upload one or more images (JPG/PNG) for inspection",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    if client is None:
        st.error("Cannot proceed. The analysis service failed to initialize.")
    else:
        for file in uploaded_files:
            st.divider()
            st.header(f"Inspection Report: {file.name}")

            # Read image
            bytes_data = file.read()
            image = Image.open(io.BytesIO(bytes_data))

            # Show original image
            st.subheader("Original Image")
            st.image(image, caption="Image submitted for inspection", use_container_width=True)
            st.markdown("---")

            # --- Run Analysis ---
            st.subheader("Detailed Visual Analysis")
            
            with st.spinner("Running high-resolution visual inspection..."):
                detailed_analysis = get_gemini_analysis(client, image)

            # Display results
            st.markdown(detailed_analysis)
            
            if "No defects detected" in detailed_analysis:
                st.info("✅ Inspection complete. The surface appears to be in good condition.")
            else:
                st.warning("⚠️ **DEFECTS IDENTIFIED.** Review the analysis above for locations requiring immediate attention.")

            # --- Note on Highlighting ---
            st.markdown(
                """
                ***
                *Note:* The inspection service performs a detailed, text-based localization. 
                For pixel-perfect **visual highlighting** (drawing boxes directly on the image), 
                a specialized, locally-run object detection model is typically used. 
                The current system provides the expert description needed for manual review.
                """
            )

else:
    st.info("👆 Upload at least one image to begin the automated inspection.")

if __name__ == '__main__':
    pass
