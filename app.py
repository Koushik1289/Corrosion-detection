import io
import json
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
import os

# --- Constants for Annotation ---
CORROSION_COLOR = (255, 69, 0)  # Red-Orange
CRACK_COLOR = (0, 255, 255)     # Cyan
DEFAULT_COLOR = (0, 255, 0)
LINE_THICKNESS = 3
DEFAULT_MODEL = 'gemini-2.5-flash'

# =========================
# GEMINI INTEGRATION (Client Setup and Structured Output)
# =========================

@st.cache_resource
def get_gemini_client():
    """Initializes the Gemini client object securely."""
    api_key = st.secrets.get("gemini_api_key")
    
    if not api_key:
        st.error("🔑 API Key Error: Please ensure `gemini_api_key` is set in `.streamlit/secrets.toml`.")
        return None
    
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        # Hide internal error details from the end-user
        st.error(f"Error connecting to the analysis service.")
        return None

def detect_and_get_boxes(client, image: Image.Image):
    """
    Prompts Gemini to output bounding box coordinates and labels in JSON format.
    
    Coordinates are returned normalized to 0-1000.
    """
    if not client:
        return None, "**Analysis failed: Service client is unavailable.**"

    # Define the structured output format using the `response_schema`
    bbox_schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "box_2d": types.Schema(
                    type=types.Type.ARRAY,
                    description="Bounding box coordinates normalized to 0-1000: [y_min, x_min, y_max, x_max].",
                    items=types.Schema(type=types.Type.INTEGER)
                ),
                "label": types.Schema(
                    type=types.Type.STRING,
                    description="The defect type: 'Corrosion', 'Crack', or other structural defect.",
                ),
                "confidence": types.Schema(
                    type=types.Type.NUMBER,
                    description="Confidence score for the detection (0.0 to 1.0).",
                ),
            },
            required=["box_2d", "label", "confidence"],
        )
    )

    prompt = (
        "Analyze the image for corrosion, cracks, or any structural defects. "
        "For every defect found, output a JSON object containing the 2D bounding box normalized to 0-1000, "
        "the label ('Corrosion', 'Crack', or specific defect name), and a confidence score. "
        "If no defects are found, return an empty array []."
    )
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=bbox_schema,
        temperature=0.1
    )

    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=[prompt, image],
            config=config,
        )
        
        # The response.text will be a valid JSON string matching the schema
        json_output = json.loads(response.text)
        
        # Get a separate summary for a nice report
        summary_prompt = "Provide a detailed textual summary of the corrosion and cracks detected in the image based on the coordinates you provided. Describe severity and general location (e.g., 'severe corrosion in the upper right'). Do NOT mention 'Gemini', 'AI', or 'model' in the output. Just provide the expert inspection report."
        
        summary_response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=[summary_prompt, image]
        )
        
        return json_output, summary_response.text

    except APIError as e:
        return None, f"API Error: Failed to get detection results. (Code: {e})"
    except json.JSONDecodeError:
        return None, "Error: Could not parse structured output from service. Try a clearer image or adjust the prompt."
    except Exception as e:
        return None, f"An unexpected error occurred during analysis: {e}"


def annotate_image(image: Image.Image, detections: list):
    """
    Draws bounding boxes on the image based on normalized coordinates.
    
    FIXED: Replaced deprecated draw.textsize() with draw.textbbox().
    """
    img_width, img_height = image.size
    annotated_image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated_image)
    
    # Try to load a standard font
    try:
        font_size = max(15, int(img_width / 40)) 
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    results_summary = []
    
    for det in detections:
        # Coordinates are normalized to 0-1000: [y_min, x_min, y_max, x_max]
        y_min_norm, x_min_norm, y_max_norm, x_max_norm = det.get('box_2d', [0, 0, 0, 0])
        label = det.get('label', 'Unknown Defect')
        confidence = det.get('confidence', 0.0)
        
        # Convert normalized coordinates (0-1000) to actual pixel coordinates
        x_min = int(x_min_norm * img_width / 1000)
        y_min = int(y_min_norm * img_height / 1000)
        x_max = int(x_max_norm * img_width / 1000)
        y_max = int(y_max_norm * img_height / 1000)
        
        # Determine color based on label
        if "corrosion" in label.lower():
            color = CORROSION_COLOR
        elif "crack" in label.lower():
            color = CRACK_COLOR
        else:
            color = DEFAULT_COLOR

        # 1. Draw Rectangle
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)], 
            outline=color, 
            width=LINE_THICKNESS
        )
        
        # 2. Draw Label Text and Background
        display_text = f"{label} ({confidence*100:.0f}%)"
        
        # Use draw.textbbox() - the modern, correct Pillow method
        left, top, right, bottom = draw.textbbox((0, 0), display_text, font=font)
        text_w = right - left
        text_h = bottom - top

        # Ensure label background is placed correctly, avoiding going off the top edge
        label_x = x_min
        label_y = max(0, y_min - text_h - 2)
        
        draw.rectangle(
            [(label_x, label_y), (label_x + text_w, label_y + text_h)], 
            fill=color
        )
        
        # Text color is white for visibility against colored background
        draw.text((label_x, label_y), display_text, fill=(255, 255, 255), font=font)
        
        results_summary.append(f"- **{label}** ({confidence*100:.0f}%) localized at: X={x_min}-{x_max}, Y={y_min}-{y_max}")

    return annotated_image, results_summary


# =========================
# STREAMLIT APP LAYOUT
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
to identify, **localize, and visually highlight** corrosion, cracks, and other defects.
The detection service runs in the background.
"""
)
st.markdown("---")

# Initialize the hidden client
client = get_gemini_client()

uploaded_file = st.file_uploader(
    "Upload an image (JPG/PNG) for defect detection",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    if client is None:
        st.error("Cannot proceed. The analysis service failed to initialize.")
    else:
        st.header("Inspection Report")

        # Read image
        image = Image.open(io.BytesIO(uploaded_file.read()))

        # --- Run Detection ---
        with st.spinner("Running high-resolution visual inspection and localization..."):
            detections, detailed_summary = detect_and_get_boxes(client, image)
        
        # Check for errors
        if isinstance(detections, str) or detections is None:
            st.error(f"Analysis Error: {detailed_summary}")
        else:
            # --- Draw Annotations ---
            annotated_image, results_list = annotate_image(image, detections)

            # --- Display Results ---
            st.subheader("Highlighted Defects")
            st.image(
                annotated_image, 
                caption=f"Visual Inspection with {len(detections)} Defects Highlighted", 
                use_container_width=True
            )
            
            st.markdown("---")
            
            st.subheader("Textual Inspection Summary")
            st.markdown(detailed_summary)
            
            st.subheader("Raw Detection Data")
            if results_list:
                 st.markdown("The following defects were precisely localized:")
                 st.code('\n'.join(results_list))
                 st.warning("⚠️ **DEFECTS IDENTIFIED.** Review the highlighted areas above for attention.")
            else:
                 st.info("✅ No structural defects requiring attention were identified.")
