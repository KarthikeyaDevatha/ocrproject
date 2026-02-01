"""
Streamlit Demo App for Handwritten OCR System.
Provides a web interface for uploading images and viewing OCR results.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.pipeline import OCRPipeline, PipelineConfig
from inference.layout import draw_detections


# Page configuration
st.set_page_config(
    page_title="Handwritten OCR System",
    page_icon="📝",
    layout="wide"
)


@st.cache_resource
def load_pipeline(device: str = "cpu", text_model_dir: str = "models/trocr_text", math_model_dir: str = "models/trocr_math") -> OCRPipeline:
    """Load OCR pipeline (cached)."""
    # Create hash for caching different models
    # Actually streamlit caches based on arguments.
    config = PipelineConfig(
        device=device,
        text_model_dir=text_model_dir,
        math_model_dir=math_model_dir,
        debug_output=True,
        save_debug_images=True
    )
    return OCRPipeline(config)


def main():
    st.title("📝 Handwritten Text & Math OCR")
    st.markdown("""
    Upload an image of handwritten notes to extract text and mathematical expressions.
    The system outputs **Markdown with LaTeX** that can be rendered directly.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model Selection
    st.sidebar.subheader("🧠 Models")
    
    # Text Model
    text_model_source = st.sidebar.radio(
        "Text Model Source",
        ["Default (Base)", "Fine-Tuned Checkpoint"],
        help="Choose between base model or your trained checkpoints"
    )
    
    text_model_dir = "models/trocr_text"
    if text_model_source == "Fine-Tuned Checkpoint":
        # Scan checkpoints
        import glob
        # Search in both default and combined directories
        checkpoints = []
        for d in ["checkpoints/trocr_text_combined", "checkpoints/trocr_text"]:
            if os.path.exists(d):
                found = sorted(glob.glob(os.path.join(d, "*")))
                checkpoints.extend([f for f in found if os.path.isdir(f)])
        
        checkpoints = sorted(list(set(checkpoints))) # Deduplicate
        
        if not checkpoints:
            st.sidebar.warning("No checkpoints found in checkpoints/trocr_text*")
            text_model_source = "Default (Base)"
        else:
            selected_ckpt_name = st.sidebar.selectbox(
                "Select Checkpoint",
                [os.path.basename(c) for c in checkpoints],
                index=len(checkpoints)-1 # Default to latest
            )
            # Find the full path for the selected name
            for c in checkpoints:
                if os.path.basename(c) == selected_ckpt_name:
                    text_model_dir = c
                    break
            st.sidebar.info(f"Using: {selected_ckpt_name}")
            
    # Math Model
    math_model_source = st.sidebar.radio(
        "Math Model Source",
        ["Default (Base Printed)", "Fine-Tuned (Math)"],
        index=0
    )
    
    math_model_dir = "models/trocr_math"
    if math_model_source == "Fine-Tuned (Math)":
         # Scan checkpoints
        import glob
        checkpoints_math = sorted(glob.glob("checkpoints/trocr_math/*"))
        checkpoints_math = [d for d in checkpoints_math if os.path.isdir(d)]
        
        if not checkpoints_math:
            st.sidebar.warning("No checkpoints found in checkpoints/trocr_math")
        else:
            selected_ckpt_math = st.sidebar.selectbox(
                "Select Math Checkpoint",
                [os.path.basename(c) for c in checkpoints_math],
                index=len(checkpoints_math)-1
            )
            math_model_dir = os.path.join("checkpoints/trocr_math", selected_ckpt_math)

    st.sidebar.divider()

    device = st.sidebar.selectbox(
        "Compute Device",
        ["cpu", "cuda"],
        index=0,
        help="Select GPU (cuda) for faster inference if available"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.1,
        help="Minimum confidence for OCR predictions"
    )
    
    enable_rerouting = st.sidebar.checkbox(
        "Enable Rerouting",
        value=True,
        help="Reroute low-confidence predictions to alternate model"
    )
    
    show_debug = st.sidebar.checkbox(
        "Show Detection Boxes",
        value=True,
        help="Display detected text and math regions"
    )
    
    # File upload
    st.header("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        help="Upload a scan or photo of handwritten notes"
    )
    
    # Sample images
    col1, col2 = st.columns(2)
    with col1:
        use_sample = st.button("Use Sample Image")
    
    # Process image
    if uploaded_file is not None or use_sample:
        # Load image
        if use_sample:
            sample_path = "samples/page.png"
            if os.path.exists(sample_path):
                image = Image.open(sample_path)
            else:
                st.warning("Sample image not found. Please generate one first.")
                st.code("python data/synthesis_script.py --generate-page")
                return
        else:
            image = Image.open(uploaded_file)
        
        # Display original image
        st.header("🖼️ Input Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process button
        if st.button("🚀 Run OCR", type="primary"):
            with st.spinner("Processing image..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name
                
                try:
                    # Load pipeline
                    pipeline = load_pipeline(device, text_model_dir, math_model_dir)
                    
                    # Update config
                    pipeline.config.confidence_threshold = confidence_threshold
                    pipeline.config.enable_rerouting = enable_rerouting
                    
                    # Process
                    start_time = time.time()
                    result = pipeline.process(tmp_path)
                    processing_time = time.time() - start_time
                    
                    # Show results
                    st.header("📊 Results")
                    
                    # Stats
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Time", f"{result.total_time:.2f}s")
                    col2.metric("Layout Detection", f"{result.layout_time:.2f}s")
                    col3.metric("OCR Processing", f"{result.ocr_time:.2f}s")
                    col4.metric("Regions", f"{result.num_text_regions} text, {result.num_math_regions} math")
                    
                    # Detection visualization
                    if show_debug and result.debug_image is not None:
                        st.subheader("🔍 Detected Regions")
                        st.image(
                            cv2.cvtColor(result.debug_image, cv2.COLOR_BGR2RGB),
                            caption="Green: Text, Blue: Math",
                            use_container_width=True
                        )
                    
                    # Raw markdown
                    st.subheader("📝 Raw Markdown Output")
                    st.code(result.markdown, language="markdown")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Markdown",
                        data=result.markdown,
                        file_name="ocr_result.md",
                        mime="text/markdown"
                    )
                    
                    # Rendered output
                    st.subheader("✨ Rendered Output")
                    st.markdown(result.markdown)
                    
                    # Detailed results
                    with st.expander("📋 Detailed Recognition Results"):
                        for i, line_result in enumerate(result.line_results):
                            st.write(f"**Line {i+1}** ({line_result.line_type})")
                            st.write(f"- Text: `{line_result.text}`")
                            st.write(f"- Confidence: {line_result.confidence:.3f}")
                            st.write(f"- BBox: {line_result.bbox}")
                            st.divider()
                    
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                
                finally:
                    # Cleanup
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.header("📖 Instructions")
    st.sidebar.markdown("""
    1. Upload an image of handwritten notes
    2. Click "Run OCR" to process
    3. View detected regions and output
    4. Download the Markdown result
    
    **Supported content:**
    - Handwritten English text
    - Mathematical expressions
    - Mixed text and math
    
    **Tips for best results:**
    - Use high-quality scans (300+ DPI)
    - Ensure good lighting
    - Keep text horizontal
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ❤️ using TrOCR + YOLOv8")


if __name__ == "__main__":
    main()
