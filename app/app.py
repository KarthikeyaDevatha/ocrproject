"""
Streamlit Demo App for Adaptive OCR Agent v2.0.

Features:
- 4-mode selector (Auto / Mathpix / TrOCR / Arithmetic)
- Mathpix API key input + mock mode toggle
- Per-line confidence display with 3-tier tags
- Pipeline log viewer
- Bounding box overlay
- Side-by-side engine comparison
- CER/WER evaluation tab
"""

import os
import sys
import tempfile
import time
import json
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from inference.hybrid_pipeline import HybridPipeline, PipelineResult
from inference.evaluator import Evaluator, compute_cer, compute_wer


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Adaptive OCR Agent v2.0",
    page_icon="🧠",
    layout="wide"
)


# ============================================================================
# CACHED RESOURCE LOADERS
# ============================================================================

@st.cache_resource
def load_hybrid_pipeline(mode: str, model_size: str) -> HybridPipeline:
    """Load hybrid pipeline (cached per mode+model)."""
    return HybridPipeline(mode=mode, trocr_model=model_size, verbose=False)


# ============================================================================
# HELPER: Draw bounding boxes on image
# ============================================================================

def draw_line_boxes(image: np.ndarray, line_results, show_tags: bool = True) -> np.ndarray:
    """Draw bounding boxes with confidence tags on the image."""
    overlay = image.copy()

    color_map = {
        "ACCEPTED": (0, 200, 0),       # Green
        "LOW_CONFIDENCE": (0, 165, 255), # Orange
        "FAILED": (0, 0, 255),          # Red
    }

    for lr in line_results:
        x1, y1, x2, y2 = lr.bbox
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue

        color = color_map.get(lr.tag, (128, 128, 128))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        if show_tags:
            label = f"{lr.tag} ({lr.confidence:.2f})"
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return overlay


# ============================================================================
# HELPER: Tag badge
# ============================================================================

def tag_badge(tag: str) -> str:
    """Return colored badge for confidence tag."""
    if tag == "ACCEPTED":
        return "🟢 ACCEPTED"
    elif tag == "LOW_CONFIDENCE":
        return "🟡 LOW CONFIDENCE"
    else:
        return "🔴 FAILED"


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.title("🧠 Adaptive OCR Agent v2.0")
    st.caption("Decision-driven, confidence-aware handwritten OCR pipeline")

    # ---- SIDEBAR ----
    st.sidebar.header("⚙️ Configuration")

    # Mode selector
    st.sidebar.subheader("🔄 Pipeline Mode")
    mode = st.sidebar.selectbox(
        "OCR Mode",
        ["auto", "trocr", "mathpix", "arithmetic"],
        index=0,
        help="Auto: smart routing | TrOCR: local offline | Mathpix: cloud API | Arithmetic: digit extraction"
    )

    st.sidebar.divider()

    # Model size
    st.sidebar.subheader("🧠 TrOCR Model")
    model_size = st.sidebar.radio(
        "Model Size",
        ["large", "base"],
        index=1,
        help="Large = accurate (CER ~2.89%), Base = fast (CER ~3.8%)"
    )

    st.sidebar.divider()

    # Mathpix config
    st.sidebar.subheader("🔑 Mathpix API")
    mathpix_id = st.sidebar.text_input(
        "App ID",
        value=os.getenv("MATHPIX_APP_ID", ""),
        type="password"
    )
    mathpix_key = st.sidebar.text_input(
        "App Key",
        value=os.getenv("MATHPIX_APP_KEY", ""),
        type="password"
    )
    mock_mode = st.sidebar.toggle(
        "Mock Mode",
        value=os.getenv("MATHPIX_MOCK", "false") == "true",
        help="Return canned response (for demo, no API call)"
    )

    # Set env vars
    if mathpix_id:
        os.environ["MATHPIX_APP_ID"] = mathpix_id
    if mathpix_key:
        os.environ["MATHPIX_APP_KEY"] = mathpix_key
    os.environ["MATHPIX_MOCK"] = "true" if mock_mode else "false"

    st.sidebar.divider()

    # Debug options
    st.sidebar.subheader("🔧 Debug")
    show_boxes = st.sidebar.checkbox("Show Detection Boxes", value=True)
    show_log = st.sidebar.checkbox("Show Pipeline Log", value=True)

    # ---- MAIN AREA ----
    # Tabs
    tab_ocr, tab_compare, tab_eval = st.tabs(["🔍 OCR", "⚖️ Compare Engines", "📊 Evaluation"])

    # ========================================================
    # TAB 1: OCR
    # ========================================================
    with tab_ocr:
        st.header("📤 Upload Image")

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "tiff", "bmp"],
            help="Upload a scan or photo of handwritten notes"
        )

        col_sample1, col_sample2 = st.columns(2)
        with col_sample1:
            use_sample = st.button("📄 Use Sample Image")

        if uploaded_file is not None or use_sample:
            # Load image
            if use_sample:
                sample_path = "samples/page.png"
                if not os.path.exists(sample_path):
                    sample_path = "samples/IMG_1771.png"
                if os.path.exists(sample_path):
                    image = Image.open(sample_path)
                else:
                    st.warning("No sample image found.")
                    return
            else:
                image = Image.open(uploaded_file)

            # Display
            st.image(image, caption="Input Image", use_container_width=True)

            if st.button("🚀 Run OCR", type="primary"):
                with st.spinner(f"Processing with mode: {mode.upper()}..."):
                    # Save temp
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        image.save(tmp.name)
                        tmp_path = tmp.name

                    try:
                        pipeline = load_hybrid_pipeline(mode, model_size)
                        result = pipeline.process(tmp_path, mode=mode)

                        # ---- RESULTS ----
                        st.header("📊 Results")

                        # Status badge
                        st.markdown(f"### {tag_badge(result.overall_tag)}")

                        # Metrics row
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Confidence", f"{result.overall_confidence:.2%}")
                        c2.metric("Engine", result.engine_used.upper())
                        c3.metric("Profile", result.profile_used.upper())
                        c4.metric("Time", f"{result.processing_time:.2f}s")

                        # Confidence bar
                        st.progress(
                            min(result.overall_confidence, 1.0),
                            text=f"Overall: {result.overall_confidence:.2%}"
                        )

                        # Stats row
                        stats = result.stats
                        if isinstance(stats, dict) and "lines_detected" in stats:
                            sc1, sc2, sc3, sc4 = st.columns(4)
                            sc1.metric("Lines", stats.get("lines_detected", 0))
                            sc2.metric("Accepted", stats.get("lines_accepted", 0))
                            sc3.metric("Low Conf", stats.get("lines_low_conf", 0))
                            sc4.metric("Retries", stats.get("retries", 0))

                        st.divider()

                        # Detection overlay
                        if show_boxes and result.lines:
                            st.subheader("🔍 Detected Lines")
                            img_cv = cv2.imread(tmp_path)
                            if img_cv is not None:
                                overlay = draw_line_boxes(img_cv, result.lines)
                                st.image(
                                    cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                                    caption="🟢 Accepted  🟡 Low Confidence  🔴 Failed",
                                    use_container_width=True
                                )

                        # OCR Output
                        st.subheader("📝 OCR Output")
                        st.code(result.text, language="text")

                        # Mathpix result
                        if result.mathpix_result:
                            st.subheader("🔬 Mathpix Details")
                            mpx = result.mathpix_result
                            if mpx.get("latex_styled"):
                                st.latex(mpx["latex_styled"])
                            if mpx.get("html"):
                                st.markdown(mpx["html"], unsafe_allow_html=True)

                        # Download
                        st.download_button(
                            "📥 Download Result",
                            data=result.text,
                            file_name="ocr_result.txt",
                            mime="text/plain"
                        )

                        # Per-line details
                        with st.expander("📋 Per-Line Details"):
                            for i, lr in enumerate(result.lines):
                                st.markdown(f"**Line {i+1}** — {tag_badge(lr.tag)}")
                                st.write(f"Text: `{lr.text}`")
                                st.write(f"Confidence: {lr.confidence:.4f} | Engine: {lr.engine_used}")
                                if lr.retried:
                                    st.caption("↩️ Retried with fallback engine")
                                st.divider()

                        # Pipeline log
                        if show_log:
                            with st.expander("📜 Pipeline Log"):
                                st.code(result.log, language="text")

                        # Full JSON
                        with st.expander("🗂️ Full JSON Output"):
                            output_dict = {
                                "text": result.text,
                                "overall_confidence": result.overall_confidence,
                                "overall_tag": result.overall_tag,
                                "engine": result.engine_used,
                                "profile": result.profile_used,
                                "mode": result.mode,
                                "processing_time": result.processing_time,
                                "features": result.features,
                                "stats": result.stats,
                                "lines": [
                                    {
                                        "text": lr.text,
                                        "confidence": lr.confidence,
                                        "tag": lr.tag,
                                        "engine": lr.engine_used,
                                        "retried": lr.retried,
                                        "bbox": list(lr.bbox)
                                    }
                                    for lr in result.lines
                                ]
                            }
                            st.json(output_dict)

                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

    # ========================================================
    # TAB 2: COMPARE ENGINES
    # ========================================================
    with tab_compare:
        st.header("⚖️ Side-by-Side Engine Comparison")
        st.caption("Run the same image through multiple engines and compare")

        compare_file = st.file_uploader(
            "Upload image for comparison",
            type=["png", "jpg", "jpeg"],
            key="compare_upload"
        )

        if compare_file:
            comp_image = Image.open(compare_file)
            st.image(comp_image, caption="Comparison Image", use_container_width=True)

            engines_to_compare = st.multiselect(
                "Select engines",
                ["trocr", "mathpix", "arithmetic"],
                default=["trocr"]
            )

            if st.button("⚡ Run Comparison", type="primary"):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    comp_image.save(tmp.name)
                    tmp_path = tmp.name

                try:
                    cols = st.columns(len(engines_to_compare))
                    for i, engine in enumerate(engines_to_compare):
                        with cols[i]:
                            st.subheader(f"🔧 {engine.upper()}")
                            with st.spinner(f"Running {engine}..."):
                                pipe = load_hybrid_pipeline(engine, model_size)
                                res = pipe.process(tmp_path, mode=engine)

                            st.markdown(f"{tag_badge(res.overall_tag)}")
                            st.metric("Confidence", f"{res.overall_confidence:.2%}")
                            st.metric("Time", f"{res.processing_time:.2f}s")
                            st.code(res.text, language="text")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    # ========================================================
    # TAB 3: EVALUATION
    # ========================================================
    with tab_eval:
        st.header("📊 CER/WER Evaluation")
        st.caption("Compute Character Error Rate and Word Error Rate against ground truth")

        eval_col1, eval_col2 = st.columns(2)

        with eval_col1:
            ocr_output = st.text_area(
                "OCR Output",
                placeholder="Paste or type OCR output here...",
                height=150
            )

        with eval_col2:
            ground_truth = st.text_area(
                "Ground Truth",
                placeholder="Paste or type expected text here...",
                height=150
            )

        if st.button("📐 Compute Metrics"):
            if ocr_output and ground_truth:
                cer = compute_cer(ocr_output, ground_truth)
                wer = compute_wer(ocr_output, ground_truth)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("CER", f"{cer:.4f}")
                mc2.metric("WER", f"{wer:.4f}")
                mc3.metric("Accuracy", f"{max(0, 1-cer):.2%}")

                if cer < 0.05:
                    st.success("Excellent accuracy!")
                elif cer < 0.15:
                    st.info("Good accuracy.")
                elif cer < 0.30:
                    st.warning("Moderate accuracy — consider preprocessing improvements.")
                else:
                    st.error("High error rate — check image quality and model selection.")
            else:
                st.warning("Please fill in both fields.")

    # ---- SIDEBAR FOOTER ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Mode Guide")
    st.sidebar.markdown("""
    | Mode | Best For |
    |------|----------|
    | **Auto** | Smart routing (default) |
    | **TrOCR** | Offline, handwriting |
    | **Mathpix** | Math/STEM (cloud) |
    | **Arithmetic** | Digit extraction |
    """)
    st.sidebar.markdown("---")
    st.sidebar.caption("Adaptive OCR Agent v2.0 • TrOCR + Mathpix + CRAFT")


if __name__ == "__main__":
    main()
