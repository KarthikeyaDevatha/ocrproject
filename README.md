# Adaptive OCR Agent v2.1 🚀

A production-grade, zero-hallucination Optical Character Recognition (OCR) pipeline for extracting math, equations, and handwriting from high-variance document images.

Built with **Streamlit, TrOCR, Mathpix, and Semantic Math Validation**, this system evaluates confidence, validates arithmetic rules, and intelligently routes image segments to the best OCR models.

## ✨ Features (v2.1)

1. **Strict Confidence Routing**: Explicit `ACCEPTED`, `RETRY_REQUIRED`, and `FAILED_EXTRACTION` labels to eliminate ambiguity.
2. **Semantic Math Validation**: Understands math! Automatically parses parsed numbers, runs division/sum/mean consistency checks, and applies validation boosts to ensure extreme accuracy.
3. **Smart Engine Dispatch**: 
   - Pure handwriting? Routes to local **TrOCR**.
   - Dense math/equations? Routes to **Mathpix** API.
   - Mixed content? Runs both and fuses results for the best outcome.
4. **Resilient Adaptive Preprocessing**: Per-crop **CLAHE** profiles specific to cleanly scanned or degraded/photographed images.
5. **Interactive UI**: A rich 4-mode Streamlit app exposing visualizations of bounding boxes, confidence metrics, logs, and side-by-side mode comparisons.

## 📦 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/KarthikeyaDevatha/ocr-handwritten.git
cd ocr-handwritten
```

### 2. Set up Python Environment
We recommend using a virtual environment (Python 3.9+).
```bash
python -m venv .venv
source .venv/bin/activate
# For Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: If you run into issues installing `craft-text-detector` on Windows, you may need C++ Build Tools installed).*

### 4. Set Environment Variables (Optional)
To use the highly-accurate **Mathpix** API engine, create a `.env` file in the root directory:
```env
MATHPIX_APP_ID=your_id_here
MATHPIX_APP_KEY=your_key_here

# To test Mathpix routing without spending API credits, enable Mock Mode:
MATHPIX_MOCK=true
```

### 5. Launch the Application
```bash
streamlit run app/app.py
```

## 🧠 Architecture Overview

`DecisionEngine` → `LineDetector (CRAFT)` → `Preprocessing` → `Orchestrator` → `PostProcessor`

- **inference/decision_engine.py**: Feature extractors (blur, math density) decide which pipeline profile handles an image via `_smart_route()`.
- **inference/confidence_gate.py**: Calculates `alpha_ratio`, evaluates beam search token probailities, and runs rule-based semantic `MathValidator` consistency checks.
- **inference/hybrid_pipeline.py**: Manages execution flow, retry-logic on failure, and multi-engine result fusion.

---
**Status**: Stable / v2.1
