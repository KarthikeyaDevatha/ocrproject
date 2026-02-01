
# Handwritten OCR System (Text & Math)

A local, privacy-focused OCR system fine-tuned for:
1.  **Handwritten English** (Scientific/Academic domain).
2.  **Mathematical Expressions** (LaTeX output).

This project presents a Document AI Agent for Handwritten Text Extraction that processes images of handwritten lab records and notes in educational environments. The system utilizes Vision Transformer-based Optical Character Recognition architecture as the core engine to extract textual content from handwritten academic documents and convert them into digital text files.

Built with **TrOCR** (Vision Encoder-Decoder) and **YOLOv8** (Layout Detection).

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Demo App
```bash
streamlit run app/app.py
```
Upload an image to see detection types (Text vs Math) and OCR results.

### 3. Inference from Command Line
```bash
python3 inference/pipeline.py --image path/to/image.png --output result.md
```

---

## 🛠 Project Structure

### Data & Training
- `data/scripts/`: Ingestion scripts for IAM, MathWriting, and Custom datasets.
- `training/`: Training scripts.
  - `train_trocr_text.py`: Fine-tune TrOCR (supports `--augment`).
  - `train_math_model.py`: Script for Math OCR (HF Streaming).

### Optimization (Accuracy Recovery)
We use a high-precision pipeline to eliminate "dot -> 0" errors:
1.  **Preprocessing**: `data/scripts/preprocess_lines.py` removes underlines and boosts dots.
2.  **Verification**: `evaluation/verify_accuracy.py` audits "Zero Hallucination Rate".
3.  **Metrics**: `evaluation/advanced_metrics.py` tracks lexical drift.

---

## 🧠 Models

| Type | Model Base | Status |
| :--- | :--- | :--- |
| **Text** | `microsoft/trocr-small-handwritten` | Fine-Tuning (Phase 2) |
| **Math** | `microsoft/trocr-base-printed` | Planned |
| **Layout** | `ultralytics/yolov8n` | Detection Ready |

---

## 🧪 Evaluation

To verify the model against specific failure modes (halucinated zeros):
```bash
python3 evaluation/verify_accuracy.py --model_path checkpoints/trocr_text_accuracy_small/epoch_1
```

---

## 📝 License
MIT License. Local Use Only.
