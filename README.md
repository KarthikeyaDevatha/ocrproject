# Nexus OCR Handwritten Scientific Document Intelligence

Production-grade modular OCR pipeline for assignment understanding with specialized pretrained models.

## Architecture

`preprocess -> layout detection -> OCR routing -> symbol correction -> semantic parser -> JSON`

Key modules:
- Preprocessing: OpenCV deskew/perspective/thresholding (`backend/pipelines/preprocessing.py`)
- Layout detector: LayoutLMv3 integration + CV fallback (`backend/pipelines/layout_detector.py`)
- OCR engines:
  - TrOCR handwriting (`backend/models/trocr_engine.py`)
  - PaddleOCR printed text (`backend/models/paddleocr_engine.py`)
  - Pix2Tex + UniMERNet equations (`backend/models/equation_engine.py`)
- Semantic parsing and problem typing (`backend/pipelines/semantic_parser.py`)
- Orchestration (`backend/pipelines/document_processor.py`)
- FastAPI API (`backend/api/main.py`)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.api.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Docker

```bash
docker-compose up --build
```

## API

- `POST /upload` -> async job id (image/pdf)
- `GET /result/{job_id}` -> result status
- `POST /process` -> sync base64 processing
- `GET /health` -> readiness

## Dataset tools

- `datasets/loaders/crohme_loader.py`
- `datasets/loaders/im2latex_loader.py`
- `datasets/loaders/publaynet_loader.py`
- `datasets/generators/synthetic_math.py`

## Notes

The system prefers pretrained models and gracefully falls back when a model dependency or weight is unavailable in local runtime.


## Training status report

```bash
python training/report_status.py
cat reports/project_status.md
```

This generates machine-readable status in `reports/project_status.json` and a human report in `reports/project_status.md`.
