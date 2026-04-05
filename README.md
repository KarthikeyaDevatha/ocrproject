# Nexus OCR

A production-grade Scientific Document Intelligence Platform that converts images or PDFs of student assignments into structured machine-readable JSON knowledge.

## Architecture
- **Preprocessing**: OpenCV normalization, deskew.
- **Layout Detection**: LayoutLMv3
- **OCR Routing**: TrOCR (Handwriting), PaddleOCR (Printed Text), Pix2Tex/UniMERNet (Equations).
- **Symbol Corrector**: LLM-assisted grammar correction.
- **Semantic Reasoner**: GPT-4o structure parsing.

## Quickstart

### Docker Compose
```bash
# Rename env file and add your LLM API Key
cp .env.example .env

# Build and start the services (API, Worker, Redis)
docker-compose up --build
```

### Local Development
```bash
pip install -r requirements.txt
uvicorn backend.api.main:app --reload
```

## API Usage
**POST /upload**
Upload a document (PDF or image). Returns a `job_id`.

**GET /result/{job_id}**
Poll for the JSON result of the async job.

**POST /process**
Send a Base64 encoded image for synchronous real-time processing.
