"""FastAPI entrypoint for async/sync document processing."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.pipelines.document_processor import DocumentProcessor

app = FastAPI(title="Nexus OCR API", version="1.1.0")
processor = DocumentProcessor()
_jobs: dict[str, dict] = {}


class ProcessRequest(BaseModel):
    image_base64: str


class RedisJobStore:
    def __init__(self) -> None:
        self.enabled = False
        self.client = None
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis

            self.client = redis.from_url(url, decode_responses=True)
            self.client.ping()
            self.enabled = True
        except Exception:
            self.enabled = False

    def set(self, job_id: str, payload: dict) -> None:
        if self.enabled and self.client is not None:
            self.client.set(f"job:{job_id}", json.dumps(payload), ex=86400)
        else:
            _jobs[job_id] = payload

    def get(self, job_id: str) -> dict | None:
        if self.enabled and self.client is not None:
            raw = self.client.get(f"job:{job_id}")
            return json.loads(raw) if raw else None
        return _jobs.get(job_id)


job_store = RedisJobStore()


@app.get("/health")
def health() -> dict:
    model_flags = {
        "layout": True,
        "handwriting": processor.router.handwriting_ocr.model is not None,
        "printed": processor.router.text_ocr.ocr is not None,
        "equation": processor.router.equation_ocr.pix2tex is not None,
    }
    return {
        "status": "ok",
        "models_loaded": any(model_flags.values()),
        "model_status": model_flags,
        "redis_enabled": job_store.enabled,
    }


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"image/png", "image/jpeg", "application/pdf"}:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    data = await file.read()
    job_id = str(uuid.uuid4())
    job_store.set(job_id, {"status": "queued", "result": None})
    background_tasks.add_task(_process_upload_job, job_id, data, file.filename)
    return {"job_id": job_id}


@app.get("/result/{job_id}")
def result(job_id: str) -> dict:
    item = job_store.get(job_id)
    if item is None:
        raise HTTPException(status_code=404, detail="job not found")
    return item


@app.post("/process")
def process(req: ProcessRequest) -> dict:
    try:
        return processor.process_base64(req.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _process_upload_job(job_id: str, data: bytes, filename: str) -> None:
    suffix = Path(filename).suffix or ".bin"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        if suffix.lower() == ".pdf":
            result = processor.process_pdf(path)
        else:
            result = processor.process_image(path)
        job_store.set(job_id, {"status": "done", "result": result})
    except Exception as exc:
        job_store.set(job_id, {"status": "failed", "error": str(exc)})


@app.post("/process/raw")
def process_raw(file: UploadFile = File(...)) -> dict:
    data = file.file.read()
    encoded = base64.b64encode(data).decode("utf-8")
    return processor.process_base64(encoded)
