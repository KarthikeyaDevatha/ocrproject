import os
import uuid
import logging
import base64
import shutil
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from redis import Redis
from rq import Queue
from rq.job import Job

# Our core processor
from backend.pipelines.document_processor import DocumentProcessor

app = FastAPI(title="Nexus OCR Scientific Document Intelligence API")

# Setup Redis Queue
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    redis_conn = Redis.from_url(REDIS_URL)
    job_queue = Queue("document_processing", connection=redis_conn)
    QUEUES_ENABLED = True
except Exception as e:
    logging.warning(f"Redis not available: {e}. Async /upload will process synchronously or fail.")
    job_queue = None
    QUEUES_ENABLED = False

# We initialize the processor globally (could be expensive without lazy loading)
processor = DocumentProcessor(api_key=os.getenv("LLM_API_KEY"))

class ProcessRequest(BaseModel):
    image_base64: str

def async_process_job(file_path: str, is_pdf: bool) -> Dict[str, Any]:
    """Background worker job."""
    try:
        if is_pdf:
            res = processor.process_pdf(file_path)
        else:
            res = processor.process_image(file_path)
            
        # Clean up tmp file
        os.remove(file_path)
        return res
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise RuntimeError(f"Processing failed: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": processor is not None,
        "queue_active": QUEUES_ENABLED
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Async processing point. Returns job_id."""
    if not QUEUES_ENABLED:
        raise HTTPException(status_code=503, detail="Redis queue not available for async processing.")
        
    file_id = str(uuid.uuid4())
    is_pdf = file.filename.lower().endswith(".pdf")
    ext = ".pdf" if is_pdf else ".jpg"
    tmp_path = f"/tmp/{file_id}{ext}"
    
    # Save file to disk for worker to pick up
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    job = job_queue.enqueue(async_process_job, args=(tmp_path, is_pdf), job_timeout=600)
    
    return {"job_id": job.get_id(), "status": "queued"}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    """Fetch structured JSON result for an async job."""
    if not QUEUES_ENABLED:
        raise HTTPException(status_code=503, detail="Redis queue not available.")

    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.is_finished:
        return {"job_id": job_id, "status": "completed", "result": job.result}
    elif job.is_failed:
        return {"job_id": job_id, "status": "failed", "error": str(job.exc_info)}
    else:
        return {"job_id": job_id, "status": "processing"}

@app.post("/process")
def process_sync(request: ProcessRequest):
    """Synchronous pipeline processing of base64 image."""
    try:
        result = processor.process_base64(request.image_base64)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Sync process error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
