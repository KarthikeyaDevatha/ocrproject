"""Master document processor orchestrating all pipeline stages."""

from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np

from backend.pipelines.layout_detector import LayoutDetector
from backend.pipelines.ocr_router import OCRRouter
from backend.pipelines.preprocessing import preprocess_pipeline
from backend.pipelines.semantic_parser import SemanticParser
from backend.pipelines.symbol_corrector import SymbolCorrector


class DocumentProcessor:
    def __init__(self) -> None:
        self.layout = LayoutDetector()
        self.router = OCRRouter()
        self.corrector = SymbolCorrector()
        self.semantic = SemanticParser()

    def process_image(self, image_path: str) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"unable to read image: {image_path}")
        return self._process_array(image)

    def process_pdf(self, pdf_path: str) -> List[dict]:
        import fitz

        doc = fitz.open(pdf_path)
        outputs = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            outputs.append(self._process_array(arr))
        return outputs

    def process_base64(self, b64_string: str) -> dict:
        blob = base64.b64decode(b64_string)
        arr = np.frombuffer(blob, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid base64 image")
        return self._process_array(image)

    def _process_array(self, image: np.ndarray) -> dict:
        start = time.time()
        preprocessed = preprocess_pipeline(image)
        if preprocessed.ndim == 2:
            routing_image = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        else:
            routing_image = preprocessed
        regions = self.layout.detect_regions(routing_image)
        ocr_results = self.router.route(routing_image, regions)

        for result in ocr_results:
            target = result.latex or result.text
            corrected = self.corrector.correction_pipeline(target, context="math")
            if result.latex:
                result.latex = corrected
            result.text = corrected

        structured = self.semantic.parse_document(ocr_results)
        structured["document_id"] = f"uuid-{uuid.uuid4()}"
        structured.setdefault("metadata", {})
        structured["metadata"]["processing_time_ms"] = int((time.time() - start) * 1000)
        return structured


def load_sample_image() -> str:
    sample = Path("samples/page.png")
    return str(sample)
