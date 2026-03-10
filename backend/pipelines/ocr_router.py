"""Route each detected region to the specialized OCR engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import numpy as np

from backend.models.equation_engine import EquationOCR
from backend.models.paddleocr_engine import PrintedTextOCR
from backend.models.trocr_engine import HandwritingOCR
from backend.models.table_engine import TableOCR
from backend.models.nougat_engine import NougatEngine
from backend.pipelines.layout_detector import Region


@dataclass
class OCRResult:
    region_id: str
    text: str
    latex: str
    confidence: float


class OCRRouter:
    def __init__(self) -> None:
        self.text_ocr = PrintedTextOCR()
        self.handwriting_ocr = HandwritingOCR()
        self.equation_ocr = EquationOCR()
        self.table_ocr = TableOCR()
        self.nougat = NougatEngine()

    def route(self, image: np.ndarray, regions: List[Region]) -> List[OCRResult]:
        results: List[OCRResult] = []
        for region in regions:
            x1, y1, x2, y2 = region.bounding_box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            text = ""
            latex = ""
            confidence = region.confidence

            if region.type in {"text", "question_block", "answer_block"}:
                text = self.text_ocr.extract_text(crop)
            elif region.type == "handwriting":
                text = "\n".join(self.handwriting_ocr.recognize_multiline(crop)).strip()
            elif region.type == "equation":
                latex = self.equation_ocr.image_to_latex(crop)
                text = latex
            elif region.type == "table":
                text = self.table_ocr.extract(crop)
            elif region.type == "diagram":
                text = self.nougat.parse(crop)
            else:
                text = self.text_ocr.extract_text(crop)

            results.append(OCRResult(region_id=region.region_id, text=text, latex=latex, confidence=confidence))
        return results

    @staticmethod
    def to_dict(results: List[OCRResult]) -> list[dict]:
        return [asdict(r) for r in results]
