"""Printed text OCR wrapper around PaddleOCR."""

from __future__ import annotations

from typing import Any, List

import cv2
import numpy as np


class PrintedTextOCR:
    def __init__(self) -> None:
        self.ocr = None
        try:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception:
            self.ocr = None

    def extract_text(self, image_crop: np.ndarray) -> str:
        results = self.extract_with_boxes(image_crop)
        return "\n".join(item["text"] for item in results)

    def extract_with_boxes(self, image_crop: np.ndarray) -> List[dict[str, Any]]:
        if self.ocr is None:
            return []
        rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB) if image_crop.ndim == 3 else image_crop
        result = self.ocr.ocr(rgb, cls=True)
        out: List[dict[str, Any]] = []
        for line in result[0] if result else []:
            box, (text, conf) = line
            out.append({"text": text.strip(), "box": box, "confidence": float(conf)})
        return out
