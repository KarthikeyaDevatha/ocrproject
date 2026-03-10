"""Layout detection with optional LayoutLMv3 backend and CV fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List
import uuid

import cv2
import numpy as np


@dataclass
class Region:
    region_id: str
    type: str
    bounding_box: list[int]
    confidence: float


class LayoutDetector:
    def __init__(self) -> None:
        self._layout_model = None
        self._layout_processor = None
        try:
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

            self._layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
            self._layout_model = LayoutLMv3ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base"
            )
        except Exception:
            self._layout_model = None

    def detect_regions(self, image: np.ndarray) -> List[Region]:
        """Detect regions. Uses CV contour segmentation with semantic heuristics."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        morph = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: list[Region] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 1200:
                continue
            rtype = self._heuristic_type(w, h, gray[y : y + h, x : x + w])
            regions.append(
                Region(
                    region_id=str(uuid.uuid4()),
                    type=rtype,
                    bounding_box=[int(x), int(y), int(x + w), int(y + h)],
                    confidence=0.7,
                )
            )

        if not regions:
            h, w = gray.shape[:2]
            regions.append(
                Region(
                    region_id=str(uuid.uuid4()),
                    type="text",
                    bounding_box=[0, 0, w, h],
                    confidence=0.5,
                )
            )
        return sorted(regions, key=lambda r: (r.bounding_box[1], r.bounding_box[0]))

    @staticmethod
    def _heuristic_type(w: int, h: int, crop: np.ndarray) -> str:
        aspect = w / max(h, 1)
        ink = float((crop < 150).sum()) / float(crop.size)
        if aspect > 6:
            return "question_block"
        if aspect < 1.3 and w > 120 and h > 120:
            return "diagram"
        if 2.0 < aspect < 6 and ink > 0.25:
            return "equation"
        if aspect > 2 and h < 90:
            return "handwriting"
        return "text"

    @staticmethod
    def to_dict(regions: List[Region]) -> list[dict]:
        return [asdict(r) for r in regions]
