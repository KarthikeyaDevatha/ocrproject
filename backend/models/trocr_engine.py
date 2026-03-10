"""Handwriting OCR powered by TrOCR with graceful fallback."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from PIL import Image


class HandwritingOCR:
    def __init__(self) -> None:
        self.processor = None
        self.model = None
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
        except Exception:
            self.processor = None
            self.model = None

    def recognize(self, image_crop: np.ndarray) -> str:
        image_crop = self._normalize(image_crop)
        if self.model is None or self.processor is None:
            return ""

        image = Image.fromarray(image_crop)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self._clean(text)

    def recognize_multiline(self, image_crop: np.ndarray) -> List[str]:
        lines = self._segment_lines(image_crop)
        return [self.recognize(line) for line in lines if line.size > 0]

    def _segment_lines(self, image_crop: np.ndarray) -> List[np.ndarray]:
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if image_crop.ndim == 3 else image_crop
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        projection = np.sum(bw > 0, axis=1)

        lines = []
        start = None
        for i, val in enumerate(projection):
            if val > 3 and start is None:
                start = i
            elif val <= 3 and start is not None:
                if i - start > 8:
                    lines.append(image_crop[start:i, :])
                start = None
        if start is not None:
            lines.append(image_crop[start:, :])
        return lines if lines else [image_crop]

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(text.replace("\n", " ").split())
