"""Equation OCR combining Pix2Tex (primary) and UniMERNet fallback hooks."""

from __future__ import annotations

import re
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class EquationOCR:
    def __init__(self) -> None:
        self.pix2tex = None
        self.unimernet = None
        try:
            from pix2tex.cli import LatexOCR

            self.pix2tex = LatexOCR()
        except Exception:
            self.pix2tex = None
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self.uni_processor = AutoProcessor.from_pretrained("WenmuZhou/UniMERNet")
            self.unimernet = AutoModelForVision2Seq.from_pretrained("WenmuZhou/UniMERNet")
        except Exception:
            self.unimernet = None
            self.uni_processor = None

    def image_to_latex(self, image_crop: np.ndarray) -> str:
        prepared = self._prepare(image_crop)
        text = self._pix2tex(prepared)
        if self.validate_latex(text):
            return text
        fallback = self._unimernet(prepared)
        return fallback if self.validate_latex(fallback) else text

    def validate_latex(self, latex: str) -> bool:
        if not latex or len(latex.strip()) < 2:
            return False
        stack = []
        pairs = {"{": "}", "(": ")", "[": "]"}
        closers = {v: k for k, v in pairs.items()}
        for ch in latex:
            if ch in pairs:
                stack.append(ch)
            elif ch in closers:
                if not stack or stack[-1] != closers[ch]:
                    return False
                stack.pop()
        if stack:
            return False
        return bool(re.search(r"[=+\\^_]|\\frac|\\sqrt|\\int|\\sum", latex))

    def _prepare(self, image: np.ndarray, size: int = 384) -> Image.Image:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        max_side = max(h, w)
        canvas = np.full((max_side, max_side), 255, dtype=np.uint8)
        y0 = (max_side - h) // 2
        x0 = (max_side - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = gray
        resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 255.0 * 255).astype(np.uint8)
        return Image.fromarray(normalized)

    def _pix2tex(self, image: Image.Image) -> str:
        if self.pix2tex is None:
            return ""
        try:
            return str(self.pix2tex(image)).strip()
        except Exception:
            return ""

    def _unimernet(self, image: Image.Image) -> str:
        if self.unimernet is None or self.uni_processor is None:
            return ""
        try:
            inputs = self.uni_processor(images=image, return_tensors="pt")
            ids = self.unimernet.generate(**inputs)
            return self.uni_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        except Exception:
            return ""
