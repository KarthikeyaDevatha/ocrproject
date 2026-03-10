"""Scientific document OCR/understanding using Meta Nougat with fallback."""

from __future__ import annotations

import numpy as np


class NougatEngine:
    def __init__(self) -> None:
        self.processor = None
        self.model = None
        try:
            from transformers import AutoProcessor, VisionEncoderDecoderModel

            self.processor = AutoProcessor.from_pretrained("facebook/nougat-base")
            self.model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
        except Exception:
            self.processor = None
            self.model = None

    def parse(self, image: np.ndarray) -> str:
        if self.processor is None or self.model is None:
            return ""
        try:
            import torch
            from PIL import Image

            pixel_values = self.processor(images=Image.fromarray(image), return_tensors="pt").pixel_values
            with torch.no_grad():
                ids = self.model.generate(pixel_values, max_new_tokens=256)
            return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        except Exception:
            return ""
