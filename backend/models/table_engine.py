"""Table extraction engine using Table Transformer with fallback."""

from __future__ import annotations

from typing import Any, List

import numpy as np


class TableOCR:
    def __init__(self) -> None:
        self.processor = None
        self.model = None
        try:
            from transformers import DetrImageProcessor, TableTransformerForObjectDetection

            self.processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
            self.model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            )
        except Exception:
            self.processor = None
            self.model = None

    def extract(self, image_crop: np.ndarray) -> str:
        boxes = self.extract_with_boxes(image_crop)
        if not boxes:
            return ""
        return "\n".join([f"table_cell_{idx}" for idx, _ in enumerate(boxes, start=1)])

    def extract_with_boxes(self, image_crop: np.ndarray) -> List[dict[str, Any]]:
        if self.model is None or self.processor is None:
            return []
        try:
            import torch
            from PIL import Image

            image = Image.fromarray(image_crop)
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            size = torch.tensor([image.size[::-1]])
            res = self.processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=size)[0]
            out = []
            for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                out.append(
                    {
                        "label": int(label.item()),
                        "confidence": float(score.item()),
                        "bbox": [round(v, 2) for v in box.tolist()],
                    }
                )
            return out
        except Exception:
            return []
