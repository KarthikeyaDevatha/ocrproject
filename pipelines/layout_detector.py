import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np

class LayoutDetector:
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base", device: str = "cpu"):
        self.device = device
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name).to(self.device)

    def detect_layout(self, image: Image.Image) -> List[Dict[str, Tuple[int, int, int, int]]]:
        """
        Detect layout regions in the document image.

        Args:
            image (PIL.Image.Image): Input document image.

        Returns:
            List[Dict[str, Tuple[int, int, int, int]]]: Detected regions with bounding boxes and labels.
        """
        # Preprocess image
        encoding = self.processor(image, return_tensors="pt").to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**encoding)

        # Extract predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

        # Map predictions to bounding boxes
        boxes = encoding["bbox"].squeeze().cpu().numpy()
        results = []
        for box, pred in zip(boxes, predictions):
            label = self.model.config.id2label[pred]
            if label != "O":  # Ignore non-entity labels
                results.append({"label": label, "bbox": tuple(box)})

        return results

if __name__ == "__main__":
    from PIL import Image
    import sys

    if len(sys.argv) < 2:
        print("Usage: python layout_detector.py path/to/image")
        sys.exit(1)

    image_path = sys.argv[1]
    image = Image.open(image_path).convert("RGB")

    detector = LayoutDetector()
    layout = detector.detect_layout(image)

    print("Detected Layout:")
    for region in layout:
        print(region)