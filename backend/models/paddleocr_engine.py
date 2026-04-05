import numpy as np
from typing import List, Dict, Any
import logging

try:
    from paddleocr import PaddleOCR
except ImportError:
    logging.warning("paddleocr not installed. PaddleOCR wrapper will fail on init.")
    PaddleOCR = None

class PrintedTextOCR:
    def __init__(self, lang: str = 'en'):
        if PaddleOCR is None:
            raise ImportError("Please install paddleocr: pip install paddleocr paddlepaddle")
            
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def extract_text(self, image_crop: np.ndarray) -> str:
        """
        Extract clean string text from image crop.
        Handles both single-line and multi-line.
        """
        result = self.ocr_engine.ocr(image_crop, cls=True)
        
        # Result is a list of lines, where each line is a list of [box, (text, confidence)]
        # Sometimes PaddleOCR returns a list of results per page, so result[0] is the page content.
        if not result or not result[0]:
            return ""
            
        page_results = result[0]
        extracted_lines = []
        
        for line in page_results:
            box = line[0]
            text, _ = line[1]
            extracted_lines.append(text)
            
        return "\n".join(extracted_lines)

    def extract_with_boxes(self, image_crop: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text along with bounding boxes and confidence scores.
        """
        result = self.ocr_engine.ocr(image_crop, cls=True)
        
        out = []
        if not result or not result[0]:
            return out
            
        page_results = result[0]
        
        for line in page_results:
            box = line[0]
            text, conf = line[1]
            out.append({
                "text": text,
                "box": box,
                "confidence": float(conf)
            })
            
        return out
