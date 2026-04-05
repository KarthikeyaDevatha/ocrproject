import cv2
import numpy as np
from typing import List, Union
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class HandwritingOCR:
    def __init__(self, model_name: str = "microsoft/trocr-large-handwritten"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def recognize(self, image_crop: np.ndarray) -> str:
        """
        Recognize text from a single line image crop.
        """
        if image_crop.size == 0:
            return ""
            
        # TrOCR expects RGB images
        if len(image_crop.shape) == 2:
            img = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2RGB)
        elif image_crop.shape[2] == 4:
            img = cv2.cvtColor(image_crop, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=128)
            
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()

    def segment_lines(self, image_crop: np.ndarray) -> List[np.ndarray]:
        """
        Naive line segmentation using horizontal projection profiles.
        In production, a robust text line segmenter like Craft or a 
        projection profile method is used. Here we use projection profiles.
        """
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop.copy()
            
        # Binarize
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection
        proj = np.sum(thresh, axis=1)
        
        # Find lines (where projection > threshold)
        threshold_val = np.max(proj) * 0.1
        line_indices = np.where(proj > threshold_val)[0]
        
        if len(line_indices) == 0:
            return [image_crop]
            
        lines = []
        start_idx = line_indices[0]
        
        for i in range(1, len(line_indices)):
            if line_indices[i] - line_indices[i - 1] > 10:  # Gap of > 10 pixels means new line
                end_idx = line_indices[i - 1]
                # Pad slightly to avoid cutting off ascenders/descenders
                y1 = max(0, start_idx - 5)
                y2 = min(image_crop.shape[0], end_idx + 5)
                lines.append(image_crop[y1:y2, :])
                start_idx = line_indices[i]
                
        # Add last line
        y1 = max(0, start_idx - 5)
        y2 = min(image_crop.shape[0], line_indices[-1] + 5)
        lines.append(image_crop[y1:y2, :])
        
        if not lines:
            return [image_crop]
            
        return lines

    def recognize_multiline(self, image_crop: np.ndarray) -> List[str]:
        """
        Segments the crop into lines, then runs TrOCR per line.
        """
        lines = self.segment_lines(image_crop)
        results = []
        for line in lines:
            text = self.recognize(line)
            if text:
                results.append(text)
        return results
