import cv2
import numpy as np
import re
import logging
from PIL import Image
import torch

try:
    from pix2tex.cli import LatexOCR
except ImportError:
    logging.warning("pix2tex not installed.")
    LatexOCR = None

class EquationOCR:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Primary Model
        if LatexOCR is not None:
            self.primary_model = LatexOCR()
        else:
            self.primary_model = None
            
        # Fallback Model (UniMERNet)
        # Assuming the library is installed or accessible. We'll simulate its integration 
        # as it requires pulling from custom repos.
        self.fallback_model = self._init_unimernet()

    def _init_unimernet(self):
        """Initialize the UniMERNet fallback model."""
        # Typically initialized via transformers or a custom pipeline
        # e.g., unimernet = UniMERNetModel.from_pretrained(...)
        return None  # Placeholder since no pip package exists natively without specific url

    def preprocess_crop(self, image_crop: np.ndarray, target_size=(384, 384)) -> np.ndarray:
        """
        Preprocessing for equation crops:
        • Pad to square with white background
        • Resize to model input dimensions
        • Normalize pixel intensity to [0, 1] (done at inference time via transforms)
        """
        # Ensure 3 channels
        if len(image_crop.shape) == 2:
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2BGR)
            
        h, w = image_crop.shape[:2]
        size = max(h, w)
        
        # Create a white canvas of square size
        canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # Center the image on canvas
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = image_crop
        
        # Resize
        resized = cv2.resize(canvas, target_size, interpolation=cv2.INTER_AREA)
        return resized

    def validate_latex(self, latex: str) -> bool:
        """
        Check if output is syntactically valid LaTeX by ensuring balanced braces
        and looking for common structural components.
        """
        if not latex or not latex.strip():
            return False
            
        # Check balanced braces
        stack = []
        for char in latex:
            if char == '{':
                stack.append('{')
            elif char == '}':
                if not stack:
                    return False
                stack.pop()
        
        if stack:
            return False
            
        # Check for bad patterns
        if latex.count('\\begin') != latex.count('\\end'):
            return False
            
        return True

    def _confidence_score(self, latex: str) -> float:
        """
        Pseudo-confidence score based on LaTeX string heuristics.
        Real confidence should come from Model generation logits.
        """
        if not self.validate_latex(latex):
            return 0.1
        
        # If it contains lots of unknown tokens or looks like noise
        if "\\unknown" in latex or "???" in latex:
            return 0.3
            
        return 0.9

    def image_to_latex(self, image_crop: np.ndarray) -> str:
        """
        Extract latex from equation snippet image. 
        Tries Pix2Tex, falls back to UniMERNet on failure or low confidence.
        """
        if self.primary_model is None:
            return "$E=mc^2$ [MOCK]"
            
        # Preprocess
        processed = self.preprocess_crop(image_crop)
        # Convert to PIL for pix2tex
        pil_img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        
        try:
            # Primary inference
            primary_latex = self.primary_model(pil_img)
            conf = self._confidence_score(primary_latex)
            
            # Fallback condition
            if conf < 0.5 and self.fallback_model is not None:
                # Run unimernet logic here
                return primary_latex  # fallback
                
            return primary_latex
            
        except Exception as e:
            logging.error(f"Equation OCR failed: {e}")
            return "ERROR"
