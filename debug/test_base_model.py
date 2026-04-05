
import sys
import os

# Add parent directory
sys.path.append(os.getcwd())

from inference.ocr_text import TextOCR, OCRResult
from PIL import Image
import numpy as np

def test():
    print("Loading TrOCR Base model...")
    try:
        ocr = TextOCR(
            model_path="models/trocr_base_handwritten",
            device="cpu"
        )
        print("Model loaded successfully!")
        
        # Test inference if sample exists
        if os.path.exists("samples/page.png"):
            print("Running inference on sample...")
            # Just create a dummy image if needed, but page.png should exist
            img = Image.open("samples/page.png").crop((0,0, 500, 100)) # Top slice
            res = ocr.recognize(img)
            print(f"Prediction: {res.text}")
            print(f"Confidence: {res.confidence}")
        else:
            print("No sample image found, skipping inference test.")
            
    except Exception as e:
        print(f"FAILED to load model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test()
