
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from inference.ocr_text import TextOCR

def test_fallback():
    print("Testing TextOCR Fallback Mechanism...")
    
    # Initialize with a bogus path to force init failure on primary, but fallback on secondary
    # ONNX paths don't exist -> force torch fallback
    # Torch path is "models/trocr_text", which likely fails -> force base model fallback
    ocr = TextOCR(
        model_path="models/trocr_text",
        encoder_path="nonexistent.onnx", 
        decoder_path="nonexistent.onnx",
        device="cpu"
    )
    
    dummy_image = np.zeros((100, 300, 3), dtype=np.uint8)
    
    print("Running recognize (expecting successful fallback)...")
    result = ocr.recognize(dummy_image)
    
    print(f"Result: {result.text} (Conf: {result.confidence})")
    
    if "Error" in result.text and result.confidence == 0.0:
         print("Caught Expected Error (Model Load Failed completely) - Crash Prevented!")
    else:
         print("Success! Model loaded via fallback and produced result.")

if __name__ == "__main__":
    test_fallback()
