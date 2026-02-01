
import sys
import os
import cv2
import numpy as np

# Add project root
sys.path.append(os.getcwd())

from inference.layout import LayoutDetector

def test_layout():
    print("Testing Layout Detection...")
    
    # Initialize detector
    detector = LayoutDetector(model_path="models/yolo/yolov8n-layout.onnx", device="cpu")
    
    if detector.session is None:
        print("Model not loaded (Fallback mode)")
    else:
        print("Model loaded successfully")

    # Create a dummy image with some "lines"
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    # Draw some black text lines
    for y in range(100, 700, 50):
        cv2.rectangle(img, (50, y), (550, y+20), (0, 0, 0), -1)
        
    # Run detection
    detections = detector.detect(img)
    print(f"Detections: {len(detections)}")
    for i, d in enumerate(detections):
        print(f" {i}: {d.class_name} ({d.confidence:.2f})")

if __name__ == "__main__":
    test_layout()
