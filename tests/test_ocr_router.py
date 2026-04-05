import pytest
import numpy as np
from backend.pipelines.ocr_router import OCRRouter

class MockPaddleEngine:
    def extract_text(self, crop):
        return "Mocked Text"

class MockTrOCREngine:
    def recognize_multiline(self, crop):
        return ["Mocked", "Handwriting"]
        
class MockEquationEngine:
    def image_to_latex(self, crop):
        return "E = mc^2"
        
def test_ocr_router_lazy_loading():
    # Setup mock models to avoid real HF instantiation
    mock_models = {
        "text": MockPaddleEngine(),
        "handwriting": MockTrOCREngine(),
        "equation": MockEquationEngine()
    }
    
    router = OCRRouter(models=mock_models)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    regions = [
        {"region_id": "r1", "type": "text", "bounding_box": [0, 0, 10, 10], "confidence": 0.9},
        {"region_id": "r2", "type": "handwriting", "bounding_box": [10, 10, 20, 20], "confidence": 0.8},
        {"region_id": "r3", "type": "equation", "bounding_box": [20, 20, 30, 30], "confidence": 0.95}
    ]
    
    results = router.route_regions(image, regions)
    
    assert len(results) == 3
    assert results[0]["text"] == "Mocked Text"
    assert results[1]["text"] == "Mocked\nHandwriting"
    assert results[2]["latex"] == "E = mc^2"
