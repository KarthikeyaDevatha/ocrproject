import numpy as np
from typing import List, Dict, Any

# We assume these models are implemented in backend.models module
# as per the architectural design. The OCR Router will lazily instantiate or 
# accept pre-instantiated engines to avoid memory bloat if some aren't used.

class OCRRouter:
    def __init__(self, models: Dict[str, Any] = None):
        """
        Initializes the OCR Router with a dictionary of instantiated engines.
        If models is None, it defaults to a lazy-loaded dict (to be implemented).
        """
        self.models = models or {}

    def get_engine(self, engine_name: str, module_path: str, class_name: str):
        """Lazy loader for engine instances."""
        if engine_name not in self.models:
            import importlib
            module = importlib.import_module(module_path)
            engine_class = getattr(module, class_name)
            self.models[engine_name] = engine_class()
        return self.models[engine_name]

    def crop_image(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Helper to crop the bounding box from the numpy image."""
        x1, y1, x2, y2 = bbox
        # Ensure bounds are within image dimensions
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]

    def route_regions(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Routes each region to the appropriate OCR engine based on region type.
        
        Args:
            image: The full preprocessed numpy image.
            regions: List of Regions from LayoutDetector.
            
        Returns:
            List of OCRResult objects.
        """
        results = []
        
        for region in regions:
            r_type = region.get("type", "text")
            bbox = region.get("bounding_box", [0, 0, image.shape[1], image.shape[0]])
            crop = self.crop_image(image, bbox)
            
            # Skip invalid crops
            if crop.size == 0:
                continue
                
            region_id = region.get("region_id", "unknown_id")
            result = {
                "region_id": region_id,
                "text": "",
                "latex": None,
                "confidence": region.get("confidence", 0.0)
            }
            
            try:
                if r_type == "text":
                    # Route to PaddleOCR
                    paddle_engine = self.get_engine("text", "backend.models.paddleocr_engine", "PrintedTextOCR")
                    result["text"] = paddle_engine.extract_text(crop)
                    
                elif r_type == "handwriting":
                    # Route to TrOCR
                    trocr_engine = self.get_engine("handwriting", "backend.models.trocr_engine", "HandwritingOCR")
                    result["text"] = trocr_engine.recognize_multiline(crop)
                    if isinstance(result["text"], list):
                        result["text"] = "\n".join(result["text"])
                        
                elif r_type == "equation":
                    # Route to Pix2Tex
                    eq_engine = self.get_engine("equation", "backend.models.equation_engine", "EquationOCR")
                    result["latex"] = eq_engine.image_to_latex(crop)
                    result["text"] = result["latex"]  # Fallback text maps to latex
                    
                elif r_type == "table":
                    # Route to Table Transformer
                    table_engine = self.get_engine("table", "backend.models.table_engine", "TableEngine")
                    result["text"] = table_engine.extract(crop)
                    
                elif r_type == "diagram":
                    # Route to DONUT
                    donut_engine = self.get_engine("diagram", "backend.models.donut_engine", "DonutEngine")
                    result["text"] = donut_engine.parse(crop)
                    
                # Optionally handle question_block and answer_block structurally, 
                # but typically they contain text/equations requiring OCR themselves
                elif r_type in ["question_block", "answer_block"]:
                    paddle_engine = self.get_engine("text", "backend.models.paddleocr_engine", "PrintedTextOCR")
                    result["text"] = paddle_engine.extract_text(crop)
                    
            except Exception as e:
                result["text"] = f"[OCR ERROR: {str(e)}]"
                
            results.append(result)
            
        return results
