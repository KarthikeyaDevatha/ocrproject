import cv2
import numpy as np
import base64
import uuid
import time
import logging
from typing import List, Dict, Any

from backend.pipelines.preprocessing import preprocess_pipeline
from backend.pipelines.layout_detector import LayoutDetector
from backend.pipelines.ocr_router import OCRRouter
from backend.pipelines.symbol_corrector import SymbolCorrector
from backend.pipelines.semantic_parser import SemanticParser

# We can reuse models to avoid massive memory loading
class DocumentProcessor:
    def __init__(self, api_key: str = None):
        self.layout_detector = LayoutDetector()
        self.ocr_router = OCRRouter() # Lazy loads models
        self.symbol_corrector = SymbolCorrector(api_key=api_key)
        self.semantic_parser = SemanticParser(api_key=api_key)

    def _dict_to_document_output(self, doc_id: str, pages: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
        """Format final JSON conforming to the schema."""
        return {
            "document_id": doc_id,
            "pages": pages,
            "metadata": meta
        }

    def _process_single_image(self, image: np.ndarray, page_num: int = 1) -> Dict[str, Any]:
        """
        Runs the full pipeline to convert a numpy image to a structured page dict.
        """
        # 1. Preprocess
        preprocessed = preprocess_pipeline(image)
        
        # 2. Layout Detection
        regions = self.layout_detector.detect_regions(preprocessed)
        
        # 3. Route to specific OCR engines
        ocr_outputs = self.ocr_router.route_regions(preprocessed, regions)
        
        # 4. Correct Symbols
        for out, region in zip(ocr_outputs, regions):
            region_type = region.get("type", "text")
            # Map region type to context for corrector
            context = "math" if region_type in ["equation", "handwriting"] else "general"
            
            raw_text = out.get("latex") if out.get("latex") else out.get("text", "")
            corrected = self.symbol_corrector.correction_pipeline(raw_text, context=context)
            
            # Map back
            if out.get("latex"):
                out["latex"] = corrected
                # Text fallbacks as well
                out["text"] = corrected
            else:
                out["text"] = corrected

        # 5. Semantic Parsing (Document scale)
        # Groups blocks into a consistent logical structure. Using LLM wrapper.
        parsed_doc = self.semantic_parser.parse_document(ocr_outputs)
        
        # Construct exact block schema array
        blocks = []
        for out, region in zip(ocr_outputs, regions):
            # Best effort to map to valid schema via our OCR properties
            raw_text = out.get("text", "")
            latex = out.get("latex", "")
            b_type = region.get("type", "text")
            
            block = {
                "block_id": out.get("region_id", str(uuid.uuid4())),
                "type": b_type,
                "raw_text": raw_text,
                "latex": latex if latex else raw_text,
                "confidence": out.get("confidence", 0.0),
                "bounding_box": region.get("bounding_box", [])
            }
            # For specific question blocks extracted during LLM step
            # We would overlay the structured parsed_doc properties if they match.
            blocks.append(block)

        return {
            "page_number": page_num,
            "blocks": blocks,
            "parsed_structure": parsed_doc # Embedded raw parse
        }

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Load an image from disk and process it.
        """
        start = time.time()
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")

        doc_id = str(uuid.uuid4())
        page_dict = self._process_single_image(image, 1)
        
        # Extract meta from parsed structure if available
        parsed = page_dict.get("parsed_structure", {})
        meta = {
            "subject": parsed.get("subject", "unknown"),
            "grade_level": parsed.get("grade_level", "unknown"),
            "total_questions": len(parsed.get("questions", [])),
            "processing_time_ms": int((time.time() - start) * 1000)
        }
        
        # Clean up
        if "parsed_structure" in page_dict:
            del page_dict["parsed_structure"]
            
        return self._dict_to_document_output(doc_id, [page_dict], meta)

    def process_base64(self, b64_string: str) -> Dict[str, Any]:
        """
        Process a base64 encoded image string.
        """
        try:
            # Handle possible dataURI scheme prepended
            if "," in b64_string:
                b64_string = b64_string.split(",")[1]
                
            img_bytes = base64.b64decode(b64_string)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Invalid decoded image.")
                
            start = time.time()
            doc_id = str(uuid.uuid4())
            page_dict = self._process_single_image(image, 1)
            
            parsed = page_dict.get("parsed_structure", {})
            meta = {
                "subject": parsed.get("subject", "unknown"),
                "grade_level": parsed.get("grade_level", "unknown"),
                "total_questions": len(parsed.get("questions", [])),
                "processing_time_ms": int((time.time() - start) * 1000)
            }
            
            if "parsed_structure" in page_dict:
                del page_dict["parsed_structure"]
                
            return self._dict_to_document_output(doc_id, [page_dict], meta)
            
        except Exception as e:
            logging.error(f"Error processing base64 image: {e}")
            raise ValueError("Failed to process base64 string.") from e

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a multi-page PDF document.
        Requires fitz (PyMuPDF) or pdf2image.
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            logging.error("pdf2image not installed. Try pip install pdf2image.")
            raise
            
        start = time.time()
        # Convert first up to 10 pages for processing
        images = convert_from_path(pdf_path, dpi=200)[:10] 
        doc_id = str(uuid.uuid4())
        
        pages = []
        for idx, pil_img in enumerate(images):
            # Convert PIL to CV2 BGR
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            page_dict = self._process_single_image(cv_img, idx + 1)
            pages.append(page_dict)
            
        # Meta from first page
        parsed = pages[0].get("parsed_structure", {}) if pages else {}
        meta = {
            "subject": parsed.get("subject", "unknown"),
            "grade_level": parsed.get("grade_level", "unknown"),
            "total_questions": sum([len(p.get("parsed_structure", {}).get("questions", [])) for p in pages]),
            "processing_time_ms": int((time.time() - start) * 1000)
        }
        
        for p in pages:
            if "parsed_structure" in p:
                del p["parsed_structure"]
                
        return self._dict_to_document_output(doc_id, pages, meta)
