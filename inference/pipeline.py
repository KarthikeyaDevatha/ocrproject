"""
Main OCR Pipeline - End-to-end inference for handwritten text and math.
Integrates layout detection, text OCR, math OCR, and document reconstruction.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.preprocess import (
    preprocess_for_layout,
    preprocess_for_ocr,
    crop_region,
    deskew_image,
    remove_underlines,
    boost_dots,
    # New advanced preprocessing
    full_preprocessing_pipeline,
    remove_noise,
    deskew_hough,
    normalize_resolution,
    detect_baselines,
    segment_text_lines
)
from inference.layout import (
    LayoutDetector,
    Detection,
    cluster_lines,
    merge_close_boxes,
    draw_detections
)
from inference.ocr_text import TextOCR, create_text_ocr
from inference.ocr_math import MathOCR, create_math_ocr
from inference.reconstruct import (
    DocumentReconstructor,
    LineResult,
    cleanup_latex
)

# Optional: Post-processing modules (graceful fallback if not available)
try:
    from inference.postprocessor import OCRPostProcessor
    from inference.lexicon import AcademicLexicon, VocabularyConstrainedDecoder
    from inference.hybrid_decoder import ConfidenceScorer, MultiPassDecoder
    HAS_POSTPROCESSOR = True
except ImportError:
    HAS_POSTPROCESSOR = False


@dataclass
class PipelineConfig:
    """Configuration for OCR pipeline."""
    # Model paths
    yolo_model: str = "models/yolo/yolov8n-layout.onnx"
    text_model_dir: str = "models/trocr_text"
    math_model_dir: str = "models/trocr_math"
    
    # Device
    device: str = "cpu"
    
    # Layout detection
    layout_conf_threshold: float = 0.5
    layout_iou_threshold: float = 0.45
    layout_input_size: int = 640
    
    # OCR
    max_new_tokens: int = 100
    num_beams: int = 1
    ocr_timeout: float = 5.0
    confidence_threshold: float = 0.4
    
    # Rerouting
    enable_rerouting: bool = True
    
    # Output
    debug_output: bool = False
    save_debug_images: bool = False


@dataclass
class PipelineResult:
    """Result from OCR pipeline."""
    markdown: str
    detections: List[Detection] = field(default_factory=list)
    line_results: List[LineResult] = field(default_factory=list)
    total_time: float = 0.0
    layout_time: float = 0.0
    ocr_time: float = 0.0
    num_text_regions: int = 0
    num_math_regions: int = 0
    debug_image: Optional[np.ndarray] = None


class OCRPipeline:
    """
    Complete OCR pipeline for handwritten notes.
    
    Pipeline steps:
    1. Preprocess (deskew, normalize)
    2. Layout detection (YOLO)
    3. Line clustering and merging
    4. Route to text/math OCR
    5. Confidence checking and rerouting
    6. Post-processing
    7. Document reconstruction
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        print("Initializing OCR Pipeline...")
        
        # Initialize layout detector
        print("Loading layout detector...")
        self.layout_detector = LayoutDetector(
            model_path=self.config.yolo_model,
            conf_threshold=self.config.layout_conf_threshold,
            iou_threshold=self.config.layout_iou_threshold,
            input_size=self.config.layout_input_size,
            device=self.config.device
        )
        
        # Initialize text OCR
        print("Loading text OCR...")
        self.text_ocr = create_text_ocr(
            model_dir=self.config.text_model_dir,
            device=self.config.device
        )
        self.text_ocr.max_new_tokens = self.config.max_new_tokens
        self.text_ocr.timeout = self.config.ocr_timeout
        self.text_ocr.confidence_threshold = self.config.confidence_threshold
        
        # Initialize math OCR
        print("Loading math OCR...")
        self.math_ocr = create_math_ocr(
            model_dir=self.config.math_model_dir,
            device=self.config.device
        )
        self.math_ocr.max_new_tokens = self.config.max_new_tokens
        self.math_ocr.timeout = self.config.ocr_timeout
        self.math_ocr.confidence_threshold = self.config.confidence_threshold
        
        # Initialize reconstructor
        self.reconstructor = DocumentReconstructor()
        
        print("Pipeline initialized.")
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and validate image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        return image
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to image."""
        # Deskew if needed
        deskewed = deskew_image(image)
        return deskewed
    
    def _run_layout_detection(
        self,
        image: np.ndarray
    ) -> List[Detection]:
        """Run layout detection and post-processing."""
        # Detect regions
        detections = self.layout_detector.detect(image)
        
        # Cluster into lines
        clusters = cluster_lines(detections)
        
        # Merge close boxes within clusters
        merged_detections = []
        for cluster in clusters:
            merged = merge_close_boxes(cluster)
            merged_detections.extend(merged)
        
        return merged_detections
    
    def _route_and_recognize(
        self,
        image: np.ndarray,
        detections: List[Detection]
    ) -> List[LineResult]:
        """
        Route detections to appropriate OCR and recognize.
        
        Args:
            image: Original image
            detections: Layout detections
        
        Returns:
            List of LineResult objects
        """
        results = []
        
        for det in detections:
            # Crop region
            region = crop_region(image, det.bbox, padding=5)
            
            if region.size == 0:
                continue
            
            # Convert BGR to RGB for OCR
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            
            # Apply Cleaning (Accuracy Phase)
            # Only for Text regions really, but harmless for Math?
            # Math might have fraction bars which are horizontal lines. DANGER.
            # Only apply if class is text_line.
            
            if det.class_name == "text_line":
                region_rgb = remove_underlines(region_rgb)
                region_rgb = boost_dots(region_rgb)
            
            # Route based on class
            if det.class_name == "text_line":
                ocr_result = self.text_ocr.recognize(region_rgb)
                
                # Reroute if low confidence and might be math
                if self.config.enable_rerouting and ocr_result.rerouted:
                    math_result = self.math_ocr.recognize(region_rgb)
                    if not math_result.discarded and math_result.confidence > ocr_result.confidence:
                        results.append(LineResult(
                            text=math_result.latex,
                            line_type='math',
                            confidence=math_result.confidence,
                            bbox=det.bbox,
                            is_display_math=math_result.is_display_math
                        ))
                        continue
                
                results.append(LineResult(
                    text=ocr_result.text,
                    line_type='text',
                    confidence=ocr_result.confidence,
                    bbox=det.bbox
                ))
            
            elif det.class_name == "math_formula":
                math_result = self.math_ocr.recognize(region_rgb)
                
                # Reroute if discarded, try text
                if self.config.enable_rerouting and math_result.discarded:
                    text_result = self.text_ocr.recognize(region_rgb)
                    if not text_result.rerouted and text_result.confidence > math_result.confidence:
                        results.append(LineResult(
                            text=text_result.text,
                            line_type='text',
                            confidence=text_result.confidence,
                            bbox=det.bbox
                        ))
                        continue
                
                latex = cleanup_latex(math_result.latex)
                results.append(LineResult(
                    text=latex,
                    line_type='math',
                    confidence=math_result.confidence,
                    bbox=det.bbox,
                    is_display_math=math_result.is_display_math
                ))
        
        return results
    
    def process(
        self,
        image_path: str,
        title: Optional[str] = None
    ) -> PipelineResult:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            title: Optional document title
        
        Returns:
            PipelineResult with markdown output and metadata
        """
        start_time = time.time()
        
        # Load image
        image = self._load_image(image_path)
        
        # Preprocess
        preprocessed = self._preprocess_image(image)
        
        # Layout detection
        layout_start = time.time()
        detections = self._run_layout_detection(preprocessed)
        layout_time = time.time() - layout_start
        
        # Count regions
        num_text = sum(1 for d in detections if d.class_name == "text_line")
        num_math = sum(1 for d in detections if d.class_name == "math_formula")
        
        print(f"Detected {num_text} text regions and {num_math} math regions")
        
        # OCR
        ocr_start = time.time()
        line_results = self._route_and_recognize(preprocessed, detections)
        ocr_time = time.time() - ocr_start
        
        # Reconstruct document
        markdown = self.reconstructor.reconstruct(line_results, title=title)
        
        total_time = time.time() - start_time
        
        # Create debug image if requested
        debug_image = None
        if self.config.debug_output or self.config.save_debug_images:
            debug_image = draw_detections(preprocessed, detections)
        
        return PipelineResult(
            markdown=markdown,
            detections=detections,
            line_results=line_results,
            total_time=total_time,
            layout_time=layout_time,
            ocr_time=ocr_time,
            num_text_regions=num_text,
            num_math_regions=num_math,
            debug_image=debug_image
        )
    
    def process_batch(
        self,
        image_paths: List[str]
    ) -> List[PipelineResult]:
        """Process multiple images."""
        results = []
        for path in image_paths:
            try:
                result = self.process(path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append(PipelineResult(
                    markdown=f"[Error: {e}]",
                    total_time=0
                ))
        return results


def main():
    """Command-line interface for OCR pipeline."""
    parser = argparse.ArgumentParser(
        description="OCR Pipeline for Handwritten Text and Math"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/demo_result.md",
        help="Path to output markdown file"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Document title"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--save-debug-image",
        type=str,
        help="Path to save debug visualization"
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="models/yolo/yolov8n-layout.onnx",
        help="Path to YOLO ONNX model"
    )
    parser.add_argument(
        "--text-model-dir",
        type=str,
        default="models/trocr_text",
        help="Path to text OCR model directory"
    )
    parser.add_argument(
        "--math-model-dir",
        type=str,
        default="models/trocr_math",
        help="Path to math OCR model directory"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="OCR confidence threshold"
    )
    parser.add_argument(
        "--no-rerouting",
        action="store_true",
        help="Disable confidence-based rerouting"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam search width (default: 1 for speed)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig(
        yolo_model=args.yolo_model,
        text_model_dir=args.text_model_dir,
        math_model_dir=args.math_model_dir,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        enable_rerouting=not args.no_rerouting,
        num_beams=args.num_beams,
        debug_output=args.debug,
        save_debug_images=args.save_debug_image is not None
    )
    
    # Initialize pipeline
    pipeline = OCRPipeline(config)
    
    # Process image
    print(f"\nProcessing: {args.image}")
    result = pipeline.process(args.image, title=args.title)
    
    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(result.markdown)
    
    print(f"\nOutput saved to: {args.output}")
    
    # Save debug image if requested
    if args.save_debug_image and result.debug_image is not None:
        cv2.imwrite(args.save_debug_image, result.debug_image)
        print(f"Debug image saved to: {args.save_debug_image}")
    
    # Print stats
    print(f"\n--- Pipeline Statistics ---")
    print(f"Total time: {result.total_time:.2f}s")
    print(f"Layout detection: {result.layout_time:.2f}s")
    print(f"OCR processing: {result.ocr_time:.2f}s")
    print(f"Text regions: {result.num_text_regions}")
    print(f"Math regions: {result.num_math_regions}")
    
    if args.debug:
        print(f"\n--- Output Preview ---")
        print(result.markdown[:500] + "..." if len(result.markdown) > 500 else result.markdown)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
