"""
Hybrid Pipeline Orchestrator v2.1 for Adaptive OCR Agent.

FIXES FROM EVALUATOR FEEDBACK:
1. Strict confidence thresholds (0.85/0.65) with unambiguous labels
2. Math-aware semantic validation with arithmetic consistency checks
3. Improved line segmentation (padding, preprocessing on line crops)
4. Smart engine routing (handwritten→TrOCR, math-heavy→Mathpix, mixed→both)
5. Multi-engine voting/fusion for RETRY_REQUIRED lines
6. Clear output labels: ACCEPTED / RETRY_REQUIRED / FAILED_EXTRACTION

Master pipeline combining:
- Decision Engine (feature-based routing)
- CRAFT/Contour Line Detection with improved padding
- Dual Preprocessing Profiles (CLAHE clean/degraded)
- TrOCR-large (local handwriting OCR)
- Mathpix API (cloud math OCR)
- Arithmetic Pipeline (digit extraction)
- Composite Confidence Gate v2 (strict thresholds + math validation)
- Voting/Fusion Layer (multi-engine consensus)
- Enhanced Post-Processing (spell + domain + math)
- Pipeline Logging
"""

import os
import cv2
import time
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Local modules
from inference.decision_engine import DecisionEngine, EngineDecision
from inference.preprocessing_profiles import full_preprocess, apply_profile, to_rgb_for_ocr
from inference.line_detector import LineDetector
from inference.confidence_gate import CompositeConfidenceGate, ConfidenceResult, MathValidator
from inference.mathpix_ocr import MathpixOCR, MathpixResult
from inference.enhanced_postprocessor import PostProcessor
from inference.pipeline_logger import PipelineLogger


@dataclass
class LineResult:
    """Result for a single detected text line."""
    text: str
    confidence: float
    tag: str  # ACCEPTED | RETRY_REQUIRED | FAILED_EXTRACTION
    engine_used: str
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    retried: bool = False
    fused: bool = False
    math_validated: bool = False
    confidence_details: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    text: str
    lines: List[LineResult]
    overall_confidence: float
    overall_tag: str
    engine_used: str
    profile_used: str
    mode: str
    features: dict
    log: str
    processing_time: float
    stats: dict = field(default_factory=dict)
    mathpix_result: Optional[dict] = None
    math_validation: Optional[dict] = None


class HybridPipeline:
    """
    Adaptive OCR Agent v2.1 — 4-mode orchestrator with semantic validation.
    
    Modes:
    - auto: Analyze image → smart routing → confidence gate → retry/fuse → validate
    - mathpix: Force Mathpix API (requires credentials)
    - trocr: Force TrOCR-large local inference
    - arithmetic: Force ArithmeticPipeline for digit extraction
    """

    def __init__(
        self,
        mode: str = "auto",
        trocr_model: str = "large",  # "base" or "large"
        verbose: bool = True
    ):
        self.mode = mode
        self.trocr_model_name = (
            "microsoft/trocr-large-handwritten" if trocr_model == "large"
            else "microsoft/trocr-base-handwritten"
        )
        
        # Components
        self.decision_engine = DecisionEngine()
        self.line_detector = LineDetector(use_craft=True, pad=10)  # Increased padding
        self.confidence_gate = CompositeConfidenceGate()
        self.math_validator = MathValidator()
        self.mathpix = MathpixOCR()
        self.postprocessor = PostProcessor()
        self.logger = PipelineLogger(verbose=verbose)
        
        # Lazy-loaded models
        self._trocr_processor = None
        self._trocr_model = None
        self._trocr_device = None
        self._torch = None

    # ========================================================================
    # MODEL LOADING
    # ========================================================================

    def _load_trocr(self):
        """Lazy-load TrOCR model with transformers 5.x compatibility."""
        if self._trocr_model is not None:
            return
        
        import torch
        from transformers import VisionEncoderDecoderModel
        
        self._torch = torch
        self.logger.info(f"Loading TrOCR: {self.trocr_model_name}", "MODEL")
        
        # Load processor with transformers 5.x compatibility
        try:
            from transformers import TrOCRProcessor
            self._trocr_processor = TrOCRProcessor.from_pretrained(
                self.trocr_model_name, use_fast=False
            )
        except (ValueError, ImportError):
            from transformers import TrOCRProcessor, AutoImageProcessor, XLMRobertaTokenizer
            self.logger.warn("Fast tokenizer failed, using slow tokenizer", "MODEL")
            image_processor = AutoImageProcessor.from_pretrained(self.trocr_model_name)
            tokenizer = XLMRobertaTokenizer.from_pretrained(self.trocr_model_name)
            self._trocr_processor = TrOCRProcessor(
                image_processor=image_processor, tokenizer=tokenizer
            )
        
        self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
            self.trocr_model_name
        )
        
        if torch.cuda.is_available():
            self._trocr_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._trocr_device = "mps"
        else:
            self._trocr_device = "cpu"
            
        self._trocr_model = self._trocr_model.to(self._trocr_device)
        self._trocr_model.eval()
        
        self.logger.info(f"TrOCR loaded on {self._trocr_device}", "MODEL")

    # ========================================================================
    # PREPROCESSING ON LINE CROPS (FIX #3: Better line capture)
    # ========================================================================

    def _preprocess_crop(self, pil_image: Image.Image, profile: str = "clean") -> Image.Image:
        """
        Apply contrast enhancement to individual line crops.
        Improves TrOCR accuracy on degraded/thin handwriting.
        """
        img_np = np.array(pil_image)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE to crop for better contrast; skip deskew for lines
        enhanced = full_preprocess(img_np, profile, do_deskew=False)
        
        return Image.fromarray(enhanced)

    # ========================================================================
    # TrOCR INFERENCE
    # ========================================================================

    def _run_trocr(
        self, pil_image: Image.Image
    ) -> Tuple[str, Optional[list], float]:
        """
        Run TrOCR on a single line crop.
        
        Returns:
            Tuple of (decoded_text, score_tensors, raw_confidence)
        """
        self._load_trocr()
        
        # Ensure RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        pixel_values = self._trocr_processor(
            pil_image, return_tensors="pt"
        ).pixel_values.to(self._trocr_device)
        
        with self._torch.no_grad():
            outputs = self._trocr_model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=128
            )
        
        generated_ids = outputs.sequences
        scores = outputs.scores if hasattr(outputs, 'scores') else None
        
        text = self._trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return text, scores, 0.0

    # ========================================================================
    # ENGINE DISPATCH
    # ========================================================================

    def _run_engine(
        self,
        engine: str,
        pil_image: Image.Image,
        image_path: Optional[str] = None
    ) -> Tuple[str, Optional[list]]:
        """Run a specific OCR engine on an image."""
        if engine == "mathpix" and image_path:
            result = self.mathpix.recognize_image(image_path)
            if result.error:
                self.logger.warn(f"Mathpix error: {result.error}", "OCR")
                return "", None
            return result.text, None
        
        elif engine == "trocr":
            text, scores, _ = self._run_trocr(pil_image)
            return text, scores
        
        elif engine == "arithmetic":
            try:
                from inference.arithmetic_pipeline import ArithmeticPipeline
                pipeline = ArithmeticPipeline()
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    pil_image.save(f.name)
                    result = pipeline.process(f.name)
                    os.unlink(f.name)
                    return result.get("final_expression", ""), None
            except Exception as e:
                self.logger.error(f"Arithmetic pipeline error: {e}", "OCR")
                return "", None
        
        else:
            text, scores, _ = self._run_trocr(pil_image)
            return text, scores

    def _get_fallback_engine(self, current_engine: str) -> Optional[str]:
        """Get fallback engine for retry."""
        fallback_map = {
            "mathpix": "trocr",
            "trocr": "mathpix" if self.mathpix.is_available else None,
            "arithmetic": "trocr",
        }
        return fallback_map.get(current_engine)

    # ========================================================================
    # SMART ENGINE ROUTING (FIX #4)
    # ========================================================================

    def _smart_route(
        self,
        features: dict,
        available_engines: List[str]
    ) -> str:
        """
        Smart engine routing based on image content analysis.
        
        Rules:
        - handwritten text only → TrOCR
        - math-heavy content → Mathpix (if available)
        - mixed content → run both engines
        - simple arithmetic → Arithmetic pipeline
        """
        math_density = features.get("math_density", 0)
        is_arithmetic = features.get("is_arithmetic", False)
        
        if is_arithmetic and "arithmetic" in available_engines:
            return "arithmetic"
        elif math_density > 0.3 and "mathpix" in available_engines:
            return "mathpix"
        elif math_density > 0.15 and "mathpix" in available_engines:
            return "both"  # Run both engines, fuse results
        else:
            return "trocr"

    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================

    def process(
        self,
        image_path: str,
        mode: Optional[str] = None,
        force_profile: Optional[str] = None
    ) -> PipelineResult:
        """
        Main pipeline entry point.
        
        Args:
            image_path: Path to input image
            mode: Override mode (auto/mathpix/trocr/arithmetic)
            force_profile: Override preprocessing profile
        
        Returns:
            PipelineResult with text, confidence, tags, and log
        """
        self.logger.reset()
        start_time = time.time()
        active_mode = mode or self.mode
        
        self.logger.info(f"Image: {os.path.basename(image_path)}", "INPUT")
        self.logger.info(f"Mode: {active_mode.upper()}", "INPUT")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return PipelineResult(
                text="", lines=[], overall_confidence=0.0,
                overall_tag="FAILED_EXTRACTION", engine_used="none",
                profile_used="none", mode=active_mode,
                features={}, log=self.logger.format_for_display(),
                processing_time=time.time() - start_time,
                stats={"error": f"Cannot read image: {image_path}"}
            )
        
        # ---- MATHPIX-ONLY MODE ----
        if active_mode == "mathpix":
            return self._process_mathpix(image_path, image, start_time)
        
        # ---- ARITHMETIC MODE ----
        if active_mode == "arithmetic":
            return self._process_arithmetic(image_path, image, start_time)
        
        # ---- AUTO / TROCR MODE ----
        # Step 1: Feature extraction + decision
        available_engines = ["trocr"]
        if self.mathpix.is_available:
            available_engines.append("mathpix")
        available_engines.append("arithmetic")
        
        decision = self.decision_engine.decide(image, available_engines)
        self.logger.log_decision(
            decision.engine, decision.profile,
            decision.reason, decision.features
        )
        
        # Handle blank image
        if decision.engine == "none":
            return PipelineResult(
                text="No readable content detected.",
                lines=[], overall_confidence=0.0,
                overall_tag="FAILED_EXTRACTION", engine_used="none",
                profile_used=decision.profile, mode=active_mode,
                features=decision.features,
                log=self.logger.format_for_display(),
                processing_time=time.time() - start_time,
                stats={"error": "Blank image"}
            )
        
        # Override profile if requested
        profile = force_profile or decision.profile
        
        # Smart routing (FIX #4)
        if active_mode == "auto":
            engine = self._smart_route(decision.features, available_engines)
        else:
            engine = "trocr"
        
        # Special case: Mathpix for full image (not per-line)
        if engine == "mathpix":
            return self._process_mathpix(image_path, image, start_time,
                                         decision=decision, profile=profile)
        
        # Step 2: Preprocess
        preprocessed = full_preprocess(image, profile)
        
        # Step 3: Line detection with improved padding
        self.logger.info("Detecting text lines", "DETECT")
        line_crops = self.line_detector.detect_and_crop(
            image_path, image=image
        )
        self.logger.info(f"Detected {len(line_crops)} text lines", "DETECT")
        
        # Step 4: Per-line OCR with confidence gate + retry + fusion
        line_results = []
        all_texts = []
        run_both = (engine == "both")
        primary_engine = "trocr"
        
        for idx, (crop_pil, bbox) in enumerate(line_crops):
            # Preprocess individual crop (FIX #3)
            enhanced_crop = self._preprocess_crop(crop_pil, profile)
            
            # Run primary engine
            text, scores = self._run_engine(primary_engine, enhanced_crop, image_path)
            confidence = self.confidence_gate.score(text, scores)
            
            retried = False
            fused = False
            final_engine = primary_engine
            
            # ---- MULTI-ENGINE VOTING/FUSION (FIX #5) ----
            if run_both and self.mathpix.is_available:
                # Run both engines unconditionally for mixed content
                alt_text, alt_scores = self._run_engine("mathpix", enhanced_crop, image_path)
                alt_confidence = self.confidence_gate.score(alt_text, alt_scores)
                
                # Fuse results
                text, confidence = self.confidence_gate.fuse_results(
                    confidence, text, alt_confidence, alt_text
                )
                fused = True
                self.logger.info(
                    f"Line {idx}: Fused TrOCR+Mathpix → {confidence.tag}", "FUSION"
                )
            
            # ---- RETRY LOGIC (FIX #1: Only retry RETRY_REQUIRED) ----
            elif confidence.tag == "RETRY_REQUIRED":
                fallback = self._get_fallback_engine(primary_engine)
                if fallback:
                    self.logger.log_retry(
                        idx, primary_engine, fallback, confidence.composite
                    )
                    alt_text, alt_scores = self._run_engine(
                        fallback, enhanced_crop, image_path
                    )
                    alt_confidence = self.confidence_gate.score(
                        alt_text, alt_scores
                    )
                    
                    # Fuse instead of simple pick (FIX #5)
                    text, confidence = self.confidence_gate.fuse_results(
                        confidence, text, alt_confidence, alt_text
                    )
                    retried = True
                    fused = True
            
            self.logger.log_confidence(
                idx, confidence.composite, confidence.tag,
                final_engine, retried
            )
            
            line_results.append(LineResult(
                text=text,
                confidence=confidence.composite,
                tag=confidence.tag,
                engine_used=final_engine,
                bbox=bbox,
                retried=retried,
                fused=fused,
                math_validated=confidence.math_validated,
                confidence_details={
                    "token_confidence": confidence.token_confidence,
                    "alpha_ratio": confidence.alpha_ratio,
                    "length_ok": confidence.length_ok,
                    "math_boost": confidence.math_boost,
                    "math_validated": confidence.math_validated,
                    "numbers_found": confidence.numbers_found,
                }
            ))
            
            if confidence.tag != "FAILED_EXTRACTION":
                all_texts.append(text)
        
        # Step 5: Post-processing
        processed_texts, pp_stats = self.postprocessor.process_lines(all_texts)
        self.logger.log_postprocess(
            pp_stats.get("spell_corrections", 0),
            pp_stats.get("lines_merged", 0)
        )
        
        # Step 6: Assemble output
        final_text = "\n".join(processed_texts)
        
        # Step 7: Full-document math validation (FIX #2)
        doc_math_boost, doc_math_details = self.math_validator.validate_arithmetic(final_text)
        
        # Overall confidence = mean of non-failed lines + doc-level math boost
        valid_scores = [lr.confidence for lr in line_results if lr.tag != "FAILED_EXTRACTION"]
        overall_confidence = (
            sum(valid_scores) / len(valid_scores)
        ) if valid_scores else 0.0
        
        # Apply document-level math boost
        overall_confidence = max(0.0, min(1.0, overall_confidence + doc_math_boost))
        
        # Overall tag with strict thresholds
        overall_tag = self.confidence_gate.classify(overall_confidence)
        
        processing_time = time.time() - start_time
        self.logger.log_final(overall_tag, processing_time, len(line_results))
        
        return PipelineResult(
            text=final_text,
            lines=line_results,
            overall_confidence=round(overall_confidence, 4),
            overall_tag=overall_tag,
            engine_used=primary_engine if not run_both else "trocr+mathpix",
            profile_used=profile,
            mode=active_mode,
            features=decision.features,
            log=self.logger.format_for_display(),
            processing_time=round(processing_time, 3),
            stats={
                "lines_detected": len(line_crops),
                "lines_accepted": sum(1 for lr in line_results if lr.tag == "ACCEPTED"),
                "lines_retry": sum(1 for lr in line_results if lr.tag == "RETRY_REQUIRED"),
                "lines_failed": sum(1 for lr in line_results if lr.tag == "FAILED_EXTRACTION"),
                "retries": sum(1 for lr in line_results if lr.retried),
                "fusions": sum(1 for lr in line_results if lr.fused),
                "math_validations": sum(1 for lr in line_results if lr.math_validated),
                "postprocessing": pp_stats
            },
            math_validation={
                "document_level": doc_math_details,
                "document_boost": round(doc_math_boost, 4),
                "numbers_in_output": doc_math_details.get("numbers_found", [])
            }
        )

    def _process_mathpix(
        self,
        image_path: str,
        image: np.ndarray,
        start_time: float,
        decision: Optional[EngineDecision] = None,
        profile: str = "clean"
    ) -> PipelineResult:
        """Process using Mathpix API (full image, not per-line)."""
        self.logger.info("Using Mathpix API for full image", "OCR")
        
        result = self.mathpix.recognize_image(image_path)
        
        if result.error:
            self.logger.error(f"Mathpix failed: {result.error}", "OCR")
            
            # Fallback to TrOCR
            self.logger.info("Falling back to TrOCR", "RETRY")
            return self.process(image_path, mode="trocr")
        
        confidence = result.confidence_rate or result.confidence
        tag = self.confidence_gate.classify(confidence)
        
        if result.is_mock:
            self.logger.info("[MOCK] Mathpix response (demo mode)", "OCR")
        
        self.logger.log_confidence(0, confidence, tag, "mathpix")
        
        processing_time = time.time() - start_time
        self.logger.log_final(tag, processing_time, 1)
        
        features = decision.features if decision else {}
        
        return PipelineResult(
            text=result.text,
            lines=[LineResult(
                text=result.text,
                confidence=confidence,
                tag=tag,
                engine_used="mathpix"
            )],
            overall_confidence=round(confidence, 4),
            overall_tag=tag,
            engine_used="mathpix",
            profile_used=profile,
            mode="mathpix",
            features=features,
            log=self.logger.format_for_display(),
            processing_time=round(processing_time, 3),
            stats={"is_mock": result.is_mock},
            mathpix_result={
                "text": result.text,
                "latex_styled": result.latex_styled,
                "html": result.html,
                "line_data": result.line_data
            }
        )

    def _process_arithmetic(
        self,
        image_path: str,
        image: np.ndarray,
        start_time: float
    ) -> PipelineResult:
        """Process using ArithmeticPipeline."""
        self.logger.info("Using ArithmeticPipeline", "OCR")
        
        try:
            from inference.arithmetic_pipeline import ArithmeticPipeline
            pipeline = ArithmeticPipeline()
            result = pipeline.process(image_path)
            
            text = result.get("final_expression", "")
            confidence = result.get("confidence", 0.0)
            tag = self.confidence_gate.classify(confidence)
            
            # Math-validate the arithmetic output
            math_boost, math_details = self.math_validator.validate_arithmetic(text)
            confidence = max(0.0, min(1.0, confidence + math_boost))
            tag = self.confidence_gate.classify(confidence)
            
            processing_time = time.time() - start_time
            self.logger.log_final(tag, processing_time, 1)
            
            return PipelineResult(
                text=text,
                lines=[LineResult(
                    text=text,
                    confidence=confidence,
                    tag=tag,
                    engine_used="arithmetic",
                    math_validated=math_details.get("validated", False)
                )],
                overall_confidence=round(confidence, 4),
                overall_tag=tag,
                engine_used="arithmetic",
                profile_used="clean",
                mode="arithmetic",
                features={},
                log=self.logger.format_for_display(),
                processing_time=round(processing_time, 3),
                stats=result,
                math_validation=math_details
            )
        except Exception as e:
            self.logger.error(f"Arithmetic pipeline error: {e}", "OCR")
            return PipelineResult(
                text="", lines=[], overall_confidence=0.0,
                overall_tag="FAILED_EXTRACTION", engine_used="arithmetic",
                profile_used="clean", mode="arithmetic",
                features={}, log=self.logger.format_for_display(),
                processing_time=time.time() - start_time,
                stats={"error": str(e)}
            )
