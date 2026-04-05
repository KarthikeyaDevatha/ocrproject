"""
Hybrid Pipeline Orchestrator for Adaptive OCR Agent v2.0.

Master pipeline combining:
- Decision Engine (feature-based routing)
- CRAFT/Contour Line Detection
- Dual Preprocessing Profiles (CLAHE clean/degraded)
- TrOCR-large (local handwriting OCR)
- Mathpix API (cloud math OCR)
- Arithmetic Pipeline (digit extraction)
- Composite Confidence Gate (3-tier tagging)
- Enhanced Post-Processing (spell + domain + math)
- Pipeline Logging

4 Modes: auto | mathpix | trocr | arithmetic
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
from inference.confidence_gate import CompositeConfidenceGate, ConfidenceResult
from inference.mathpix_ocr import MathpixOCR, MathpixResult
from inference.enhanced_postprocessor import PostProcessor
from inference.pipeline_logger import PipelineLogger


@dataclass
class LineResult:
    """Result for a single detected text line."""
    text: str
    confidence: float
    tag: str  # ACCEPTED | LOW_CONFIDENCE | FAILED
    engine_used: str
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    retried: bool = False
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


class HybridPipeline:
    """
    Adaptive OCR Agent — 4-mode orchestrator.
    
    Modes:
    - auto: Analyze image → decide engine → confidence gate → retry if needed
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
        self.line_detector = LineDetector(use_craft=True)
        self.confidence_gate = CompositeConfidenceGate()
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
        
        self._trocr_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._trocr_model = self._trocr_model.to(self._trocr_device)
        self._trocr_model.eval()
        
        self.logger.info(f"TrOCR loaded on {self._trocr_device}", "MODEL")

    # ========================================================================
    # TrOCR INFERENCE
    # ========================================================================

    def _run_trocr(
        self, pil_image: Image.Image
    ) -> Tuple[str, Optional[list], float]:
        """
        Run TrOCR on a single line crop.
        
        Args:
            pil_image: PIL Image of a single text line
        
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
        """
        Run a specific OCR engine on an image.
        
        Returns:
            Tuple of (text, scores)
        """
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
                overall_tag="FAILED", engine_used="none",
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
                overall_tag="FAILED", engine_used="none",
                profile_used=decision.profile, mode=active_mode,
                features=decision.features,
                log=self.logger.format_for_display(),
                processing_time=time.time() - start_time,
                stats={"error": "Blank image"}
            )
        
        # Override profile if requested
        profile = force_profile or decision.profile
        engine = decision.engine if active_mode == "auto" else "trocr"
        
        # Special case: Mathpix for full image (not per-line)
        if engine == "mathpix":
            return self._process_mathpix(image_path, image, start_time,
                                         decision=decision, profile=profile)
        
        # Step 2: Preprocess
        preprocessed = full_preprocess(image, profile)
        
        # Step 3: Line detection
        self.logger.info("Detecting text lines", "DETECT")
        line_crops = self.line_detector.detect_and_crop(
            image_path, image=image
        )
        self.logger.info(f"Detected {len(line_crops)} text lines", "DETECT")
        
        # Step 4: Per-line OCR with confidence gate
        line_results = []
        all_texts = []
        
        for idx, (crop_pil, bbox) in enumerate(line_crops):
            # Run primary engine
            text, scores = self._run_engine(engine, crop_pil, image_path)
            confidence = self.confidence_gate.score(text, scores)
            
            retried = False
            
            # Confidence gate: retry if needed
            if self.confidence_gate.should_retry(confidence):
                fallback = self._get_fallback_engine(engine)
                if fallback:
                    self.logger.log_retry(
                        idx, engine, fallback, confidence.composite
                    )
                    alt_text, alt_scores = self._run_engine(
                        fallback, crop_pil, image_path
                    )
                    alt_confidence = self.confidence_gate.score(
                        alt_text, alt_scores
                    )
                    
                    # Pick best
                    choice = self.confidence_gate.pick_best(
                        confidence, alt_confidence
                    )
                    if choice == 1:
                        text, scores = alt_text, alt_scores
                        confidence = alt_confidence
                        engine = fallback
                        retried = True
            
            self.logger.log_confidence(
                idx, confidence.composite, confidence.tag,
                engine, retried
            )
            
            line_results.append(LineResult(
                text=text,
                confidence=confidence.composite,
                tag=confidence.tag,
                engine_used=engine,
                bbox=bbox,
                retried=retried,
                confidence_details={
                    "token_confidence": confidence.token_confidence,
                    "alpha_ratio": confidence.alpha_ratio,
                    "length_ok": confidence.length_ok
                }
            ))
            
            if confidence.tag != "FAILED":
                all_texts.append(text)
        
        # Step 5: Post-processing
        processed_texts, pp_stats = self.postprocessor.process_lines(all_texts)
        self.logger.log_postprocess(
            pp_stats.get("spell_corrections", 0),
            pp_stats.get("lines_merged", 0)
        )
        
        # Step 6: Assemble output
        final_text = "\n".join(processed_texts)
        
        # Overall confidence = mean of accepted/low-confidence lines
        valid_scores = [lr.confidence for lr in line_results if lr.tag != "FAILED"]
        overall_confidence = (
            sum(valid_scores) / len(valid_scores)
        ) if valid_scores else 0.0
        
        # Overall tag
        if all(lr.tag == "ACCEPTED" for lr in line_results):
            overall_tag = "ACCEPTED"
        elif any(lr.tag == "FAILED" for lr in line_results):
            overall_tag = "LOW_CONFIDENCE"
        else:
            overall_tag = "ACCEPTED" if overall_confidence >= 0.72 else "LOW_CONFIDENCE"
        
        processing_time = time.time() - start_time
        self.logger.log_final(overall_tag, processing_time, len(line_results))
        
        return PipelineResult(
            text=final_text,
            lines=line_results,
            overall_confidence=round(overall_confidence, 4),
            overall_tag=overall_tag,
            engine_used=engine,
            profile_used=profile,
            mode=active_mode,
            features=decision.features,
            log=self.logger.format_for_display(),
            processing_time=round(processing_time, 3),
            stats={
                "lines_detected": len(line_crops),
                "lines_accepted": sum(1 for lr in line_results if lr.tag == "ACCEPTED"),
                "lines_low_conf": sum(1 for lr in line_results if lr.tag == "LOW_CONFIDENCE"),
                "lines_failed": sum(1 for lr in line_results if lr.tag == "FAILED"),
                "retries": sum(1 for lr in line_results if lr.retried),
                "postprocessing": pp_stats
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
            
            processing_time = time.time() - start_time
            self.logger.log_final(tag, processing_time, 1)
            
            return PipelineResult(
                text=text,
                lines=[LineResult(
                    text=text,
                    confidence=confidence,
                    tag=tag,
                    engine_used="arithmetic"
                )],
                overall_confidence=round(confidence, 4),
                overall_tag=tag,
                engine_used="arithmetic",
                profile_used="clean",
                mode="arithmetic",
                features={},
                log=self.logger.format_for_display(),
                processing_time=round(processing_time, 3),
                stats=result
            )
        except Exception as e:
            self.logger.error(f"Arithmetic pipeline error: {e}", "OCR")
            return PipelineResult(
                text="", lines=[], overall_confidence=0.0,
                overall_tag="FAILED", engine_used="arithmetic",
                profile_used="clean", mode="arithmetic",
                features={}, log=self.logger.format_for_display(),
                processing_time=time.time() - start_time,
                stats={"error": str(e)}
            )
