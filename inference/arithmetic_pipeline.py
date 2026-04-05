"""
Zero-Hallucination Arithmetic OCR Pipeline.
End-to-end pipeline for handwritten arithmetic expression recognition.

Uses TrOCR-large-handwritten for OCR, with multi-layer post-processing:
1. Image preprocessing (grayscale, denoise, adaptive threshold)
2. Controlled OCR (TrOCR-large only, no LaTeX models)
3. Token cleaning (strip LaTeX, non-numeric noise)
4. Similarity-based correction (Levenshtein distance)
5. Hard constraint validation
6. Structure reconstruction (from numbers only)
7. Mathematical validation
8. Confidence scoring
9. Self-correction loop (3 passes)
10. Final JSON output

CONSTRAINTS:
- NEVER trust OCR structure
- NEVER allow LaTeX hallucination
- ONLY allow digits + basic operators
- ALWAYS apply correction before validation
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field, asdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.preprocess import (
    full_preprocessing_pipeline,
    binarize_adaptive,
    enhance_contrast,
    remove_noise,
    deskew_image
)
from inference.token_corrector import (
    TokenCleaner,
    SimilarityCorrector,
    CorrectedToken
)
from inference.arithmetic_validator import (
    ArithmeticValidator,
    ArithmeticConfidenceScorer,
    ArithmeticResult,
    ValidationResult
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ArithmeticPipelineConfig:
    """Configuration for arithmetic OCR pipeline."""
    # Model
    model_name: str = "microsoft/trocr-large-handwritten"

    # Domain constraints
    number_range: Tuple[int, int] = (0, 100)
    allowed_operators: List[str] = field(default_factory=lambda: ['+', '-', '/', '='])
    min_numbers: int = 2

    # Confidence
    confidence_threshold: float = 0.6

    # Self-correction
    max_passes: int = 3

    # Device
    device: str = "cpu"

    # Preprocessing
    denoise_method: str = "bilateral"
    apply_adaptive_threshold: bool = True

    # Debug
    debug: bool = False


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class ArithmeticPipeline:
    """
    Zero-hallucination OCR pipeline for handwritten arithmetic expressions.

    Processes images through 10 steps:
    1. Preprocess → 2. OCR → 3. Clean → 4. Correct →
    5. Validate → 6. Reconstruct → 7. Math Validate →
    8. Score → 9. Self-Correct → 10. Output
    """

    def __init__(self, config: Optional[ArithmeticPipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or ArithmeticPipelineConfig()

        # Initialize components
        self.cleaner = TokenCleaner()
        self.corrector = SimilarityCorrector(
            number_range=self.config.number_range,
            allowed_operators=self.config.allowed_operators
        )
        self.validator = ArithmeticValidator(
            number_range=self.config.number_range,
            allowed_operators=self.config.allowed_operators,
            min_numbers=self.config.min_numbers
        )
        self.scorer = ArithmeticConfidenceScorer(
            confidence_threshold=self.config.confidence_threshold
        )

        # Lazy-load OCR model
        self._model = None
        self._processor = None
        self._torch = None

    def _load_ocr_model(self):
        """Lazy-load TrOCR-large-handwritten model."""
        if self._model is not None:
            return

        import torch
        from transformers import VisionEncoderDecoderModel

        self._torch = torch

        print(f"Loading TrOCR model: {self.config.model_name}")

        # Load processor with transformers 5.x compatibility
        try:
            from transformers import TrOCRProcessor
            self._processor = TrOCRProcessor.from_pretrained(
                self.config.model_name, use_fast=False
            )
        except (ValueError, ImportError):
            # Fallback: build processor from components with explicit slow tokenizer
            from transformers import TrOCRProcessor, AutoImageProcessor, XLMRobertaTokenizer
            print(f"Fast tokenizer failed, loading slow tokenizer for: {self.config.model_name}")
            image_processor = AutoImageProcessor.from_pretrained(self.config.model_name)
            tokenizer = XLMRobertaTokenizer.from_pretrained(self.config.model_name)
            self._processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

        self._model = VisionEncoderDecoderModel.from_pretrained(
            self.config.model_name
        )

        # Set device
        if self.config.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.cuda()
            self._device = "cuda"
        else:
            self._device = "cpu"

        self._model.eval()
        print(f"TrOCR model loaded on {self._device}")

    # ========================================================================
    # STEP 1: PREPROCESS IMAGE
    # ========================================================================

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing: grayscale → denoise → adaptive threshold.
        Normalize contrast and remove artifacts.

        Args:
            image: Input BGR image

        Returns:
            Preprocessed image (RGB for OCR)
        """
        # Full preprocessing pipeline
        processed = full_preprocessing_pipeline(
            image,
            denoise=True,
            deskew=True,
            normalize_res=True,
            enhance=True,
            denoise_method=self.config.denoise_method
        )

        # Additional: Apply adaptive thresholding for cleaner digits
        if self.config.apply_adaptive_threshold:
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed

            binary = binarize_adaptive(gray, block_size=15, c=4)

            # Convert back to 3-channel for OCR model
            processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        # Ensure RGB
        if len(processed.shape) == 3 and processed.shape[2] == 3:
            # Check if BGR (OpenCV default)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        elif len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

        return processed

    # ========================================================================
    # STEP 2: OCR EXECUTION (CONTROLLED)
    # ========================================================================

    def _run_ocr(self, image: np.ndarray) -> str:
        """
        Run TrOCR-large-handwritten on preprocessed image.
        Strictly no LaTeX models. Extract raw tokens only.

        Args:
            image: Preprocessed RGB image

        Returns:
            Raw OCR text output
        """
        self._load_ocr_model()

        pil_image = Image.fromarray(image)

        pixel_values = self._processor(
            images=pil_image.convert('RGB'),
            return_tensors="pt"
        ).pixel_values

        if self._device == "cuda":
            pixel_values = pixel_values.cuda()

        with self._torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values,
                max_new_tokens=100,
                num_beams=4,
                repetition_penalty=2.0,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return text.strip()

    # ========================================================================
    # STEP 3-4: TOKEN CLEANING + CORRECTION
    # ========================================================================

    def _clean_and_correct(
        self, raw_text: str
    ) -> Tuple[List[CorrectedToken], List[str], str]:
        """
        Clean raw OCR output and correct tokens.

        Args:
            raw_text: Raw OCR output string

        Returns:
            Tuple of (corrected_tokens, raw_tokens, cleaned_text)
        """
        # Step 3: Token cleaning
        cleaned = self.cleaner.clean_raw_output(raw_text)
        raw_tokens = self.cleaner.tokenize(cleaned)

        if self.config.debug:
            print(f"  Raw OCR: '{raw_text}'")
            print(f"  Cleaned: '{cleaned}'")
            print(f"  Tokens:  {raw_tokens}")

        # Step 4: Similarity-based correction
        corrected = self.corrector.correct_all(raw_tokens)

        if self.config.debug:
            for ct in corrected:
                if ct.original != ct.corrected:
                    print(f"  Corrected: '{ct.original}' → '{ct.corrected}' "
                          f"(confidence={ct.confidence:.2f})")

        return corrected, raw_tokens, cleaned

    # ========================================================================
    # STEPS 5-8: VALIDATE, RECONSTRUCT, COMPUTE, SCORE
    # ========================================================================

    def _validate_and_compute(
        self,
        corrected_tokens: List[CorrectedToken],
        raw_text: str
    ) -> ArithmeticResult:
        """
        Validate tokens, reconstruct expression, compute result, score output.

        Args:
            corrected_tokens: Corrected token list
            raw_text: Original raw OCR text

        Returns:
            ArithmeticResult
        """
        # Step 5: Hard constraint validation
        validation = self.validator.validate_tokens(corrected_tokens)
        numbers = validation.valid_numbers

        if self.config.debug:
            print(f"  Valid numbers: {numbers}")
            print(f"  Validation: {'PASS' if validation.is_valid else 'FAIL'}")
            if validation.reasons:
                for r in validation.reasons:
                    print(f"    - {r}")

        # Check for minimum numbers
        if len(numbers) < self.config.min_numbers:
            return ArithmeticResult(
                status="retry",
                numbers=numbers,
                expression="",
                result=None,
                confidence=0.0,
                reason=f"Insufficient numbers: {len(numbers)} < {self.config.min_numbers}"
            )

        # Step 6: Structure reconstruction (NEVER trust OCR structure)
        expression = self.validator.reconstruct_expression(numbers)

        # Step 7: Mathematical validation
        result, error = self.validator.compute_result(numbers)

        if error:
            return ArithmeticResult(
                status="error",
                numbers=numbers,
                expression=expression,
                result=None,
                confidence=0.0,
                reason=f"Math error: {error}"
            )

        # Step 8: Confidence scoring
        confidence, score_details = self.scorer.score(
            corrected_tokens, validation, numbers, self.config.number_range
        )

        if self.config.debug:
            print(f"  Expression: {expression}")
            print(f"  Result: {result}")
            print(f"  Confidence: {confidence}")
            print(f"  Score details: {score_details}")

        # Check confidence threshold
        if self.scorer.should_retry(confidence):
            return ArithmeticResult(
                status="retry",
                numbers=numbers,
                expression=expression,
                result=result,
                confidence=confidence,
                reason="low confidence OCR"
            )

        return ArithmeticResult(
            status="success",
            numbers=numbers,
            expression=expression,
            result=result,
            confidence=confidence
        )

    # ========================================================================
    # STEP 9: SELF-CORRECTION LOOP
    # ========================================================================

    def _self_correction_pass(
        self,
        result: ArithmeticResult,
        corrected_tokens: List[CorrectedToken],
        raw_text: str,
        pass_number: int
    ) -> Tuple[ArithmeticResult, List[str]]:
        """
        Execute one self-correction pass.

        Pass 1: Generate output
        Pass 2: Critically check for hallucination, invalid tokens, wrong reconstruction
        Pass 3: Fix all issues and return final answer

        Args:
            result: Current ArithmeticResult
            corrected_tokens: Current corrected tokens
            raw_text: Original raw OCR text
            pass_number: 1, 2, or 3

        Returns:
            Tuple of (updated_result, issues_found)
        """
        issues = []

        if pass_number == 1:
            # Pass 1: Just return the initial result
            return result, []

        elif pass_number == 2:
            # Pass 2: Critical check

            # Check for hallucination in raw text
            has_hallucination, violations = self.validator.check_hallucination(raw_text)
            if has_hallucination:
                issues.append(f"Hallucination detected: {violations}")

            # Check for invalid tokens
            for ct in corrected_tokens:
                if ct.token_type == 'discarded':
                    issues.append(f"Invalid token discarded: '{ct.original}'")
                elif ct.confidence < 0.5:
                    issues.append(
                        f"Low confidence correction: '{ct.original}' → "
                        f"'{ct.corrected}' ({ct.confidence:.2f})"
                    )

            # Check reconstruction validity
            if result.numbers:
                # Verify all numbers are in range
                for n in result.numbers:
                    if not (self.config.number_range[0] <= n <= self.config.number_range[1]):
                        issues.append(f"Out-of-range number in output: {n}")

                # Verify result is mathematically correct
                if result.result is not None:
                    expected = sum(result.numbers) / len(result.numbers)
                    if abs(result.result - expected) > 0.01:
                        issues.append(
                            f"Math inconsistency: result={result.result}, "
                            f"expected={expected}"
                        )

            return result, issues

        elif pass_number == 3:
            # Pass 3: Fix issues

            # Re-filter numbers to ensure range compliance
            fixed_numbers = [
                n for n in result.numbers
                if self.config.number_range[0] <= n <= self.config.number_range[1]
            ]

            if len(fixed_numbers) < self.config.min_numbers:
                return ArithmeticResult(
                    status="retry",
                    numbers=fixed_numbers,
                    expression="",
                    result=None,
                    confidence=0.0,
                    reason="Insufficient valid numbers after correction"
                ), issues

            # Rebuild expression
            expression = self.validator.reconstruct_expression(fixed_numbers)
            math_result, error = self.validator.compute_result(fixed_numbers)

            if error:
                return ArithmeticResult(
                    status="error",
                    numbers=fixed_numbers,
                    expression=expression,
                    result=None,
                    confidence=0.0,
                    reason=f"Math error after correction: {error}"
                ), issues

            # Recompute confidence
            confidence, _ = self.scorer.score(
                corrected_tokens,
                self.validator.validate_tokens(corrected_tokens),
                fixed_numbers,
                self.config.number_range
            )

            fixed_result = ArithmeticResult(
                status="success" if not self.scorer.should_retry(confidence) else "retry",
                numbers=fixed_numbers,
                expression=expression,
                result=math_result,
                confidence=confidence,
                reason="" if not self.scorer.should_retry(confidence) else "low confidence after correction"
            )

            return fixed_result, issues

        return result, issues

    # ========================================================================
    # STEP 10: FINAL OUTPUT
    # ========================================================================

    def _format_output(self, result: ArithmeticResult, pass_details: List[Dict] = None) -> Dict:
        """
        Format final output as JSON-serializable dict.

        Args:
            result: ArithmeticResult
            pass_details: Optional details from self-correction passes

        Returns:
            JSON-serializable dict
        """
        output = {
            "status": result.status,
            "numbers": result.numbers,
            "expression": result.expression,
            "result": result.result,
            "confidence": result.confidence
        }

        if result.reason:
            output["reason"] = result.reason

        if pass_details:
            output["pass_details"] = pass_details

        return output

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def process(self, image_path: str) -> Dict:
        """
        Process a single image through the complete arithmetic pipeline.

        Args:
            image_path: Path to input image

        Returns:
            JSON-serializable dict with results
        """
        if not os.path.exists(image_path):
            return {
                "status": "error",
                "numbers": [],
                "expression": "",
                "result": None,
                "confidence": 0.0,
                "reason": f"Image not found: {image_path}"
            }

        image = cv2.imread(image_path)
        if image is None:
            return {
                "status": "error",
                "numbers": [],
                "expression": "",
                "result": None,
                "confidence": 0.0,
                "reason": f"Could not read image: {image_path}"
            }

        return self.process_from_array(image)

    def process_from_array(self, image: np.ndarray) -> Dict:
        """
        Process a numpy array through the complete arithmetic pipeline.

        Args:
            image: Input image (BGR numpy array)

        Returns:
            JSON-serializable dict with results
        """
        start_time = time.time()
        pass_details = []

        if self.config.debug:
            print("=" * 60)
            print("ARITHMETIC OCR PIPELINE")
            print("=" * 60)

        # Step 1: Preprocess
        if self.config.debug:
            print("\n[STEP 1] Preprocessing...")
        preprocessed = self._preprocess_image(image)

        # Step 2: OCR
        if self.config.debug:
            print("\n[STEP 2] Running TrOCR-large...")
        raw_text = self._run_ocr(preprocessed)

        # Steps 3-4: Clean and Correct
        if self.config.debug:
            print("\n[STEPS 3-4] Cleaning and correcting...")
        corrected_tokens, raw_tokens, cleaned_text = self._clean_and_correct(raw_text)

        # Steps 5-8: Validate, Reconstruct, Compute, Score
        if self.config.debug:
            print("\n[STEPS 5-8] Validating and computing...")
        result = self._validate_and_compute(corrected_tokens, raw_text)

        # Step 9: Self-correction loop (3 passes)
        if self.config.debug:
            print("\n[STEP 9] Self-correction loop...")

        for pass_num in range(1, self.config.max_passes + 1):
            if self.config.debug:
                print(f"\n  Pass {pass_num}:")

            result, issues = self._self_correction_pass(
                result, corrected_tokens, raw_text, pass_num
            )

            pass_detail = {
                "pass": pass_num,
                "status": result.status,
                "issues_found": len(issues),
                "issues": issues
            }
            pass_details.append(pass_detail)

            if self.config.debug and issues:
                for issue in issues:
                    print(f"    Issue: {issue}")

        # Step 10: Final output
        if self.config.debug:
            print("\n[STEP 10] Final output")

        output = self._format_output(result, pass_details if self.config.debug else None)

        elapsed = time.time() - start_time
        output["processing_time_seconds"] = round(elapsed, 3)

        if self.config.debug:
            print(f"\nResult: {json.dumps(output, indent=2)}")
            print(f"Total time: {elapsed:.3f}s")

        return output

    def process_tokens_only(self, raw_text: str) -> Dict:
        """
        Process raw OCR text through the correction pipeline (no image/OCR step).
        Useful for testing the post-processing stack independently.

        Args:
            raw_text: Raw OCR output string

        Returns:
            JSON-serializable dict with results
        """
        pass_details = []

        # Steps 3-4
        corrected_tokens, raw_tokens, cleaned_text = self._clean_and_correct(raw_text)

        # Steps 5-8
        result = self._validate_and_compute(corrected_tokens, raw_text)

        # Step 9: Self-correction loop
        for pass_num in range(1, self.config.max_passes + 1):
            result, issues = self._self_correction_pass(
                result, corrected_tokens, raw_text, pass_num
            )
            pass_details.append({
                "pass": pass_num,
                "status": result.status,
                "issues_found": len(issues),
                "issues": issues
            })

        # Step 10
        output = self._format_output(result, pass_details if self.config.debug else None)
        return output


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for arithmetic OCR pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Zero-Hallucination Arithmetic OCR Pipeline"
    )
    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.6,
        help="Confidence threshold for retry"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to save JSON output"
    )

    args = parser.parse_args()

    config = ArithmeticPipelineConfig(
        device=args.device,
        debug=args.debug,
        confidence_threshold=args.confidence_threshold
    )

    pipeline = ArithmeticPipeline(config)
    result = pipeline.process(args.image)

    # Output
    output_json = json.dumps(result, indent=2)
    print(output_json)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(output_json)
        print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
