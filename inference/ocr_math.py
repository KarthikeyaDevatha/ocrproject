"""
Math OCR module using Pix2Text-MFR ONNX model.
Handles handwritten mathematical expression recognition.
"""

import os
import re
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Tuple, Optional, List
from dataclasses import dataclass
from transformers import TrOCRProcessor, AutoImageProcessor


def _load_trocr_processor(model_name_or_path: str) -> TrOCRProcessor:
    """
    Load TrOCRProcessor with transformers 5.x compatibility.
    Explicitly loads the slow tokenizer to avoid the sentencepiece/tiktoken
    fast tokenizer conversion bug in transformers >= 5.0.
    """
    try:
        # Try normal loading first (works on older transformers)
        return TrOCRProcessor.from_pretrained(model_name_or_path, use_fast=False)
    except (ValueError, ImportError):
        # Fallback: build processor from components with explicit slow tokenizer
        from transformers import XLMRobertaTokenizer
        print(f"Fast tokenizer failed, loading slow tokenizer for: {model_name_or_path}")
        image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name_or_path)
        return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


@dataclass
class MathOCRResult:
    """Math OCR prediction result."""
    latex: str
    confidence: float
    inference_time: float
    discarded: bool = False
    is_display_math: bool = False


class MathOCR:
    """Math formula recognition using ONNX Runtime."""
    
    # Pattern to check if output looks like math
    MATH_PATTERN = re.compile(r'[\\\_\^{}]')
    
    def __init__(
        self,
        encoder_path: str = "models/trocr_math/encoder_model.onnx",
        decoder_path: str = "models/trocr_math/decoder_model.onnx",
        tokenizer_path: str = "models/trocr_math/tokenizer",
        device: str = "cpu",
        max_new_tokens: int = 150,
        num_beams: int = 4,
        timeout: float = 5.0,
        confidence_threshold: float = 0.4,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize Math OCR engine.
        
        Args:
            encoder_path: Path to encoder ONNX model
            decoder_path: Path to decoder ONNX model
            tokenizer_path: Path to tokenizer directory
            device: 'cpu' or 'cuda'
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width
            timeout: Maximum inference time
            confidence_threshold: Minimum confidence threshold
            input_size: Input image size (height, width)
        """
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.timeout = timeout
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        
        # Load ONNX sessions
        self.encoder_session = None
        self.decoder_session = None
        
        if os.path.exists(encoder_path):
            self.encoder_session = ort.InferenceSession(encoder_path, providers=providers)
        
        if os.path.exists(decoder_path):
            self.decoder_session = ort.InferenceSession(decoder_path, providers=providers)
        
        # Load tokenizer
        # Check if local tokenizer has required files
        local_tokenizer_valid = (
            os.path.exists(tokenizer_path) and 
            os.path.exists(os.path.join(tokenizer_path, "preprocessor_config.json"))
        )
        
        if local_tokenizer_valid:
            self.processor = _load_trocr_processor(tokenizer_path)
        else:
            print(f"Loading default processor for math (tokenizer not found or incomplete at {tokenizer_path})")
            self.processor = _load_trocr_processor("microsoft/trocr-small-handwritten")
        
        # Fallback mode
        self.use_torch_fallback = self.encoder_session is None or self.decoder_session is None
        
        if self.use_torch_fallback:
            print("ONNX models not found, using PyTorch fallback for math")
            self._init_torch_model()
    
    def _init_torch_model(self):
        """Initialize PyTorch model as fallback."""
        try:
            from transformers import VisionEncoderDecoderModel
            import torch
            
            self.torch_model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-small-handwritten"
            )
            self.torch_model.eval()
            self.device = "cpu"
            
            if torch.cuda.is_available():
                self.torch_model = self.torch_model.cuda()
                self.device = "cuda"
        except Exception as e:
            print(f"Could not load PyTorch model: {e}")
            self.torch_model = None
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for math OCR.
        
        Args:
            image: Input image
        
        Returns:
            Preprocessed pixel values
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Resize to input size
        pil_image = pil_image.convert('RGB').resize(
            (self.input_size[1], self.input_size[0]),
            Image.Resampling.LANCZOS
        )
        
        pixel_values = self.processor(
            images=pil_image,
            return_tensors="np"
        ).pixel_values
        
        return pixel_values
    
    def _greedy_decode_onnx(
        self,
        encoder_output: np.ndarray
    ) -> Tuple[List[int], float]:
        """Greedy decoding with ONNX models."""
        decoder_input_ids = np.array([[self.processor.tokenizer.cls_token_id]], dtype=np.int64)
        
        generated_tokens = []
        log_probs = []
        
        start_time = time.time()
        
        for _ in range(self.max_new_tokens):
            if time.time() - start_time > self.timeout:
                print("Warning: Math decoding timeout")
                break
            
            decoder_inputs = {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_output
            }
            
            input_names = [i.name for i in self.decoder_session.get_inputs()]
            filtered_inputs = {k: v for k, v in decoder_inputs.items() if k in input_names}
            
            outputs = self.decoder_session.run(None, filtered_inputs)
            logits = outputs[0]
            
            next_token_logits = logits[0, -1, :]
            exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            next_token = int(np.argmax(probs))
            log_prob = float(np.log(probs[next_token] + 1e-10))
            
            if next_token == self.processor.tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token)
            log_probs.append(log_prob)
            
            decoder_input_ids = np.concatenate([
                decoder_input_ids,
                np.array([[next_token]], dtype=np.int64)
            ], axis=1)
        
        avg_log_prob = np.mean(log_probs) if log_probs else 0.0
        
        return generated_tokens, avg_log_prob
    
    def _validate_math_output(self, latex: str) -> bool:
        """
        Validate that output looks like mathematical LaTeX.
        
        Args:
            latex: Generated LaTeX string
        
        Returns:
            True if valid math-like output
        """
        # Must contain some math-like characters
        if not self.MATH_PATTERN.search(latex):
            return False
        
        # Check brace balance
        open_braces = latex.count('{')
        close_braces = latex.count('}')
        if open_braces != close_braces:
            return False
        
        return True
    
    def _postprocess_latex(self, latex: str) -> str:
        """
        Clean up and fix common LaTeX issues.
        
        Args:
            latex: Raw LaTeX string
        
        Returns:
            Cleaned LaTeX
        """
        latex = latex.strip()
        
        # Remove $ delimiters if present
        if latex.startswith('$') and latex.endswith('$'):
            latex = latex[1:-1]
        
        # Balance braces
        open_braces = latex.count('{')
        close_braces = latex.count('}')
        
        if open_braces > close_braces:
            latex += '}' * (open_braces - close_braces)
        elif close_braces > open_braces:
            latex = '{' * (close_braces - open_braces) + latex
        
        # Common fixes
        latex = re.sub(r'\\frac\s*{', r'\\frac{', latex)
        latex = re.sub(r'\\sqrt\s*{', r'\\sqrt{', latex)
        
        return latex
    
    def _is_display_math(self, latex: str) -> bool:
        """
        Determine if expression should be display math.
        
        Args:
            latex: LaTeX string
        
        Returns:
            True if should use display math ($$...$$)
        """
        display_patterns = [
            r'\\frac',
            r'\\sum',
            r'\\prod',
            r'\\int',
            r'\\lim',
            r'\\begin{',
            r'\\matrix',
        ]
        
        for pattern in display_patterns:
            if re.search(pattern, latex):
                return True
        
        return False
    
    def recognize(self, image: np.ndarray) -> MathOCRResult:
        """
        Recognize mathematical expression in image.
        
        Args:
            image: Input image
        
        Returns:
            MathOCRResult with LaTeX and metadata
        """
        start_time = time.time()
        
        # Convert if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Preprocess
        pixel_values = self.preprocess(pil_image)
        
        if self.use_torch_fallback and self.torch_model is not None:
            return self._recognize_torch(pil_image, start_time)
        
        if self.encoder_session is None:
            return MathOCRResult(
                latex="[Math OCR model not loaded]",
                confidence=0.0,
                inference_time=time.time() - start_time,
                discarded=True
            )
        
        # Run encoder
        encoder_output = self.encoder_session.run(
            None,
            {"pixel_values": pixel_values}
        )[0]
        
        # Decode
        token_ids, avg_log_prob = self._greedy_decode_onnx(encoder_output)
        
        # Decode to text
        latex = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Postprocess
        latex = self._postprocess_latex(latex)
        
        inference_time = time.time() - start_time
        confidence = np.exp(avg_log_prob)
        
        # Validate output
        discarded = not self._validate_math_output(latex) or confidence < self.confidence_threshold
        
        # Determine display vs inline
        is_display = self._is_display_math(latex)
        
        return MathOCRResult(
            latex=latex,
            confidence=float(confidence),
            inference_time=inference_time,
            discarded=discarded,
            is_display_math=is_display
        )
    
    def _recognize_torch(self, image: Image.Image, start_time: float) -> MathOCRResult:
        """Fallback recognition using PyTorch."""
        import torch
        
        # Resize
        image = image.resize((self.input_size[1], self.input_size[0]), Image.Resampling.LANCZOS)
        
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        
        if self.device == "cuda":
            pixel_values = pixel_values.cuda()
        
        with torch.no_grad():
            generated_ids = self.torch_model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams
            )
        
        latex = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        latex = self._postprocess_latex(latex)
        
        is_valid = self._validate_math_output(latex)
        is_display = self._is_display_math(latex)
        
        return MathOCRResult(
            latex=latex,
            confidence=0.9 if is_valid else 0.3,
            inference_time=time.time() - start_time,
            discarded=not is_valid,
            is_display_math=is_display
        )
    
    def batch_recognize(
        self,
        images: List[np.ndarray]
    ) -> List[MathOCRResult]:
        """Recognize math in multiple images."""
        return [self.recognize(img) for img in images]


def create_math_ocr(
    model_dir: str = "models/trocr_math",
    device: str = "cpu"
) -> MathOCR:
    """
    Factory function to create MathOCR instance.
    
    Args:
        model_dir: Directory containing ONNX models
        device: 'cpu' or 'cuda'
    
    Returns:
        Configured MathOCR instance
    """
    return MathOCR(
        encoder_path=os.path.join(model_dir, "encoder_model.onnx"),
        decoder_path=os.path.join(model_dir, "decoder_model.onnx"),
        tokenizer_path=os.path.join(model_dir, "tokenizer"),
        device=device
    )
