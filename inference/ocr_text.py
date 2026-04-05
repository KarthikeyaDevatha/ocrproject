"""
Text OCR module using TrOCR ONNX model.
Handles handwritten English text recognition with confidence scoring.
"""

import os
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
class OCRResult:
    """OCR prediction result."""
    text: str
    confidence: float
    inference_time: float
    rerouted: bool = False


class TextOCR:
    """TrOCR-based text recognition using ONNX Runtime."""
    
    def __init__(
        self,
        encoder_path: str = "models/trocr_text/encoder_model.onnx",
        decoder_path: str = "models/trocr_text/decoder_model.onnx",
        tokenizer_path: str = "models/trocr_text/tokenizer",
        model_path: Optional[str] = None,
        device: str = "cpu",
        max_new_tokens: int = 100,
        num_beams: int = 4,
        timeout: float = 5.0,
        confidence_threshold: float = 0.4
    ):
        """
        Initialize Text OCR engine.
        
        Args:
            encoder_path: Path to encoder ONNX model
            decoder_path: Path to decoder ONNX model
            tokenizer_path: Path to tokenizer directory
            model_path: Path to PyTorch model (if fallback used)
            device: 'cpu' or 'cuda'
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width
            timeout: Maximum inference time in seconds
            confidence_threshold: Minimum average log probability
        """
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.timeout = timeout
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        self.torch_model = None
        
        # Setup providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        
        # Load ONNX sessions
        self.encoder_session = None
        self.decoder_session = None
        
        if os.path.exists(encoder_path):
            self.encoder_session = ort.InferenceSession(encoder_path, providers=providers)
        
        if os.path.exists(decoder_path):
            self.decoder_session = ort.InferenceSession(decoder_path, providers=providers)
        
        # Load tokenizer/processor
        # Check if local tokenizer has required files
        local_tokenizer_valid = (
            os.path.exists(tokenizer_path) and 
            os.path.exists(os.path.join(tokenizer_path, "preprocessor_config.json"))
        )
        
        if local_tokenizer_valid:
            self.processor = _load_trocr_processor(tokenizer_path)
        else:
            # Fallback to large model for maximum accuracy
            print(f"Loading default processor (tokenizer not found or incomplete at {tokenizer_path})")
            self.processor = _load_trocr_processor("microsoft/trocr-large-handwritten")
        
        # Fallback to HuggingFace if ONNX not available
        self.use_torch_fallback = self.encoder_session is None or self.decoder_session is None
        
        if self.use_torch_fallback:
            print("ONNX models not found, using PyTorch fallback")
            self._init_torch_model()
    
    def _init_torch_model(self):
        """Initialize PyTorch model as fallback."""
        try:
            from transformers import VisionEncoderDecoderModel
            import torch
            
            # Try custom path first, then default to HuggingFace
            path = self.model_path if self.model_path and os.path.exists(os.path.join(self.model_path, "config.json")) else "microsoft/trocr-large-handwritten"
            print(f"Loading PyTorch model from: {path}")
            
            self.torch_model = VisionEncoderDecoderModel.from_pretrained(path)
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
        Preprocess image for TrOCR.
        
        Args:
            image: Input image (RGB numpy array or PIL Image)
        
        Returns:
            Preprocessed pixel values
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        pixel_values = self.processor(
            images=image.convert('RGB'),
            return_tensors="np"
        ).pixel_values
        
        return pixel_values
    
    def _greedy_decode_onnx(
        self,
        encoder_output: np.ndarray
    ) -> Tuple[List[int], float]:
        """
        Greedy decoding with ONNX models.
        
        Args:
            encoder_output: Encoder hidden states
        
        Returns:
            Tuple of (token_ids, average_log_probability)
        """
        # Initialize decoder input
        decoder_input_ids = np.array([[self.processor.tokenizer.cls_token_id]], dtype=np.int64)
        
        generated_tokens = []
        log_probs = []
        
        start_time = time.time()
        
        for _ in range(self.max_new_tokens):
            # Check timeout
            if time.time() - start_time > self.timeout:
                print("Warning: Decoding timeout reached")
                break
            
            # Run decoder
            decoder_inputs = {
                "input_ids": decoder_input_ids,
                "attention_mask": np.ones_like(decoder_input_ids, dtype=np.int64),
                "encoder_hidden_states": encoder_output
            }
            
            # Handle different decoder input configurations
            input_names = [i.name for i in self.decoder_session.get_inputs()]
            filtered_inputs = {k: v for k, v in decoder_inputs.items() if k in input_names}
            
            outputs = self.decoder_session.run(None, filtered_inputs)
            logits = outputs[0]
            
            # Get next token
            next_token_logits = logits[0, -1, :]
            
            # Softmax to get probabilities
            exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            next_token = int(np.argmax(probs))
            log_prob = float(np.log(probs[next_token] + 1e-10))
            
            # Check for EOS
            if next_token == self.processor.tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token)
            log_probs.append(log_prob)
            
            # Update decoder input
            decoder_input_ids = np.concatenate([
                decoder_input_ids,
                np.array([[next_token]], dtype=np.int64)
            ], axis=1)
        
        avg_log_prob = np.mean(log_probs) if log_probs else 0.0
        
        return generated_tokens, avg_log_prob
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Recognize text in image.
        
        Args:
            image: Input image (RGB or BGR numpy array, or PIL Image)
        
        Returns:
            OCRResult with recognized text and confidence
        """
        start_time = time.time()
        
        # Convert BGR to RGB if needed
        if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
            # Check if likely BGR (OpenCV format)
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.fromarray(image).convert('RGB')
        
        # Preprocess
        pixel_values = self.preprocess(pil_image)
        
        if self.use_torch_fallback and self.torch_model is not None:
            return self._recognize_torch(pil_image, start_time)
        
        if self.encoder_session is None:
            return OCRResult(
                text="[OCR model not loaded]",
                confidence=0.0,
                inference_time=time.time() - start_time
            )
        
        try:
            # Run encoder
            encoder_output = self.encoder_session.run(
                None,
                {"pixel_values": pixel_values}
            )[0]
            
            # Decode
            token_ids, avg_log_prob = self._greedy_decode_onnx(encoder_output)
            
            # Decode tokens to text
            text = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)
            
            inference_time = time.time() - start_time
            
            # Check confidence
            confidence = np.exp(avg_log_prob)  # Convert log prob to probability
            rerouted = confidence < self.confidence_threshold
            
            return OCRResult(
                text=text.strip(),
                confidence=float(confidence),
                inference_time=inference_time,
                rerouted=rerouted
            )
        except Exception as e:
            print(f"ONNX inference failed: {e}. Falling back to PyTorch.")
            self.use_torch_fallback = True
            if self.torch_model is None:
                self._init_torch_model()
            return self._recognize_torch(pil_image, start_time)
    
    def _recognize_torch(self, image: Image.Image, start_time: float) -> OCRResult:
        """Fallback recognition using PyTorch model."""
        import torch
        
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        
        if self.device == "cuda":
            pixel_values = pixel_values.cuda()
        
        with torch.no_grad():
            generated_ids = self.torch_model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                repetition_penalty=2.0,      # Penalize repetition strongly
                length_penalty=1.0,          # Penalize very long sequences
                early_stopping=True,
                no_repeat_ngram_size=3       # Prevent repeating 3-grams
            )
        
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return OCRResult(
            text=text.strip(),
            confidence=0.9,  # Approximate confidence for torch fallback
            inference_time=time.time() - start_time
        )
    
    def batch_recognize(
        self,
        images: List[np.ndarray]
    ) -> List[OCRResult]:
        """
        Recognize text in multiple images.
        
        Args:
            images: List of input images
        
        Returns:
            List of OCRResult objects
        """
        return [self.recognize(img) for img in images]


def create_text_ocr(
    model_dir: str = "models/trocr_text",
    device: str = "cpu"
) -> TextOCR:
    """
    Factory function to create TextOCR instance.
    
    Args:
        model_dir: Directory containing ONNX models and tokenizer
        device: 'cpu' or 'cuda'
    
    Returns:
        Configured TextOCR instance
    """
    return TextOCR(
        encoder_path=os.path.join(model_dir, "encoder_model.onnx"),
        decoder_path=os.path.join(model_dir, "decoder_model.onnx"),
        tokenizer_path=os.path.join(model_dir, "tokenizer"),
        model_path=model_dir,  # Use model_dir as checkpoint path for fallback
        device=device
    )
