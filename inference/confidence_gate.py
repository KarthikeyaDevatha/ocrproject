"""
Composite Confidence Gate for Adaptive OCR Agent.

Combines three signals to produce a composite confidence score:
1. Token probability (from model.generate output_scores)
2. Character composition (penalise excessive symbols)
3. Output length plausibility

Output tags: ACCEPTED / LOW_CONFIDENCE / FAILED
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Thresholds — calibrate on your validation set (10+ samples)
DEFAULT_ACCEPTED_THRESHOLD = 0.72
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.50


@dataclass
class ConfidenceResult:
    """Result of confidence gate evaluation."""
    composite: float
    token_confidence: float
    alpha_ratio: float
    length_ok: bool
    length_score: float
    tag: str  # "ACCEPTED" | "LOW_CONFIDENCE" | "FAILED"


class CompositeConfidenceGate:
    """
    Multi-signal confidence scorer for OCR output.
    
    Signals:
    1. Token probability: mean max softmax from beam search scores
    2. Alpha ratio: fraction of alphanumeric/space chars (penalise junk)
    3. Length plausibility: is output length within expected range?
    
    Thresholds are empirically derived from validation set.
    """

    def __init__(
        self,
        accepted_threshold: float = DEFAULT_ACCEPTED_THRESHOLD,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        expected_min_chars: int = 2,
        expected_max_chars: int = 500,
        token_weight: float = 0.5,
        alpha_weight: float = 0.3,
        length_weight: float = 0.2
    ):
        self.accepted_threshold = accepted_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.expected_min_chars = expected_min_chars
        self.expected_max_chars = expected_max_chars
        self.token_weight = token_weight
        self.alpha_weight = alpha_weight
        self.length_weight = length_weight

    def compute_token_confidence(self, scores: Optional[list] = None) -> float:
        """
        Compute mean token probability from beam search scores.
        
        Args:
            scores: List of score tensors from model.generate(output_scores=True)
        
        Returns:
            Mean max softmax probability across tokens
        """
        if not scores:
            return 0.5  # Default when scores unavailable
        
        try:
            import torch
            token_probs = []
            for s in scores:
                if isinstance(s, torch.Tensor):
                    probs = torch.softmax(s, dim=-1)
                    max_prob = probs.max().item()
                    token_probs.append(max_prob)
            
            if token_probs:
                return sum(token_probs) / len(token_probs)
        except (ImportError, Exception):
            pass
        
        return 0.5

    def compute_alpha_ratio(self, text: str) -> float:
        """
        Compute ratio of alphanumeric + space characters to total.
        Low ratio indicates garbled/garbage output.
        
        Args:
            text: Decoded OCR output text
        
        Returns:
            Ratio [0, 1]
        """
        if not text:
            return 0.0
        
        valid_chars = sum(
            1 for c in text
            if c.isalnum() or c.isspace() or c in '.,;:!?-+=/()[]{}$\\'
        )
        return valid_chars / len(text)

    def compute_length_score(self, text: str) -> Tuple[bool, float]:
        """
        Check if output length is within expected range.
        
        Args:
            text: Decoded OCR output text
        
        Returns:
            Tuple of (is_ok, score)
        """
        text_len = len(text.strip())
        is_ok = self.expected_min_chars <= text_len <= self.expected_max_chars
        score = 1.0 if is_ok else 0.4
        
        # Gradual penalty for very short text
        if text_len < self.expected_min_chars:
            score = max(0.1, text_len / self.expected_min_chars * 0.4)
        
        return is_ok, score

    def score(
        self,
        decoded_text: str,
        scores: Optional[list] = None
    ) -> ConfidenceResult:
        """
        Compute composite confidence score from all signals.
        
        Args:
            decoded_text: Decoded OCR output text
            scores: Optional beam search score tensors
        
        Returns:
            ConfidenceResult with composite score and tag
        """
        # Signal 1: Token probability
        token_confidence = self.compute_token_confidence(scores)
        
        # Signal 2: Character composition
        alpha_ratio = self.compute_alpha_ratio(decoded_text)
        
        # Signal 3: Length plausibility
        length_ok, length_score = self.compute_length_score(decoded_text)
        
        # Composite score
        composite = (
            self.token_weight * token_confidence +
            self.alpha_weight * alpha_ratio +
            self.length_weight * length_score
        )
        composite = round(composite, 4)
        
        # Classify
        tag = self.classify(composite)
        
        return ConfidenceResult(
            composite=composite,
            token_confidence=round(token_confidence, 4),
            alpha_ratio=round(alpha_ratio, 4),
            length_ok=length_ok,
            length_score=round(length_score, 4),
            tag=tag
        )

    def classify(self, score: float) -> str:
        """
        Classify confidence score into 3-tier tag.
        
        Args:
            score: Composite confidence score
        
        Returns:
            "ACCEPTED", "LOW_CONFIDENCE", or "FAILED"
        """
        if score >= self.accepted_threshold:
            return "ACCEPTED"
        elif score >= self.low_confidence_threshold:
            return "LOW_CONFIDENCE"
        else:
            return "FAILED"

    def should_retry(self, result: ConfidenceResult) -> bool:
        """
        Determine if a retry with alternate engine is warranted.
        
        Args:
            result: ConfidenceResult from scoring
        
        Returns:
            True if retry recommended
        """
        return result.tag != "ACCEPTED"

    def pick_best(
        self, result_a: ConfidenceResult, result_b: ConfidenceResult
    ) -> int:
        """
        Compare two results and return index of the better one.
        
        Returns:
            0 for result_a, 1 for result_b
        """
        return 0 if result_a.composite >= result_b.composite else 1
