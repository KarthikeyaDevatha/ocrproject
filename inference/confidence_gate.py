"""
Composite Confidence Gate v2 for Adaptive OCR Agent.

FIXES FROM EVALUATOR FEEDBACK:
1. Strict threshold-based decision logic (no ambiguity)
2. Math-aware semantic validation (sum/mean consistency checks)
3. Structured output labels: ACCEPTED / RETRY_REQUIRED / FAILED_EXTRACTION
4. Domain-specific number extraction and arithmetic verification

Combines signals to produce a composite confidence score:
1. Token probability (from model.generate output_scores)
2. Character composition (penalise excessive symbols)
3. Output length plausibility
4. Math semantic validation (NEW)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


# ============================================================================
# STRICT THRESHOLDS — NO AMBIGUITY
# ============================================================================
ACCEPT_THRESHOLD = 0.85       # >= 0.85 → ACCEPTED
RETRY_THRESHOLD = 0.65        # 0.65–0.85 → RETRY_REQUIRED
# < 0.65 → FAILED_EXTRACTION


@dataclass
class ConfidenceResult:
    """Result of confidence gate evaluation."""
    composite: float
    token_confidence: float
    alpha_ratio: float
    length_ok: bool
    length_score: float
    tag: str  # "ACCEPTED" | "RETRY_REQUIRED" | "FAILED_EXTRACTION"
    math_boost: float = 0.0
    math_validated: bool = False
    numbers_found: list = field(default_factory=list)


class MathValidator:
    """
    Semantic math validator.
    
    Extracts numbers from OCR output and verifies arithmetic consistency:
    - Checks if sum/mean/product computations match stated results
    - Boosts confidence when math checks pass
    - Penalises when math is inconsistent
    """

    # Patterns for extracting numbers from text
    NUMBER_PATTERN = re.compile(r'-?\d+\.?\d*')
    
    # Patterns for detecting arithmetic operations
    SUM_PATTERN = re.compile(r'(?:sum|total)\s*[=:]\s*(-?\d+\.?\d*)', re.IGNORECASE)
    MEAN_PATTERN = re.compile(r'(?:mean|average|avg)\s*[=:]\s*(-?\d+\.?\d*)', re.IGNORECASE)
    EQUALS_PATTERN = re.compile(r'=\s*(-?\d+\.?\d*)\s*$')
    ADDITION_PATTERN = re.compile(r'^[\d\s\+\-\.]+$')

    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text."""
        matches = self.NUMBER_PATTERN.findall(text)
        numbers = []
        for m in matches:
            try:
                numbers.append(float(m))
            except ValueError:
                continue
        return numbers

    def validate_arithmetic(self, text: str) -> Tuple[float, Dict]:
        """
        Validate arithmetic consistency of OCR output.
        
        Returns:
            Tuple of (boost_score, validation_details)
            boost_score: -0.15 to +0.15 adjustment to confidence
        """
        details = {
            "numbers_found": [],
            "sum_check": None,
            "mean_check": None,
            "equation_check": None,
            "is_math": False,
            "validated": False
        }

        numbers = self.extract_numbers(text)
        details["numbers_found"] = numbers

        if len(numbers) < 2:
            return 0.0, details

        # Does the text look like math?
        has_operators = bool(re.search(r'[\+\-\*\/\=]', text))
        has_math_words = bool(re.search(
            r'\b(sum|total|mean|average|avg|result|answer)\b', text, re.IGNORECASE
        ))
        
        if not has_operators and not has_math_words:
            return 0.0, details

        details["is_math"] = True
        boost = 0.0

        # Check 1: Addition chain (e.g., "25 + 30 + 28 = 209")
        # Look for = result at the end
        eq_match = self.EQUALS_PATTERN.search(text)
        if eq_match and '+' in text:
            stated_result = float(eq_match.group(1))
            # Numbers before the = sign
            before_eq = text[:text.rfind('=')]
            addends = self.extract_numbers(before_eq)
            
            if addends:
                computed_sum = sum(addends)
                tolerance = max(1.0, abs(computed_sum) * 0.05)  # 5% tolerance
                
                if abs(computed_sum - stated_result) <= tolerance:
                    boost += 0.12
                    details["sum_check"] = "PASS"
                    details["equation_check"] = f"{computed_sum} ≈ {stated_result}"
                else:
                    boost -= 0.05
                    details["sum_check"] = "FAIL"
                    details["equation_check"] = f"{computed_sum} ≠ {stated_result}"

        # Check 2: Sum = N stated explicitly
        sum_match = self.SUM_PATTERN.search(text)
        if sum_match and numbers:
            stated_sum = float(sum_match.group(1))
            # Exclude the sum value itself from addends
            addends = [n for n in numbers if n != stated_sum]
            if addends:
                computed = sum(addends)
                if abs(computed - stated_sum) <= max(1.0, abs(computed) * 0.05):
                    boost += 0.10
                    details["sum_check"] = "PASS"
                else:
                    boost -= 0.05
                    details["sum_check"] = "FAIL"

        # Check 3: Mean = N stated explicitly
        mean_match = self.MEAN_PATTERN.search(text)
        if mean_match and numbers:
            stated_mean = float(mean_match.group(1))
            candidates = [n for n in numbers if n != stated_mean]
            if candidates:
                computed_mean = sum(candidates) / len(candidates)
                if abs(computed_mean - stated_mean) <= max(0.5, abs(computed_mean) * 0.05):
                    boost += 0.10
                    details["mean_check"] = "PASS"
                else:
                    boost -= 0.05
                    details["mean_check"] = "FAIL"

        # Check 4: Simple equation like "209 / 7 = 29.857"
        div_match = re.search(r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)', text)
        if div_match:
            a, b, c = float(div_match.group(1)), float(div_match.group(2)), float(div_match.group(3))
            if b != 0:
                computed = a / b
                if abs(computed - c) <= max(0.01, abs(computed) * 0.02):
                    boost += 0.12
                    details["equation_check"] = f"{a}/{b}={computed:.3f} ≈ {c}"

        details["validated"] = boost > 0
        
        # Clamp boost
        boost = max(-0.15, min(0.15, boost))
        return boost, details

    def is_math_content(self, text: str) -> bool:
        """Quick check if text contains math content."""
        math_signals = [
            re.search(r'[\+\-\*\/\=]', text),
            re.search(r'\b\d+\b.*\b\d+\b', text),
            re.search(r'\b(sum|mean|total|avg)\b', text, re.IGNORECASE),
        ]
        return sum(1 for s in math_signals if s) >= 2


class CompositeConfidenceGate:
    """
    Multi-signal confidence scorer with math-aware semantic validation.
    
    Signals:
    1. Token probability: mean max softmax from beam search scores
    2. Alpha ratio: fraction of alphanumeric/space chars (penalise junk)
    3. Length plausibility: is output length within expected range?
    4. Math validation: arithmetic consistency checks (NEW)
    
    Labels (strict):
    - ACCEPTED (>= 0.85): high confidence, no retry needed
    - RETRY_REQUIRED (0.65–0.85): run alternate engine
    - FAILED_EXTRACTION (< 0.65): flag for manual review
    """

    def __init__(
        self,
        accepted_threshold: float = ACCEPT_THRESHOLD,
        retry_threshold: float = RETRY_THRESHOLD,
        expected_min_chars: int = 2,
        expected_max_chars: int = 500,
        token_weight: float = 0.45,
        alpha_weight: float = 0.25,
        length_weight: float = 0.15,
        math_weight: float = 0.15
    ):
        self.accepted_threshold = accepted_threshold
        self.retry_threshold = retry_threshold
        self.expected_min_chars = expected_min_chars
        self.expected_max_chars = expected_max_chars
        self.token_weight = token_weight
        self.alpha_weight = alpha_weight
        self.length_weight = length_weight
        self.math_weight = math_weight
        self.math_validator = MathValidator()

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
        Compute ratio of valid characters to total.
        Allows alphanumeric, spaces, math operators, and common punctuation.
        """
        if not text:
            return 0.0
        
        valid_chars = sum(
            1 for c in text
            if c.isalnum() or c.isspace() or c in '.,;:!?-+=/()[]{}$\\'
        )
        return valid_chars / len(text)

    def compute_length_score(self, text: str) -> Tuple[bool, float]:
        """Check if output length is within expected range."""
        text_len = len(text.strip())
        is_ok = self.expected_min_chars <= text_len <= self.expected_max_chars
        score = 1.0 if is_ok else 0.4
        
        if text_len < self.expected_min_chars:
            score = max(0.1, text_len / self.expected_min_chars * 0.4)
        
        return is_ok, score

    def score(
        self,
        decoded_text: str,
        scores: Optional[list] = None
    ) -> ConfidenceResult:
        """
        Compute composite confidence score from all signals including math validation.
        """
        # Signal 1: Token probability
        token_confidence = self.compute_token_confidence(scores)
        
        # Signal 2: Character composition
        alpha_ratio = self.compute_alpha_ratio(decoded_text)
        
        # Signal 3: Length plausibility
        length_ok, length_score = self.compute_length_score(decoded_text)
        
        # Signal 4: Math semantic validation (NEW)
        math_boost, math_details = self.math_validator.validate_arithmetic(decoded_text)
        math_base = 0.5  # neutral baseline
        if math_details["validated"]:
            math_base = 1.0  # math checks passed
        elif math_details["is_math"] and not math_details["validated"]:
            math_base = 0.3  # math detected but failed checks
        
        # Composite score with math component
        composite = (
            self.token_weight * token_confidence +
            self.alpha_weight * alpha_ratio +
            self.length_weight * length_score +
            self.math_weight * math_base
        )
        
        # Apply math boost/penalty on top
        composite += math_boost
        composite = max(0.0, min(1.0, composite))
        composite = round(composite, 4)
        
        # Classify with strict thresholds
        tag = self.classify(composite)
        
        return ConfidenceResult(
            composite=composite,
            token_confidence=round(token_confidence, 4),
            alpha_ratio=round(alpha_ratio, 4),
            length_ok=length_ok,
            length_score=round(length_score, 4),
            tag=tag,
            math_boost=round(math_boost, 4),
            math_validated=math_details.get("validated", False),
            numbers_found=math_details.get("numbers_found", [])
        )

    def classify(self, score: float) -> str:
        """
        Classify with strict, unambiguous thresholds.
        
        >= 0.85 → ACCEPTED
        0.65–0.85 → RETRY_REQUIRED
        < 0.65 → FAILED_EXTRACTION
        """
        if score >= self.accepted_threshold:
            return "ACCEPTED"
        elif score >= self.retry_threshold:
            return "RETRY_REQUIRED"
        else:
            return "FAILED_EXTRACTION"

    def should_retry(self, result: ConfidenceResult) -> bool:
        """Only retry RETRY_REQUIRED lines. ACCEPTED pass through. FAILED are flagged."""
        return result.tag == "RETRY_REQUIRED"

    def pick_best(
        self, result_a: ConfidenceResult, result_b: ConfidenceResult
    ) -> int:
        """
        Compare two results using composite + math validation.
        Prefer math-validated results even if raw confidence is slightly lower.
        """
        score_a = result_a.composite + (0.05 if result_a.math_validated else 0)
        score_b = result_b.composite + (0.05 if result_b.math_validated else 0)
        return 0 if score_a >= score_b else 1

    def fuse_results(
        self,
        result_a: ConfidenceResult,
        text_a: str,
        result_b: ConfidenceResult,
        text_b: str
    ) -> Tuple[str, ConfidenceResult]:
        """
        Voting/fusion layer for multi-engine results.
        
        Strategy:
        1. If both agree on text → boost confidence
        2. If math-validated result exists → prefer it
        3. Otherwise → pick higher confidence
        """
        # Check text agreement (normalized)
        norm_a = re.sub(r'\s+', ' ', text_a.strip().lower())
        norm_b = re.sub(r'\s+', ' ', text_b.strip().lower())
        
        if norm_a == norm_b:
            # Consensus! Boost the better one
            best_idx = self.pick_best(result_a, result_b)
            chosen_result = result_a if best_idx == 0 else result_b
            chosen_text = text_a if best_idx == 0 else text_b
            
            # Consensus boost
            boosted = ConfidenceResult(
                composite=min(1.0, chosen_result.composite + 0.10),
                token_confidence=chosen_result.token_confidence,
                alpha_ratio=chosen_result.alpha_ratio,
                length_ok=chosen_result.length_ok,
                length_score=chosen_result.length_score,
                tag=self.classify(min(1.0, chosen_result.composite + 0.10)),
                math_boost=chosen_result.math_boost,
                math_validated=chosen_result.math_validated,
                numbers_found=chosen_result.numbers_found
            )
            return chosen_text, boosted
        
        # No consensus → prefer math-validated
        if result_a.math_validated and not result_b.math_validated:
            return text_a, result_a
        if result_b.math_validated and not result_a.math_validated:
            return text_b, result_b
        
        # Neither validated or both → pick higher confidence
        best_idx = self.pick_best(result_a, result_b)
        return (text_a, result_a) if best_idx == 0 else (text_b, result_b)
