"""
Arithmetic Validator for Zero-Hallucination OCR Pipeline.
Provides hard constraint validation, expression reconstruction,
mathematical computation, and confidence scoring.
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from inference.token_corrector import CorrectedToken


@dataclass
class ValidationResult:
    """Result from token validation."""
    is_valid: bool
    reasons: List[str]
    valid_numbers: List[int]
    valid_operators: List[str]


@dataclass
class ArithmeticResult:
    """Result from arithmetic computation."""
    status: str  # 'success', 'retry', 'error'
    numbers: List[int]
    expression: str
    result: Optional[float]
    confidence: float
    reason: str = ""
    pass_details: Optional[List[Dict]] = None


class ArithmeticValidator:
    """
    Validates corrected tokens against hard constraints and
    reconstructs arithmetic expressions from extracted numbers.
    """

    def __init__(
        self,
        number_range: Tuple[int, int] = (0, 100),
        allowed_operators: List[str] = None,
        min_numbers: int = 2
    ):
        """
        Args:
            number_range: Valid number range (inclusive)
            allowed_operators: List of valid operators
            min_numbers: Minimum number of valid numbers required
        """
        self.number_range = number_range
        self.allowed_operators = allowed_operators or ['+', '-', '/', '=']
        self.min_numbers = min_numbers

    def validate_tokens(
        self,
        corrected_tokens: List[CorrectedToken]
    ) -> ValidationResult:
        """
        Validate corrected tokens against hard constraints.

        Rejects if:
        - Any token not in allowed vocabulary
        - Numbers outside expected range
        - Operator set violated
        - Less than min_numbers valid numbers detected

        Args:
            corrected_tokens: List of CorrectedToken objects

        Returns:
            ValidationResult with validity status and reasons
        """
        reasons = []
        valid_numbers = []
        valid_operators = []

        for ct in corrected_tokens:
            if ct.token_type == 'number':
                try:
                    val = int(ct.corrected)
                    if self.number_range[0] <= val <= self.number_range[1]:
                        valid_numbers.append(val)
                    else:
                        reasons.append(
                            f"Number {val} outside range "
                            f"[{self.number_range[0]}, {self.number_range[1]}]"
                        )
                except ValueError:
                    reasons.append(f"Invalid number token: {ct.corrected}")

            elif ct.token_type == 'operator':
                if ct.corrected in self.allowed_operators:
                    valid_operators.append(ct.corrected)
                else:
                    reasons.append(
                        f"Operator '{ct.corrected}' not in allowed set "
                        f"{self.allowed_operators}"
                    )

            elif ct.token_type == 'discarded':
                reasons.append(f"Discarded token: '{ct.original}'")

        # Check minimum number count
        if len(valid_numbers) < self.min_numbers:
            reasons.append(
                f"Only {len(valid_numbers)} valid numbers found, "
                f"need at least {self.min_numbers}"
            )

        is_valid = len(reasons) == 0 and len(valid_numbers) >= self.min_numbers

        return ValidationResult(
            is_valid=is_valid,
            reasons=reasons,
            valid_numbers=valid_numbers,
            valid_operators=valid_operators
        )

    def reconstruct_expression(self, numbers: List[int]) -> str:
        """
        Reconstruct mean expression from extracted numbers.
        NEVER trusts OCR structure — builds from numbers only.

        Args:
            numbers: List of valid integers

        Returns:
            Expression string, e.g. "(25 + 30 + 28) / 3"
        """
        if not numbers:
            return ""

        if len(numbers) == 1:
            return str(numbers[0])

        sum_expr = " + ".join(str(n) for n in numbers)
        count = len(numbers)
        return f"({sum_expr}) / {count}"

    def compute_result(self, numbers: List[int]) -> Tuple[Optional[float], str]:
        """
        Compute the mean of extracted numbers.

        Args:
            numbers: List of valid integers

        Returns:
            Tuple of (result_value, error_message)
        """
        if not numbers:
            return None, "No numbers to compute"

        if len(numbers) == 0:
            return None, "Division by zero: no numbers"

        try:
            total = sum(numbers)
            count = len(numbers)
            result = total / count
            return round(result, 4), ""
        except ZeroDivisionError:
            return None, "Division by zero"
        except Exception as e:
            return None, f"Computation error: {str(e)}"

    def check_hallucination(self, raw_text: str) -> Tuple[bool, List[str]]:
        """
        Final hallucination check on raw text.

        Args:
            raw_text: Raw OCR output

        Returns:
            Tuple of (has_hallucination, list_of_violations)
        """
        violations = []

        # Banned patterns
        banned = [
            (r'\\frac', 'LaTeX fraction'),
            (r'\\ldots', 'LaTeX ellipsis'),
            (r'\\times', 'LaTeX times'),
            (r'\\sqrt', 'LaTeX sqrt'),
            (r'\\sum', 'LaTeX sum'),
            (r'\\int', 'LaTeX integral'),
            (r'\\begin', 'LaTeX environment'),
            (r'\\end', 'LaTeX environment'),
            (r'\$', 'Dollar sign (LaTeX delimiter)'),
            (r'\{', 'Curly brace (LaTeX)'),
            (r'\}', 'Curly brace (LaTeX)'),
        ]

        for pattern, name in banned:
            if re.search(pattern, raw_text):
                violations.append(name)

        return len(violations) > 0, violations


class ArithmeticConfidenceScorer:
    """
    Computes confidence score for the arithmetic pipeline output.
    Scores based on token correction quality, number consistency,
    and structural validity.
    """

    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold

    def score(
        self,
        corrected_tokens: List[CorrectedToken],
        validation_result: ValidationResult,
        numbers: List[int],
        number_range: Tuple[int, int] = (0, 100)
    ) -> Tuple[float, Dict]:
        """
        Compute overall confidence score.

        Args:
            corrected_tokens: List of corrected tokens
            validation_result: Result from validation
            numbers: Extracted numbers
            number_range: Expected number range

        Returns:
            Tuple of (confidence_score, details_dict)
        """
        details = {}

        # 1. Token correction confidence (avg confidence of corrections)
        if corrected_tokens:
            token_confidences = [ct.confidence for ct in corrected_tokens]
            token_score = sum(token_confidences) / len(token_confidences)
        else:
            token_score = 0.0
        details['token_correction_score'] = round(token_score, 4)

        # 2. Number consistency score
        if numbers:
            # Check if numbers are within expected range
            in_range = sum(
                1 for n in numbers
                if number_range[0] <= n <= number_range[1]
            )
            range_score = in_range / len(numbers)

            # Check for reasonable spread (not all same, not too wild)
            if len(numbers) > 1:
                mean = sum(numbers) / len(numbers)
                variance = sum((n - mean) ** 2 for n in numbers) / len(numbers)
                std_dev = variance ** 0.5
                # Reasonable spread: std_dev between 0 and 30
                spread_score = max(0.0, 1.0 - (std_dev / 50.0))
            else:
                spread_score = 0.8

            number_score = (range_score * 0.6 + spread_score * 0.4)
        else:
            number_score = 0.0
        details['number_consistency_score'] = round(number_score, 4)

        # 3. Structural validity score
        if validation_result.is_valid:
            structure_score = 1.0
        else:
            # Partial credit based on how many constraints passed
            total_issues = len(validation_result.reasons)
            structure_score = max(0.0, 1.0 - (total_issues * 0.2))
        details['structural_validity_score'] = round(structure_score, 4)

        # 4. Number count score (more numbers = more confident in mean)
        if len(numbers) >= 3:
            count_score = 1.0
        elif len(numbers) == 2:
            count_score = 0.7
        elif len(numbers) == 1:
            count_score = 0.4
        else:
            count_score = 0.0
        details['number_count_score'] = round(count_score, 4)

        # Weighted overall score
        overall = (
            token_score * 0.3 +
            number_score * 0.3 +
            structure_score * 0.25 +
            count_score * 0.15
        )
        details['overall_confidence'] = round(overall, 4)
        details['meets_threshold'] = overall >= self.confidence_threshold

        return round(overall, 4), details

    def should_retry(self, confidence: float) -> bool:
        """Check if confidence is below threshold and should retry."""
        return confidence < self.confidence_threshold
