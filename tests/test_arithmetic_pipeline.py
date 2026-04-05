"""
Tests for the Zero-Hallucination Arithmetic OCR Pipeline.
Tests cover:
- TokenCleaner: LaTeX stripping, noise removal, tokenization
- SimilarityCorrector: digit/operator correction via Levenshtein
- ArithmeticValidator: validation, reconstruction, computation
- ArithmeticConfidenceScorer: scoring logic
- ArithmeticPipeline: end-to-end with synthetic image and text-only mode
"""

import pytest
import numpy as np
import cv2
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.token_corrector import (
    TokenCleaner,
    SimilarityCorrector,
    CorrectedToken,
    levenshtein_distance,
    similarity_score,
    DIGIT_CONFUSIONS,
    OPERATOR_CONFUSIONS
)
from inference.arithmetic_validator import (
    ArithmeticValidator,
    ArithmeticConfidenceScorer,
    ArithmeticResult,
    ValidationResult
)
from inference.arithmetic_pipeline import (
    ArithmeticPipeline,
    ArithmeticPipelineConfig
)


# ============================================================================
# HELPERS
# ============================================================================

def make_synthetic_image(text: str, width: int = 600, height: int = 100) -> np.ndarray:
    """Create a synthetic image with text rendered via OpenCV."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.putText(img, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return img


# ============================================================================
# LEVENSHTEIN DISTANCE TESTS
# ============================================================================

class TestLevenshteinDistance:
    def test_identical(self):
        assert levenshtein_distance("30", "30") == 0

    def test_single_substitution(self):
        assert levenshtein_distance("30", "20") == 1

    def test_single_insertion(self):
        assert levenshtein_distance("3", "30") == 1

    def test_single_deletion(self):
        assert levenshtein_distance("30", "3") == 1

    def test_completely_different(self):
        assert levenshtein_distance("abc", "xyz") == 3

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3


class TestSimilarityScore:
    def test_identical(self):
        assert similarity_score("30", "30") == 1.0

    def test_similar(self):
        score = similarity_score("30", "20")
        assert 0.0 < score < 1.0

    def test_different(self):
        score = similarity_score("abc", "xyz")
        assert score == 0.0

    def test_empty(self):
        assert similarity_score("", "") == 1.0
        assert similarity_score("abc", "") == 0.0


# ============================================================================
# TOKEN CLEANER TESTS
# ============================================================================

class TestTokenCleaner:
    @pytest.fixture
    def cleaner(self):
        return TokenCleaner()

    # --- LaTeX stripping ---

    def test_strip_frac(self, cleaner):
        result = cleaner.clean_raw_output("\\frac{25}{5}")
        # Should strip LaTeX, keep any remaining digits
        assert "\\frac" not in result
        assert "{" not in result
        assert "}" not in result

    def test_strip_times(self, cleaner):
        result = cleaner.clean_raw_output("25 \\times 5")
        assert "\\times" not in result

    def test_strip_ldots(self, cleaner):
        result = cleaner.clean_raw_output("1 + 2 + \\ldots + 10")
        assert "\\ldots" not in result

    def test_strip_sqrt(self, cleaner):
        result = cleaner.clean_raw_output("\\sqrt{16}")
        assert "\\sqrt" not in result

    def test_strip_mixed_latex(self, cleaner):
        raw = "\\frac{25 + 30}{2} = \\ldots"
        result = cleaner.clean_raw_output(raw)
        assert "\\" not in result
        assert "{" not in result
        assert "}" not in result

    def test_strip_dollar_signs(self, cleaner):
        result = cleaner.clean_raw_output("$25 + 30$")
        assert "$" not in result

    # --- Noise removal ---

    def test_remove_non_numeric(self, cleaner):
        result = cleaner.clean_raw_output("abc 25 def 30 ghi")
        # Non-digit non-operator chars should be removed
        assert "abc" not in result
        assert "def" not in result

    def test_preserve_numbers(self, cleaner):
        result = cleaner.clean_raw_output("25 + 30 + 28")
        assert "25" in result
        assert "30" in result
        assert "28" in result

    def test_preserve_operators(self, cleaner):
        result = cleaner.clean_raw_output("25 + 30 / 2 = 27.5")
        assert "+" in result
        assert "/" in result
        assert "=" in result

    # --- Digit confusion correction ---

    def test_O_to_0(self, cleaner):
        result = cleaner.clean_raw_output("3O")
        assert "30" in result

    def test_l_to_1(self, cleaner):
        result = cleaner.clean_raw_output("l5")
        assert "15" in result

    # --- Tokenization ---

    def test_tokenize_simple(self, cleaner):
        tokens = cleaner.tokenize("25 + 30 + 28")
        assert "25" in tokens
        assert "30" in tokens
        assert "28" in tokens
        assert "+" in tokens

    def test_tokenize_empty(self, cleaner):
        tokens = cleaner.tokenize("")
        assert tokens == []

    def test_tokenize_mixed(self, cleaner):
        tokens = cleaner.tokenize("25 + 30 / 2 = 27.5")
        assert "25" in tokens
        assert "30" in tokens

    # --- Hallucination detection ---

    def test_detect_hallucination(self, cleaner):
        has_h, patterns = cleaner.has_hallucination("\\frac{25}{5}")
        assert has_h is True
        assert len(patterns) > 0

    def test_no_hallucination(self, cleaner):
        has_h, patterns = cleaner.has_hallucination("25 + 30 + 28")
        assert has_h is False

    def test_detect_structural_noise(self, cleaner):
        has_h, patterns = cleaner.has_hallucination("25 + {30}")
        assert has_h is True


# ============================================================================
# SIMILARITY CORRECTOR TESTS
# ============================================================================

class TestSimilarityCorrector:
    @pytest.fixture
    def corrector(self):
        return SimilarityCorrector(
            number_range=(0, 100),
            allowed_operators=['+', '-', '/', '='],
            similarity_threshold=0.5
        )

    # --- Number correction ---

    def test_correct_valid_number(self, corrector):
        result = corrector.correct_number("30")
        assert result is not None
        assert result.corrected == "30"
        assert result.confidence == 1.0

    def test_correct_number_from_string(self, corrector):
        result = corrector.correct_number("25")
        assert result is not None
        assert result.corrected == "25"

    def test_correct_similar_number(self, corrector):
        # "3O" after cleaning would become "30" already in cleaner,
        # but test direct correction
        result = corrector.correct_number("31")
        assert result is not None
        assert result.corrected == "31"

    def test_reject_out_of_range(self, corrector):
        result = corrector.correct_number("150")
        # Should map to nearest in range or reject
        if result is not None:
            val = int(result.corrected)
            assert 0 <= val <= 100

    # --- Operator correction ---

    def test_correct_valid_operator(self, corrector):
        result = corrector.correct_operator("+")
        assert result is not None
        assert result.corrected == "+"
        assert result.confidence == 1.0

    def test_correct_division(self, corrector):
        result = corrector.correct_operator("/")
        assert result is not None
        assert result.corrected == "/"

    def test_correct_equals(self, corrector):
        result = corrector.correct_operator("=")
        assert result is not None
        assert result.corrected == "="

    # --- Token correction ---

    def test_correct_token_number(self, corrector):
        result = corrector.correct_token("42")
        assert result.token_type == 'number'
        assert result.corrected == "42"

    def test_correct_token_operator(self, corrector):
        result = corrector.correct_token("+")
        assert result.token_type == 'operator'
        assert result.corrected == "+"

    def test_discard_invalid(self, corrector):
        result = corrector.correct_token("@#@#")
        assert result.token_type == 'discarded'

    # --- Batch correction ---

    def test_correct_all(self, corrector):
        tokens = ["25", "+", "30", "+", "28"]
        results = corrector.correct_all(tokens)
        numbers = corrector.extract_numbers(results)
        assert 25 in numbers
        assert 30 in numbers
        assert 28 in numbers

    def test_extract_numbers(self, corrector):
        tokens = ["25", "+", "30", "/", "2"]
        results = corrector.correct_all(tokens)
        numbers = corrector.extract_numbers(results)
        assert len(numbers) >= 2


# ============================================================================
# ARITHMETIC VALIDATOR TESTS
# ============================================================================

class TestArithmeticValidator:
    @pytest.fixture
    def validator(self):
        return ArithmeticValidator(
            number_range=(0, 100),
            allowed_operators=['+', '-', '/', '='],
            min_numbers=2
        )

    # --- Token validation ---

    def test_validate_valid_tokens(self, validator):
        tokens = [
            CorrectedToken("25", "25", "number", 1.0, 0),
            CorrectedToken("+", "+", "operator", 1.0, 0),
            CorrectedToken("30", "30", "number", 1.0, 0),
        ]
        result = validator.validate_tokens(tokens)
        assert result.is_valid is True
        assert len(result.valid_numbers) == 2

    def test_validate_insufficient_numbers(self, validator):
        tokens = [
            CorrectedToken("25", "25", "number", 1.0, 0),
        ]
        result = validator.validate_tokens(tokens)
        assert result.is_valid is False
        assert any("need at least" in r for r in result.reasons)

    def test_validate_out_of_range(self, validator):
        tokens = [
            CorrectedToken("25", "25", "number", 1.0, 0),
            CorrectedToken("150", "150", "number", 1.0, 0),
        ]
        result = validator.validate_tokens(tokens)
        # 150 is out of range [0, 100]
        assert any("outside range" in r for r in result.reasons)

    # --- Expression reconstruction ---

    def test_reconstruct_mean(self, validator):
        numbers = [25, 30, 28, 35, 32, 30, 29]
        expr = validator.reconstruct_expression(numbers)
        assert "25 + 30 + 28 + 35 + 32 + 30 + 29" in expr
        assert "/ 7" in expr

    def test_reconstruct_two_numbers(self, validator):
        numbers = [25, 30]
        expr = validator.reconstruct_expression(numbers)
        assert "25 + 30" in expr
        assert "/ 2" in expr

    def test_reconstruct_single(self, validator):
        numbers = [42]
        expr = validator.reconstruct_expression(numbers)
        assert expr == "42"

    def test_reconstruct_empty(self, validator):
        expr = validator.reconstruct_expression([])
        assert expr == ""

    # --- Mathematical computation ---

    def test_compute_mean(self, validator):
        numbers = [25, 30, 28, 35, 32, 30, 29]
        result, error = validator.compute_result(numbers)
        assert error == ""
        expected = sum(numbers) / len(numbers)
        assert abs(result - expected) < 0.01

    def test_compute_two_numbers(self, validator):
        numbers = [20, 40]
        result, error = validator.compute_result(numbers)
        assert result == 30.0
        assert error == ""

    def test_compute_empty(self, validator):
        result, error = validator.compute_result([])
        assert result is None
        assert "No numbers" in error

    # --- Hallucination check ---

    def test_detect_frac_hallucination(self, validator):
        has_h, violations = validator.check_hallucination("\\frac{25}{5}")
        assert has_h is True
        assert "LaTeX fraction" in violations

    def test_detect_ldots_hallucination(self, validator):
        has_h, violations = validator.check_hallucination("1 + \\ldots + 10")
        assert has_h is True

    def test_no_hallucination(self, validator):
        has_h, violations = validator.check_hallucination("25 + 30 + 28")
        assert has_h is False
        assert len(violations) == 0


# ============================================================================
# CONFIDENCE SCORER TESTS
# ============================================================================

class TestArithmeticConfidenceScorer:
    @pytest.fixture
    def scorer(self):
        return ArithmeticConfidenceScorer(confidence_threshold=0.6)

    def test_high_confidence(self, scorer):
        tokens = [
            CorrectedToken("25", "25", "number", 1.0, 0),
            CorrectedToken("+", "+", "operator", 1.0, 0),
            CorrectedToken("30", "30", "number", 1.0, 0),
        ]
        validation = ValidationResult(True, [], [25, 30], ['+'])
        confidence, details = scorer.score(tokens, validation, [25, 30])
        assert confidence > 0.5
        assert details['meets_threshold'] is True

    def test_low_confidence(self, scorer):
        tokens = [
            CorrectedToken("abc", "25", "number", 0.2, 3),
        ]
        validation = ValidationResult(False, ["Only 1 number"], [25], [])
        confidence, details = scorer.score(tokens, validation, [25])
        assert confidence < 0.6
        assert details['meets_threshold'] is False

    def test_should_retry(self, scorer):
        assert scorer.should_retry(0.3) is True
        assert scorer.should_retry(0.8) is False
        assert scorer.should_retry(0.6) is False


# ============================================================================
# ARITHMETIC PIPELINE TESTS (TEXT-ONLY MODE)
# ============================================================================

class TestArithmeticPipelineTextOnly:
    """Test the pipeline using process_tokens_only() — no model required."""

    @pytest.fixture
    def pipeline(self):
        config = ArithmeticPipelineConfig(debug=False)
        return ArithmeticPipeline(config)

    def test_clean_arithmetic(self, pipeline):
        result = pipeline.process_tokens_only("25 + 30 + 28 + 35 + 32")
        assert result['status'] == 'success'
        assert len(result['numbers']) == 5
        assert result['result'] is not None
        assert result['confidence'] > 0

    def test_latex_hallucination_stripped(self, pipeline):
        result = pipeline.process_tokens_only("\\frac{25 + 30}{2}")
        # LaTeX should be stripped, numbers extracted
        assert "\\frac" not in result.get('expression', '')

    def test_mixed_digits_operators(self, pipeline):
        result = pipeline.process_tokens_only("25 + 30 + 28 + 35 + 32 + 30 + 29")
        assert result['status'] == 'success'
        assert len(result['numbers']) == 7
        expected_mean = (25 + 30 + 28 + 35 + 32 + 30 + 29) / 7
        assert abs(result['result'] - expected_mean) < 0.01

    def test_single_number_insufficient(self, pipeline):
        result = pipeline.process_tokens_only("42")
        # Only 1 number, needs at least 2
        assert result['status'] in ['retry', 'error']

    def test_empty_input(self, pipeline):
        result = pipeline.process_tokens_only("")
        assert result['status'] in ['retry', 'error']

    def test_all_latex_rejected(self, pipeline):
        result = pipeline.process_tokens_only("\\frac{\\sqrt{x}}{\\ldots}")
        # After stripping, nothing valid should remain
        assert result['status'] in ['retry', 'error']

    def test_numbers_in_noisy_text(self, pipeline):
        result = pipeline.process_tokens_only("abc 25 def + 30 ghi + 28")
        assert result['status'] == 'success'
        assert 25 in result['numbers']
        assert 30 in result['numbers']
        assert 28 in result['numbers']


# ============================================================================
# END-TO-END PIPELINE TEST (SYNTHETIC IMAGE)
# ============================================================================

class TestArithmeticPipelineEndToEnd:
    """
    End-to-end tests using synthetic images.
    These tests require the TrOCR model — skip if not available.
    """

    @pytest.fixture
    def pipeline(self):
        config = ArithmeticPipelineConfig(debug=False)
        p = ArithmeticPipeline(config)
        return p

    def test_synthetic_image_processable(self, pipeline):
        """Test that a synthetic image can be preprocessed without error."""
        img = make_synthetic_image("25 + 30 + 28")
        # Just test preprocessing step
        preprocessed = pipeline._preprocess_image(img)
        assert preprocessed is not None
        assert len(preprocessed.shape) == 3
        assert preprocessed.shape[2] == 3

    @pytest.mark.skipif(
        not os.environ.get("RUN_MODEL_TESTS"),
        reason="Model tests require RUN_MODEL_TESTS=1"
    )
    def test_end_to_end_synthetic(self, pipeline):
        """Full end-to-end test with model (requires model download)."""
        img = make_synthetic_image("25 + 30 + 28")

        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            result = pipeline.process(f.name)
            os.unlink(f.name)

        assert result['status'] in ['success', 'retry']
        assert 'numbers' in result
        assert 'confidence' in result


# ============================================================================
# INTEGRATION: FULL PIPELINE CONSISTENCY
# ============================================================================

class TestPipelineConsistency:
    """Verify pipeline output format consistency."""

    @pytest.fixture
    def pipeline(self):
        config = ArithmeticPipelineConfig(debug=False)
        return ArithmeticPipeline(config)

    def test_output_has_required_keys(self, pipeline):
        result = pipeline.process_tokens_only("25 + 30 + 28")
        required_keys = ['status', 'numbers', 'expression', 'result', 'confidence']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_output_json_serializable(self, pipeline):
        import json
        result = pipeline.process_tokens_only("25 + 30 + 28")
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_deterministic_output(self, pipeline):
        """Same input should produce same output."""
        r1 = pipeline.process_tokens_only("25 + 30 + 28")
        r2 = pipeline.process_tokens_only("25 + 30 + 28")
        assert r1['numbers'] == r2['numbers']
        assert r1['result'] == r2['result']
        assert r1['expression'] == r2['expression']

    def test_status_values(self, pipeline):
        """Status should be one of: success, retry, error."""
        r1 = pipeline.process_tokens_only("25 + 30 + 28")
        assert r1['status'] in ['success', 'retry', 'error']

        r2 = pipeline.process_tokens_only("")
        assert r2['status'] in ['success', 'retry', 'error']
