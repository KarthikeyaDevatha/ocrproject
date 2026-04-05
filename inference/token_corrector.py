"""
Token Corrector for Zero-Hallucination Arithmetic OCR Pipeline.
Provides LaTeX stripping, token cleaning, and similarity-based digit/operator correction.
"""

import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CorrectedToken:
    """Result of correcting a single token."""
    original: str
    corrected: str
    token_type: str  # 'number', 'operator', 'discarded'
    confidence: float  # 0.0 - 1.0 based on edit distance
    edit_distance: int


# ============================================================================
# LATEX / HALLUCINATION PATTERNS TO STRIP
# ============================================================================

LATEX_PATTERNS = [
    r'\\frac\{[^}]*\}\{[^}]*\}',   # \frac{...}{...}
    r'\\frac',                       # standalone \frac
    r'\\sqrt\{[^}]*\}',             # \sqrt{...}
    r'\\sqrt',                       # standalone \sqrt
    r'\\times',                      # \times
    r'\\div',                        # \div
    r'\\cdot',                       # \cdot
    r'\\ldots',                      # \ldots
    r'\\dots',                       # \dots
    r'\\pm',                         # \pm
    r'\\mp',                         # \mp
    r'\\leq',                        # \leq
    r'\\geq',                        # \geq
    r'\\neq',                        # \neq
    r'\\approx',                     # \approx
    r'\\sum',                        # \sum
    r'\\prod',                       # \prod
    r'\\int',                        # \int
    r'\\lim',                        # \lim
    r'\\infty',                      # \infty
    r'\\alpha',                      # greek letters
    r'\\beta',
    r'\\gamma',
    r'\\delta',
    r'\\theta',
    r'\\pi',
    r'\\sigma',
    r'\\mu',
    r'\\lambda',
    r'\\begin\{[^}]*\}',            # \begin{...}
    r'\\end\{[^}]*\}',              # \end{...}
    r'\\left',                       # \left
    r'\\right',                      # \right
    r'\\text\{[^}]*\}',             # \text{...}
    r'\\mathrm\{[^}]*\}',           # \mathrm{...}
    r'\\mathbf\{[^}]*\}',           # \mathbf{...}
    r'\\overline\{[^}]*\}',         # \overline{...}
    r'\\bar\{[^}]*\}',              # \bar{...}
    r'\\hat\{[^}]*\}',              # \hat{...}
    r'\\vec\{[^}]*\}',              # \vec{...}
    r'\\[a-zA-Z]+',                  # catch-all: any remaining \command
]

# Characters that are structural LaTeX noise
STRUCTURAL_NOISE = set('{}[]$\\^_&~')

# Common OCR character confusions for digits
DIGIT_CONFUSIONS = {
    'O': '0', 'o': '0',
    'Q': '0',
    'D': '0',
    'l': '1', 'I': '1', 'i': '1', '|': '1',
    'Z': '2', 'z': '2',
    'E': '3',
    'A': '4',
    'S': '5', 's': '5',
    'G': '6', 'b': '6',
    'T': '7',
    'B': '8',
    'g': '9', 'q': '9',
}

# Common OCR confusions for operators
OPERATOR_CONFUSIONS = {
    '×': '+', 'x': '+', 'X': '+', '*': '+',
    '÷': '/', '\\': '/',
    '—': '-', '–': '-', '−': '-', '_': '-',
    ':': '/',
    '==': '=', '⁼': '=',
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # insertions, deletions, substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity_score(s1: str, s2: str) -> float:
    """
    Compute similarity between two strings.
    Returns float in [0, 1] where 1.0 = identical.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    dist = levenshtein_distance(s1, s2)
    return 1.0 - (dist / max_len)


class TokenCleaner:
    """
    Cleans raw OCR output by stripping LaTeX artifacts,
    structural noise, and non-numeric characters.
    """

    def __init__(self):
        # Compile all LaTeX patterns for performance
        self._latex_regex = re.compile(
            '|'.join(LATEX_PATTERNS),
            re.IGNORECASE
        )
        # Pattern for valid tokens: numbers and operators
        self._valid_token_pattern = re.compile(
            r'(\d+\.?\d*|[+\-*/=÷×])'
        )

    def clean_raw_output(self, text: str) -> str:
        """
        Strip all LaTeX artifacts and structural noise from raw OCR output.

        Args:
            text: Raw OCR output string

        Returns:
            Cleaned string with only potential digits and operators
        """
        if not text:
            return ""

        cleaned = text

        # Step 1: Remove LaTeX commands
        cleaned = self._latex_regex.sub(' ', cleaned)

        # Step 2: Remove structural noise characters
        cleaned = ''.join(
            c if c not in STRUCTURAL_NOISE else ' '
            for c in cleaned
        )

        # Step 3: Apply character-level digit confusion correction
        corrected_chars = []
        for c in cleaned:
            if c in DIGIT_CONFUSIONS:
                corrected_chars.append(DIGIT_CONFUSIONS[c])
            elif c in OPERATOR_CONFUSIONS:
                corrected_chars.append(OPERATOR_CONFUSIONS[c])
            else:
                corrected_chars.append(c)
        cleaned = ''.join(corrected_chars)

        # Step 4: Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Step 5: Remove any remaining non-digit, non-operator characters
        # Keep: digits, +, -, *, /, =, spaces, dots
        cleaned = re.sub(r'[^\d+\-*/=.\s]', '', cleaned)

        # Step 6: Collapse multiple operators
        cleaned = re.sub(r'([+\-*/=])\s*([+\-*/=])', r'\1', cleaned)

        return cleaned.strip()

    def tokenize(self, cleaned_text: str) -> List[str]:
        """
        Split cleaned text into atomic tokens (numbers and operators).

        Args:
            cleaned_text: Output from clean_raw_output()

        Returns:
            List of token strings
        """
        if not cleaned_text:
            return []

        tokens = self._valid_token_pattern.findall(cleaned_text)
        return [t.strip() for t in tokens if t.strip()]

    def has_hallucination(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if raw OCR text contains hallucinated patterns.

        Args:
            text: Raw OCR output

        Returns:
            Tuple of (has_hallucination, list_of_found_patterns)
        """
        found = []

        # Check for LaTeX commands
        latex_matches = self._latex_regex.findall(text)
        if latex_matches:
            found.extend(latex_matches)

        # Check for structural characters
        for c in text:
            if c in STRUCTURAL_NOISE:
                found.append(f'structural:{c}')

        return len(found) > 0, list(set(found))


class SimilarityCorrector:
    """
    Corrects OCR tokens by mapping them to the nearest valid token
    using Levenshtein distance-based similarity.
    """

    def __init__(
        self,
        number_range: Tuple[int, int] = (0, 100),
        allowed_operators: List[str] = None,
        similarity_threshold: float = 0.5
    ):
        """
        Args:
            number_range: Valid number range (inclusive)
            allowed_operators: List of valid operators
            similarity_threshold: Minimum similarity to accept correction
        """
        self.number_range = number_range
        self.allowed_operators = allowed_operators or ['+', '-', '/', '=']
        self.similarity_threshold = similarity_threshold

        # Build valid number vocabulary
        self.valid_numbers = [
            str(n) for n in range(number_range[0], number_range[1] + 1)
        ]

    def correct_token(self, token: str) -> CorrectedToken:
        """
        Map a single token to the nearest valid token.

        Args:
            token: Raw token string

        Returns:
            CorrectedToken with correction details
        """
        token = token.strip()

        if not token:
            return CorrectedToken(
                original=token, corrected='',
                token_type='discarded', confidence=0.0, edit_distance=0
            )

        # Try as number first
        number_result = self.correct_number(token)
        if number_result is not None:
            return number_result

        # Try as operator
        operator_result = self.correct_operator(token)
        if operator_result is not None:
            return operator_result

        # No valid mapping found
        return CorrectedToken(
            original=token, corrected='',
            token_type='discarded', confidence=0.0, edit_distance=len(token)
        )

    def correct_number(self, token: str) -> Optional[CorrectedToken]:
        """
        Correct a token to the nearest valid number.

        Args:
            token: Token string that might be a number

        Returns:
            CorrectedToken if a valid number mapping found, else None
        """
        # Direct parse attempt
        try:
            val = int(float(token))
            if self.number_range[0] <= val <= self.number_range[1]:
                return CorrectedToken(
                    original=token, corrected=str(val),
                    token_type='number', confidence=1.0, edit_distance=0
                )
        except (ValueError, OverflowError):
            pass

        # Similarity-based correction
        best_match = None
        best_sim = 0.0
        best_dist = float('inf')

        for valid_num in self.valid_numbers:
            sim = similarity_score(token, valid_num)
            dist = levenshtein_distance(token, valid_num)

            if sim > best_sim or (sim == best_sim and dist < best_dist):
                best_sim = sim
                best_dist = dist
                best_match = valid_num

        if best_match is not None and best_sim >= self.similarity_threshold:
            return CorrectedToken(
                original=token, corrected=best_match,
                token_type='number', confidence=best_sim,
                edit_distance=best_dist
            )

        return None

    def correct_operator(self, token: str) -> Optional[CorrectedToken]:
        """
        Correct a token to the nearest valid operator.

        Args:
            token: Token string that might be an operator

        Returns:
            CorrectedToken if a valid operator mapping found, else None
        """
        # Direct match
        if token in self.allowed_operators:
            return CorrectedToken(
                original=token, corrected=token,
                token_type='operator', confidence=1.0, edit_distance=0
            )

        # Check confusion table
        if token in OPERATOR_CONFUSIONS:
            corrected = OPERATOR_CONFUSIONS[token]
            if corrected in self.allowed_operators:
                return CorrectedToken(
                    original=token, corrected=corrected,
                    token_type='operator', confidence=0.8,
                    edit_distance=1
                )

        # Similarity-based correction
        best_match = None
        best_sim = 0.0

        for op in self.allowed_operators:
            sim = similarity_score(token, op)
            if sim > best_sim:
                best_sim = sim
                best_match = op

        if best_match is not None and best_sim >= self.similarity_threshold:
            return CorrectedToken(
                original=token, corrected=best_match,
                token_type='operator', confidence=best_sim,
                edit_distance=levenshtein_distance(token, best_match)
            )

        return None

    def correct_all(self, tokens: List[str]) -> List[CorrectedToken]:
        """
        Correct a list of tokens.

        Args:
            tokens: List of raw token strings

        Returns:
            List of CorrectedToken objects (discarded tokens excluded)
        """
        results = []
        for token in tokens:
            result = self.correct_token(token)
            if result.token_type != 'discarded':
                results.append(result)
        return results

    def extract_numbers(self, corrected_tokens: List[CorrectedToken]) -> List[int]:
        """
        Extract only valid numbers from corrected tokens.

        Args:
            corrected_tokens: List of CorrectedToken objects

        Returns:
            List of integers
        """
        numbers = []
        for ct in corrected_tokens:
            if ct.token_type == 'number':
                try:
                    numbers.append(int(ct.corrected))
                except ValueError:
                    continue
        return numbers
