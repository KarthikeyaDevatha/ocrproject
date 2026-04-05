"""
Enhanced Post-Processor for Adaptive OCR Agent.

Features:
- Regex-based artifact removal (hashtags, LaTeX fragments)
- Domain-aware spell correction (pyspellchecker + custom wordlist)
- Line merging for broken text
- Math expression normalization
"""

import re
from typing import List, Tuple, Optional


# Domain wordlist — prevents spell-correction of valid technical terms
DOMAIN_WORDS = {
    # Chemistry / Lab
    "titration", "molarity", "beaker", "reagent", "centrifuge",
    "naoh", "hcl", "ph", "pipette", "nacl", "burette", "aliquot",
    "precipitate", "filtrate", "solvent", "solute", "molar",
    # Math
    "integral", "derivative", "summation", "polynomial", "coefficient",
    "quadratic", "exponential", "logarithm", "factorial", "permutation",
    "numerator", "denominator",
    # Tech / OCR
    "trocr", "cer", "wer", "ocr", "clahe", "mathpix",
}


class PostProcessor:
    """
    Enhanced OCR post-processing pipeline.
    
    Steps:
    1. Strip known artifacts (#, repeated symbols)
    2. Clean LaTeX fragments if not expected
    3. Spell-check with domain awareness
    4. Merge broken lines
    5. Normalize math expressions
    """

    def __init__(self, use_spellcheck: bool = True):
        self.use_spellcheck = use_spellcheck
        self._spell = None
        self._spell_loaded = False

    def _load_spellchecker(self):
        """Lazy-load pyspellchecker with domain words."""
        if self._spell_loaded:
            return
        
        try:
            from spellchecker import SpellChecker
            self._spell = SpellChecker()
            self._spell.word_frequency.load_words(DOMAIN_WORDS)
            self._spell_loaded = True
        except ImportError:
            self._spell = None
            self._spell_loaded = True  # Don't retry

    # ========================================================================
    # STEP 1: Artifact removal
    # ========================================================================

    def remove_artifacts(self, text: str) -> str:
        """Remove common OCR artifacts."""
        # Remove standalone hashtags (TrOCR artifact)
        text = re.sub(r'\s*#\s*', ' ', text)
        
        # Remove repeated punctuation (e.g., "..." becomes ".")
        text = re.sub(r'([.!?,;:]){3,}', r'\1', text)
        
        # Remove leading/trailing junk
        text = re.sub(r'^[\s#*\-_=]+', '', text)
        text = re.sub(r'[\s#*\-_=]+$', '', text)
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    # ========================================================================
    # STEP 2: LaTeX fragment cleanup
    # ========================================================================

    def clean_latex_fragments(self, text: str) -> str:
        """Remove unwanted LaTeX commands from non-math output."""
        # Common hallucinated LaTeX commands
        latex_patterns = [
            r'\\frac\{[^}]*\}\{[^}]*\}',  # \frac{a}{b}
            r'\\ldots',                       # \ldots
            r'\\times',                       # \times
            r'\\cdot',                        # \cdot
            r'\\sqrt\{[^}]*\}',              # \sqrt{x}
            r'\\sum',                         # \sum
            r'\\int',                         # \int
            r'\\[a-zA-Z]+',                  # Any \command
        ]
        
        for pattern in latex_patterns:
            text = re.sub(pattern, '', text)
        
        # Clean up orphaned braces
        text = re.sub(r'[{}]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    # ========================================================================
    # STEP 3: Spell correction (domain-aware)
    # ========================================================================

    def correct_spelling(self, text: str) -> Tuple[str, int]:
        """
        Domain-aware spell correction.
        
        Args:
            text: Input text
        
        Returns:
            Tuple of (corrected_text, num_corrections)
        """
        if not self.use_spellcheck:
            return text, 0
        
        self._load_spellchecker()
        if self._spell is None:
            return text, 0
        
        words = text.split()
        corrected = []
        corrections = 0
        
        for word in words:
            # Skip short words, numbers, and punctuation
            clean = re.sub(r'[^\w]', '', word).lower()
            
            if len(clean) < 3 or clean.isdigit():
                corrected.append(word)
                continue
            
            # Check if known word
            if clean in self._spell or clean in DOMAIN_WORDS:
                corrected.append(word)
                continue
            
            # Try correction
            suggestion = self._spell.correction(clean)
            if suggestion and suggestion != clean:
                # Preserve original casing pattern
                if word[0].isupper() and len(suggestion) > 0:
                    suggestion = suggestion[0].upper() + suggestion[1:]
                corrected.append(suggestion)
                corrections += 1
            else:
                corrected.append(word)
        
        return " ".join(corrected), corrections

    # ========================================================================
    # STEP 4: Line merging
    # ========================================================================

    def merge_broken_lines(self, lines: List[str]) -> List[str]:
        """
        Merge lines that were incorrectly split.
        A line ending without sentence-ending punctuation is merged with the next.
        
        Args:
            lines: List of text lines
        
        Returns:
            Merged lines
        """
        if not lines:
            return lines
        
        merged = []
        buffer = lines[0]
        
        for i in range(1, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            
            # Merge if previous line doesn't end with sentence punctuation
            if buffer and not re.search(r'[.!?:;]$', buffer.strip()):
                buffer = buffer.strip() + " " + line
            else:
                merged.append(buffer.strip())
                buffer = line
        
        if buffer:
            merged.append(buffer.strip())
        
        return merged

    # ========================================================================
    # STEP 5: Math normalization
    # ========================================================================

    def normalize_math(self, text: str) -> str:
        """
        Basic math expression normalization.
        - Balance braces/brackets
        - Standardize operator spacing
        """
        # Standardize operators with spaces
        text = re.sub(r'\s*([+\-=])\s*', r' \1 ', text)
        
        # Fix multiplication
        text = text.replace('×', '*')
        text = text.replace('÷', '/')
        
        # Balance parentheses
        open_count = text.count('(')
        close_count = text.count(')')
        if open_count > close_count:
            text += ')' * (open_count - close_count)
        elif close_count > open_count:
            text = '(' * (close_count - open_count) + text
        
        return text

    # ========================================================================
    # FULL PIPELINE
    # ========================================================================

    def process(
        self,
        text: str,
        is_math: bool = False,
        clean_latex: bool = False,
        do_spellcheck: bool = True
    ) -> Tuple[str, dict]:
        """
        Full post-processing pipeline.
        
        Args:
            text: Raw OCR output text
            is_math: Whether the content is expected to be math
            clean_latex: Whether to strip LaTeX fragments
            do_spellcheck: Whether to run spell correction
        
        Returns:
            Tuple of (cleaned_text, stats_dict)
        """
        stats = {"artifacts_removed": False, "latex_cleaned": False,
                 "spell_corrections": 0, "math_normalized": False}
        
        original = text
        
        # Step 1: Artifact removal
        text = self.remove_artifacts(text)
        stats["artifacts_removed"] = text != original
        
        # Step 2: LaTeX cleanup (skip if math mode)
        if clean_latex and not is_math:
            before = text
            text = self.clean_latex_fragments(text)
            stats["latex_cleaned"] = text != before
        
        # Step 3: Spell correction
        if do_spellcheck and not is_math:
            text, corrections = self.correct_spelling(text)
            stats["spell_corrections"] = corrections
        
        # Step 5: Math normalization
        if is_math:
            text = self.normalize_math(text)
            stats["math_normalized"] = True
        
        return text, stats

    def process_lines(
        self,
        lines: List[str],
        is_math: bool = False,
        merge: bool = True
    ) -> Tuple[List[str], dict]:
        """
        Process multiple lines with optional merging.
        
        Returns:
            Tuple of (processed_lines, aggregate_stats)
        """
        total_stats = {"artifacts_removed": 0, "latex_cleaned": 0,
                       "spell_corrections": 0, "lines_merged": 0}
        
        # Process each line
        processed = []
        for line in lines:
            cleaned, stats = self.process(line, is_math=is_math)
            if cleaned:
                processed.append(cleaned)
                if stats["artifacts_removed"]:
                    total_stats["artifacts_removed"] += 1
                if stats["latex_cleaned"]:
                    total_stats["latex_cleaned"] += 1
                total_stats["spell_corrections"] += stats["spell_corrections"]
        
        # Merge broken lines
        if merge:
            before_count = len(processed)
            processed = self.merge_broken_lines(processed)
            total_stats["lines_merged"] = before_count - len(processed)
        
        return processed, total_stats
