"""
Evaluation Module for Adaptive OCR Agent.

Computes CER (Character Error Rate) and WER (Word Error Rate) for OCR outputs.
Includes Tesseract baseline comparison and multi-engine evaluation.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EvalSample:
    """A single evaluation sample."""
    image_path: str
    ground_truth: str
    predictions: Dict[str, str] = field(default_factory=dict)  # engine -> text


@dataclass
class EngineMetrics:
    """Metrics for a single engine."""
    engine_name: str
    mean_cer: float
    mean_wer: float
    samples_evaluated: int
    per_sample_cer: List[float] = field(default_factory=list)
    per_sample_wer: List[float] = field(default_factory=list)


def compute_cer(prediction: str, reference: str) -> float:
    """
    Compute Character Error Rate (CER).
    
    CER = (insertions + deletions + substitutions) / total_reference_chars
    
    Uses edit distance (Levenshtein) at character level.
    
    Args:
        prediction: OCR output text
        reference: Ground truth text
    
    Returns:
        CER as float [0, ∞). 0 = perfect, 1.0 = 100% error
    """
    if not reference:
        return 0.0 if not prediction else 1.0
    
    # Levenshtein distance at character level
    pred_chars = list(prediction)
    ref_chars = list(reference)
    
    distance = _levenshtein_distance(pred_chars, ref_chars)
    return distance / len(ref_chars)


def compute_wer(prediction: str, reference: str) -> float:
    """
    Compute Word Error Rate (WER).
    
    WER = (insertions + deletions + substitutions) / total_reference_words
    
    Args:
        prediction: OCR output text
        reference: Ground truth text
    
    Returns:
        WER as float [0, ∞). 0 = perfect, 1.0 = 100% error
    """
    if not reference.strip():
        return 0.0 if not prediction.strip() else 1.0
    
    pred_words = prediction.strip().split()
    ref_words = reference.strip().split()
    
    distance = _levenshtein_distance(pred_words, ref_words)
    return distance / len(ref_words)


def _levenshtein_distance(seq1: list, seq2: list) -> int:
    """Standard Levenshtein distance (edit distance)."""
    m, n = len(seq1), len(seq2)
    
    # Use O(n) space
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr_row[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,       # deletion
                curr_row[j - 1] + 1,    # insertion
                prev_row[j - 1] + cost  # substitution
            )
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[n]


class Evaluator:
    """
    Multi-engine OCR evaluator.
    
    Compares: Tesseract (baseline) | TrOCR-base | TrOCR-large | Hybrid Auto
    Produces: CER/WER per engine, comparison table, formatted report.
    """

    def __init__(self):
        self.results: List[EngineMetrics] = []

    def evaluate_single(
        self,
        prediction: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Evaluate a single prediction against reference.
        
        Returns:
            {"cer": float, "wer": float}
        """
        return {
            "cer": round(compute_cer(prediction, reference), 4),
            "wer": round(compute_wer(prediction, reference), 4)
        }

    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions.
        
        Returns:
            {"mean_cer": float, "mean_wer": float, "per_sample": [...]}
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        per_sample = []
        cer_list = []
        wer_list = []
        
        for pred, ref in zip(predictions, references):
            metrics = self.evaluate_single(pred, ref)
            per_sample.append(metrics)
            cer_list.append(metrics["cer"])
            wer_list.append(metrics["wer"])
        
        return {
            "mean_cer": round(sum(cer_list) / len(cer_list), 4) if cer_list else 0.0,
            "mean_wer": round(sum(wer_list) / len(wer_list), 4) if wer_list else 0.0,
            "std_cer": round(_std(cer_list), 4),
            "std_wer": round(_std(wer_list), 4),
            "samples": len(cer_list),
            "per_sample": per_sample
        }

    def run_tesseract_baseline(
        self,
        image_path: str
    ) -> Optional[str]:
        """
        Run Tesseract OCR as baseline.
        
        Returns:
            OCR text or None if Tesseract unavailable
        """
        try:
            import pytesseract
            text = pytesseract.image_to_string(image_path)
            return text.strip()
        except ImportError:
            return None
        except Exception as e:
            print(f"Tesseract error: {e}")
            return None

    def compare_engines(
        self,
        samples: List[EvalSample],
        engines: Optional[List[str]] = None
    ) -> Dict[str, EngineMetrics]:
        """
        Compare multiple engines on the same set of samples.
        
        Args:
            samples: List of EvalSample with predictions per engine
            engines: Engine names to compare
        
        Returns:
            Dict of engine_name -> EngineMetrics
        """
        if engines is None:
            # Discover from first sample
            engines = list(samples[0].predictions.keys()) if samples else []
        
        results = {}
        
        for engine in engines:
            cer_list = []
            wer_list = []
            
            for sample in samples:
                pred = sample.predictions.get(engine, "")
                ref = sample.ground_truth
                
                cer_list.append(compute_cer(pred, ref))
                wer_list.append(compute_wer(pred, ref))
            
            results[engine] = EngineMetrics(
                engine_name=engine,
                mean_cer=round(sum(cer_list) / len(cer_list), 4) if cer_list else 0.0,
                mean_wer=round(sum(wer_list) / len(wer_list), 4) if wer_list else 0.0,
                samples_evaluated=len(samples),
                per_sample_cer=[round(c, 4) for c in cer_list],
                per_sample_wer=[round(w, 4) for w in wer_list]
            )
        
        return results

    def generate_report(
        self,
        engine_metrics: Dict[str, EngineMetrics]
    ) -> str:
        """
        Generate a formatted evaluation report.
        
        Returns:
            Markdown table string
        """
        lines = [
            "# OCR Evaluation Report",
            "",
            "| Engine | Mean CER | Mean WER | Samples |",
            "|--------|----------|----------|---------|"
        ]
        
        for name, metrics in sorted(
            engine_metrics.items(),
            key=lambda x: x[1].mean_cer
        ):
            lines.append(
                f"| {name} | {metrics.mean_cer:.4f} | "
                f"{metrics.mean_wer:.4f} | {metrics.samples_evaluated} |"
            )
        
        lines.append("")
        lines.append(f"*Lower is better. CER=0 means perfect character accuracy.*")
        
        return "\n".join(lines)


def _std(values: list) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5
