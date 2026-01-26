"""
Character Error Rate (CER) computation for text OCR evaluation.
"""

import os
import sys
import json
import argparse
from typing import List, Tuple, Dict

from tqdm import tqdm


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load samples from JSONL manifest."""
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def compute_cer(prediction: str, reference: str) -> float:
    """
    Compute Character Error Rate between prediction and reference.
    
    Args:
        prediction: Predicted text
        reference: Ground truth text
    
    Returns:
        CER value (0.0 to 1.0+)
    """
    if not reference:
        return 0.0 if not prediction else 1.0
    
    distance = _levenshtein_distance(prediction, reference)
    return distance / len(reference)


def compute_wer(prediction: str, reference: str) -> float:
    """
    Compute Word Error Rate between prediction and reference.
    
    Args:
        prediction: Predicted text
        reference: Ground truth text
    
    Returns:
        WER value (0.0 to 1.0+)
    """
    if not reference:
        return 0.0 if not prediction else 1.0
    
    pred_words = prediction.split()
    ref_words = reference.split()
    
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    
    distance = _levenshtein_distance(' '.join(pred_words), ' '.join(ref_words))
    # Word-level: count edits on word list
    # Simpler approach: use word-based edit distance
    distance = _levenshtein_word_distance(pred_words, ref_words)
    return distance / len(ref_words)


def _levenshtein_word_distance(s1: List[str], s2: List[str]) -> int:
    """Compute Levenshtein distance at word level."""
    if len(s1) < len(s2):
        return _levenshtein_word_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, w1 in enumerate(s1):
        current_row = [i + 1]
        for j, w2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (w1 != w2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]







def evaluate_predictions(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Evaluate a list of predictions against references.
    
    Args:
        predictions: List of predicted texts
        references: List of ground truth texts
    
    Returns:
        Dictionary with evaluation metrics
    """
    assert len(predictions) == len(references), "Mismatched lengths"
    
    cer_scores = []
    wer_scores = []
    exact_matches = 0
    
    for pred, ref in zip(predictions, references):
        cer = compute_cer(pred, ref)
        wer = compute_wer(pred, ref)
        
        cer_scores.append(cer)
        wer_scores.append(wer)
        
        if pred.strip() == ref.strip():
            exact_matches += 1
    
    n = len(predictions)
    
    return {
        "cer_mean": sum(cer_scores) / n if n > 0 else 0,
        "cer_median": sorted(cer_scores)[n // 2] if n > 0 else 0,
        "wer_mean": sum(wer_scores) / n if n > 0 else 0,
        "wer_median": sorted(wer_scores)[n // 2] if n > 0 else 0,
        "exact_match_rate": exact_matches / n if n > 0 else 0,
        "num_samples": n
    }


def run_evaluation(
    manifest_path: str,
    model_dir: str,
    mode_filter: str = "text",
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Run full evaluation on a manifest.
    
    Args:
        manifest_path: Path to test manifest
        model_dir: Path to OCR model directory
        mode_filter: Filter samples by mode
        device: Compute device
    
    Returns:
        Evaluation metrics
    """
    # Import OCR module
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from inference.ocr_text import create_text_ocr
    from inference.preprocess import preprocess_for_ocr
    
    # Load samples
    samples = load_manifest(manifest_path)
    
    if mode_filter:
        samples = [s for s in samples if s.get('mode') == mode_filter]
    
    print(f"Evaluating on {len(samples)} samples...")
    
    # Initialize OCR
    ocr = create_text_ocr(model_dir=model_dir, device=device)
    
    predictions = []
    references = []
    
    for sample in tqdm(samples, desc="Running OCR"):
        image_path = sample['image_path']
        ground_truth = sample['ground_truth_text']
        
        try:
            image = preprocess_for_ocr(image_path)
            result = ocr.recognize(image)
            predictions.append(result.text)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            predictions.append("")
        
        references.append(ground_truth)
    
    # Compute metrics
    metrics = evaluate_predictions(predictions, references)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute CER for text OCR")
    parser.add_argument("--manifest", type=str, required=True, help="Path to test manifest")
    parser.add_argument("--model-dir", type=str, default="models/trocr_text", help="OCR model directory")
    parser.add_argument("--mode-filter", type=str, default="text", help="Filter by mode")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    metrics = run_evaluation(
        manifest_path=args.manifest,
        model_dir=args.model_dir,
        mode_filter=args.mode_filter,
        device=args.device
    )
    
    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
