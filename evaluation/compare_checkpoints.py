"""
Compare checkpoints to find the best performing model.
"""

import os
import sys
import json
import argparse
from typing import List, Dict
from pathlib import Path

from tqdm import tqdm


def find_checkpoints(checkpoint_dir: str) -> List[str]:
    """
    Find all checkpoint directories.
    
    Args:
        checkpoint_dir: Base checkpoint directory
    
    Returns:
        List of checkpoint paths
    """
    checkpoints = []
    
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains model files
            if any(f.endswith('.bin') or f.endswith('.safetensors') or f.endswith('.onnx') 
                   for f in os.listdir(item_path)):
                checkpoints.append(item_path)
    
    return sorted(checkpoints)


def evaluate_checkpoint(
    checkpoint_path: str,
    manifest_path: str,
    model_type: str = "text",
    device: str = "cpu",
    max_samples: int = 100
) -> Dict[str, float]:
    """
    Evaluate a single checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        manifest_path: Path to validation manifest
        model_type: 'text' or 'math'
        device: Compute device
        max_samples: Maximum samples to evaluate
    
    Returns:
        Evaluation metrics
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if model_type == "text":
        from inference.ocr_text import TextOCR
        from evaluation.compute_cer import compute_cer, compute_wer
        
        ocr = TextOCR(
            encoder_path=os.path.join(checkpoint_path, "encoder_model.onnx"),
            decoder_path=os.path.join(checkpoint_path, "decoder_model.onnx"),
            tokenizer_path=checkpoint_path,
            model_path=checkpoint_path, # Pass checkpoint path for fallback loading
            device=device
        )
        
        metric_fn = lambda pred, ref: {
            "cer": compute_cer(pred, ref),
            "wer": compute_wer(pred, ref)
        }
        
    else:  # math
        from inference.ocr_math import MathOCR
        from evaluation.compute_token_distance import normalized_edit_distance
        
        ocr = MathOCR(
            encoder_path=os.path.join(checkpoint_path, "encoder_model.onnx"),
            decoder_path=os.path.join(checkpoint_path, "decoder_model.onnx"),
            tokenizer_path=checkpoint_path,
            device=device
        )
        
        metric_fn = lambda pred, ref: {
            "token_distance": normalized_edit_distance(pred, ref)
        }
    
    # Load samples
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            mode = sample.get('mode', 'text')
            if (model_type == "text" and mode == "text") or \
               (model_type == "math" and mode in ["math", "mixed"]):
                samples.append(sample)
                if len(samples) >= max_samples:
                    break
    
    if not samples:
        return {"error": "No samples found"}
    
    from inference.preprocess import preprocess_for_ocr
    
    all_metrics = []
    
    for sample in samples:
        try:
            image = preprocess_for_ocr(sample['image_path'])
            result = ocr.recognize(image)
            
            pred_text = result.text if model_type == "text" else result.latex
            ref_text = sample['ground_truth_text']
            
            metrics = metric_fn(pred_text, ref_text)
            all_metrics.append(metrics)
            
        except Exception as e:
            continue
    
    if not all_metrics:
        return {"error": "All samples failed"}
    
    # Aggregate metrics
    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        aggregated[f"{key}_mean"] = sum(values) / len(values)
    
    aggregated["num_evaluated"] = len(all_metrics)
    
    return aggregated


def compare_checkpoints(
    checkpoint_dir: str,
    manifest_path: str,
    model_type: str = "text",
    device: str = "cpu",
    max_samples: int = 100
) -> List[Dict]:
    """
    Compare all checkpoints in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        manifest_path: Path to validation manifest
        model_type: 'text' or 'math'
        device: Compute device
        max_samples: Max samples per checkpoint
    
    Returns:
        List of results for each checkpoint
    """
    checkpoints = find_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    results = []
    
    for ckpt in tqdm(checkpoints, desc="Evaluating checkpoints"):
        print(f"\nEvaluating: {os.path.basename(ckpt)}")
        
        metrics = evaluate_checkpoint(
            ckpt, manifest_path, model_type, device, max_samples
        )
        
        results.append({
            "checkpoint": ckpt,
            "name": os.path.basename(ckpt),
            **metrics
        })
    
    return results


def find_best_checkpoint(
    results: List[Dict],
    metric: str = "cer_mean",
    lower_is_better: bool = True
) -> Dict:
    """
    Find the best checkpoint based on a metric.
    
    Args:
        results: List of evaluation results
        metric: Metric to compare
        lower_is_better: Whether lower values are better
    
    Returns:
        Best checkpoint result
    """
    valid_results = [r for r in results if metric in r]
    
    if not valid_results:
        return None
    
    if lower_is_better:
        best = min(valid_results, key=lambda x: x[metric])
    else:
        best = max(valid_results, key=lambda x: x[metric])
    
    return best


def main():
    parser = argparse.ArgumentParser(description="Compare model checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--manifest", type=str, required=True, help="Validation manifest")
    parser.add_argument("--model-type", type=str, choices=["text", "math"], default="text")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    results = compare_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        manifest_path=args.manifest,
        model_type=args.model_type,
        device=args.device,
        max_samples=args.max_samples
    )
    
    if not results:
        print("No results to compare")
        return
    
    # Print comparison table
    print("\n=== Checkpoint Comparison ===")
    
    if args.model_type == "text":
        metric_key = "cer_mean"
        print(f"{'Checkpoint':<30} {'CER':<10} {'WER':<10} {'Samples':<10}")
        print("-" * 60)
        
        for r in results:
            cer = r.get('cer_mean', 'N/A')
            wer = r.get('wer_mean', 'N/A')
            samples = r.get('num_evaluated', 'N/A')
            
            cer_str = f"{cer:.4f}" if isinstance(cer, float) else str(cer)
            wer_str = f"{wer:.4f}" if isinstance(wer, float) else str(wer)
            
            print(f"{r['name']:<30} {cer_str:<10} {wer_str:<10} {samples:<10}")
    else:
        metric_key = "token_distance_mean"
        print(f"{'Checkpoint':<30} {'Token Dist':<15} {'Samples':<10}")
        print("-" * 55)
        
        for r in results:
            dist = r.get('token_distance_mean', 'N/A')
            samples = r.get('num_evaluated', 'N/A')
            
            dist_str = f"{dist:.4f}" if isinstance(dist, float) else str(dist)
            
            print(f"{r['name']:<30} {dist_str:<15} {samples:<10}")
    
    # Find best
    best = find_best_checkpoint(results, metric_key, lower_is_better=True)
    
    if best:
        print(f"\n*** Best checkpoint: {best['name']} ({metric_key}: {best.get(metric_key, 'N/A'):.4f})")
        print(f"    Path: {best['checkpoint']}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "results": results,
                "best": best
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
