"""Debug script to see actual predictions vs ground truth."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.ocr_text import create_text_ocr
from inference.preprocess import preprocess_for_ocr


def main():
    manifest_path = "data/manifests/val.jsonl"
    model_dir = "models/trocr_text"
    mode_filter = "text"
    
    # Load samples
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            if mode_filter is None or sample.get('mode') == mode_filter:
                samples.append(sample)
    
    print(f"Found {len(samples)} text samples")
    
    # Initialize OCR
    ocr = create_text_ocr(model_dir=model_dir, device="cpu")
    
    print("\n" + "="*80)
    print("PREDICTIONS vs GROUND TRUTH")
    print("="*80)
    
    for i, sample in enumerate(samples[:5]):  # First 5
        image_path = sample['image_path']
        ground_truth = sample['ground_truth_text']
        
        try:
            image = preprocess_for_ocr(image_path)
            result = ocr.recognize(image)
            pred = result.text
        except Exception as e:
            pred = f"[ERROR: {e}]"
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Image: {image_path}")
        print(f"Ground Truth: '{ground_truth}'")
        print(f"Prediction:   '{pred}'")
        print(f"Match: {'YES' if pred.strip() == ground_truth.strip() else 'NO'}")


if __name__ == "__main__":
    main()
