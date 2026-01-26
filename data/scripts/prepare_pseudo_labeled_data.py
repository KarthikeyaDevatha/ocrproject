
import os
import sys
import argparse
import json
import cv2
import logging
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.layout import LayoutDetector, cluster_lines, merge_close_boxes
from inference.ocr_text import create_text_ocr
from inference.preprocess import crop_region, deskew_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_page_image(
    image_path: str, 
    layout_detector: LayoutDetector, 
    text_ocr, 
    output_dir: str
) -> list[dict]:
    """
    Detect lines in a page, crop them, license them, and return manifest entries.
    """
    records = []
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read {image_path}")
            return []
            
        # Deskew
        image = deskew_image(image)
        
        # Detect Layout
        detections = layout_detector.detect(image)
        
        # We generally only want text lines for this specific training task, 
        # but the request was generic "TRAIN DATA". 
        # Let's filter for text_line and possibly math_formula if we want to mix.
        # For now, let's process 'text_line' for the text model.
        
        # Cluster/Merge logic from pipeline.py
        clusters = cluster_lines(detections)
        merged_detections = []
        for cluster in clusters:
            merged_detections.extend(merge_close_boxes(cluster))
            
        base_name = Path(image_path).stem
        
        for i, det in enumerate(merged_detections):
            if det.class_name != "text_line":
                continue
                
            # Crop
            region = crop_region(image, det.bbox, padding=5)
            if region.size == 0:
                continue
                
            # OCR for Pseudo-label
            # Convert to RGB for OCR (TrOCR expects RGB)
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            result = text_ocr.recognize(region_rgb)
            
            # Save crop
            crop_filename = f"{base_name}_line_{i:03d}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, region) # Save as BGR (OpenCV default)
            
            # Record
            record = {
                "image_path": os.path.abspath(crop_path),
                "ground_truth_text": result.text,
                "mode": "text",
                "confidence": result.confidence,
                "pseudo_labeled": True,
                "source_page": base_name
            }
            records.append(record)
            
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        
    return records

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labeled line data from page images.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing page images (e.g. data/processed/pdf_data)")
    parser.add_argument("--output-dir", type=str, default="data/processed/pseudo_lines", help="Output for line crops")
    parser.add_argument("--manifest-output", type=str, default="data/manifests/pseudo_train.jsonl", help="Output manifest")
    parser.add_argument("--yolo-model", type=str, default="models/yolo/yolov8n-layout.onnx")
    parser.add_argument("--text-model-dir", type=str, default="models/trocr_text")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest_output), exist_ok=True)
    
    # Init Models
    logger.info("Loading models...")
    layout_detector = LayoutDetector(
        model_path=args.yolo_model, 
        device=args.device
    )
    
    text_ocr = create_text_ocr(
        model_dir=args.text_model_dir, 
        device=args.device
    )
    
    # Collect images (recursive)
    image_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
                
    logger.info(f"Found {len(image_files)} images to process.")
    
    all_records = []
    
    for img_path in tqdm(image_files, desc="Pseudo-labeling"):
        records = process_page_image(img_path, layout_detector, text_ocr, args.output_dir)
        all_records.extend(records)
        
    # Write manifest
    with open(args.manifest_output, 'w') as f:
        for r in all_records:
            f.write(json.dumps(r) + '\n')
            
    logger.info(f"Generated {len(all_records)} line samples.")
    logger.info(f"Manifest: {args.manifest_output}")

if __name__ == "__main__":
    main()
