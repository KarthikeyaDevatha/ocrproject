
import os
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm
from pdf2image import convert_from_path
import pypdf
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str, page_num: int) -> str:
    """Extract text from a specific page of a PDF using pypdf."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        if len(reader.pages) > page_num:
            text = reader.pages[page_num].extract_text()
            return text.strip() if text else ""
        return ""
    except Exception as e:
        logger.warning(f"Failed to extract text from {pdf_path} page {page_num}: {e}")
        return ""

def process_pdf(pdf_path: str, output_dir: str, rel_path_start: str) -> list[dict]:
    """
    Convert a PDF to images and create manifest entries.
    
    Args:
        pdf_path: Path to the source PDF.
        output_dir: Root directory for saving processed images.
        rel_path_start: Directory to calculate relative paths from (usually project root).
        
    Returns:
        List of manifest entries.
    """
    records = []
    pdf_name = Path(pdf_path).stem
    
    # Create specific output folder for this PDF to avoid overwriting
    pdf_output_dir = os.path.join(output_dir, pdf_name)
    os.makedirs(pdf_output_dir, exist_ok=True)

    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        for i, image in enumerate(images):
            # Extract text (ground truth attempt)
            # text = extract_text_from_pdf(pdf_path, i)
            # NOTE: For now, avoiding direct text extraction if alignment isn't guaranteed. 
            # But the plan said "attempt to extract text".
            # The issue is that pypdf extraction might not align perfectly with the image if it's not a native digital PDF.
            # Let's try to extract it.
            
            text = extract_text_from_pdf(pdf_path, i)
            
            if not text:
                logger.info(f"No text extracted for {pdf_name} page {i}. Marking as needs_labeling.")
                # We can't train without text, but we can generate the entry for inference/labeling.
                # For this script's purpose of "improving training data", we might want to flag these.
                gt_text = ""
            else:
                gt_text = text

            image_filename = f"{pdf_name}_page_{i:03d}.jpg"
            image_path = os.path.join(pdf_output_dir, image_filename)
            
            # Save image
            image.save(image_path, "JPEG")
            
            # Create record
            # We want absolute path for training usually, or relative to script execution?
            # The existing scripts seem to use absolute paths or paths compatible with the loader.
            # train_trocr_text.py uses Image.open(sample['image_path']) directly.
            
            record = {
                "image_path": os.path.abspath(image_path),
                "ground_truth_text": gt_text,
                "mode": "pdf_page" if gt_text else "needs_labeling",
                "source_pdf": os.path.basename(pdf_path),
                "page": i
            }
            records.append(record)
            
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        
    return records

def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to images and generate training manifest.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", type=str, default="data/processed/pdf_data", help="Output directory for images")
    parser.add_argument("--manifest-output", type=str, default="data/manifests/pdf_train.jsonl", help="Output path for manifest file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs to process")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest_output), exist_ok=True)
    
    # Find all PDFs recursively
    pdf_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    
    if args.limit:
        pdf_files = pdf_files[:args.limit]
        
    logger.info(f"Found {len(pdf_files)} PDF files to process.")
    
    all_records = []
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        records = process_pdf(pdf_path, args.output_dir, os.getcwd())
        all_records.extend(records)
        
    # Write manifest
    # Filtering out items with no text if we want to be strict, but for now let's keep all
    # and let the user decide how to handle empty GT.
    
    valid_records = [r for r in all_records if r['ground_truth_text']]
    needs_labeling = [r for r in all_records if not r['ground_truth_text']]
    
    logger.info(f"Processed {len(all_records)} pages.")
    logger.info(f"  With extracted text: {len(valid_records)}")
    logger.info(f"  Empty/Failed text: {len(needs_labeling)}")
    
    with open(args.manifest_output, 'w') as f:
        for record in all_records:
            f.write(json.dumps(record) + '\n')
            
    logger.info(f"Manifest saved to {args.manifest_output}")

if __name__ == "__main__":
    main()
