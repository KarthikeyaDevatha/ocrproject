
import os
import argparse
import sys
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Add parent dir to path for imports if needed, 
# but we can use standard transformers inference here.

def normalize_text(text):
    return text.lower().strip()

def check_for_errors(text):
    issues = []
    # 1. Check for '0' in likely text context
    if '0' in text:
        # Heuristic: if contains no other digits?
        # Actually strict rule: text shouldn't have isolated 0s if it's "Enzymes..."
        issues.append("Contains '0'")
        
    # 2. Check for lexical drift "catalog"
    if "catalog" in text.lower():
        issues.append("Lexical Drift: 'catalog' found (expected 'catalyse/catalyze')")
        
    # 3. Check for "reaction" vs "reaction."
    # If duplicates dots? ".."
    if ".." in text:
        issues.append("Double dots")
        
    return issues

def verify(args):
    print(f"Loading model from {args.model_path}...")
    try:
        # Try loading both from checkpoint
        try:
            processor = TrOCRProcessor.from_pretrained(args.model_path)
        except:
            print("Processor not found in checkpoint, loading from base 'microsoft/trocr-small-handwritten'...")
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
            
        model = VisionEncoderDecoderModel.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # List of specific test images (from user upload or knowledge)
    # The user uploaded 3 images. 
    # "/Users/karthikeyadevatha/.gemini/antigravity/brain/37bac856-6f9c-4c8b-a7bd-cadac79ee6de/uploaded_media_0_1769403048596.jpg" etc.
    # We should scan for them or use a directory.
    
    images = []
    if args.image_path:
        images.append(args.image_path)
    else:
        # Look for the uploaded media in artifacts dir
        artifact_dir = "/Users/karthikeyadevatha/.gemini/antigravity/brain/37bac856-6f9c-4c8b-a7bd-cadac79ee6de"
        for f in os.listdir(artifact_dir):
            if f.startswith("uploaded_media") and f.endswith(".jpg"):
                images.append(os.path.join(artifact_dir, f))
    
    print(f"Found {len(images)} test images.")
    
    for img_path in images:
        print(f"\nTesting: {os.path.basename(img_path)}")
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
            print(f"Prediction: {generated_text}")
            
            issues = check_for_errors(generated_text)
            if issues:
                print(f"❌ FAILED: {', '.join(issues)}")
            else:
                print("✅ PASSED: No forbidden errors found.")
                
        except Exception as e:
            print(f"Error processing image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint (e.g. checkpoints/trocr_text_accuracy/epoch_1)")
    parser.add_argument("--image_path", type=str, help="Specific image to test")
    args = parser.parse_args()
    verify(args)
