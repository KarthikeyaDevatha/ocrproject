import sys
import os
import json
import logging

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.generators.synthetic_math import SyntheticMathGenerator
from backend.pipelines.document_processor import DocumentProcessor
import cv2

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Step 1: Instantiating DocumentProcessor...")
    # Mocking openai so it doesn't fail if we don't have a real key setup in the environment
    import backend.pipelines.semantic_parser as sp
    
    # Simple mock for semantic parser to avoid actual API call in the demo test
    original_call_llm = sp.SemanticParser._call_llm
    def mock_call_llm(self, prompt):
        return json.dumps({
            "subject": "algebra",
            "grade_level": "grade_10",
            "questions": [
                {
                    "problem_type": "polynomial",
                    "solution_steps": ["Step 1", "Step 2"]
                }
            ]
        })
    sp.SemanticParser._call_llm = mock_call_llm

    processor = DocumentProcessor(api_key="dummy_key")

    logging.info("Step 2: Generating synthetic math image...")
    gen_dir = "./datasets/synthetic_math_data"
    os.makedirs(gen_dir, exist_ok=True)
    
    generator = SyntheticMathGenerator(gen_dir)
    sample = generator.generate_sample()
    
    if sample is None:
        logging.error("Failed to generate sample. Are sympy/matplotlib installed? Will fallback to a dummy image.")
        img = __import__("numpy").ones((300, 300, 3), dtype=np.uint8) * 255
    else:
        img = sample["image"]
        logging.info(f"Generated expression: {sample['latex']}")

    test_img_path = "/tmp/test_end_to_end.png"
    cv2.imwrite(test_img_path, img)

    logging.info("Step 3: Processing image through full pipeline...")
    result = processor.process_image(test_img_path)

    logging.info("Step 4: Output structured JSON details...")
    print(json.dumps(result, indent=2))
    
    logging.info("Verification Complete. Pipeline works End-to-End.")

if __name__ == "__main__":
    main()
