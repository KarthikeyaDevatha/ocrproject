
import sys
import os
import unittest
import numpy as np
import shutil
from PIL import Image

# Add project root
sys.path.append(os.getcwd())

from inference.ocr_text import TextOCR, OCRResult

class TestTextOCR(unittest.TestCase):
    
    def setUp(self):
        # Create dummy models dir for testing
        os.makedirs("debug/mock_models", exist_ok=True)
        
    def tearDown(self):
        # Cleanup
        if os.path.exists("debug/mock_models"):
            shutil.rmtree("debug/mock_models")

    def test_initialization_defaults(self):
        """Test simple initialization."""
        print("\n[Test] Initialization Defaults")
        ocr = TextOCR(device="cpu")
        self.assertIsNotNone(ocr)
        if ocr.encoder_session is None:
            print(" -> ONNX not found, fallback mode active")
            self.assertTrue(ocr.use_torch_fallback)

    def test_preprocess(self):
        """Test preprocessing logic."""
        print("\n[Test] Preprocessing")
        ocr = TextOCR(device="cpu")
        
        # Test 3-channel
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pixels = ocr.preprocess(img)
        self.assertEqual(pixels.shape, (1, 3, 384, 384)) # TrOCR default resize
        
        # Test grayscale
        img_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        pixels = ocr.preprocess(img_gray)
        self.assertEqual(pixels.shape, (1, 3, 384, 384))

    def test_recognition_valid_image_fallback(self):
        """Test recognition with actual execution (Torch Fallback)."""
        print("\n[Test] Recognition (Valid Image) - Fallback Path")
        # Initialize with invalid ONNX paths to force fallback
        ocr = TextOCR(
            encoder_path="invalid.onnx",
            decoder_path="invalid.onnx",
            device="cpu"
        )
        
        # Create a simple white image with black text-like noise
        img = np.ones((64, 200, 3), dtype=np.uint8) * 255
        img[20:40, 50:150] = 0
        
        result = ocr.recognize(img)
        
        print(f" -> Result: '{result.text}' (Conf: {result.confidence:.2f})")
        
        self.assertIsInstance(result, OCRResult)
        self.assertIsNotNone(result.text)
        self.assertGreaterEqual(result.confidence, 0.0)

    def test_recognition_crash_recovery(self):
        """Test recovery when even Fallback fails."""
        print("\n[Test] Crash Recovery")
        
        # Point to a directory that exists but has no model
        # AND mock _init_torch_model to fail differently or simulate corruption?
        # Actually my code handles "Model Load Error".
        
        # Initialize
        ocr = TextOCR(
            model_path="debug/mock_models", # Exists but empty
            encoder_path="invalid",
            decoder_path="invalid",
            device="cpu"
        )
        
        # The _init_torch_model will try to load from "debug/mock_models", fail.
        # Then try "microsoft/trocr-small-handwritten", succeed.
        # So this should SUCCEED via secondary fallback.
        
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        result = ocr.recognize(img)
        print(f" -> Secondary Fallback Result: '{result.text}'")
        self.assertNotEqual(result.text, "[Error: Text Recognition Failed - Model Load Error]")

    def test_input_types(self):
        """Test PIL vs Numpy inputs."""
        print("\n[Test] Input Types")
        ocr = TextOCR(device="cpu")
        
        # PIL
        img_pil = Image.new('RGB', (100, 100), color='white')
        res_pil = ocr.recognize(img_pil)
        
        # Numpy
        img_np = np.array(img_pil)
        res_np = ocr.recognize(img_np)
        
        self.assertIsNotNone(res_pil)
        self.assertIsNotNone(res_np)

if __name__ == '__main__':
    unittest.main()
