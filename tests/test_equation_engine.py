import pytest
import numpy as np
from backend.models.equation_engine import EquationOCR

def test_equation_ocr_init():
    engine = EquationOCR()
    assert engine is not None

def test_preprocess_crop():
    engine = EquationOCR()
    # Create non-square image
    img = np.ones((100, 200, 3), dtype=np.uint8) * 128
    
    # Process it
    processed = engine.preprocess_crop(img, target_size=(384, 384))
    
    # Target size is 384x384
    assert processed.shape == (384, 384, 3)
    
def test_validate_latex():
    engine = EquationOCR()
    
    assert engine.validate_latex("\\frac{1}{2}") == True
    assert engine.validate_latex("\\frac{1}{2") == False
    assert engine.validate_latex("\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}") == True
    assert engine.validate_latex("\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 ") == False
