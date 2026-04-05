import pytest
import numpy as np
import cv2
from backend.pipelines.preprocessing import (
    deskew_image,
    correct_perspective,
    denoise,
    adaptive_threshold,
    normalize_contrast,
    preprocess_pipeline
)

@pytest.fixture
def sample_image():
    # Create a simple dummy image (white background, black text/shapes)
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Test Text", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return img

def test_deskew_image(sample_image):
    # Rotate the image slightly to simulate skew
    center = (250, 250)
    M = cv2.getRotationMatrix2D(center, 5, 1.0)
    skewed = cv2.warpAffine(sample_image, M, (500, 500), borderValue=(255, 255, 255))
    
    deskewed = deskew_image(skewed)
    assert deskewed is not None
    assert isinstance(deskewed, np.ndarray)
    assert deskewed.shape == sample_image.shape

def test_correct_perspective(sample_image):
    # Pass a valid image
    corrected = correct_perspective(sample_image)
    assert corrected is not None
    # For a flat image with no contour rect, it returns the same image
    assert corrected.shape == sample_image.shape

def test_denoise(sample_image):
    denoised = denoise(sample_image)
    assert denoised is not None
    assert isinstance(denoised, np.ndarray)

def test_normalize_contrast(sample_image):
    normalized = normalize_contrast(sample_image)
    assert normalized is not None
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == sample_image.shape

def test_adaptive_threshold(sample_image):
    thresh = adaptive_threshold(sample_image)
    assert thresh is not None
    assert isinstance(thresh, np.ndarray)
    # Threshold returns a single channel image
    assert len(thresh.shape) == 2

def test_preprocess_pipeline(sample_image):
    final = preprocess_pipeline(sample_image)
    assert final is not None
    assert isinstance(final, np.ndarray)
    # Pipeline ends with threshold so should be 2D
    assert len(final.shape) == 2
