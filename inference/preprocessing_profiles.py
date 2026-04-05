"""
Preprocessing Profiles for Adaptive OCR Agent.

Two deterministic profiles selected by the Decision Engine:
- Profile A (clean): for scanned documents, good lighting
- Profile B (degraded): for photographed notes, shadows, faded ink
"""

import cv2
import numpy as np
from typing import Tuple


def profile_clean(image: np.ndarray) -> np.ndarray:
    """
    Profile A — Clean scans (controlled lighting, scanner input).
    
    Pipeline: grayscale → denoise(light) → CLAHE(2.0) → Otsu binarization
    
    Args:
        image: BGR or grayscale input image
    
    Returns:
        Preprocessed grayscale image
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Light denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # CLAHE — adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Otsu binarization
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return binary


def profile_degraded(image: np.ndarray) -> np.ndarray:
    """
    Profile B — Degraded / photographed (shadows, faded ink, stains).
    
    Pipeline: grayscale → denoise(aggressive) → CLAHE(3.5, small tiles) →
              adaptive threshold → morphological close
    
    Args:
        image: BGR or grayscale input image
    
    Returns:
        Preprocessed grayscale image
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Aggressive denoising for degraded images
    denoised = cv2.fastNlMeansDenoising(
        gray, h=20, templateWindowSize=7, searchWindowSize=21
    )
    
    # CLAHE with stronger clip limit and smaller tiles for local enhancement
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(denoised)
    
    # Adaptive threshold — handles uneven lighting better than Otsu
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    
    # Morphological close — reconnect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return closed


def apply_profile(image: np.ndarray, profile: str) -> np.ndarray:
    """
    Apply the named preprocessing profile.
    
    Args:
        image: Input BGR or grayscale image
        profile: "clean" or "degraded"
    
    Returns:
        Preprocessed image
    """
    if profile == "degraded":
        return profile_degraded(image)
    return profile_clean(image)


def to_rgb_for_ocr(preprocessed: np.ndarray) -> np.ndarray:
    """
    Convert preprocessed grayscale/binary image to RGB for TrOCR input.
    
    Args:
        preprocessed: Grayscale or binary image
    
    Returns:
        3-channel RGB image
    """
    if len(preprocessed.shape) == 2:
        return cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
    elif preprocessed.shape[2] == 1:
        return cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
    return preprocessed


def deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Deskew image by detecting text line angle.
    
    Args:
        image: Grayscale input image
        max_angle: Maximum correction angle (beyond this, skip)
    
    Returns:
        Deskewed image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    
    if len(coords) < 10:
        return image
    
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle -= 90
    
    if abs(angle) > max_angle:
        return image
    
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )


def full_preprocess(
    image: np.ndarray,
    profile: str = "clean",
    do_deskew: bool = True
) -> np.ndarray:
    """
    Full preprocessing pipeline: deskew → profile → RGB conversion.
    
    Args:
        image: Input BGR image
        profile: "clean" or "degraded"
        do_deskew: Whether to apply deskewing
    
    Returns:
        Preprocessed RGB image ready for OCR
    """
    # Step 1: Deskew
    if do_deskew:
        image = deskew(image)
    
    # Step 2: Apply profile
    processed = apply_profile(image, profile)
    
    # Step 3: Convert to RGB for OCR models
    rgb = to_rgb_for_ocr(processed)
    
    return rgb
