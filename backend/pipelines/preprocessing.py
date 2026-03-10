"""Image preprocessing utilities for document OCR pipelines."""

from __future__ import annotations

import cv2
import numpy as np


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Deskew image using min-area-rect angle estimation on foreground pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    inv = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(inv > 0))
    if coords.size == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    flags = cv2.INTER_CUBIC
    border = cv2.BORDER_REPLICATE
    if image.ndim == 3:
        return cv2.warpAffine(image, matrix, (w, h), flags=flags, borderMode=border)
    return cv2.warpAffine(gray, matrix, (w, h), flags=flags, borderMode=border)


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """Apply perspective correction from largest page-like contour."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    page = None
    for contour in contours[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            page = approx.reshape(4, 2)
            break

    if page is None:
        return image

    rect = _order_points(page)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def denoise(image: np.ndarray) -> np.ndarray:
    """Denoise image while preserving edges."""
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Adaptive thresholding for robust binarization under uneven lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )


def normalize_contrast(image: np.ndarray) -> np.ndarray:
    """Normalize contrast via CLAHE."""
    if image.ndim == 2:
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_pipeline(image: np.ndarray) -> np.ndarray:
    """Chain preprocessing steps in order optimized for document quality."""
    step = correct_perspective(image)
    step = deskew_image(step)
    step = denoise(step)
    step = normalize_contrast(step)
    step = adaptive_threshold(step)
    return step


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect
