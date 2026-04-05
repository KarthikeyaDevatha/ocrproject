"""
Decision Engine for Adaptive OCR Agent.

Analyzes image features and selects the optimal OCR engine + preprocessing profile.
Features: math symbol density, blur score, text complexity, blank detection, skew angle.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ImageFeatures:
    """Extracted image features for decision making."""
    blur_score: float = 0.0
    mean_intensity: float = 0.0
    std_intensity: float = 0.0
    math_density: float = 0.0
    text_density: float = 0.0
    line_count: int = 0
    skew_angle: float = 0.0
    is_blank: bool = False
    is_arithmetic: bool = False
    edge_density: float = 0.0
    contour_count: int = 0
    image_quality: str = "unknown"  # "clean" | "degraded"


@dataclass
class EngineDecision:
    """Decision output from the engine selector."""
    engine: str            # "mathpix" | "trocr" | "arithmetic"
    profile: str           # "clean" | "degraded"
    reason: str
    features: dict = field(default_factory=dict)
    confidence: float = 0.0


class DecisionEngine:
    """
    Analyzes input images and decides:
    1. Which preprocessing profile to use (clean vs degraded)
    2. Which OCR engine to route to (Mathpix / TrOCR / Arithmetic)
    
    Uses lightweight CV features — no ML model required.
    """

    def __init__(
        self,
        blur_threshold: float = 80.0,
        math_density_threshold: float = 0.3,
        blank_threshold: float = 0.02,
        std_threshold: float = 40.0,
        skew_warning_angle: float = 30.0
    ):
        self.blur_threshold = blur_threshold
        self.math_density_threshold = math_density_threshold
        self.blank_threshold = blank_threshold
        self.std_threshold = std_threshold
        self.skew_warning_angle = skew_warning_angle

    # ========================================================================
    # FEATURE EXTRACTION
    # ========================================================================

    def compute_blur_score(self, gray: np.ndarray) -> float:
        """
        Compute blur score using Laplacian variance.
        Higher = sharper. Below ~80 indicates significant blur.
        """
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def compute_math_density(self, gray: np.ndarray) -> float:
        """
        Estimate math symbol density using edge analysis.
        Math expressions have high edge density in localized regions
        (fractions, integrals, superscripts create dense edge clusters).
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges (math symbols cluster)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Math indicators: small, dense contours with high aspect ratio variation
        h, w = gray.shape
        total_area = h * w
        
        small_dense_count = 0
        total_contour_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            total_contour_area += area
            
            if area < total_area * 0.01:  # Small contours
                x, y, cw, ch = cv2.boundingRect(cnt)
                if ch > 0 and cw > 0:
                    aspect = max(cw, ch) / min(cw, ch)
                    if aspect > 2.0:  # Tall/narrow or wide/short = math-like
                        small_dense_count += 1
        
        # Density = ratio of small dense math-like contours to total
        if len(contours) == 0:
            return 0.0
        
        density = small_dense_count / max(len(contours), 1)
        edge_coverage = total_contour_area / total_area
        
        # Combined score: high density + moderate edge coverage = math
        return min(1.0, density * 0.7 + edge_coverage * 0.3)

    def compute_text_density(self, binary: np.ndarray) -> float:
        """Ratio of text pixels to total pixels."""
        text_pixels = np.sum(binary < 128)  # Dark pixels
        total = binary.size
        return text_pixels / total if total > 0 else 0.0

    def estimate_line_count(self, binary: np.ndarray) -> int:
        """Estimate number of text lines using horizontal projection."""
        # Horizontal projection
        h_proj = np.sum(binary < 128, axis=1)
        
        # Smooth projection
        kernel_size = max(3, binary.shape[0] // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = np.convolve(h_proj, np.ones(kernel_size) / kernel_size, mode='same')
        
        # Count transitions (peaks = lines)
        threshold = np.max(smoothed) * 0.1
        above = smoothed > threshold
        
        # Count rising edges
        transitions = np.diff(above.astype(int))
        line_count = np.sum(transitions == 1)
        
        return max(1, line_count)

    def estimate_skew_angle(self, binary: np.ndarray) -> float:
        """Estimate skew angle using minimum area rectangle."""
        coords = np.column_stack(np.where(binary < 128))
        if len(coords) < 10:
            return 0.0
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle -= 90
        
        return abs(angle)

    def detect_arithmetic_pattern(self, binary: np.ndarray) -> bool:
        """
        Detect if image likely contains simple arithmetic.
        Heuristic: few text lines, low text density, structured layout.
        """
        lines = self.estimate_line_count(binary)
        density = self.compute_text_density(binary)
        
        # Arithmetic: typically 1-3 lines, low density, structured
        return lines <= 5 and density < 0.15

    def detect_blank(self, gray: np.ndarray, binary: np.ndarray) -> bool:
        """Detect if image is effectively blank (no meaningful content)."""
        std = np.std(gray)
        # Use binary image for text computation, not grayscale!
        text_density = self.compute_text_density(binary)
        # Handwriting lines are very thin. Even 0.5% is enough to not be blank.
        return std < 10.0 or text_density < 0.005

    # ========================================================================
    # FEATURE AGGREGATION
    # ========================================================================

    def extract_features(self, image: np.ndarray) -> ImageFeatures:
        """
        Extract all decision features from an image.
        
        Args:
            image: BGR or grayscale image
        
        Returns:
            ImageFeatures dataclass
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binary for text analysis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        features = ImageFeatures()
        features.blur_score = self.compute_blur_score(gray)
        features.mean_intensity = float(np.mean(gray))
        features.std_intensity = float(np.std(gray))
        features.math_density = self.compute_math_density(gray)
        features.text_density = self.compute_text_density(binary)
        features.line_count = self.estimate_line_count(binary)
        features.skew_angle = self.estimate_skew_angle(binary)
        features.is_blank = self.detect_blank(gray, binary)
        features.is_arithmetic = self.detect_arithmetic_pattern(binary)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features.edge_density = float(np.sum(edges > 0) / edges.size)
        
        # Contour count
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features.contour_count = len(contours)
        
        # Quality classification
        features.image_quality = self.select_profile(
            features.blur_score,
            features.mean_intensity,
            features.std_intensity
        )
        
        return features

    # ========================================================================
    # PROFILE SELECTION
    # ========================================================================

    def select_profile(
        self,
        blur_score: float,
        mean_intensity: float,
        std_intensity: float
    ) -> str:
        """
        Select preprocessing profile based on image quality.
        
        Returns:
            "clean" or "degraded"
        """
        if blur_score < self.blur_threshold or std_intensity < self.std_threshold:
            return "degraded"
        return "clean"

    # ========================================================================
    # DECISION
    # ========================================================================

    def decide(
        self,
        image: np.ndarray,
        available_engines: Optional[List[str]] = None
    ) -> EngineDecision:
        """
        Analyze image and decide which OCR engine + profile to use.
        
        Args:
            image: Input image (BGR or grayscale)
            available_engines: List of available engines
                             (e.g., ["mathpix", "trocr", "arithmetic"])
        
        Returns:
            EngineDecision with engine, profile, and reasoning
        
        Raises:
            ValueError: If image is blank
        """
        if available_engines is None:
            available_engines = ["trocr", "arithmetic"]
        
        features = self.extract_features(image)
        features_dict = {
            "blur_score": features.blur_score,
            "mean_intensity": features.mean_intensity,
            "std_intensity": features.std_intensity,
            "math_density": features.math_density,
            "text_density": features.text_density,
            "line_count": features.line_count,
            "skew_angle": features.skew_angle,
            "is_blank": features.is_blank,
            "is_arithmetic": features.is_arithmetic,
            "edge_density": features.edge_density,
            "contour_count": features.contour_count,
            "image_quality": features.image_quality
        }
        
        profile = features.image_quality
        
        # Guard: blank image
        if features.is_blank:
            return EngineDecision(
                engine="none",
                profile=profile,
                reason="Blank image detected — no readable content",
                features=features_dict,
                confidence=0.0
            )
        
        # Guard: extreme skew
        if features.skew_angle > self.skew_warning_angle:
            # Still process but note the warning
            pass
        
        # Decision logic
        if features.math_density > self.math_density_threshold and "mathpix" in available_engines:
            return EngineDecision(
                engine="mathpix",
                profile=profile,
                reason=f"math_density={features.math_density:.2f} > {self.math_density_threshold}",
                features=features_dict,
                confidence=0.8
            )
        elif features.is_arithmetic and "arithmetic" in available_engines:
            return EngineDecision(
                engine="arithmetic",
                profile=profile,
                reason="Arithmetic pattern detected (few lines, low density)",
                features=features_dict,
                confidence=0.7
            )
        elif "trocr" in available_engines:
            reason = "default fast path"
            if features.blur_score < self.blur_threshold:
                reason = f"blur_score={features.blur_score:.1f} < {self.blur_threshold} (degraded)"
            return EngineDecision(
                engine="trocr",
                profile=profile,
                reason=reason,
                features=features_dict,
                confidence=0.6
            )
        else:
            return EngineDecision(
                engine="trocr",
                profile=profile,
                reason="No specialized engine available, using TrOCR",
                features=features_dict,
                confidence=0.5
            )
