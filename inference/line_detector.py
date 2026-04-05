"""
CRAFT-based Line Detector for Adaptive OCR Agent.

Uses CRAFT (Character-Region Awareness For Text) for text line detection,
with fallback to contour-based detection if CRAFT is unavailable.

Crops individual text lines from full-page images for per-line TrOCR inference.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional


def _sort_boxes_top_to_bottom(
    boxes: List[Tuple[int, int, int, int]],
    y_tolerance: int = 20
) -> List[Tuple[int, int, int, int]]:
    """Sort bounding boxes top-to-bottom, left-to-right within same line."""
    if not boxes:
        return boxes
    
    # Sort by y_center first
    boxes_with_center = [(b, (b[1] + b[3]) / 2) for b in boxes]
    boxes_with_center.sort(key=lambda x: x[1])
    
    # Group into lines by Y proximity
    lines = []
    current_line = [boxes_with_center[0]]
    
    for i in range(1, len(boxes_with_center)):
        box_data = boxes_with_center[i]
        prev_y = sum(b[1] for b in current_line) / len(current_line)
        
        if abs(box_data[1] - prev_y) <= y_tolerance:
            current_line.append(box_data)
        else:
            lines.append(current_line)
            current_line = [box_data]
    
    if current_line:
        lines.append(current_line)
    
    # Sort each line left-to-right, then flatten
    sorted_boxes = []
    for line in lines:
        line.sort(key=lambda x: x[0][0])  # Sort by x1
        sorted_boxes.extend([b[0] for b in line])
    
    return sorted_boxes


def _merge_overlapping_boxes(
    boxes: List[Tuple[int, int, int, int]],
    y_tolerance: int = 15,
    x_gap_tolerance: int = 30
) -> List[Tuple[int, int, int, int]]:
    """
    Merge bounding boxes that belong to the same text line.
    Uses Y-center proximity and X-gap to merge horizontally adjacent boxes.
    """
    if not boxes:
        return boxes
    
    # Group by Y-center proximity (same as group_into_lines from notebook)
    boxes_sorted = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    
    merged_lines = []
    current_group = [boxes_sorted[0]]
    
    for i in range(1, len(boxes_sorted)):
        box = boxes_sorted[i]
        prev_y = sum((b[1] + b[3]) / 2 for b in current_group) / len(current_group)
        curr_y = (box[1] + box[3]) / 2
        
        if abs(curr_y - prev_y) <= y_tolerance:
            current_group.append(box)
        else:
            # Merge current group into single box
            x1 = min(b[0] for b in current_group)
            y1 = min(b[1] for b in current_group)
            x2 = max(b[2] for b in current_group)
            y2 = max(b[3] for b in current_group)
            merged_lines.append((x1, y1, x2, y2))
            current_group = [box]
    
    # Don't forget last group
    if current_group:
        x1 = min(b[0] for b in current_group)
        y1 = min(b[1] for b in current_group)
        x2 = max(b[2] for b in current_group)
        y2 = max(b[3] for b in current_group)
        merged_lines.append((x1, y1, x2, y2))
    
    return merged_lines


class LineDetector:
    """
    Text line detector using CRAFT or contour-based fallback.
    
    Primary: CRAFT text detector (lightweight CNN for detection only)
    Fallback: OpenCV contour-based detection (no extra dependencies)
    """

    def __init__(self, use_craft: bool = True, y_tolerance: int = 20, pad: int = 5):
        self.use_craft = use_craft
        self.y_tolerance = y_tolerance
        self.pad = pad
        self._craft = None
        self._craft_available = None

    def _check_craft(self) -> bool:
        """Check if CRAFT is importable."""
        if self._craft_available is not None:
            return self._craft_available
        
        try:
            from craft_text_detector import Craft
            self._craft_available = True
        except ImportError:
            self._craft_available = False
        
        return self._craft_available

    def detect_lines_craft(
        self, image_path: str
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect text lines using CRAFT text detector.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            List of (x1, y1, x2, y2) bounding boxes
        """
        from craft_text_detector import Craft
        import torch
        
        craft = Craft(
            output_dir=None,
            crop_type="box",
            cuda=torch.cuda.is_available()
        )
        
        try:
            result = craft.detect_text(image_path)
            boxes = result["boxes"]
            
            bboxes = []
            for box in boxes:
                x1 = int(min(p[0] for p in box))
                y1 = int(min(p[1] for p in box))
                x2 = int(max(p[0] for p in box))
                y2 = int(max(p[1] for p in box))
                
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    bboxes.append((x1, y1, x2, y2))
            
            return bboxes
        
        finally:
            craft.unload_craftnet_model()
            craft.unload_refinenet_model()

    def detect_lines_contour(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Fallback: detect text lines using OpenCV contours + horizontal projection.
        No external dependencies required.
        
        Args:
            image: BGR or grayscale image
        
        Returns:
            List of (x1, y1, x2, y2) bounding boxes
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal dilation to connect text on same line
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        dilated = cv2.dilate(binary, kernel, iterations=3)
        
        # Slight vertical dilation to catch ascenders/descenders
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        dilated = cv2.dilate(dilated, kernel_v, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = gray.shape
        bboxes = []
        
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            
            # Filter tiny noise
            if cw < w * 0.05 or ch < 5:
                continue
            
            bboxes.append((x, y, x + cw, y + ch))
        
        return bboxes

    def detect_and_crop(
        self,
        image_path: str,
        image: Optional[np.ndarray] = None,
        merge_lines: bool = True
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Detect text lines and crop them from the image.
        
        Args:
            image_path: Path to image file
            image: Optional preloaded image (BGR numpy array)
            merge_lines: Whether to merge overlapping boxes into lines
        
        Returns:
            List of (cropped_PIL_image, (x1, y1, x2, y2)) tuples
        """
        # Load image if not provided
        if image is None:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
        
        h, w = image.shape[:2]
        
        # Try CRAFT first, fall back to contour
        if self.use_craft and self._check_craft():
            try:
                bboxes = self.detect_lines_craft(image_path)
            except Exception as e:
                print(f"CRAFT failed ({e}), using contour fallback")
                bboxes = self.detect_lines_contour(image)
        else:
            bboxes = self.detect_lines_contour(image)
        
        # Handle no detections
        if not bboxes:
            # Return full image as single line
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return [(pil_image, (0, 0, w, h))]
        
        # Merge into lines
        if merge_lines:
            bboxes = _merge_overlapping_boxes(bboxes, self.y_tolerance)
        
        # Sort top-to-bottom
        bboxes = _sort_boxes_top_to_bottom(bboxes, self.y_tolerance)
        
        # Crop lines
        crops = []
        for (x1, y1, x2, y2) in bboxes:
            # Add padding
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = min(w, x2 + self.pad)
            y2 = min(h, y2 + self.pad)
            
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crops.append((pil_crop, (x1, y1, x2, y2)))
        
        return crops if crops else [(
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            (0, 0, w, h)
        )]

    def get_bounding_boxes(
        self,
        image_path: str,
        image: Optional[np.ndarray] = None
    ) -> List[Tuple[int, int, int, int]]:
        """
        Get merged bounding boxes only (no cropping).
        Useful for drawing overlays on the UI.
        """
        if image is None:
            image = cv2.imread(image_path)
        
        if self.use_craft and self._check_craft():
            try:
                bboxes = self.detect_lines_craft(image_path)
            except Exception:
                bboxes = self.detect_lines_contour(image)
        else:
            bboxes = self.detect_lines_contour(image)
        
        return _merge_overlapping_boxes(bboxes, self.y_tolerance)
