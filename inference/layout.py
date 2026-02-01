"""
Layout detection module using YOLOv8-Nano ONNX.
Detects text lines and math formulas in document images.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    
    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2
    
    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]


class LayoutDetector:
    """YOLOv8 ONNX-based layout detector."""
    
    CLASS_NAMES = {0: "text_line", 1: "math_formula"}
    
    def __init__(
        self,
        model_path: str = "models/yolo/yolov8n-layout.onnx",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        device: str = "cpu"
    ):
        """
        Initialize layout detector.
        
        Args:
            model_path: Path to ONNX model
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            input_size: Model input size
            device: 'cpu' or 'cuda'
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        
        # Setup ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        
        if os.path.exists(model_path):
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
        else:
            print(f"Warning: Model not found at {model_path}. Using fallback detection.")
            self.session = None
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Preprocess image for YOLO inference.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Tuple of (input_tensor, metadata)
        """
        original_shape = image.shape[:2]
        
        # Letterbox resize
        img, ratio, (pad_w, pad_h) = self._letterbox(
            image, 
            new_shape=(self.input_size, self.input_size)
        )
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        
        # HWC to NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        
        metadata = {
            "original_shape": original_shape,
            "ratio": ratio,
            "pad": (pad_w, pad_h)
        }
        
        return img, metadata
    
    def _letterbox(
        self,
        image: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize implementation."""
        shape = image.shape[:2]
        
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
        
        return image, r, (int(dw), int(dh))
    
    def postprocess(
        self,
        output: np.ndarray,
        metadata: dict
    ) -> List[Detection]:
        """
        Postprocess YOLO output to get detections.
        
        Args:
            output: Model output
            metadata: Preprocessing metadata
        
        Returns:
            List of Detection objects
        """
        predictions = np.squeeze(output)
        
        # YOLO output format: (num_detections, 4 + num_classes)
        # or transposed depending on model
        if predictions.ndim == 2:
            if predictions.shape[0] > predictions.shape[1]:
                predictions = predictions.T
        
        # Extract boxes and scores
        boxes = predictions[:4, :].T  # x_center, y_center, width, height
        scores = predictions[4:, :].T if predictions.shape[0] > 4 else None
        
        if scores is None or len(scores) == 0:
            return []
        
        # Get class predictions
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert from center format to corner format
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        if len(indices) == 0:
            return []
        
        indices = indices.flatten()
        
        # Scale boxes back to original image
        ratio = metadata["ratio"]
        pad_w, pad_h = metadata["pad"]
        orig_h, orig_w = metadata["original_shape"]
        
        detections = []
        for i in indices:
            x1, y1, x2, y2 = boxes_xyxy[i]
            
            # Remove padding and scale
            x1 = int((x1 - pad_w) / ratio)
            y1 = int((y1 - pad_h) / ratio)
            x2 = int((x2 - pad_w) / ratio)
            y2 = int((y2 - pad_h) / ratio)
            
            # Clamp to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            class_id = int(class_ids[i])
            
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=self.CLASS_NAMES.get(class_id, "unknown"),
                confidence=float(confidences[i])
            ))
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection on image.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            List of Detection objects
        """
        if self.session is None:
            return self._fallback_detection(image)
        
        # Preprocess
        input_tensor, metadata = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess(outputs[0], metadata)
        
        # Fallback if no detections found (e.g. model weak or domain mismatch)
        if not detections:
            # print("Warning: No layout detected by model -> Using CV fallback")
            return self._fallback_detection(image)
        
        return detections
    
    def _fallback_detection(self, image: np.ndarray) -> List[Detection]:
        """
        Fallback line detection using traditional CV.
        Used when ONNX model is not available.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter small boxes
            if w > 50 and h > 15:
                # Heuristic: math tends to be taller relative to width
                aspect_ratio = w / h if h > 0 else 0
                is_math = h > 50 or aspect_ratio < 3
                
                detections.append(Detection(
                    bbox=(x, y, x + w, y + h),
                    class_id=1 if is_math else 0,
                    class_name="math_formula" if is_math else "text_line",
                    confidence=0.8
                ))
        
        return detections


def cluster_lines(
    detections: List[Detection],
    overlap_threshold: float = 0.5
) -> List[List[Detection]]:
    """
    Cluster detections into reading lines based on vertical overlap.
    
    Args:
        detections: List of detections
        overlap_threshold: Minimum vertical overlap ratio
    
    Returns:
        List of detection clusters (one per line)
    """
    if not detections:
        return []
    
    # Sort by y coordinate
    sorted_dets = sorted(detections, key=lambda d: d.center_y)
    
    clusters = [[sorted_dets[0]]]
    
    for det in sorted_dets[1:]:
        merged = False
        
        for cluster in clusters:
            # Check overlap with cluster items
            for c_det in cluster:
                y1_overlap = max(det.bbox[1], c_det.bbox[1])
                y2_overlap = min(det.bbox[3], c_det.bbox[3])
                
                overlap = max(0, y2_overlap - y1_overlap)
                min_height = min(det.height, c_det.height)
                
                if min_height > 0 and overlap / min_height >= overlap_threshold:
                    cluster.append(det)
                    merged = True
                    break
            
            if merged:
                break
        
        if not merged:
            clusters.append([det])
    
    # Sort items within each cluster by x coordinate
    for cluster in clusters:
        cluster.sort(key=lambda d: d.center_x)
    
    # Sort clusters by y coordinate
    clusters.sort(key=lambda c: min(d.center_y for d in c))
    
    return clusters


def merge_close_boxes(
    detections: List[Detection],
    x_gap_threshold: int = 20
) -> List[Detection]:
    """
    Merge horizontally close boxes of the same type.
    
    Args:
        detections: List of detections
        x_gap_threshold: Maximum horizontal gap to merge
    
    Returns:
        Merged detections
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by x
    sorted_dets = sorted(detections, key=lambda d: d.bbox[0])
    
    merged = [sorted_dets[0]]
    
    for det in sorted_dets[1:]:
        last = merged[-1]
        
        # Check if same class and close enough
        gap = det.bbox[0] - last.bbox[2]
        
        if det.class_id == last.class_id and gap <= x_gap_threshold:
            # Merge boxes
            new_bbox = (
                min(last.bbox[0], det.bbox[0]),
                min(last.bbox[1], det.bbox[1]),
                max(last.bbox[2], det.bbox[2]),
                max(last.bbox[3], det.bbox[3])
            )
            merged[-1] = Detection(
                bbox=new_bbox,
                class_id=last.class_id,
                class_name=last.class_name,
                confidence=max(last.confidence, det.confidence)
            )
        else:
            merged.append(det)
    
    return merged


def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    show_labels: bool = True
) -> np.ndarray:
    """
    Draw detection boxes on image.
    
    Args:
        image: Input image
        detections: List of detections
        show_labels: Whether to show class labels
    
    Returns:
        Image with drawn detections
    """
    img = image.copy()
    
    colors = {
        0: (0, 255, 0),    # text_line: green
        1: (255, 0, 0),    # math_formula: blue
    }
    
    for det in detections:
        color = colors.get(det.class_id, (128, 128, 128))
        x1, y1, x2, y2 = det.bbox
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        if show_labels:
            label = f"{det.class_name}: {det.confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img
