import os
import json
import cv2
import numpy as np
from typing import Tuple, List, Dict, Generator

class PubLayNetLoader:
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Loads PubLayNet Document Layout dataset.
        Annotations are in COCO format.
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, split)
        self.annotation_file = os.path.join(data_dir, f"{split}.json")
        
        self.images_meta = {}
        self.annotations_by_image = {}
        self.categories = {}
        self.image_ids = []
        
        self._load_annotations()

    def _load_annotations(self):
        if not os.path.exists(self.annotation_file):
            return
            
        with open(self.annotation_file, "r") as f:
            data = json.load(f)
            
        for cat in data.get("categories", []):
            self.categories[cat["id"]] = cat["name"]
            
        for img in data.get("images", []):
            self.images_meta[img["id"]] = img
            self.annotations_by_image[img["id"]] = []
            self.image_ids.append(img["id"])
            
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in self.annotations_by_image:
                self.annotations_by_image[img_id].append(ann)

    def get_sample(self, idx: int) -> Tuple[np.ndarray, List[Dict]]:
        if idx < 0 or idx >= len(self.image_ids):
            raise IndexError("Index out of bounds")
            
        img_id = self.image_ids[idx]
        img_meta = self.images_meta[img_id]
        
        img_path = os.path.join(self.image_dir, img_meta["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            raise IOError(f"Failed to load image: {img_path}")
            
        raw_anns = self.annotations_by_image[img_id]
        regions = []
        for ann in raw_anns:
            # COCO bbox: [x, y, width, height]
            bbox = ann["bbox"]
            cat_name = self.categories.get(ann["category_id"], "unknown")
            
            # Convert to [x1, y1, x2, y2]
            regions.append({
                "type": cat_name,
                "bounding_box": [
                    int(bbox[0]), 
                    int(bbox[1]), 
                    int(bbox[0] + bbox[2]), 
                    int(bbox[1] + bbox[3])
                ]
            })
            
        return image, regions

    def load_generator(self) -> Generator[Tuple[np.ndarray, List[Dict]], None, None]:
        for idx in range(len(self.image_ids)):
            try:
                yield self.get_sample(idx)
            except IOError:
                continue
