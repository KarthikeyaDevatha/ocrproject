import os
import cv2
import numpy as np
from typing import Tuple, Generator

class CrohmeLoader:
    def __init__(self, data_dir: str):
        """
        Loads the CROHME dataset (Handwritten math expressions)
        Format varies, assuming standard INKML parsed to images or raw PNGs mapped to labels.
        Assuming pre-rendered images and a labels.txt format for simplicity.
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.label_file = os.path.join(data_dir, "labels.txt")
        self.samples = self._load_manifest()

    def _load_manifest(self):
        samples = []
        if not os.path.exists(self.label_file):
            return samples
            
        with open(self.label_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    samples.append({
                        "filename": parts[0],
                        "latex": parts[1]
                    })
        return samples

    def get_sample(self, idx: int) -> Tuple[np.ndarray, str]:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError("Index out of bounds")
            
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample["filename"])
        image = cv2.imread(img_path)
        if image is None:
            raise IOError(f"Failed to load image: {img_path}")
            
        return image, sample["latex"]

    def load_generator(self) -> Generator[Tuple[np.ndarray, str], None, None]:
        for idx in range(len(self.samples)):
            try:
                yield self.get_sample(idx)
            except IOError:
                continue
