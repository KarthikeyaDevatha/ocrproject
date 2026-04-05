import os
import cv2
import numpy as np
from typing import Tuple, Generator

class Im2LatexLoader:
    def __init__(self, data_dir: str):
        """
        Loads the Im2LaTeX-100K dataset.
        Format typically contains formulas.norm.lst, images directory, and train/val/test splits.
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "formula_images")
        self.formulas_file = os.path.join(data_dir, "formulas.norm.lst")
        self.split_file = os.path.join(data_dir, "im2latex_train.lst")
        
        self.formulas = self._load_formulas()
        self.samples = self._load_split()

    def _load_formulas(self):
        formulas = []
        if not os.path.exists(self.formulas_file):
            return formulas
        with open(self.formulas_file, "r") as f:
            formulas = [line.strip() for line in f]
        return formulas

    def _load_split(self):
        samples = []
        if not os.path.exists(self.split_file):
            return samples
        with open(self.split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # Format: formula_idx image_name render_type
                    form_idx = int(parts[0])
                    img_name = parts[1]
                    if form_idx < len(self.formulas):
                        samples.append({
                            "filename": img_name,
                            "latex": self.formulas[form_idx]
                        })
        return samples

    def get_sample(self, idx: int) -> Tuple[np.ndarray, str]:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError("Index out of bounds")
            
        sample = self.samples[idx]
        # Images usually have .png extension in dir despite split file
        img_filename = sample["filename"]
        if not img_filename.endswith('.png'):
            img_filename += ".png"
            
        img_path = os.path.join(self.image_dir, img_filename)
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
