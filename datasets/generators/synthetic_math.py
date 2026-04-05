import os
import random
import logging
import uuid
import numpy as np
import cv2

try:
    import sympy
except ImportError:
    sympy = None

try:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
except ImportError:
    plt = None

try:
    import albumentations as A
except ImportError:
    A = None

class SyntheticMathGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, "images")
        self.label_file = os.path.join(output_dir, "labels.txt")
        
        os.makedirs(self.image_dir, exist_ok=True)
        
        if sympy is None or plt is None or A is None:
            logging.warning("Missing synthetic generation dependencies. Please install: sympy, matplotlib, albumentations")

    def random_polynomial(self):
        x, y = sympy.symbols('x y')
        deg_x = random.randint(1, 4)
        deg_y = random.randint(1, 4)
        expr = sum(random.randint(-10, 10) * (x**i) * (y**j) 
                   for i in range(deg_x+1) for j in range(deg_y+1))
        # Add basic equation
        return sympy.Eq(expr, random.randint(-50, 50))
        
    def random_integral(self):
        x = sympy.symbols('x')
        expr = random.randint(1, 10) * x**random.randint(1, 3) + random.randint(1, 5) * sympy.sin(x)
        return sympy.Integral(expr, (x, 0, random.randint(1, 10)))
        
    def random_matrix(self):
        n = random.randint(2, 4)
        m = random.randint(2, 4)
        return sympy.Matrix(n, m, lambda i, j: random.randint(-10, 10))
        
    def render_latex_to_image(self, latex_str: str, font_size: int = 16) -> np.ndarray:
        if plt is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        # Matplotlib latex rendering
        fig = plt.figure(figsize=(4, 1), dpi=200)
        fig.text(0.5, 0.5, f"${latex_str}$", size=font_size, ha='center', va='center')
        
        # Draw on canvas
        fig.canvas.draw()
        
        # Convert to numpy array
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        # Crop white borders
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        coords = cv2.findNonZero(255 - gray)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y-10:y+h+10, x-10:x+w+10]
            
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    def apply_degradations(self, img: np.ndarray) -> np.ndarray:
        if A is None:
            return img
            
        transforms = A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3)
        ])
        
        augmented = transforms(image=img)
        return augmented['image']

    def generate_sample(self):
        if sympy is None:
            return None
            
        # Select expression type
        choices = [self.random_polynomial, self.random_integral, self.random_matrix]
        expr_func = random.choice(choices)
        expr = expr_func()
        
        # Format latex
        latex_str = sympy.latex(expr)
        
        # Render clean image
        clean_img = self.render_latex_to_image(latex_str, font_size=random.randint(14, 22))
        
        # Apply noise/degradation
        noisy_img = self.apply_degradations(clean_img)
        
        return {"image": noisy_img, "latex": latex_str}
        
    def generate_dataset(self, num_samples: int):
        logging.info(f"Generating {num_samples} synthetic math samples...")
        
        with open(self.label_file, "a") as f:
            for i in range(num_samples):
                sample = self.generate_sample()
                if sample is None:
                    continue
                    
                filename = f"synthetic_{uuid.uuid4().hex[:8]}.png"
                img_path = os.path.join(self.image_dir, filename)
                
                # Save image
                cv2.imwrite(img_path, sample["image"])
                
                # Save label mapping with tab separation
                f.write(f"{filename}\t{sample['latex']}\n")
                
                if (i+1) % 100 == 0:
                    logging.info(f"Generated {i+1}/{num_samples} samples.")
                    
        logging.info("Generation complete.")

if __name__ == "__main__":
    generator = SyntheticMathGenerator("./datasets/synthetic_math_data")
    generator.generate_dataset(10)  # Demo 10 samples
