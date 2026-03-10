"""Synthetic math sample generation pipeline."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def random_expression() -> sp.Expr:
    x = sp.symbols("x")
    exprs = [
        x**2 + random.randint(1, 9) * x + random.randint(1, 9),
        sp.integrate(random.randint(1, 5) * x**2, x),
        sp.Matrix([[random.randint(1, 9), random.randint(1, 9)], [random.randint(1, 9), random.randint(1, 9)]]),
    ]
    return random.choice(exprs)


def render_latex_to_image(latex: str, out_path: str) -> None:
    fig = plt.figure(figsize=(4, 1.2), dpi=150)
    fig.text(0.05, 0.4, f"${latex}$", fontsize=18)
    plt.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def degrade_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    angle = random.uniform(-5, 5)
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def generate_dataset(output_dir: str, n_samples: int = 100000) -> None:
    out = Path(output_dir)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    labels = out / "labels.txt"

    with labels.open("w", encoding="utf-8") as f:
        for i in range(n_samples):
            expr = random_expression()
            latex = sp.latex(expr)
            tmp = images / f"tmp_{i}.png"
            final = images / f"sample_{i}.png"
            render_latex_to_image(latex, str(tmp))
            degraded = degrade_image(str(tmp))
            cv2.imwrite(str(final), degraded)
            tmp.unlink(missing_ok=True)
            f.write(f"{final.name}\t{latex}\n")
