"""Im2LaTeX-100K loader."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple


def load_im2latex(root: str, split: str = "train") -> Iterator[Tuple[str, str]]:
    root_path = Path(root)
    mapping = root_path / f"{split}.lst"
    formulas = (root_path / "formulas.norm.lst").read_text(encoding="utf-8").splitlines()
    for line in mapping.read_text(encoding="utf-8").splitlines():
        idx, image_rel = line.split()
        yield str(root_path / "images" / image_rel), formulas[int(idx)]
