"""CROHME 2019 loader yielding (image_path, latex_label)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple


def load_crohme(root: str) -> Iterator[Tuple[str, str]]:
    root_path = Path(root)
    for inkml in root_path.rglob("*.inkml"):
        latex = ""
        text = inkml.read_text(encoding="utf-8", errors="ignore")
        if "<annotation type=\"truth\">" in text:
            latex = text.split("<annotation type=\"truth\">", 1)[1].split("</annotation>", 1)[0].strip()
        image_path = str(inkml.with_suffix(".png"))
        yield image_path, latex
