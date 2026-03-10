"""PubLayNet COCO-style loader yielding (image_path, regions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Tuple


def load_publaynet(annotation_file: str, image_root: str) -> Iterator[Tuple[str, list[dict]]]:
    data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data["images"]}
    grouped: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann)

    for image_id, anns in grouped.items():
        img = images[image_id]
        path = str(Path(image_root) / img["file_name"])
        regions = [
            {
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
            }
            for ann in anns
        ]
        yield path, regions
