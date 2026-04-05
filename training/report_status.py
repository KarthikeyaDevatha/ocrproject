"""Project training/status reporter for OCR platform."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict

REQUIRED_DATASETS = {
    "CROHME": ["data/crohme", "datasets/crohme"],
    "Im2LaTeX-100K": ["data/im2latex", "datasets/im2latex"],
    "HME100K": ["data/hme100k", "datasets/hme100k"],
    "NTCIR-12": ["data/ntcir12", "datasets/ntcir12"],
    "PubLayNet": ["data/publaynet", "datasets/publaynet"],
    "DocBank": ["data/docbank", "datasets/docbank"],
    "arXiv": ["data/arxiv", "datasets/arxiv"],
    "IAM": ["data/iam", "datasets/iam"],
    "CVL": ["data/cvl", "datasets/cvl"],
    "Bentham": ["data/bentham", "datasets/bentham"],
}

DEPENDENCIES = [
    "numpy",
    "cv2",
    "fastapi",
    "torch",
    "transformers",
    "paddleocr",
    "pix2tex",
    "redis",
]


def module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def detect_datasets() -> Dict[str, dict]:
    out = {}
    for ds, candidates in REQUIRED_DATASETS.items():
        found = None
        for c in candidates:
            p = Path(c)
            if p.exists():
                found = p
                break
        out[ds] = {
            "available": found is not None,
            "path": str(found) if found else "",
        }
    return out


def main() -> None:
    deps = {name: module_available(name) for name in DEPENDENCIES}
    datasets = detect_datasets()

    training_ready = all(deps.get(k, False) for k in ["torch", "transformers", "numpy"]) and any(
        v["available"] for v in datasets.values()
    )

    report = {
        "dependency_status": deps,
        "dataset_status": datasets,
        "training_ready": training_ready,
        "recommendations": [
            "Install missing dependencies from requirements.txt.",
            "Download at least one core dataset (CROHME or Im2LaTeX-100K) before training.",
            "Use training/train_trocr_text.py and training/train_math_model.py for first-pass fine-tuning.",
        ],
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/project_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Project Training Status", "", f"Training ready: **{training_ready}**", "", "## Dependencies"]
    for name, ok in deps.items():
        lines.append(f"- {'✅' if ok else '❌'} {name}")
    lines.append("\n## Datasets")
    for ds, info in datasets.items():
        if info["available"]:
            lines.append(f"- ✅ {ds}: `{info['path']}`")
        else:
            lines.append(f"- ❌ {ds}: not found")
    lines.append("\n## Next Actions")
    for rec in report["recommendations"]:
        lines.append(f"- {rec}")
    Path("reports/project_status.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
