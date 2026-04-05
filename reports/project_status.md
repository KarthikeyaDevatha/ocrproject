# Project Training Status

Training ready: **False**

## Dependencies
- ❌ numpy
- ❌ cv2
- ❌ fastapi
- ❌ torch
- ❌ transformers
- ❌ paddleocr
- ❌ pix2tex
- ❌ redis

## Datasets
- ❌ CROHME: not found
- ❌ Im2LaTeX-100K: not found
- ❌ HME100K: not found
- ❌ NTCIR-12: not found
- ❌ PubLayNet: not found
- ❌ DocBank: not found
- ❌ arXiv: not found
- ❌ IAM: not found
- ❌ CVL: not found
- ❌ Bentham: not found

## Next Actions
- Install missing dependencies from requirements.txt.
- Download at least one core dataset (CROHME or Im2LaTeX-100K) before training.
- Use training/train_trocr_text.py and training/train_math_model.py for first-pass fine-tuning.