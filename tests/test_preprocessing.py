import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("numpy")

from backend.pipelines.preprocessing import (
    adaptive_threshold,
    correct_perspective,
    denoise,
    deskew_image,
    normalize_contrast,
    preprocess_pipeline,
)


def test_preprocess_functions_shape():
    img = cv2.imread("samples/page.png")
    assert img is not None
    assert deskew_image(img).ndim in (2, 3)
    assert correct_perspective(img).ndim in (2, 3)
    assert denoise(img).shape[:2] == img.shape[:2]
    assert adaptive_threshold(img).ndim == 2
    assert normalize_contrast(img).shape[:2] == img.shape[:2]


def test_preprocess_pipeline_output():
    img = cv2.imread("samples/page.png")
    out = preprocess_pipeline(img)
    assert out.ndim == 2
    assert out.shape[0] > 0 and out.shape[1] > 0
