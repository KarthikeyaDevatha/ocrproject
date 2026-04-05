import pytest

cv2 = pytest.importorskip("cv2")

from backend.models.equation_engine import EquationOCR


def test_validate_latex():
    engine = EquationOCR()
    assert engine.validate_latex(r"x^2+1=0")
    assert not engine.validate_latex(r"x^{2")


def test_equation_image_to_latex_returns_str():
    engine = EquationOCR()
    img = cv2.imread("samples/page.png", cv2.IMREAD_GRAYSCALE)
    crop = img[0:128, 0:256]
    out = engine.image_to_latex(crop)
    assert isinstance(out, str)
