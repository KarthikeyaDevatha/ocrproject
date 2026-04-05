import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")

from backend.pipelines.document_processor import DocumentProcessor


def test_process_image_runs():
    processor = DocumentProcessor()
    out = processor.process_image("samples/page.png")
    assert "document_id" in out
    assert "pages" in out
    assert "metadata" in out
