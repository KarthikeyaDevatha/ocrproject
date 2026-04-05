import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data

def test_process_sync_invalid_base64():
    response = client.post("/process", json={"image_base64": "invalid_base64_string!"})
    # Should catch the base64 decode error or cv2 decode error
    assert response.status_code in [400, 500]
