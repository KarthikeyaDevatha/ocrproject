import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from backend.api.main import app


def test_health():
    client = TestClient(app)
    res = client.get('/health')
    assert res.status_code == 200
    assert res.json()['status'] == 'ok'
