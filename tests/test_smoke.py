"""Depth smoke tests — verify core endpoints respond."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def patch_models():
    """Patch ML model loading so tests run without torch/GPU."""
    with patch("depth_app.estimator") as mock_est:
        mock_est.loaded_models.return_value = []
        mock_est.VALID_MODELS = {"small"}
        yield mock_est


@pytest.fixture()
def client(patch_models):
    from depth_app import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_cors_headers(client):
    r = client.options(
        "/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code in (200, 204)
