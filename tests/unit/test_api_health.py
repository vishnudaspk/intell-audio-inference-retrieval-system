"""
Unit tests for FastAPI health and endpoint validation.
"""

from fastapi.testclient import TestClient

from backend.api import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "database" in data["services"]


def test_api_get_nonexistent_asset():
    response = client.get("/api/v1/assets/non-existent-uuid")
    assert response.status_code == 404
