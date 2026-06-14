"""API route tests using FastAPI TestClient."""

from fastapi.testclient import TestClient


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_readiness(client: TestClient):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_validate_valid_records(client: TestClient, sample_responses):
    resp = client.post("/api/v1/survey/validate", json={"responses": sample_responses})
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid_count"] == 3
    assert data["invalid_count"] == 0
    assert data["pass_rate"] == 1.0


def test_validate_empty_text(client: TestClient):
    resp = client.post(
        "/api/v1/survey/validate",
        json={"responses": [{"response_id": "x", "text": ""}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["invalid_count"] == 1


def test_analyze_empty_payload(client: TestClient):
    resp = client.post(
        "/api/v1/survey/analyze",
        json={"responses": [{"response_id": "bad", "text": ""}]},
    )
    # No valid records → 422
    assert resp.status_code == 422
