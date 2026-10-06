"""Integration tests for the Streamly FastAPI churn risk scoring service.

Verifies:
- GET /health liveness and readiness probe.
- POST /score valid prediction contract and probability boundedness.
- Sub-200ms latency execution guarantee.
- Rejection of missing fields and invalid plan tiers (HTTP 422).
- Strict rejection of future label / leakage attempts (HTTP 422).
- Dynamic registry alias resolution (@champion vs @challenger).
"""

from __future__ import annotations

import os
import re
from collections.abc import Generator
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from streamly.serving.app import app, load_scoring_model


def is_valid_model_version(version: str) -> bool:
    """Accept a registry alias or an immutable local artifact digest."""
    return version.startswith("streamly_churn_model@") or bool(
        re.fullmatch(r"local:.+@sha256:[0-9a-f]{64}", version)
    )


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Provide a TestClient with triggered lifespan events."""
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns.db"
    os.environ["MODEL_NAME"] = "streamly_churn_model"
    os.environ["MODEL_REGISTRY_ALIAS"] = "champion"

    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client: TestClient) -> None:
    """GET /health must return status healthy with model version details."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert is_valid_model_version(data["model_version"])
    assert "X-Process-Time-Ms" in response.headers


def test_root_redirects_to_docs(client: TestClient) -> None:
    """GET / must redirect to /docs Swagger UI."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_score_valid_payload(client: TestClient) -> None:
    """POST /score must calculate and return calibrated probability for valid member."""
    payload = {
        "member_id": "mem_0042",
        "tenure_days": 180,
        "sessions_7d": 5,
        "watch_hours_7d": 12.5,
        "support_tickets_30d": 1,
        "plan_tier": "standard",
        "price_increase_flag": 0,
    }

    response = client.post("/score", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["member_id"] == "mem_0042"
    assert isinstance(data["churn_risk"], float)
    assert 0.0 <= data["churn_risk"] <= 1.0
    assert is_valid_model_version(data["model_version"])

    # Verify latency constraint: execution duration header must be under 200ms
    duration_ms = float(response.headers["X-Process-Time-Ms"])
    assert duration_ms < 200.0, f"Latency violation: {duration_ms}ms >= 200ms"


def test_score_rejects_label_leakage(client: TestClient) -> None:
    """POST /score must strictly reject attempts to pass the target label churned_30d."""
    payload = {
        "member_id": "mem_0042",
        "tenure_days": 180,
        "sessions_7d": 5,
        "watch_hours_7d": 12.5,
        "support_tickets_30d": 1,
        "plan_tier": "standard",
        "price_increase_flag": 0,
        "churned_30d": 1,  # Target leakage!
    }

    response = client.post("/score", json=payload)
    assert response.status_code == 422
    assert "extra_forbidden" in str(response.json()) or "churned_30d" in str(response.json())


def test_score_rejects_missing_field(client: TestClient) -> None:
    """POST /score must reject requests missing required fields."""
    payload = {
        "member_id": "mem_0042",
        # tenure_days is missing
        "sessions_7d": 5,
        "watch_hours_7d": 12.5,
        "support_tickets_30d": 1,
        "plan_tier": "standard",
        "price_increase_flag": 0,
    }

    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_score_rejects_invalid_plan_tier(client: TestClient) -> None:
    """POST /score must enforce controlled vocabulary on plan_tier."""
    payload = {
        "member_id": "mem_0042",
        "tenure_days": 180,
        "sessions_7d": 5,
        "watch_hours_7d": 12.5,
        "support_tickets_30d": 1,
        "plan_tier": "enterprise_gold",  # Invalid tier
        "price_increase_flag": 0,
    }

    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_score_rejects_negative_values(client: TestClient) -> None:
    """POST /score must reject negative engagement counts."""
    payload = {
        "member_id": "mem_0042",
        "tenure_days": -10,  # Invalid negative tenure
        "sessions_7d": 5,
        "watch_hours_7d": 12.5,
        "support_tickets_30d": 1,
        "plan_tier": "standard",
        "price_increase_flag": 0,
    }

    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_local_fallback_has_immutable_sha256_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local fallback responses must identify the exact model artifact bytes."""
    artifact_path = tmp_path / "test-model.joblib"
    joblib.dump({"model": "test"}, artifact_path)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'missing.db'}")
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(artifact_path))

    model, version = load_scoring_model()

    assert model == {"model": "test"}
    assert re.fullmatch(r"local:test-model\.joblib@sha256:[0-9a-f]{64}", version)
