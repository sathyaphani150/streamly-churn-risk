"""Cross-platform smoke test for the Compose-hosted Streamly API."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

BASE_URL = os.getenv("STREAMLY_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MEMBER_PAYLOAD: dict[str, Any] = {
    "member_id": "compose_smoke_test",
    "tenure_days": 210,
    "sessions_7d": 8,
    "watch_hours_7d": 16.5,
    "support_tickets_30d": 0,
    "plan_tier": "premium",
    "price_increase_flag": 0,
}


def read_json(request: str | urllib.request.Request) -> dict[str, Any]:
    """Send an HTTP request and decode its JSON object response."""
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
        payload: Any = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("Expected the API to return a JSON object.")
    return payload


def wait_until_healthy(attempts: int = 30, delay_seconds: float = 2.0) -> dict[str, Any]:
    """Wait for Compose startup and return the first healthy response."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            health = read_json(f"{BASE_URL}/health")
            if health.get("status") == "healthy" and health.get("model_loaded") is True:
                return health
            last_error = RuntimeError(f"Service is not ready: {health}")
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            last_error = exc
        time.sleep(delay_seconds)
    raise RuntimeError(f"Service did not become healthy at {BASE_URL}: {last_error}")


def main() -> None:
    """Verify health, model identity, and one real scoring request."""
    health = wait_until_healthy()
    request = urllib.request.Request(
        f"{BASE_URL}/score",
        data=json.dumps(MEMBER_PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    score = read_json(request)
    churn_risk = score.get("churn_risk")
    if not isinstance(churn_risk, int | float) or not 0.0 <= churn_risk <= 1.0:
        raise RuntimeError(f"Invalid scoring response: {score}")
    if not score.get("model_version"):
        raise RuntimeError(f"Scoring response has no model identity: {score}")

    print(f"[smoke] Healthy service: {health}")
    print(f"[smoke] Valid prediction: {score}")


if __name__ == "__main__":
    main()
