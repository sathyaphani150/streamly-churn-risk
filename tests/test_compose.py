"""Static contract tests for the reviewer-facing Docker Compose service."""

from pathlib import Path
from typing import Any, cast

import yaml


def load_api_service() -> dict[str, Any]:
    """Load the Compose API service as a YAML mapping."""
    compose = cast(dict[str, Any], yaml.safe_load(Path("compose.yaml").read_text(encoding="utf-8")))
    return cast(dict[str, Any], compose["services"]["api"])


def test_compose_builds_runtime_and_exposes_local_api() -> None:
    """A fresh clone must build the runtime image and expose its API locally."""
    service = load_api_service()

    assert service["build"]["target"] == "runtime"
    assert service["ports"] == ["127.0.0.1:${STREAMLY_API_PORT:-8000}:8000"]
    assert service["environment"]["STREAMLY_ENV"] == "${STREAMLY_COMPOSE_ENV:-dev}"
    assert "healthcheck" in service


def test_compose_enforces_runtime_hardening() -> None:
    """Compose must preserve least privilege and block privilege escalation."""
    service = load_api_service()

    assert service["user"] == "10001:10001"
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
