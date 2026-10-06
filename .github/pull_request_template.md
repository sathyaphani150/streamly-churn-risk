## Problem
Streamly requires near-real-time churn risk prediction (<200ms) with anti-leakage data contracts and automated model gating.

## Approach
- Pandera data contracts enforcing anti-leakage constraints.
- Reproducible DVC pipeline (prepare -> train -> evaluate).
- MLflow experiment tracking and threshold-based promotion (@champion/@challenger).
- FastAPI real-time scoring microservice and multi-stage hardened Docker container.

## How to Run
- uv sync --extra dev
- uv run pytest
- docker build -t streamly-churn:latest .

## Risks & Mitigations
- Feature drift / target leakage: blocked by Pandera schema quality gate.
- Degraded model promotion: blocked by thresholds.yaml gate.
- Container cold start: mitigated by lifespan pre-warming.

## Evidence
- All unit and integration tests passing.
- Ruff & strict Mypy clean.
- Multi-stage non-root container builds and passes /health probe.
