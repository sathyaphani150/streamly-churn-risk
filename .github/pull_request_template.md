## Problem
Streamly requires near-real-time churn risk prediction (<200ms) with anti-leakage data contracts and automated model gating.

## Approach
- Pandera data contracts enforcing anti-leakage constraints.
- Reproducible DVC pipeline (prepare -> train -> evaluate).
- MLflow experiment tracking, threshold gates, and audited @challenger/@champion governance.
- Enforced dev/ci/prod profiles with production failing closed.
- FastAPI scoring service, Docker Compose reviewer path, and hardened non-root image.
- Blocking Trivy scan and generated SPDX-JSON SBOM retained as CI evidence.

## How to Run
- uv sync --extra dev
- uv run python scripts/verify.py
- docker compose up --build --wait
- docker compose exec api python /app/scripts/smoke_test.py

## Risks & Mitigations
- Feature drift / target leakage: blocked by Pandera schema quality gate.
- Degraded or unauthorized model promotion: blocked by thresholds and alias authorization gates.
- Container cold start: mitigated by lifespan pre-warming.
- Newly disclosed dependency CVEs: blocking Trivy scan is evaluated on every image build.

## Evidence
- All unit and integration tests passing.
- Ruff & strict Mypy clean.
- DVC reports data and pipeline stages are up to date.
- Multi-stage non-root Compose service passes `/health` and `/score` smoke tests.
- Trivy reports no fixable HIGH/CRITICAL findings; Syft SBOM is attached by CI.
