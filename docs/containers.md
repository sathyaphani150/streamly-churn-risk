# Containerisation & Image Governance Strategy

This document details the container architecture, security hardening, vulnerability scanning, and Software Bill of Materials (SBOM) strategy for the Streamly churn risk prediction microservice.

---

## 1. Multi-Stage Dockerfile Architecture

The container is built using a multi-stage `Dockerfile` separating the compilation environment from the final runtime image:

```
+-------------------------------------------------------------+
| Stage 1: builder (python:3.11-slim-bookworm + uv)           |
| - Layer 1: COPY pyproject.toml uv.lock                      |
| - RUN uv sync --frozen --no-dev --no-install-project        |
| - Layer 2: COPY src/ & RUN uv sync --frozen --no-dev       |
+-------------------------------------------------------------+
                              |
                     COPY --from=builder
                              v
+-------------------------------------------------------------+
| Stage 2: runtime (python:3.11-slim-bookworm)                |
| - No compiler, no package manager tools                     |
| - COPY /app/.venv & /app/src                                |
| - USER appuser (UID 10001)                                  |
| - HEALTHCHECK /health                                       |
| - ENTRYPOINT uvicorn streamly.serving.app:app               |
+-------------------------------------------------------------+
```

### Key Benefits
1. **Layer Caching Optimization**: Manifest files (`pyproject.toml`, `uv.lock`) are copied and installed *before* application source code. When business logic changes in `src/`, Docker reuses the cached dependency layer, reducing build times from minutes to seconds.
2. **Attack Surface Reduction**: Build tools (`uv`, compilation toolchains) are discarded and never shipped to production.
3. **Slim Footprint**: The final image contains only the compiled virtual environment, runtime Python interpreter, and source files.

---

## 2. Hardened Security Profile

### Non-Root Execution (Least Privilege)
Containers executing as `root` represent a critical security vulnerability (container breakouts can compromise the host kernel).
- Dedicated group and user created: `appgroup:10001` and `appuser:10001`.
- Shell disabled (`--shell /bin/false`) and home directory omitted (`--no-create-home`).
- Directive `USER appuser` drops all capabilities before exposing port `8000`.

### Base Image Digest Pinning
- The builder and runtime both use Debian Bookworm slim and are pinned to the exact multi-platform
  image digest resolved during the verified build.
- The external `uv` image used by the builder is also digest-pinned:
  ```dockerfile
  FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89
  COPY --from=ghcr.io/astral-sh/uv:0.6.1@sha256:90daa0b4d74ea55c7b8e06d25d3826b1eac66e7994387248e6173dd2b66668e2 /uv /uvx /bin/
  ```
- The base interpreter's preinstalled `pip`, `setuptools`, and `wheel` copies are removed because
  installation tooling is unnecessary at runtime. If an application dependency requires packaging
  libraries, only its locked, scanned virtual-environment versions remain.

---

## 3. Vulnerability Scanning (Trivy & Grype)

The CI pipeline now executes Trivy `v0.75.0` against the built image. It fails the container job
when a fixable HIGH or CRITICAL vulnerability is found, while ignoring findings that have no vendor
fix. The Trivy action itself is pinned to its immutable `v0.36.0` commit rather than a mutable tag.
The JSON report is retained for 30 days even when the blocking scan fails.

The final hardened image was also scanned locally with this exact policy: Trivy exited `0` with
zero fixable HIGH or CRITICAL findings. An earlier scan found two vulnerable packaging utilities
from the base interpreter; removing those unused system copies eliminated the findings without
changing the locked application environment copied from the builder.

### Recommended CI Command (Trivy)
```bash
# Scan container image for CRITICAL and HIGH severity vulnerabilities
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:0.75.0 image \
  --scanners vuln --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 \
  streamly-churn:local
```
- **Exit Code 1**: Fails the CI pipeline if an unpatched vulnerability exceeds the risk threshold.
- **Ignore Unfixed**: Can be toggled (`--ignore-unfixed`) to focus exclusively on actionable vulnerabilities with vendor fixes.

---

## 4. Software Bill of Materials (SBOM)

### What is an SBOM?
A **Software Bill of Materials (SBOM)** is an authoritative, machine-readable inventory of all third-party software components, libraries, transitive dependencies, and licenses bundled into a container image. It is a cornerstone of modern software supply chain security (NIST SP 800-218, Executive Order 14028) that enables instant vulnerability triage (e.g. Log4Shell style zero-days).

### Generating an SBOM with Syft
Anchore **Syft** generates standard SPDX and CycloneDX format SBOMs:

```bash
# Generate a full SPDX-JSON SBOM from the built image
syft streamly-churn:latest -o spdx-json=sbom.spdx.json

# Or human-readable table summary
syft streamly-churn:latest
```

CI executes the digest-pinned Anchore SBOM action with Syft `v1.54.1`, writes
`reports/security/sbom.spdx.json`, and archives it alongside the Trivy JSON report, Docker build
log, and immutable image ID as the `container-security-evidence` artifact for 30 days.

---

## 5. Local Build & Run Instructions

### 1. Build and Start with Docker Compose

This is the recommended fresh-clone and reviewer path. It selects the safe `dev` runtime profile,
builds the multi-stage image, runs as UID/GID `10001`, drops all Linux capabilities, prevents
privilege escalation, mounts a small temporary filesystem, and waits for `/health`:

```bash
docker compose up --build --wait
```

If host port `8000` is busy, set `STREAMLY_API_PORT` before starting Compose; the container still
listens on its internal port `8000`:

```powershell
$env:STREAMLY_API_PORT = "8081"
docker compose up --build --wait
```

No `.env`, MLflow server, or host model volume is required for this self-contained demonstration.
`STREAMLY_COMPOSE_ENV` is deliberately separate from the host's `STREAMLY_ENV`, preventing an
unrelated shell variable from accidentally placing the review container in production mode.

### 2. Verify Health & Live Inference

Run the portable smoke test from a second terminal:

```bash
docker compose exec api python /app/scripts/smoke_test.py
```

Or inspect each endpoint manually:

```bash
# Check container health status
curl -i http://localhost:8000/health

# Call live scoring endpoint
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "member_id": "mem_prod_001",
    "tenure_days": 210,
    "sessions_7d": 8,
    "watch_hours_7d": 16.5,
    "support_tickets_30d": 0,
    "plan_tier": "premium",
    "price_increase_flag": 0
  }'
```

Confirm Compose reports a healthy container and that the process is non-root:

```bash
docker compose ps
docker compose exec api id -u
# Expected UID: 10001
```

Stop and remove the service without deleting any project data:

```bash
docker compose down
```

When connected to MLflow, `model_version` reports the configured alias, such as
`streamly_churn_model@champion`. The self-contained fallback reports the exact artifact digest as
`local:baseline_model.joblib@sha256:<digest>`, so the response still identifies immutable model
bytes when a registry is unavailable.

Verified standalone-container response:

```json
{
  "member_id": "demo_member",
  "churn_risk": 0.4336,
  "model_version": "local:baseline_model.joblib@sha256:307981abb90b3f20f8d6fea53b0e6885beffb85965fec0b90ec2845e8c4204c6"
}
```

The same verification confirmed that the running process uses UID `10001`.
