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

### Base Image Tag Pinning
- Pinned to Debian Bookworm slim (`python:3.11-slim-bookworm`).
- In production deployment manifests, immutable cryptographic sha256 digests are used:
  ```dockerfile
  FROM python:3.11-slim-bookworm@sha256:d8b74685f4019a3b2b40656a59600a94bca99ab85c1ff94e094eb9726dc6a0c5
  ```

---

## 3. Vulnerability Scanning (Trivy & Grype)

Image vulnerability scanning is integrated into the automated CI pipeline before any image is tagged or pushed to a container registry.

### Recommended CI Command (Trivy)
```bash
# Scan container image for CRITICAL and HIGH severity vulnerabilities
trivy image --severity HIGH,CRITICAL --exit-code 1 streamly-churn:latest
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

The resulting `sbom.spdx.json` is archived as a build artifact in CI alongside the container image digest.

---

## 5. Local Build & Run Instructions

### 1. Build the Container Image
```bash
docker build -t streamly-churn:latest .
```

### 2. Run the Container
```bash
docker run -d \
  --name streamly-api \
  -p 8000:8000 \
  -e STREAMLY_ENV=prod \
  -e MODEL_REGISTRY_ALIAS=champion \
  streamly-churn:latest
```

### 3. Verify Health & Live Inference
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
