# ==============================================================================
# Streamly Production Churn Scoring Service Dockerfile
#
# Architectural Highlights:
# 1. Multi-Stage Build: Separates build tooling (uv, compilers) from slim runtime.
# 2. Layered Caching: Dependencies installed from uv.lock before copying source.
# 3. Least-Privilege Security: Runs under unprivileged non-root user (appuser:10001).
# 4. Built-in Container Healthcheck: Automated probing of /health for orchestrators.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Dependency Resolution
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Install uv directly from verified official multi-arch binary
COPY --from=ghcr.io/astral-sh/uv:0.6.1 /uv /uvx /bin/

WORKDIR /app

# Enable bytecode pre-compilation for instant container startup latency
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Layer 1: Copy only dependency manifests to leverage Docker build cache
COPY pyproject.toml uv.lock ./

# Install production dependencies into isolated virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# Layer 2: Copy application code and perform final project installation
COPY src/ ./src/
COPY README.md ./
RUN uv sync --frozen --no-dev

# ------------------------------------------------------------------------------
# Stage 2: Minimal Hardened Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Configure hardened runtime environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    STREAMLY_ENV="prod" \
    MODEL_NAME="streamly_churn_model" \
    MODEL_REGISTRY_ALIAS="champion"

# Create dedicated non-root user and group (Principle of Least Privilege)
RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid 10001 --no-create-home --shell /bin/false appuser

# Copy virtual environment and project binaries from builder stage
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --chown=appuser:appgroup src/ /app/src/

# Copy local fallback model artifact for air-gapped / offline deployments
COPY --chown=appuser:appgroup models/baseline_model.joblib /app/models/baseline_model.joblib

# Drop all root privileges
USER appuser

# Expose HTTP port
EXPOSE 8000

# Automated container healthcheck validating the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production server entrypoint with standard ASGI uvicorn
ENTRYPOINT ["uvicorn", "streamly.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
