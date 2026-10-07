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
FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89 AS builder

# Install uv directly from verified official multi-arch binary
COPY --from=ghcr.io/astral-sh/uv:0.6.1@sha256:90daa0b4d74ea55c7b8e06d25d3826b1eac66e7994387248e6173dd2b66668e2 /uv /uvx /bin/

WORKDIR /app

# Enable bytecode pre-compilation for instant container startup latency
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Layer 1: Copy only dependency manifests to leverage Docker build cache
COPY pyproject.toml uv.lock ./

# Install production dependencies into isolated virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# Layer 2: Copy application code, configurations, and perform final project installation
COPY src/ ./src/
COPY configs/ ./configs/
COPY README.md params.yaml ./
RUN uv sync --frozen --no-dev

# Generate local fallback model artifact inside image for self-contained portability
RUN /app/.venv/bin/python -m streamly.data.make_dataset && \
    /app/.venv/bin/python -m streamly.pipeline.prepare && \
    /app/.venv/bin/python -m streamly.pipeline.train

# ------------------------------------------------------------------------------
# Stage 2: Minimal Hardened Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89 AS runtime

WORKDIR /app

# Package installers and their vendored build libraries are unnecessary at
# runtime. Removing them reduces both attack surface and fixable CVEs.
RUN python -m pip uninstall --yes pip setuptools wheel

# Configure hardened runtime environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    STREAMLY_ENV="prod"

# Create dedicated non-root user and group (Principle of Least Privilege)
RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid 10001 --no-create-home --shell /bin/false appuser

# Copy virtual environment and project binaries from builder stage
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup scripts/smoke_test.py /app/scripts/smoke_test.py
COPY --chown=appuser:appgroup params.yaml ./

# Copy fallback model artifact generated during build
COPY --from=builder --chown=appuser:appgroup /app/models /app/models

# Drop all root privileges
USER appuser

# Expose HTTP port
EXPOSE 8000

# Automated container healthcheck validating the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production server entrypoint with standard ASGI uvicorn
ENTRYPOINT ["/app/.venv/bin/python", "-m", "uvicorn", "streamly.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
