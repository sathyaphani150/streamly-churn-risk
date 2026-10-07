"""FastAPI real-time churn risk prediction service for Streamly.

Exposes:
- GET /health: Health check and active model version inspection.
- POST /score: Validates raw member payload, enforces anti-leakage contracts,
  and returns calibrated churn risk probability in <200ms.
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from streamly.config import EnvironmentConfig, load_environment_config
from streamly.data.validation import validate_serving_data
from streamly.features.builder import build_serving_features


class ScoreRequest(BaseModel):
    """Member snapshot request payload for real-time churn risk scoring.

    Enforces non-negotiable contract:
    - Member features available at request time only.
    - Zero future labels or post-churn leakage fields (extra="forbid").
    """

    model_config = ConfigDict(extra="forbid")

    member_id: str = Field(..., description="Unique member identifier (e.g. 'mem_0042')")
    tenure_days: int = Field(..., ge=0, description="Account age in days")
    sessions_7d: int = Field(..., ge=0, description="App sessions over previous 7 days")
    watch_hours_7d: float = Field(..., ge=0.0, description="Watch time in hours over previous 7 days")
    support_tickets_30d: int = Field(..., ge=0, description="Support inquiries over previous 30 days")
    plan_tier: Literal["basic", "standard", "premium"] = Field(
        ..., description="Subscription tier controlled vocabulary"
    )
    price_increase_flag: int = Field(
        ..., ge=0, le=1, description="Binary flag indicating whether price hike is active"
    )


class ScoreResponse(BaseModel):
    """Churn risk scoring response conforming to Streamly SLA."""

    member_id: str = Field(..., description="Echoed member identifier")
    churn_risk: float = Field(..., ge=0.0, le=1.0, description="Predicted churn probability [0, 1]")
    model_version: str = Field(..., description="Identifier or alias of serving model artifact")


class HealthResponse(BaseModel):
    """Liveness and readiness health payload."""

    status: str
    environment: str
    model_name: str
    model_version: str
    model_loaded: bool


def local_artifact_version(artifact_path: Path) -> str:
    """Return an immutable SHA-256 identifier for a local model artifact."""
    with open(artifact_path, "rb") as artifact_file:
        digest = hashlib.file_digest(artifact_file, "sha256").hexdigest()
    return f"local:{artifact_path.name}@sha256:{digest}"


def load_scoring_model(settings: EnvironmentConfig | None = None) -> tuple[Any, str]:
    """Load model artifact from MLflow Model Registry via alias or local fallback.

    Returns:
        tuple[Any, str]: (fitted_pipeline, version_identifier_string).
    """
    runtime = settings or load_environment_config()
    model_name = runtime.model_name
    target_alias = runtime.model_registry_alias
    tracking_uri = runtime.mlflow_tracking_uri
    local_artifact_path = runtime.model_artifact_path

    # 1. Primary path: Attempt MLflow Model Registry alias resolution
    model_uri = f"models:/{model_name}@{target_alias}"
    should_try_mlflow = tracking_uri is not None
    if tracking_uri is not None and tracking_uri.startswith("sqlite:///"):
        sqlite_file = Path(tracking_uri.replace("sqlite:///", ""))
        if not sqlite_file.exists():
            should_try_mlflow = False

    if should_try_mlflow and tracking_uri is not None:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"[serving] Resolving model from MLflow Registry: {model_uri}...")
            model = mlflow.sklearn.load_model(model_uri)
            version_tag = f"{model_name}@{target_alias}"
            print(f"[serving] Successfully loaded model from {version_tag}")
            return model, version_tag
        except Exception as exc:
            print(f"[serving] Warning: Failed to load from MLflow Registry ({exc}).", file=sys.stderr)

    # 2. Resilient fallback path: Local serialised artifact (for air-gapped / test environments)
    if runtime.allow_local_model_fallback and local_artifact_path.exists():
        model = joblib.load(local_artifact_path)
        version_tag = local_artifact_version(local_artifact_path)
        print(f"[serving] Loaded fallback model from {local_artifact_path}")
        return model, version_tag

    raise RuntimeError(
        f"Unable to load scoring model from MLflow URI '{model_uri}'. Local fallback "
        f"is {'enabled' if runtime.allow_local_model_fallback else 'disabled'} for "
        f"environment '{runtime.environment}'."
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage model lifecycle: warm up model artifact on startup."""
    print("[serving] Initializing Streamly Churn Scoring API...")
    settings: EnvironmentConfig | None = None
    try:
        settings = load_environment_config()
        app.state.settings = settings
        model, model_version = load_scoring_model(settings)
        # Warm up feature builder and model pipeline to eliminate first-request cold-start latency
        warmup_df = pd.DataFrame(
            [
                {
                    "member_id": "warmup",
                    "tenure_days": 10,
                    "sessions_7d": 1,
                    "watch_hours_7d": 1.0,
                    "support_tickets_30d": 0,
                    "plan_tier": "basic",
                    "price_increase_flag": 0,
                }
            ]
        )
        warmup_features = build_serving_features(warmup_df)
        model.predict_proba(warmup_features)
        print("[serving] Model pipeline and schemas warmed up successfully.")

        app.state.model = model
        app.state.model_version = model_version
        app.state.is_ready = True
    except Exception as exc:
        print(f"[serving] FATAL: Could not initialize model: {exc}", file=sys.stderr)
        app.state.model = None
        app.state.model_version = "uninitialized"
        app.state.is_ready = False
        if settings is None or settings.environment == "prod":
            raise

    yield

    print("[serving] Shutting down Streamly Churn Scoring API...")


app = FastAPI(
    title="Streamly Churn Risk Scoring API",
    description="Low-latency near-real-time churn risk prediction microservice",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Any) -> Response:
    """Capture execution duration to enforce and monitor the 200ms latency budget."""
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Readiness and liveness probe for orchestrators and load balancers."""
    is_ready = getattr(app.state, "is_ready", False)
    settings: EnvironmentConfig = getattr(app.state, "settings", load_environment_config())
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        environment=settings.environment,
        model_name=settings.model_name,
        model_version=getattr(app.state, "model_version", "unknown"),
        model_loaded=is_ready,
    )


@app.post(
    "/score",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def score_member(request: ScoreRequest) -> ScoreResponse:
    """Score member churn risk probability in near-real-time (<200ms).

    Enforces data contract via shared feature builder, preventing train/serve skew
    and strictly disallowing label leakage.
    """
    if not getattr(app.state, "is_ready", False) or app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring service is not ready. Model artifact is uninitialized.",
        )

    # 1. Convert validated Pydantic model to single-row DataFrame
    payload_dict = request.model_dump()
    raw_df = pd.DataFrame([payload_dict])

    # 2. Defense-in-depth: Run Pandera serving contract validation
    try:
        validate_serving_data(raw_df)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Serving data contract validation failed: {exc}",
        ) from exc

    # 3. Build identical feature vector via shared feature builder
    features_df = build_serving_features(raw_df)

    # 4. Predict calibrated probability
    try:
        probs = app.state.model.predict_proba(features_df)
        churn_risk = float(probs[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {exc}",
        ) from exc

    return ScoreResponse(
        member_id=request.member_id,
        churn_risk=round(churn_risk, 4),
        model_version=str(app.state.model_version),
    )
