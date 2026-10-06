# Streamly Churn-Risk Scoring Platform

A production-shaped, audit-ready machine learning operations (MLOps) platform built for Streamly subscription streaming services. 

Delivers near-real-time churn risk predictions within a 200 ms service target over HTTP using model artifacts versioned and governed through Pandera data contracts, DVC pipelines, MLflow experiment tracking, automated quality gates, and hardened multi-stage Docker containers.

---

## 1. Quickstart: From Zero to `curl`

### Prerequisites
- Python 3.11+
- `uv` package manager (`pip install uv` or `winget install astral-sh.uv`)
- Git

### Step-by-Step Walkthrough

```bash
# 1. Clone repository
git clone https://github.com/sathyaphani150/streamly-churn-risk.git
cd streamly-churn-risk

# 2. Set up deterministic virtual environment from uv.lock
uv sync --extra dev

# 3. Configure local environment variables
cp .env.example .env

# 4. Materialize the deterministic sample data (required on a fresh clone)
uv run python -m streamly.data.make_dataset

# 5. Verify code quality gates (Linting & Strict Type Checking)
uv run ruff check .
uv run mypy src tests

# 6. Reproduce and verify the complete DVC pipeline
uv run dvc repro
uv run dvc status

# 7. Run the data contract directly and execute all 59 tests
uv run python -m streamly.data.validation
uv run pytest

# 8. Track a candidate and assign the CI-safe challenger alias after it passes
uv run python -m streamly.tracking.experiment --run-name production_candidate
uv run python -m streamly.tracking.promotion --run-id <RUN_ID> --alias challenger

# 9. Manual production approval only: move an approved run to champion
uv run python -m streamly.tracking.promotion --run-id <APPROVED_RUN_ID> --alias champion

# 10. Start the real-time scoring microservice
uv run uvicorn streamly.serving.app:app --port 8000
```

### Test Live Scoring with `curl`
In another terminal, send a member snapshot:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "member_id": "mem_0042",
    "tenure_days": 180,
    "sessions_7d": 5,
    "watch_hours_7d": 12.5,
    "support_tickets_30d": 1,
    "plan_tier": "standard",
    "price_increase_flag": 0
  }'
```

**Response (HTTP 200 OK — Latency: ~15ms)**:
```json
{
  "member_id": "mem_0042",
  "churn_risk": 0.4336,
  "model_version": "streamly_churn_model@champion"
}
```

When the registry is unavailable, the self-contained container reports an immutable fallback ID
such as `local:baseline_model.joblib@sha256:<64-hex-digest>` instead of an alias.

---

## 2. Sample Data and DVC Storage

The assessment sample is synthetic, so it contains no customer information. Running
`python -m streamly.data.make_dataset` always creates 10,000 member snapshots using random seed
`42`. Tenure follows a bounded exponential distribution; sessions use a Poisson distribution;
watch time is correlated with sessions; and support tickets, plan tier, and price-increase status
use fixed categorical probabilities.

The `churned_30d` label is sampled from a documented logistic risk function using only fields
available at snapshot/request time, plus random noise. It never uses cancellation dates, reasons,
or any future outcome as an input feature. The resulting Parquet file is represented in Git by
`data/raw/streamly_churn_sample.parquet.dvc` with MD5
`275ac17edd883e88de83ee71589d2b92`.

The assessment uses `.dvc_remote/` as a local DVC remote. That directory is intentionally
gitignored, so a fresh clone uses the deterministic generator before `dvc repro`. In production,
the local remote would be replaced with shared object storage, for example:

```bash
# Install and lock the relevant DVC remote extra in the project first (for example dvc-s3).
dvc remote add -d production s3://streamly-dvc/churn-risk
dvc push
```

CI or runtime credentials would be supplied through environment variables or a secret manager;
access keys must not be written to `.dvc/config` or committed to Git.

---

## 3. Reviewer Demo Pointers

| Requirement | Command / Pointer | Evidence / Location |
| :--- | :--- | :--- |
| **Reproducible Pipeline** | `uv run dvc repro` | [`dvc.yaml`](dvc.yaml), [`dvc.lock`](dvc.lock) |
| **MLflow Experiment UI** | `uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000` | Open `http://127.0.0.1:5000` |
| **Champion Model ID** | Version 1 (Logistic Regression) | `streamly_churn_model@champion` (ROC-AUC: 0.8038, PR-AUC: 0.6657, Brier: 0.1653) |
| **Challenger Model ID** | Version 3 (Random Forest) | `streamly_churn_model@challenger` (ROC-AUC: 0.7791, Prec@R60: 0.6015) |
| **Promotion Thresholds**| [`configs/thresholds.yaml`](configs/thresholds.yaml) | `min_roc_auc: 0.75`, `min_pr_auc: 0.60`, `min_prec@r60: 0.55`, `max_brier: 0.20` |
| **Live Serving Service**| `http://127.0.0.1:8000/docs` | Interactive Swagger UI + `/health` and `/score` |
| **Containerization** | `docker build -t streamly-churn:latest .` | Multi-stage, non-root user (`appuser:10001`) |

---

## 4. Architecture & System Flow

```
[Raw Member Data] (data/raw/streamly_churn_sample.parquet)
       │
       ▼ (Contract validation: Pandera)
[Validated Parquet Dataset] 
       │
       ▼ (DVC Stage 1: prepare)
[Stratified Train/Test Splits] (80/20 train.parquet & test.parquet)
       │
       ▼ (DVC Stage 2: train & Shared Feature Builder)
[Trained Model Pipeline] (models/baseline_model.joblib)
       │
       ▼ (DVC Stage 3 & MLflow Experiment Tracking)
[Metrics, Lineage Hash, Signatures & Artifacts] 
       │
       ▼ (Promotion Quality Gate: configs/thresholds.yaml)
[MLflow Model Registry] ──(@champion / @challenger)──┐
                                                     │
                                                     ▼
                                      [FastAPI Microservice] 
                                      (POST /score <200ms)
                                                     │
                                                     ▼
                                      [Production Telemetry & Drift Monitoring]
```

---

## 5. Key Engineering Guarantees

### Zero Train/Serve Skew (Shared Feature Builder)
Feature transformations are implemented in a single unified module: [`src/streamly/features/builder.py`](src/streamly/features/builder.py).
Both offline training (`build_training_features`) and online inference (`build_serving_features`) share identical logic for:
- One-hot encoding of controlled subscription tiers (`plan_tier ∈ {"basic", "standard", "premium"}`).
- Laplace-smoothed engagement ratios: `watch_hours_per_session = (watch_hours_7d + 1.0) / (sessions_7d + 1.0)`.
- Removal of identity fields (`member_id`).

### Anti-Leakage Data Contracts (Pandera & Pydantic)
The system enforces strict schema separation:
- **`TRAINING_SCHEMA`**: Requires `churned_30d` binary target.
- **`SERVING_SCHEMA`**: Strict schema explicitly rejecting any future labels or cancellation indicators (e.g. `churned_30d`, `cancel_reason`, `churn_date`) with HTTP 422 errors.

### Cryptographic Lineage Tracking
The exact MD5 digest of the raw Parquet file tracked by DVC (`275ac17edd883e88de83ee71589d2b92`) is captured and tagged to every MLflow run (`dataset_dvc_hash`), ensuring complete traceability from production predictions back to raw data.

---

## 6. Production Documentation Suite

Detailed architecture specifications and operational guides are documented in [`docs/`](docs/):
- [**`docs/design.md`**](docs/design.md): System architecture, end-to-end lifecycle, ownership matrix (RACI), environment differences, and Definition-of-Done checklist.
- [**`docs/lineage.md`**](docs/lineage.md): Data and model lineage line-of-custody specification and DVC Directed Acyclic Graph (DAG).
- [**`docs/containers.md`**](docs/containers.md): Multi-stage container architecture, non-root security policy, Trivy scanning, and Syft Software Bill of Materials (SBOM) guide.
- [**`docs/branch-protection.md`**](docs/branch-protection.md): Trunk-based development policies, PR requirements, conventional commits, and registry alias governance.
