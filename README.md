# Streamly Churn-Risk Scoring Platform

A production-shaped, audit-ready machine learning operations (MLOps) platform built for Streamly subscription streaming services. 

Delivers near-real-time churn risk predictions within a 200 ms service target over HTTP using model artifacts versioned and governed through Pandera data contracts, DVC pipelines, MLflow experiment tracking, automated quality gates, and hardened multi-stage Docker containers.

---

## 1. Quickstart: From Zero to `curl`

### Prerequisites
- Python 3.11+
- `uv` package manager (`pip install uv` or `winget install astral-sh.uv`)
- Git
- Docker Desktop or Docker Engine with Compose v2

### Fastest reviewer path: Docker Compose

The container includes a deterministic fallback model, so no `.env`, local Python installation,
MLflow server, or DVC remote is required for this demonstration:

```bash
docker compose up --build --wait
```

If port `8000` is already occupied, choose another host port (PowerShell example):

```powershell
$env:STREAMLY_API_PORT = "8081"
docker compose up --build --wait
```

In a second terminal, run the portable health-and-scoring smoke test:

```bash
docker compose exec api python /app/scripts/smoke_test.py
```

Stop and remove the local service with `docker compose down`.

Every pull request also builds this image, generates an SPDX-JSON SBOM with Syft, and runs a
blocking Trivy scan for fixable HIGH and CRITICAL vulnerabilities. GitHub Actions retains the SBOM,
scan JSON, image ID, and build log as the `container-security-evidence` artifact.

### Run CI checks before pushing

Do not execute `.github/workflows/ci.yml` directly. GitHub Actions supplies the remote runner, but
local development and CI both call the same version-controlled verification program:

```bash
uv run python scripts/verify.py
```

It stops on the first failure and checks Ruff, strict Mypy, deterministic data generation, the
training-data contract, DVC reproduction and status, and the complete test suite. A successful run
also creates `coverage.xml` and `junit-report.xml`, which GitHub Actions archives as evidence.

### Step-by-Step Walkthrough

```bash
# 1. Clone repository
git clone https://github.com/sathyaphani150/streamly-churn-risk.git
cd streamly-churn-risk

# 2. Set up deterministic virtual environment from uv.lock
uv sync --extra dev

# 3. Run the complete local/CI verification contract
uv run python scripts/verify.py

# 4. Track a candidate and assign the CI-safe challenger alias after it passes
uv run python -m streamly.tracking.experiment --run-name production_candidate
uv run python -m streamly.tracking.promotion --run-id <RUN_ID> --alias challenger

# 5. Manual production approval only: move an approved run to champion
uv run python -m streamly.tracking.promotion --run-id <APPROVED_RUN_ID> --alias champion --approved-by <REVIEWER_ID>

# 6. Start the real-time scoring microservice
uv run uvicorn streamly.serving.app:app --port 8000
```

The default profile is [`configs/environments/dev.yaml`](configs/environments/dev.yaml), so a
`.env` file is not required. Select another profile with `STREAMLY_ENV=ci` or
`STREAMLY_ENV=prod`. Environment variables listed in [`.env.example`](.env.example) are optional
deployment-time overrides. Production deliberately fails to start unless
`MLFLOW_TRACKING_URI` is supplied, and it never falls back to the bundled local model.

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
| **Complete Pre-Push Gate** | `uv run python scripts/verify.py` | Same seven quality gates used by CI |
| **MLflow Experiment UI** | `uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000` | Open `http://127.0.0.1:5000` |
| **Champion Model ID** | Version 1 / Run `de3365da2aa241858427c0464009da1c` | `streamly_churn_model@champion` (ROC-AUC: 0.8038, PR-AUC: 0.6657, Brier: 0.1653) |
| **Challenger Model ID** | Version 3 (Random Forest) | `streamly_churn_model@challenger` (ROC-AUC: 0.7791, Prec@R60: 0.6015) |
| **Promotion Thresholds**| [`configs/thresholds.yaml`](configs/thresholds.yaml) | `min_roc_auc: 0.75`, `min_pr_auc: 0.60`, `min_prec@r60: 0.55`, `max_brier: 0.20` |
| **Live Serving Service**| `http://127.0.0.1:8000/docs` | Interactive Swagger UI + `/health` and `/score` |
| **Containerization** | `docker compose up --build --wait` | Multi-stage, non-root UID `10001`, health check, Trivy scan, and Syft SBOM |

The thresholds are deliberately above weak/random behavior without being tuned to the held-out
sample: ROC-AUC `0.75` requires useful ranking, PR-AUC `0.60` requires substantial lift over the
roughly 33% churn prevalence, precision `0.55` at recall `0.60` limits wasted retention outreach,
and Brier score `0.20` requires reasonably calibrated probabilities. The champion clears all four
with `0.8038`, `0.6657`, `0.6082`, and `0.1653`, respectively.

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
