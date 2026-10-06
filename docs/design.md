# Streamly Churn Risk Scoring System — Production Architecture

This document describes the production architecture, ownership model, repository layout, and Definition-of-Done (DoD) for Streamly's real-time churn risk prediction platform.

---

## 1. System Lifecycle

```
[Raw Member Data] 
       │
       ▼ (Contract validation: Pandera)
[Validated Parquet Dataset] 
       │
       ▼ (DVC Stage 1: prepare)
[Stratified Train/Test Splits] 
       │
       ▼ (DVC Stage 2: train & Shared Feature Builder)
[Fitted Candidate Model Pipeline] 
       │
       ▼ (DVC Stage 3 & MLflow Experiment Tracking)
[Evaluation Metrics & Artifacts] 
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

### End-to-End Operational Lifecycle:
1. **Raw Extract & Ingestion**: Data Engineering generates daily snapshots of member engagement. The raw dataset is tracked via DVC (`data/raw/streamly_churn_sample.parquet.dvc`), keeping large binary files out of Git.
2. **Schema Contract Gate**: Pandera schemas ([`validation.py`](../src/streamly/data/validation.py)) validate types, ranges, non-nullability, and controlled vocabularies (`basic`, `standard`, `premium`). Any future label leakage (e.g. `churned_30d` or cancellation reasons in serving data) is blocked with non-zero exit codes.
3. **Reproducible Pipeline**: DVC orchestrates `prepare` (80/20 stratified split) $\rightarrow$ `train` $\rightarrow$ `evaluate`.
4. **Experiment Tracking**: MLflow tracks all candidate hyperparameters, evaluation metrics (ROC-AUC, PR-AUC, Precision@Recall $\ge$ 60%, Brier Score), confusion matrices, and the cryptographic DVC dataset hash lineage tag.
5. **Quality Gate & Promotion**: The candidate model is evaluated against [`configs/thresholds.yaml`](../configs/thresholds.yaml). Passing automation may assign `@challenger`; moving an approved run to the production `@champion` alias requires manual owner sign-off.
6. **Low-Latency Serving**: A containerized FastAPI service loads `models:/streamly_churn_model@champion`, transforms raw request JSON via the shared [`build_serving_features`](../src/streamly/features/builder.py) function, and enforces the 200 ms scoring target.
7. **Lightweight Monitoring**: The service logs request latency via the `X-Process-Time-Ms` header, while `/health` provides readiness/liveness status for orchestrators.

---

## 2. Ownership & RACI Matrix

| Functional Stage | Responsible (R) | Accountable (A) | Consulted (C) | Informed (I) |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Member Ingestion** | Data Engineering | Data Engineering Lead | ML Engineers | Product Owner |
| **Data Contracts & Schemas** | ML Engineers | Senior ML Engineer | Data Engineering | Security / Legal |
| **Model Development & Pipeline** | ML Engineers | Senior ML Engineer | Retention Analytics | Product Owner |
| **Experiment Evaluation & Review** | ML Engineers | Lead ML Engineer | Retention Lead | Growth Team |
| **Model Registry Promotion** | Lead ML Engineer | Head of Machine Learning | Retention Lead | DevOps / SRE |
| **Scoring Microservice & Docker** | MLOps / Backend | Platform Lead | ML Engineers | DevOps / SRE |
| **Production Runtime & Monitoring**| SRE / Platform | SRE Manager | ML Engineers | Customer Support |

---

## 3. Repository Architecture & File Taxonomy

```
mlops_poc/
├── .github/workflows/ci.yml       # Trunk-based CI: lint, typecheck, DQ gate, test suite, Docker
├── configs/
│   └── thresholds.yaml            # Non-negotiable evaluation thresholds and registry config
├── data/
│   ├── raw/                       # Raw snapshot Parquet (Gitignored, tracked by DVC)
│   └── processed/                 # Train/test Parquet splits (Gitignored, DVC stage outputs)
├── docs/
│   ├── design.md                  # System architecture, lifecycle, ownership, DoD (this file)
│   ├── lineage.md                 # Data and model lineage specification + DVC DAG
│   ├── containers.md              # Containerization, security hardening, and SBOM strategy
│   └── branch-protection.md       # Trunk-based Git and branch protection policies
├── models/                        # Serialized fallback model artifacts (Gitignored)
├── src/streamly/
│   ├── data/                      # Synthetic data generator & Pandera schema validation
│   ├── features/                  # Shared feature builder preventing train/serve skew
│   ├── models/                    # Model factory (LR, RF, HGB) and evaluation metrics
│   ├── pipeline/                  # DVC pipeline stages (prepare, train, evaluate)
│   ├── tracking/                  # MLflow experiment tracking & promotion quality gate
│   └── serving/                   # FastAPI real-time scoring microservice (POST /score)
├── tests/                         # Unit and integration tests for all critical gates
├── .dockerignore                  # Docker build cache and file exclusions
├── .env.example                   # Environment configuration template (zero secrets in Git)
├── Dockerfile                     # Multi-stage, non-root hardened production image
├── dvc.yaml & dvc.lock            # Reproducible DVC pipeline definition and lockfile
├── params.yaml                    # Hyperparameters and split configuration
├── pyproject.toml                 # Dependencies, tool configuration, PEP 621 metadata
└── uv.lock                        # Deterministic cross-platform dependency lockfile
```

### Git vs DVC Separation:
- **Tracked in Git**: Source code (`src/`), tests (`tests/`), configuration (`configs/`, `params.yaml`), DVC pointer files (`.dvc`), and pipeline manifests (`dvc.yaml`, `dvc.lock`).
- **Tracked in DVC**: Large binary files (`data/raw/*.parquet`, `data/processed/*.parquet`, `models/*.joblib`), protected by remote cache.

---

## 4. Environment Differentiation

| Aspect | Development (`dev`) | Continuous Integration (`ci`) | Production (`prod`) |
| :--- | :--- | :--- | :--- |
| **Tracking URI** | Local SQLite (`sqlite:///mlruns.db`) | Ephemeral SQLite / Hosted MLflow | Managed MLflow (PostgreSQL + S3) |
| **DVC Remote** | Local directory (`.dvc_remote/`) | Read-only object storage cache | Secure Cloud Bucket (`s3://streamly-dvc/`) |
| **Model Registry Alias**| `@challenger` or `@champion` | Gated candidate alias | **`@champion`** (strictly protected) |
| **Secrets / Credentials**| Local `.env` (gitignored) | GitHub Actions Encrypted Secrets | AWS Secrets Manager / Vault |
| **Error Handling** | Detailed tracebacks enabled | Verbose test reporting | Sanitized responses, structured JSON logs |

---

## 5. Definition-of-Done (DoD) Checklist for Production

Before any model or code modification is released to production, all items must be satisfied:

- [x] **Clean Repository Hygiene**: Strict typechecking (`mypy --strict`), linting (`ruff`), zero secrets in Git.
- [x] **Reproducible Environment**: Dependencies pinned in `uv.lock`, installs without floating versions.
- [x] **Anti-Leakage Data Contracts**: Schemas enforce non-nullability, ranges, vocabulary, and reject labels at serve time.
- [x] **Train/Serve Parity**: Exactly one shared feature builder produces identical matrices for train and serve.
- [x] **Data & Code Version Separation**: Datasets versioned via DVC; pipeline stages reproduce cleanly via `dvc repro`.
- [x] **Lineage & Artifact Audit**: MLflow logs parameters, scalar metrics, confusion matrix, signature, and DVC hash.
- [x] **Passed Promotion Gate**: Candidate model meets all metric thresholds in `configs/thresholds.yaml`.
- [x] **Protected Registry Aliasing**: Assigned `@champion` (active) and `@challenger` (shadow).
- [x] **Sub-200ms Scoring SLA**: Microservice tests enforce the `POST /score` latency budget and expose request timing.
- [x] **Hardened Container**: Built via multi-stage Dockerfile, executes under unprivileged non-root user `appuser`.
- [x] **Automated CI & Branch Protection**: Green CI running lint, types, contracts, test suite, and Docker build.
