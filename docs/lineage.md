# Data & Model Lineage Specification

This document captures the end-to-end reproducible line of custody for Streamly churn risk prediction models, from raw member snapshots to production container inference.

---

## 1. End-to-End Line of Custody

The lineage architecture links version-controlled Git source code, DVC-tracked datasets, MLflow tracked experiment artifacts, and the active production container:

```
[Raw Snapshot Parquet] (data/raw/streamly_churn_sample.parquet)
       |
       |--> Tracked via DVC pointer: MD5 hash `275ac17edd883e88de83ee71589d2b92`
       v
[DVC Pipeline Stages] (dvc.yaml -> dvc.lock)
  ├── 1. prepare  --> data/processed/train.parquet + test.parquet (stratified 80/20)
  ├── 2. train    --> models/baseline_model.joblib (reproducible seed=42)
  └── 3. evaluate --> metrics.json (ROC-AUC: 0.8038, PR-AUC: 0.6657, Brier: 0.1653)
       |
       v
[MLflow Experiment Tracking] (Experiment: streamly-churn-risk)
  ├── Lineage Tag: dataset_dvc_hash = 275ac17edd883e88de83ee71589d2b92
  ├── Parameters: algorithm, max_iter, solver, split random_state
  ├── Metrics: roc_auc, pr_auc, precision_at_recall_60, brier_score, confusion_matrix
  ├── Signature: 9-column schema -> float64 probability [0, 1]
  └── Artifacts: model/, evaluation/confusion_matrix.json, features/feature_columns.json
       |
       v
[Evaluation Quality Gate] (src/streamly/tracking/promotion.py)
  ├── Loads thresholds from configs/thresholds.yaml
  ├── Validates candidate run against non-negotiable gates
  └── If passed: registers model in MLflow Model Registry
       |
       v
[MLflow Model Registry] (Catalog: streamly_churn_model)
  ├── Version 1: @champion  (StandardScaler + LogisticRegression)
  ├── Version 3: @challenger (RandomForestClassifier)
       |
       v
[FastAPI Microservice / Docker Container] (POST /score)
  ├── Resolves: models:/streamly_churn_model@champion
  ├── Transforms features via shared builder (zero skew)
  └── Answers real-time scoring requests in <200ms
```

---

## 2. DVC Pipeline Directed Acyclic Graph (DAG)

The declared execution stages in [`dvc.yaml`](../dvc.yaml) form a deterministic DAG:

```
+--------------------------------------------+ 
| data\raw\streamly_churn_sample.parquet.dvc | 
+--------------------------------------------+ 
                       *                       
                       *                       
                       *                       
                  +---------+                  
                  | prepare |                  
                  +---------+                  
                  *         **                 
                **            *                
               *               **              
         +-------+               *             
         | train |             **              
         +-------+            *                
                  *         **                 
                   **     **                   
                     *   *                     
                 +----------+                  
                 | evaluate |                  
                 +----------+                  
```

### Stage Summary

| Stage | Input Dependencies (`deps`) | Parameter Dependencies (`params`) | Generated Outputs (`outs`) |
| :--- | :--- | :--- | :--- |
| **`prepare`** | `data/raw/streamly_churn_sample.parquet`, `src/streamly/pipeline/prepare.py` | `prepare.test_size`, `prepare.random_state` | `data/processed/train.parquet`, `data/processed/test.parquet` |
| **`train`** | `data/processed/train.parquet`, `src/streamly/pipeline/train.py`, `src/streamly/models/baseline.py`, `src/streamly/features/builder.py` | `train.model_type`, `train.random_state`, and all model hyperparameters | `models/baseline_model.joblib` |
| **`evaluate`** | `data/processed/test.parquet`, `models/baseline_model.joblib`, `src/streamly/pipeline/evaluate.py`, `src/streamly/models/evaluation.py` | `evaluate.threshold`, `evaluate.target_recall` | `metrics.json` |

---

## 3. How to Verify Lineage Locally

### Fresh-clone data materialization

The configured assessment remote (`../.dvc_remote`) is deliberately local and is not available
to somebody cloning the Git repository on another machine. A reviewer can reproduce the exact
sample from its source code and fixed seed before running the DVC pipeline:

```bash
uv run python -m streamly.data.make_dataset
uv run dvc repro
uv run dvc status
```

The final command must print `Data and pipelines are up to date.` In a real shared environment,
the first command would be replaced by `dvc pull` from an access-controlled S3/GCS remote.

### 1. Inspect Pipeline Reproducibility
```bash
# Verify that all stages are up-to-date with zero pending changes
uv run dvc status
```

### 2. Inspect Dataset Hash
```bash
# Inspect DVC tracking pointer
cat data/raw/streamly_churn_sample.parquet.dvc
```

### 3. Verify MLflow Lineage Tag
```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlruns.db")
model = client.get_model_version_by_alias("streamly_churn_model", "champion")
run = client.get_run(model.run_id)

print(f"Model Version: {model.version}")
print(f"Originating Run ID: {model.run_id}")
print(f"Dataset DVC Hash: {run.data.tags.get('dataset_dvc_hash')}")
```
Output:
```text
Model Version: 1
Originating Run ID: de3365da2aa241858427c0464009da1c
Dataset DVC Hash: 275ac17edd883e88de83ee71589d2b92
```
This confirms that the exact bytes of the training sample can be cryptographically linked to the binary serving live predictions in production.
