"""Quality and Promotion Gate for Streamly MLflow Model Registry.

Enforces configurable evaluation thresholds before registering models
and moving deployment aliases (@champion / @production).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Final

import mlflow
import yaml
from mlflow.tracking import MlflowClient

DEFAULT_THRESHOLDS_PATH: Final[Path] = Path("configs/thresholds.yaml")


class PromotionGateError(Exception):
    """Raised when a candidate model fails evaluation thresholds and cannot be promoted."""


def load_promotion_config(config_path: Path = DEFAULT_THRESHOLDS_PATH) -> dict[str, Any]:
    """Load evaluation thresholds and registry configuration from YAML.

    Args:
        config_path: Path to thresholds.yaml.

    Returns:
        dict containing 'thresholds' and 'registry' blocks.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Promotion config not found at: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    return config


def evaluate_quality_gate(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Evaluate candidate model metrics against configured quality thresholds.

    Checks:
    - roc_auc >= min_roc_auc
    - pr_auc >= min_pr_auc
    - precision_at_recall_60 >= min_precision_at_recall_60
    - brier_score <= max_brier_score

    Args:
        metrics: Actual scalar metrics from evaluation.
        thresholds: Target threshold requirements.

    Returns:
        tuple[bool, dict]: (all_passed, detailed_checks_dict).
    """
    checks: dict[str, dict[str, Any]] = {}
    all_passed = True

    # 1. Check ROC-AUC
    if "min_roc_auc" in thresholds:
        actual = metrics.get("roc_auc", 0.0)
        required = thresholds["min_roc_auc"]
        passed = actual >= required
        checks["roc_auc"] = {"actual": actual, "required": f">= {required}", "passed": passed}
        if not passed:
            all_passed = False

    # 2. Check PR-AUC
    if "min_pr_auc" in thresholds:
        actual = metrics.get("pr_auc", 0.0)
        required = thresholds["min_pr_auc"]
        passed = actual >= required
        checks["pr_auc"] = {"actual": actual, "required": f">= {required}", "passed": passed}
        if not passed:
            all_passed = False

    # 3. Check Precision at Recall >= 60%
    if "min_precision_at_recall_60" in thresholds:
        actual = metrics.get("precision_at_recall_60", 0.0)
        required = thresholds["min_precision_at_recall_60"]
        passed = actual >= required
        checks["precision_at_recall_60"] = {
            "actual": actual,
            "required": f">= {required}",
            "passed": passed,
        }
        if not passed:
            all_passed = False

    # 4. Check Brier Score (Loss: lower is better)
    if "max_brier_score" in thresholds:
        actual = metrics.get("brier_score", 1.0)
        required = thresholds["max_brier_score"]
        passed = actual <= required
        checks["brier_score"] = {"actual": actual, "required": f"<= {required}", "passed": passed}
        if not passed:
            all_passed = False

    return all_passed, checks


def promote_model_to_registry(
    run_id: str,
    model_name: str,
    target_alias: str = "champion",
    tracking_uri: str | None = None,
) -> str:
    """Register trained model artifact and assign deployment alias in MLflow Registry.

    Args:
        run_id: MLflow run_id containing the trained model artifact.
        model_name: Registered model catalog name.
        target_alias: Deployment alias (e.g. 'champion' or 'production').
        tracking_uri: Tracking backend URI.

    Returns:
        str: Created model version identifier.
    """
    uri: str = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlruns.db"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    # 1. Register the model artifact from the run
    model_uri = f"runs:/{run_id}/model"
    print(f"[registry] Registering model from {model_uri} under name '{model_name}'...")
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    version_str = str(model_version.version)

    # 2. Assign target deployment alias (e.g. @champion)
    clean_alias = target_alias.lstrip("@")
    print(f"[registry] Assigning alias '@{clean_alias}' to {model_name} version {version_str}...")
    client.set_registered_model_alias(
        name=model_name,
        alias=clean_alias,
        version=version_str,
    )

    # 3. Tag the originating run with promotion audit metadata
    client.set_tag(run_id, "promotion.status", "PROMOTED")
    client.set_tag(run_id, "promotion.model_name", model_name)
    client.set_tag(run_id, "promotion.model_version", version_str)
    client.set_tag(run_id, "promotion.alias", f"@{clean_alias}")

    return version_str


def gate_and_promote(
    run_id: str,
    config_path: Path = DEFAULT_THRESHOLDS_PATH,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Execute evaluation quality gate on a run and promote to registry if thresholds pass.

    Args:
        run_id: MLflow run ID to evaluate.
        config_path: Path to thresholds.yaml.
        tracking_uri: MLflow tracking URI.

    Returns:
        dict[str, Any] with gate results and promotion status.

    Raises:
        PromotionGateError: If any threshold is violated.
    """
    uri: str = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlruns.db"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    # 1. Fetch run and metrics
    run = client.get_run(run_id)
    actual_metrics = run.data.metrics

    # 2. Load thresholds configuration
    config = load_promotion_config(config_path)
    thresholds = config.get("thresholds", {})
    registry_cfg = config.get("registry", {})

    model_name = os.getenv("MODEL_NAME") or registry_cfg.get("model_name", "streamly_churn_model")
    target_alias = os.getenv("MODEL_REGISTRY_ALIAS") or registry_cfg.get("target_alias", "champion")

    # 3. Evaluate Quality Gate
    all_passed, checks = evaluate_quality_gate(actual_metrics, thresholds)

    print("\n" + "=" * 65)
    print(f"        STREAMLY MODEL PROMOTION QUALITY GATE (Run: {run_id[:8]})")
    print("=" * 65)
    for metric_name, detail in checks.items():
        status = "PASS [OK]" if detail["passed"] else "FAIL [X]"
        print(f"  {metric_name:<25} Actual: {detail['actual']:<7} Req: {detail['required']:<8} -> {status}")
    print("-" * 65)

    if not all_passed:
        failed_metrics = [k for k, v in checks.items() if not v["passed"]]
        client.set_tag(run_id, "promotion.status", "REJECTED")
        client.set_tag(run_id, "promotion.rejection_reason", f"Failed: {failed_metrics}")
        raise PromotionGateError(
            f"Quality gate REJECTED promotion for run {run_id}. Failed metrics: {failed_metrics}"
        )

    # 4. Gate passed -> Promote to Registry and assign alias
    print("Quality gate PASSED all required thresholds!")
    version = promote_model_to_registry(
        run_id=run_id,
        model_name=model_name,
        target_alias=target_alias,
        tracking_uri=uri,
    )
    print(f"SUCCESS: Promoted {model_name} v{version} to @{target_alias.lstrip('@')}")
    print("=" * 65 + "\n")

    return {
        "status": "PROMOTED",
        "run_id": run_id,
        "model_name": model_name,
        "version": version,
        "alias": f"@{target_alias.lstrip('@')}",
        "checks": checks,
    }


def main() -> None:
    """CLI entrypoint for running model promotion quality gate."""
    parser = argparse.ArgumentParser(description="Streamly Model Promotion Gate")
    parser.add_argument("--run-id", type=str, required=True, help="MLflow run ID to evaluate")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_THRESHOLDS_PATH,
        help="Path to thresholds.yaml",
    )

    args = parser.parse_args()

    try:
        gate_and_promote(run_id=args.run_id, config_path=args.config_path)
        sys.exit(0)
    except PromotionGateError as exc:
        print(f"\nPROMOTION BLOCKED: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
