"""Quality and Promotion Gate for Streamly MLflow Model Registry.

Enforces configurable evaluation thresholds before registering models
and moving deployment aliases (@champion / @production).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import mlflow
from mlflow.tracking import MlflowClient

from streamly.config import load_environment_config, load_promotion_settings

DEFAULT_THRESHOLDS_PATH: Final[Path] = Path("configs/thresholds.yaml")
PROTECTED_ALIASES: Final[frozenset[str]] = frozenset({"champion", "production"})


class PromotionGateError(Exception):
    """Raised when a candidate model fails evaluation thresholds and cannot be promoted."""


def validate_promotion_authorization(
    target_alias: str,
    environment: str,
    approved_by: str | None,
) -> tuple[str, str]:
    """Validate alias governance and return normalized alias and audit actor."""
    clean_alias = target_alias.lstrip("@").lower()
    clean_approver = approved_by.strip() if approved_by else ""
    clean_environment = environment.strip().lower()

    if not clean_alias:
        raise PromotionGateError("A non-empty model registry alias is required.")
    if clean_environment not in {"dev", "ci", "prod"}:
        raise PromotionGateError(f"Unsupported promotion environment '{environment}'.")

    if clean_alias in PROTECTED_ALIASES:
        if clean_environment == "ci":
            raise PromotionGateError(
                f"CI is not authorized to assign protected alias '@{clean_alias}'."
            )
        if not clean_approver:
            raise PromotionGateError(
                f"Protected alias '@{clean_alias}' requires --approved-by with the reviewer identity."
            )

    actor = clean_approver or f"{clean_environment}-automation"
    return clean_alias, actor


def load_promotion_config(config_path: Path = DEFAULT_THRESHOLDS_PATH) -> dict[str, Any]:
    """Load evaluation thresholds and registry configuration from YAML.

    Args:
        config_path: Path to thresholds.yaml.

    Returns:
        dict containing 'thresholds' and 'registry' blocks.
    """
    return load_promotion_settings(config_path).model_dump()


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
    target_alias: str = "challenger",
    tracking_uri: str | None = None,
    environment: str = "dev",
    approved_by: str | None = None,
) -> str:
    """Register trained model artifact and assign deployment alias in MLflow Registry.

    Args:
        run_id: MLflow run_id containing the trained model artifact.
        model_name: Registered model catalog name.
        target_alias: Deployment alias (e.g. 'champion' or 'production').
        tracking_uri: Tracking backend URI.
        environment: Runtime environment requesting the promotion.
        approved_by: Human reviewer identity for protected aliases.

    Returns:
        str: Created model version identifier.
    """
    normalized_environment = environment.strip().lower()
    clean_alias, audit_actor = validate_promotion_authorization(
        target_alias=target_alias,
        environment=normalized_environment,
        approved_by=approved_by,
    )
    uri: str = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlruns.db"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    # 1. Reuse an existing version for this run, or register it exactly once.
    existing_versions = client.search_model_versions(f"name='{model_name}'")
    matching_versions = [version for version in existing_versions if version.run_id == run_id]
    if matching_versions:
        model_version = max(matching_versions, key=lambda version: int(version.version))
        version_str = str(model_version.version)
        print(
            f"[registry] Reusing {model_name} version {version_str} already linked to run {run_id}."
        )
    else:
        model_uri = f"runs:/{run_id}/model"
        print(f"[registry] Registering model from {model_uri} under name '{model_name}'...")
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        version_str = str(model_version.version)

    # 2. Assign target deployment alias (e.g. @champion)
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
    client.set_tag(run_id, "promotion.environment", normalized_environment)
    client.set_tag(run_id, "promotion.approved_by", audit_actor)
    client.set_tag(run_id, "promotion.timestamp_utc", datetime.now(UTC).isoformat())
    client.set_model_version_tag(model_name, version_str, "promotion.environment", normalized_environment)
    client.set_model_version_tag(model_name, version_str, "promotion.approved_by", audit_actor)
    client.set_model_version_tag(
        model_name,
        version_str,
        "promotion.timestamp_utc",
        datetime.now(UTC).isoformat(),
    )

    return version_str


def gate_and_promote(
    run_id: str,
    config_path: Path = DEFAULT_THRESHOLDS_PATH,
    target_alias: str | None = None,
    tracking_uri: str | None = None,
    approved_by: str | None = None,
) -> dict[str, Any]:
    """Execute evaluation quality gate on a run and promote to registry if thresholds pass.

    Args:
        run_id: MLflow run ID to evaluate.
        config_path: Path to thresholds.yaml.
        target_alias: Optional alias to assign (defaults to config or env, e.g. 'champion' or 'challenger').
        tracking_uri: MLflow tracking URI.
        approved_by: Human reviewer identity required for protected aliases.

    Returns:
        dict[str, Any] with gate results and promotion status.

    Raises:
        PromotionGateError: If any threshold is violated.
    """
    runtime = load_environment_config()
    config = load_promotion_config(config_path)
    thresholds = config.get("thresholds", {})
    registry_cfg = config.get("registry", {})

    model_name = os.getenv("MODEL_NAME") or registry_cfg.get("model_name", "streamly_churn_model")
    effective_alias = target_alias or os.getenv("MODEL_REGISTRY_ALIAS") or registry_cfg.get(
        "target_alias", "challenger"
    )
    clean_alias, audit_actor = validate_promotion_authorization(
        target_alias=effective_alias,
        environment=runtime.environment,
        approved_by=approved_by,
    )

    uri = tracking_uri or runtime.mlflow_tracking_uri
    if uri is None:
        raise PromotionGateError("MLflow tracking URI is required for model promotion.")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    # 1. Fetch run and metrics
    run = client.get_run(run_id)
    actual_metrics = run.data.metrics

    # 2. Evaluate Quality Gate
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

    # 3. Gate passed -> Promote to Registry and assign alias
    print("Quality gate PASSED all required thresholds!")
    version = promote_model_to_registry(
        run_id=run_id,
        model_name=model_name,
        target_alias=clean_alias,
        tracking_uri=uri,
        environment=runtime.environment,
        approved_by=audit_actor,
    )
    print(f"SUCCESS: Promoted {model_name} v{version} to @{effective_alias.lstrip('@')}")
    print("=" * 65 + "\n")

    return {
        "status": "PROMOTED",
        "run_id": run_id,
        "model_name": model_name,
        "version": version,
        "alias": f"@{effective_alias.lstrip('@')}",
        "approved_by": audit_actor,
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
    parser.add_argument(
        "--alias",
        type=str,
        default=None,
        help="Registry alias to assign on promotion (e.g. champion, challenger)",
    )
    parser.add_argument(
        "--approved-by",
        type=str,
        default=None,
        help="Reviewer identity; required when assigning champion or production",
    )

    args = parser.parse_args()

    try:
        gate_and_promote(
            run_id=args.run_id,
            config_path=args.config_path,
            target_alias=args.alias,
            approved_by=args.approved_by,
        )
        sys.exit(0)
    except PromotionGateError as exc:
        print(f"\nPROMOTION BLOCKED: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
