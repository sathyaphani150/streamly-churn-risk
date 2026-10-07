"""Run the same deterministic quality gates locally and in GitHub Actions."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VerificationStep:
    """One named, shell-free verification command."""

    name: str
    command: tuple[str, ...]


def build_verification_steps(
    python_executable: str = sys.executable,
) -> tuple[VerificationStep, ...]:
    """Return the ordered local/CI verification contract."""
    python_module = (python_executable, "-m")
    return (
        VerificationStep("Ruff linting", (*python_module, "ruff", "check", ".")),
        VerificationStep("Strict Mypy analysis", (*python_module, "mypy", "src", "tests", "scripts")),
        VerificationStep(
            "Deterministic sample materialization",
            (*python_module, "streamly.data.make_dataset"),
        ),
        VerificationStep(
            "Training data contract",
            (
                *python_module,
                "streamly.data.validation",
                "--data-path",
                "data/raw/streamly_churn_sample.parquet",
                "--mode",
                "train",
            ),
        ),
        VerificationStep("DVC pipeline reproduction", (*python_module, "dvc", "repro")),
        VerificationStep("DVC clean-status assertion", (*python_module, "dvc", "status")),
        VerificationStep(
            "Unit and integration tests",
            (
                *python_module,
                "pytest",
                "--cov-report=xml:coverage.xml",
                "--cov-report=term-missing",
                "--junitxml=junit-report.xml",
            ),
        ),
    )


def run_verification(steps: Iterable[VerificationStep]) -> None:
    """Run every verification step and stop immediately on the first failure."""
    for position, step in enumerate(steps, start=1):
        print(f"\n[verify {position}] {step.name}", flush=True)
        subprocess.run(step.command, cwd=REPOSITORY_ROOT, check=True)


def main() -> None:
    """CLI entry point used identically by developers and CI."""
    steps = build_verification_steps()
    try:
        run_verification(steps)
    except subprocess.CalledProcessError as exc:
        print(f"\n[verify] FAILED with exit code {exc.returncode}.", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc

    print(f"\n[verify] PASSED all {len(steps)} quality gates.", flush=True)


if __name__ == "__main__":
    main()
