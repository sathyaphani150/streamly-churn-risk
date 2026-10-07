"""Tests for the shared local and CI verification runner."""

import subprocess
from pathlib import Path

import pytest

from scripts.verify import VerificationStep, build_verification_steps, run_verification


def test_verification_contract_contains_all_required_gates() -> None:
    """The shared runner must cover code, data, reproducibility, and tests."""
    commands = [" ".join(step.command) for step in build_verification_steps("python")]
    combined = "\n".join(commands)

    assert "ruff check ." in combined
    assert "mypy src tests scripts" in combined
    assert "streamly.data.validation" in combined
    assert "dvc repro" in combined
    assert "dvc status" in combined
    assert "pytest" in combined
    assert "coverage.xml" in combined
    assert "junit-report.xml" in combined


def test_verification_stops_at_first_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing gate must stop later commands and return a non-zero result."""
    executed: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], *, cwd: Path, check: bool) -> None:
        assert cwd.is_dir()
        assert check is True
        executed.append(command)
        if command[-1] == "fail":
            raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(subprocess, "run", fake_run)
    steps = (
        VerificationStep("pass", ("verify", "pass")),
        VerificationStep("fail", ("verify", "fail")),
        VerificationStep("must not run", ("verify", "later")),
    )

    with pytest.raises(subprocess.CalledProcessError):
        run_verification(steps)

    assert executed == [("verify", "pass"), ("verify", "fail")]
