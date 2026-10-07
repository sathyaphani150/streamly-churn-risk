"""CI policy tests for executable container security evidence."""

from pathlib import Path


def test_ci_executes_pinned_trivy_blocking_scan() -> None:
    """CI must fail on fixable HIGH or CRITICAL image vulnerabilities."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25" in workflow
    assert "version: v0.75.0" in workflow
    assert "severity: HIGH,CRITICAL" in workflow
    assert "ignore-unfixed: true" in workflow
    assert 'exit-code: "1"' in workflow
    assert "reports/security/trivy-report.json" in workflow


def test_ci_generates_and_archives_pinned_sbom() -> None:
    """CI must create an SPDX SBOM and retain all container-security evidence."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "anchore/sbom-action@66cbf4bc1f1c0d2edc94016e65bc221b6bb0ad6c" in workflow
    assert "syft-version: v1.54.1" in workflow
    assert "format: spdx-json" in workflow
    assert "reports/security/sbom.spdx.json" in workflow
    assert "container-security-evidence" in workflow
    assert "reports/security/image-id.txt" in workflow
