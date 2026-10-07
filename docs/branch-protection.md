# Trunk-Based Development & Branch Protection Policy

This document declares the production governance policy for the Streamly codebase, protecting the `main` branch from regressions, schema drift, uncalibrated models, and unreviewed changes.

---

## 1. Branching Strategy: Trunk-Based Development

Streamly follows **Trunk-Based Development** with short-lived feature branches:

1. **Short-Lived Branches**: Developers branch directly from `main` (branch lifespan $\le 2$ days).
   - Naming convention: `feat/<name>`, `fix/<issue>`, `chore/<task>`.
2. **Conventional Commits**: Every commit adheres to the [Conventional Commits](https://www.conventionalcommits.org/) specification:
   - `feat(...)`: New model algorithm, pipeline stage, or API endpoint.
   - `fix(...)`: Bug fix or schema edge case correction.
   - `test(...)`: Adding or updating test fixtures and suites.
   - `ci(...)`: GitHub Actions workflow and pipeline updates.
   - `docs(...)`: Architectural design notes and guides.
3. **Pull Requests into `main`**: All modifications enter `main` exclusively through Pull Requests. Direct pushes to `main` are strictly blocked.

---

## 2. GitHub Branch Protection Policy for `main`

The following settings are the expected protection policy for the `main` branch (and should be
enforced in GitHub wherever repository permissions allow):

| Policy Setting | Value | Rationale |
| :--- | :--- | :--- |
| **Require pull request before merging** | **Enabled** | Enforces code review and peer accountability. |
| **Required approving reviews** | **At least 1 Senior Engineer** | Prevents lone-wolf changes to serving contracts or evaluation thresholds. |
| **Dismiss stale pull request approvals when new commits are pushed** | **Enabled** | Ensures any late edits are re-reviewed. |
| **Require status checks to pass before merging** | **Enabled** | Code must pass automated quality gates. |
| **Require branches to be up to date before merging** | **Enabled** | Guarantees test suite ran against the exact merged tip of `main`. |
| **Do not allow bypassing the above settings** | **Enabled** | Applies universally, including repository administrators. |
| **Restrict who can push to matching branches** | **No direct push allowed** | Forces all changes through the audited PR pipeline. |

---

## 3. Mandatory CI Status Checks (Quality Gates)

Before a PR can be merged into `main`, the following automated jobs must report **GREEN**:

1. **`Code Hygiene, Contracts & Test Suite`**:
   - Developers and CI both run `uv run python scripts/verify.py`; there is no separate local approximation of the workflow.
   - The shared command enforces Ruff, strict Mypy, deterministic sample materialization, the Pandera training-data contract, DVC reproduction and clean status, and the complete Pytest suite.
   - Coverage XML and JUnit XML reports are generated locally and archived by GitHub Actions.
2. **`Multi-stage Docker Build Verification`**:
   - Proves container compiles from `uv.lock` and packages clean without dependency conflicts.

---

## 4. Model Registry Alias Governance (`@champion` / `@production`)

Just as `main` is protected from unreviewed code, the **`@champion`** and **`@production`** aliases in the MLflow Model Registry are protected assets:

- **Automated Gating**: Models can only receive the `@challenger` alias automatically upon clearing `configs/thresholds.yaml` in CI.
- **Promotion to `@champion`**: Requires manual sign-off followed by an explicit `promotion --alias champion --approved-by <REVIEWER_ID>` command. The command rejects protected aliases when `STREAMLY_ENV=ci` and rejects anonymous protected promotions in every environment.
- **Audit Lineage**: Every alias assignment records the `run_id`, environment, normalized alias, approver identity, and UTC timestamp as MLflow metadata. This application-level guard complements—not replaces—access control on the shared MLflow service.
