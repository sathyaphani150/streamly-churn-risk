# Streamly Churn-Risk Scoring System

A production-shaped MLOps churn prediction platform for Streamly subscription streaming service.

## Project Architecture & Quickstart

### Prerequisites
- Python 3.11+
- `uv` (recommended package manager) or standard `pip`
- Git

### 1. Environment Setup
```bash
# Create and sync virtual environment with all runtime + dev dependencies
uv sync --extra dev

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
```

### 2. Code Quality & Testing
```bash
# Run linter and formatting check
uv run ruff check .
uv run ruff format --check .

# Run static type checks
uv run mypy src tests

# Run unit and integration tests with coverage
uv run pytest
```

### 3. Environment Configuration
```bash
# Initialize local environment variables
cp .env.example .env
```
