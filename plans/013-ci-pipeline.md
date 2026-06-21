# Plan 013: CI Pipeline Setup (F2)

**Commit**: `6d7af02`  
**Status**: TODO  
**Effort**: M (~2 h)  
**Risk**: LOW (new CI config, no code changes)

## Problem

The repo has no CI pipeline. Tests only run when manually invoked. There's no linting, no type checking, no cross-doc drift checking. Drift accumulates silently between sessions (as demonstrated by the D1-D8 doc fixes).

## Fix

### Step 1: Create GitHub Actions workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --group dev

      - name: Lint with ruff
        run: uv run ruff check src/ tests/ --output-format=github

      - name: Type check with pyright (optional)
        run: uv run pyright src/ --ignoreexternal
        continue-on-error: true  # pyright may fail without GPU deps

      - name: Run tests
        run: uv run python -m pytest tests/ -v --tb=short
```

### Step 2: Add ruff + pyright dev dependencies

Add to `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    "pytest>=9.1.0",
    "ruff>=0.2.0",
    "pyright>=1.1.300",
]
```

### Step 3: Add linting config

Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "S"]
ignore = ["S101"]  # allow assert for tests
```

### Step 3: Verify

```bash
# Check the workflow is syntactically valid
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('YAML OK')"

# Run tests locally (should pass)
uv run python -m pytest tests/ -v --tb=short
# Expected: all tests pass

# Run ruff check
uv run ruff check src/ tests/
# Expected: no errors or warnings
```

## Files to Create/Modify

- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/` directory
- Modify: `pyproject.toml` (add ruff config)
