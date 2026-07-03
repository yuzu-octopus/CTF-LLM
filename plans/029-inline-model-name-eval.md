# Plan 029: Inline _model_name in eval.py to break circular import

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report.
>
> **Drift check (run first)**: `git diff --stat 087ce44..HEAD -- <in-scope paths>`
> If any in-scope file changed, compare excerpts against live code before proceeding.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: tech-debt
- **Planned at**: commit `087ce44`, 2025-06-15

## Why this matters

`src/eval.py:46-50` has `_model_name()` which imports from `src.train` to load config. This creates a circular dependency chain. The function is trivial — it just reads a YAML file.

## Current state

```python
def _model_name(model_key: str) -> str:
    from src.train import load_config
    config = load_config(model_key)
    return config.get("model", config).get("name", model_key)
```

## Scope

**In scope:** `src/eval.py` only

## Steps

### Step 1: Replace _model_name with direct YAML loading

```python
def _model_name(model_key: str) -> str:
    from pathlib import Path
    config_path = Path("configs") / f"{model_key}.yaml"
    if not config_path.exists():
        config_path = Path("config.yaml")
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("model", cfg).get("name", model_key)
```

**Verify:** `uv run python -m pytest tests/test_eval.py -v` → all pass

## Done criteria
- [ ] eval.py does not import from src.train
- [ ] All tests pass

## STOP conditions

- In-scope code changed since plan was written
- Verification fails twice after fix attempt
- Fix requires touching out-of-scope files

## Verification

```bash
uv run python -m pytest tests/ -v --tb=short
uv run ruff check src/ tests/
```
