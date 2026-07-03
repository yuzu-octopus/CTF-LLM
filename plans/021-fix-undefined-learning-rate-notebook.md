# Plan 021: Fix missing LEARNING_RATE in generated notebook

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report.
>
> **Drift check (run first)**: `git diff --stat 087ce44..HEAD -- <in-scope paths>`
> If any in-scope file changed, compare excerpts against live code before proceeding.

## Status

- **Priority**: P0
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `087ce44`, 2025-06-15

## Why this matters

`scripts/generate_notebook.py:336` uses `LEARNING_RATE` in the SFTConfig, but Section 3 of the generated notebook never assigns this variable. Any Colab training run crashes with `NameError: name 'LEARNING_RATE' is not defined`.

## Current state

Section 3 of the generated notebook reads `configs/<model>.yaml` but never extracts `learning_rate` to a named variable. The training cell references `LEARNING_RATE` but it's undefined.

## Scope

**In scope:** `scripts/generate_notebook.py` only (regenerate notebook after)

## Steps

### Step 1: Add `LEARNING_RATE` assignment in Section 3 config cell

In `scripts/generate_notebook.py`, find the Section 3 cell source (around line 200+) and add after the existing config extraction:
```python
'LEARNING_RATE = float(tc.get("learning_rate", 2e-4))',
```

### Step 2: Regenerate notebook

```bash
python3 scripts/generate_notebook.py
```

**Verify:** `uv run python -m pytest tests/ -v --tb=short` → all pass

## Done criteria
- [ ] `grep "LEARNING_RATE" notebooks/self_contained.ipynb` shows definition and usage
- [ ] `python3 scripts/generate_notebook.py` → exit 0

## STOP conditions

- In-scope code changed since plan was written
- Verification fails twice after fix attempt
- Fix requires touching out-of-scope files

## Verification

```bash
uv run python -m pytest tests/ -v --tb=short
uv run ruff check src/ tests/
```
