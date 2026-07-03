# Plan 024: Fix bare except swallowing contamination check failures

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report.
>
> **Drift check (run first)**: `git diff --stat 087ce44..HEAD -- <in-scope paths>`
> If any in-scope file changed, compare excerpts against live code before proceeding.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `087ce44`, 2025-06-15

## Why this matters

`src/eval.py:275` has `except Exception: pass` on the contamination check. When the check fails, users see no warning and think they're safe.

## Current state

```python
def check_contamination(benchmarks):
    try:
        # ... hash comparison logic ...
    except Exception:
        pass
```

## Scope

**In scope:** `src/eval.py` only

## Steps

### Step 1: Log exception instead of swallowing

```python
except Exception as e:
    print(f"  Warning: contamination check failed: {e}")
```

**Verify:** `uv run python -m pytest tests/test_eval_orchestration.py -v` → all pass

## Done criteria
- [ ] No bare `except Exception: pass` in eval.py
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
