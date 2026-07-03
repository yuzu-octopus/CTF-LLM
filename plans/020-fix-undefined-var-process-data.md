# Plan 020: Fix undefined variable in process_data error handler

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

`src/process_data.py:111` prints `{i+1}` in the error handler, but the loop uses `for line in line_iter` without `enumerate()`. If any JSONL line is malformed, the error handler itself crashes with `NameError`, killing the entire processing run silently.

## Current state

- `src/process_data.py:108-112` — the except block references undefined `i`:
  ```python
  except Exception as e:
      skipped += 1
      if skipped <= 5:
          print(f"  Warning: skipped line {i+1}: {str(e)[:100]}")
      continue
  ```

## Scope

**In scope:** `src/process_data.py` only

## Steps

### Step 1: Replace `i+1` with `skipped + count + 1`

Edit `src/process_data.py:111`:
```python
# Before
print(f"  Warning: skipped line {i+1}: {str(e)[:100]}")
# After — compute a line number from counters
print(f"  Warning: skipped line {count + skipped}: {str(e)[:100]}")
```

**Verify:** `uv run python -m pytest tests/test_process_data.py -v` → all pass

## Done criteria
- [ ] `uv run python -m pytest tests/ -v --tb=short` → all 64 pass
- [ ] No `NameError` in error handler path

## STOP conditions

- In-scope code changed since plan was written
- Verification fails twice after fix attempt
- Fix requires touching out-of-scope files

## Verification

```bash
uv run python -m pytest tests/ -v --tb=short
uv run ruff check src/ tests/
```
