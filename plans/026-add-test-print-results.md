# Plan 026: Add tests for eval print_results bucketing logic

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report.
>
> **Drift check (run first)**: `git diff --stat 087ce44..HEAD -- <in-scope paths>`
> If any in-scope file changed, compare excerpts against live code before proceeding.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: tests
- **Planned at**: commit `087ce44`, 2025-06-15

## Why this matters

`src/eval.py:348-438` contains ~90 lines of bucket computation, Wilson CI, and balanced accuracy calculation with zero test coverage. Regressions in eval statistics would be invisible.

## Current state

- `tests/test_eval_orchestration.py` exists with 8 tests but none cover `print_results`
- The function mixes I/O (printing) with computation (bucketing, CI calculation)

## Scope

**In scope:** `tests/test_eval_orchestration.py` only (add new test class)

## Steps

### Step 1: Extract bucketing logic into a testable function

In `src/eval.py`, extract the bucketing computation from `print_results` into a pure function `_compute_buckets(results)` that returns the bucket dict. Have `print_results` call it.

### Step 2: Write tests for bucketing

Add to `tests/test_eval_orchestration.py`:
- Test single category/difficulty bucket
- Test Wilson CI bounds (known input → known output)
- Test balanced accuracy with unequal bucket sizes
- Test empty results (n=0 edge case)

**Verify:** `uv run python -m pytest tests/test_eval_orchestration.py -v` → all pass

## Done criteria
- [ ] `_compute_buckets()` is a pure function
- [ ] ≥4 new tests covering bucketing and CI
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
