# Plan 027: Add tests for grade_code restricted exec

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

`src/eval.py:161-180` runs user-submitted code in a sandboxed exec with a hand-rolled `safe_builtins` dict. This is never validated — we don't know if the builtins actually block dangerous operations or if missing builtins cause `NameError` crashes.

## Current state

```python
safe_builtins = {
    "range": range, "len": len, "int": int, "float": float,
    "str": str, "bool": bool, "list": list, "dict": dict,
    ...
}
local_env = {"__builtins__": safe_builtins}
exec(candidate, local_env)
```

## Scope

**In scope:** `tests/test_eval.py` only

## Steps

### Step 1: Add sandbox validation tests

- Test that `exec` runs valid code (happy path exists already)
- Test that dangerous builtins like `open`, `__import__`, `eval` are NOT available
- Test that code using available builtins (len, range, etc.) works
- Test that code using unavailable builtins raises appropriate error
- Test that code with side effects (file access, network) is blocked

**Verify:** `uv run python -m pytest tests/test_eval.py -v` → all pass

## Done criteria
- [ ] ≥3 new tests for grade_code sandbox
- [ ] Tests confirm dangerous builtins are not in scope
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
