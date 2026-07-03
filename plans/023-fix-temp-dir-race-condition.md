# Plan 023: Fix temp dir race condition in parallel cloning

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

`src/build_dataset.py:367` uses `tempfile.gettempdir()/{repo['name']}` which is the same path across concurrent invocations. Two parallel builds clobber each other's cloned repos.

## Current state

```python
# src/build_dataset.py:367
def _clone_one(repo):
    p = f"{tempfile.gettempdir()}/{repo['name']}"
    return repo, clone_repo(repo["url"], p), p
```

## Scope

**In scope:** `src/build_dataset.py` only

## Steps

### Step 1: Use unique temp subdirectory per invocation

```python
import uuid
# In build_writeups_dataset(), before cloning:
_run_id = uuid.uuid4().hex[:8]

# Then in _clone_one:
def _clone_one(repo):
    p = f"{tempfile.gettempdir()}/ctf-llm-{_run_id}/{repo['name']}"
    return repo, clone_repo(repo["url"], p), p
```

**Verify:** `uv run python -m pytest tests/ -v --tb=short` → all pass

## Done criteria
- [ ] No shared temp paths between invocations
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
