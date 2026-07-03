# Plan 028: Parallelize doc scraping in build_dataset.py

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
- **Category**: perf
- **Planned at**: commit `087ce44`, 2025-06-15

## Why this matters

`src/build_dataset.py:417-449` processes 12 `DOC_SOURCES` serially via network requests. Each doc is independent. The writeup scraping already uses `ThreadPoolExecutor` for cloning — same pattern should apply here.

## Current state

```python
for doc_idx, doc in enumerate(DOC_SOURCES):
    # ... sequential HTTP request per doc ...
```

## Scope

**In scope:** `src/build_dataset.py` only

## Steps

### Step 1: Wrap iteration in ThreadPoolExecutor

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _fetch_one(doc):
    # ... existing scraping logic ...
    return doc['name'], examples

with ThreadPoolExecutor(max_workers=6) as pool:
    futures = {pool.submit(_fetch_one, doc): doc for doc in DOC_SOURCES}
    for f in as_completed(futures):
        name, examples = f.result()
        all_examples.extend(examples[:max_per_doc])
        print(f"  ✓ {name}: {len(examples)} examples")
```

**Verify:** `uv run python -m pytest tests/test_build_dataset.py -v` → all pass

## Done criteria
- [ ] Doc sources are fetched in parallel
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
