# Plan 010: Dead Code Cleanup (F5)

**Commit**: `6d7af02`  
**Status**: TODO  
**Effort**: S (~5 min)  
**Risk**: NONE (removing dead code only)

## Problem

`src/build_dataset.py:283` defines `extract_from_huggingface()` but it's never called anywhere in the codebase. It's a leftover from when HF extraction was refactored to `download_datasets.py`.

## Current State

```python
# src/build_dataset.py:283 — never called
def extract_from_huggingface(dataset_name: str, max_samples: int = 5000) -> list:
```

```bash
# No callers exist
$ grep -rn 'extract_from_huggingface' src/
src/build_dataset.py:283:def extract_from_huggingface(dataset_name: str, max_samples: int = 5000) -> list:
```

## Fix

### Step 1: Remove the dead function

Delete lines 283 through the end of the function definition (including its docstring and body).

```bash
# Find the exact line range
grep -n 'def extract_from_huggingface' src/build_dataset.py
```

The function likely ends at some closing line. Remove the full block.

### Step 2: Verify

```bash
# Confirm no remaining references
grep -rn 'extract_from_huggingface' src/

# Expected output: empty (no matches)

# Run existing tests to confirm nothing broke
uv run python -m pytest tests/test_build_dataset.py -v
# Expected: all tests pass
```

## Files to Modify

- `src/build_dataset.py`
