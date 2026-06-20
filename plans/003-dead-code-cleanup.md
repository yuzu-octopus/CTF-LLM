# Plan 003: Dead Code Cleanup (3, 4, N5)

**Commit**: `5f04de4`  
**Status**: TODO  
**Effort**: S (~15 min)  
**Risk**: LOW (removing dead code)

## Problem

1. **#3**: `train.py:259-262` — `stage1_config` dict built but never consumed by `train()`
2. **#4**: `build_dataset.py:555` — `--source merge` registered in argparse but has no handler
3. **N5**: `generate_notebook.py:100` — `src_funcs["extract_writeup"]` extracted but never used

## Current State

```python
# train.py:259-262 (dead code)
stage1_config = {
    "model": {**config.get("model", {}), "r": 8, "lora_alpha": 16},
    "training": {**config.get("training", {}), "num_train_epochs": 1},
}
# This dict is never passed to train()

# build_dataset.py:555 (--source merge no handler)
choices=["writeups", "docs", "ctfd", "all", "merge"]
# No `elif args.source == "merge":` branch exists

# generate_notebook.py:100 (dead extraction)
src_funcs["extract_writeup"] = adapt_for_notebook(
    extract_function("build_dataset", "extract_writeups_from_repo"))
# Never referenced in cells list
```

## Fix

### Step 1: Remove dead stage1_config from train.py

Delete lines 259-262 (the `stage1_config` dict construction). The `train()` function reads config from YAML via `load_config()` independently.

### Step 2: Remove "merge" from argparse choices in build_dataset.py

Change:
```python
choices=["writeups", "docs", "ctfd", "all", "merge"]
```
To:
```python
choices=["writeups", "docs", "ctfd", "all"]
```

### Step 3: Remove dead extract_writeup from generate_notebook.py

Delete line 100:
```python
src_funcs["extract_writeup"] = adapt_for_notebook(
    extract_function("build_dataset", "extract_writeups_from_repo"))
```

## Verification

```bash
# Verify stage1_config is gone
grep -n "stage1_config" src/train.py
# Expected: no output

# Verify merge removed from choices
grep -n '"merge"' src/build_dataset.py
# Expected: no output

# Verify extract_writeup removed from generator
grep -n 'extract_writeup' scripts/generate_notebook.py
# Expected: no output (or only in cells that define it, not in src_funcs)

# Run tests
uv run python -m pytest tests/ -v
```

## Files to Modify

- `src/train.py`
- `src/build_dataset.py`
- `scripts/generate_notebook.py`
