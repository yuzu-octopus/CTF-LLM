# Plan 008: Data Quality Hardening (12, 13)

**Commit**: `5f04de4`  
**Status**: TODO  
**Effort**: S (~15 min)  
**Risk**: LOW (error handling + config fix)

## Problem

1. **#12**: `process_data.py:107-108` — `except Exception as e: continue` silently drops malformed JSONL lines with no logging
2. **#13**: `config.yaml` has `num_train_epochs: 3` for all models, but AGENTS.md says fast=1 epoch, quality=2 epochs

## Current State

```python
# process_data.py:107-108
except Exception as e:
    continue  # silently drops the line
```

```yaml
# config.yaml (all models)
training:
  num_train_epochs: 3  # should match fast/quality modes
```

## Fix

### Step 1: Add logging for dropped lines in process_data.py

```python
except Exception as e:
    skipped += 1
    if skipped <= 5:  # log first 5 errors
        print(f"  Warning: skipped line {i+1}: {str(e)[:100]}")
    continue
```

### Step 2: Add warning when many lines are skipped

After the loop:
```python
if skipped > 0:
    print(f"  Warning: {skipped}/{total} lines skipped due to errors")
```

### Step 3: Fix config.yaml epochs

Change `num_train_epochs: 3` to `num_train_epochs: 2` (the more conservative default that matches quality mode). Users can override via notebook MODE or CLI args.

## Verification

```bash
# Verify logging exists
grep -n "skipped" src/process_data.py
# Expected: at least one match

# Verify config.yaml epochs
grep "num_train_epochs" config.yaml
# Expected: 2 (not 3)

# Run tests
uv run python -m pytest tests/ -v
```

## Files to Modify

- `src/process_data.py`
- `config.yaml`
