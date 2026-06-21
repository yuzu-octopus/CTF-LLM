# Plan 009: Config Drift Fix (F3)

**Commit**: `6d7af02`  
**Status**: ✅ DONE (44d12cb)  
**Effort**: S (~30 min)  
**Risk**: LOW (config values only, no code changes)

## Problem

`config.yaml` and `configs/*.yaml` both default to quality-mode LoRA values (r=32, alpha=64). Fast-mode values (r=8, alpha=16) only exist in the notebook `MODEL_CONFIGS` dict. Adding a new model means updating 3 places (config.yaml + configs/*.yaml + notebook) and they'll drift.

Specifically:
- `config.yaml:20-21` — gemma4 has `r: 32, lora_alpha: 64` (quality mode only)
- `config.yaml:122-123` — qwen35-4b has `r: 8, lora_alpha: 16` (accidentally matches fast mode)
- `configs/gemma4.yaml` — same quality-mode values
- `configs/qwen35-4b.yaml` — matches fast mode only

The configs are supposed to be fallback-quality defaults. Fast mode overrides them in the notebook, but a user running `./finetune.sh gemma4 --train` without explicitly setting MODE gets quality-mode params even if they intended fast mode.

## Fix

### Step 1: Document that configs/*.yaml are quality-mode defaults

Add a comment at the top of each `configs/*.yaml`:
```yaml
# Default: quality mode (r=32, alpha=64). Fast mode (r=8, alpha=16) is
# controlled by the notebook MODEL_CONFIGS dict; override via --mode fast.
```

Also add a comment at `config.yaml:7`:
```yaml
# WARNING: config.yaml values are quality-mode defaults. The notebook
# MODEL_CONFIGS dict provides per-mode overrides (fast/quality).
# Keep both in sync when adding a new model.
```

### Step 2: Verify all 4 configs/*.yaml match config.yaml for quality mode

Check `configs/gemma4.yaml`, `configs/gemma4-12b.yaml`, `configs/qwen35.yaml`, `configs/qwen35-4b.yaml` — their `r` and `lora_alpha` should match the corresponding entries in `config.yaml` under `models:` (these are the quality-mode defaults).

The current state:
| Model | config.yaml r/alpha | configs/*.yaml r/alpha | Match? |
|-------|---------------------|------------------------|--------|
| gemma4 | 32/64 | 32/64 | ✓ |
| gemma4-12b | 32/64 | 32/64 | ✓ |
| qwen35 | 32/64 | 32/64 | ✓ |
| qwen35-4b | 8/16 | 8/16 | ✓ |

All currently match. Just add the documentation comments.

### Step 3: Add mode override via env var (for colab-exec pipeline)

Currently, mode is only controllable in the notebook. Add env-var wiring to `finetune.sh` so `MODE=fast ./finetune.sh gemma4 --train` works:

```bash
# In finetune.sh, after ACTION= parsing (around line 15):
# Mode override (default: quality)
MODE="${MODE:-quality}"
```

Then ensure the env var is passed through in the colab exec command. This avoids any `$2` collision since MODE is an env var, not a positional arg.

### Step 4: Verify

```bash
# Verify comments exist
grep -c "quality-mode" config.yaml configs/*.yaml
# Expected: at least 1 per file (5 total across 5 files)

# Verify r/alpha values are consistent
grep -E '^\s+r: |lora_alpha:' config.yaml configs/*.yaml
# Expected: all consistent pairs

# Verify MODE env var is referenced
grep -n 'MODE' finetune.sh
# Expected: at least the MODE="${MODE:-quality}" line
```

## Files to Modify

- `config.yaml` — add warning comment
- `configs/gemma4.yaml` — add mode comment
- `configs/gemma4-12b.yaml` — add mode comment
- `configs/qwen35.yaml` — add mode comment
- `configs/qwen35-4b.yaml` — add mode comment
- `finetune.sh` — add `--mode` CLI arg (optional stretch goal)
