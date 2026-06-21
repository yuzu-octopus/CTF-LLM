# Plan 006: Docs Stale Values (8, 9)

**Commit**: `5f04de4`  
**Status**: ✅ DONE (9b71fcc)  
**Effort**: S (~20 min)  
**Risk**: LOW (documentation only)

## Problem

1. **#8**: `TRAINING.md` shows wrong hyperparameters — says "16-bit LoRA" for Gemma 4 E4B (actual: 4-bit QLoRA), missing Gemma 4 12B entirely, wrong epochs
2. **#9**: `README.md:186-198` config examples show stale values — `lora_alpha: 32` (actual: 64), `learning_rate: 2e-4` (actual: 1e-4)

## Current State

**TRAINING.md:123**:
```
| Gemma 4 E4B | 16-bit LoRA | r=32, alpha=64 |
```
Actual `configs/gemma4.yaml`: `load_in_4bit: true` (4-bit QLoRA)

**README.md:186-198**:
```yaml
models:
  gemma4:
    r: 32
    lora_alpha: 32       # actual: 64
    learning_rate: 2e-4  # actual: 1e-4
```

## Fix

### Step 1: Update TRAINING.md

- Change "16-bit LoRA" to "4-bit QLoRA" for Gemma 4 E4B
- Add Gemma 4 12B row with VRAM warning (>=20GB, not T4-compatible)
- Update epochs to match AGENTS.md table (fast=1, quality=2)

### Step 2: Update README.md config examples

Replace the stale config.yaml example with actual current values:
```yaml
default_model: gemma4

models:
  gemma4:
    name: unsloth/gemma-4-E4B-it
    r: 32
    lora_alpha: 64
    max_seq_length: 4096
    batch_size: 1
    load_in_4bit: true
    
training:
  num_train_epochs: 3
  learning_rate: 1.0e-4
  gradient_accumulation_steps: 4
```

Also fix the configs example:
```yaml
# configs/gemma4.yaml
model:
  name: unsloth/gemma-4-E4B-it
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  r: 32
  lora_alpha: 64
```

## Verification

```bash
# Verify TRAINING.md mentions 4-bit for gemma4
grep -n "4-bit QLoRA" TRAINING.md
# Expected: at least one match

# Verify README config examples match actual values
grep -n "lora_alpha: 64" README.md
grep -n "learning_rate: 1.0e-4" README.md
# Expected: matches in config examples

# Verify gemma4-12b is documented
grep -n "gemma4-12b" TRAINING.md
# Expected: at least one match
```

## Files to Modify

- `TRAINING.md`
- `README.md`
