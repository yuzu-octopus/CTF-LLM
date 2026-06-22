# Plan 018: Training Quality & Performance

**Commit**: `0831d17`
**Status**: TODO
**Effort**: L (~4 h)
**Risk**: MEDIUM (hyperparameter changes affect model quality)

## Problems

15 issues in training methodology and performance. Key items:

**CLI path vs notebook divergence (critical gap)**: `train.py` (used by `finetune.sh`) lacks:
- `weight_decay` → 0.0, no regularization on 2500-example dataset
- `max_grad_norm` → fp16 QLoRA needs gradient clipping
- `lr_scheduler_type` → defaults to "linear", notebook uses "cosine"
- `neftune_noise_alpha` → absent from CLI path, notebook sets alpha=5
- These make `finetune.sh --train` produce measurably worse models than the notebook

**Config issues**:
- `qwen35-4b.yaml` has r=8 (fast mode), no quality-mode config file exists
- `lora_alpha/r` ratio is 2:1, Unsloth recommends 1:1 (alpha == r)
- LR is uniform 1e-4 across all models; Unsloth uses 2e-4 as default
- `num_train_epochs=2` (quality) is low end for LoRA convergence

## Fix

### Step 1: Add missing SFTConfig params to train.py (Unsloth-recommended values)

In `src/train.py` around line 199, add to the SFTConfig constructor:

```python
weight_decay=training_config.get("weight_decay", 0.001),
max_grad_norm=training_config.get("max_grad_norm", 0.3),
lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
neftune_noise_alpha=training_config.get("neftune_noise_alpha", 5),
```

Values sourced from Unsloth's official Gemma 4 and Qwen 3.5 fine-tuning guides. Also add these to the config files so they can be overridden per-model:

```yaml
# In configs/gemma4.yaml training: section
weight_decay: 0.001
max_grad_norm: 0.3
lr_scheduler_type: cosine
neftune_noise_alpha: 5
```

### Step 2: Add lr_scheduler_type to configs

Add to all 4 config files. Consistent across models.

### Step 3: Fix qwen35-4b config to represent quality defaults

Per Unsloth's recommendation (alpha == r), update quality mode to use 1:1 ratio:
`configs/qwen35-4b.yaml`: change `r: 8` to `r: 16`, `lora_alpha: 16` to `lora_alpha: 16` (1:1, same value). Also set `use_rslora: false` (Unsloth's own default).

Update all 4 configs to use 1:1 alpha:r ratio:
| Model | Current alpha:r | Change to |
|-------|----------------|-----------|
| gemma4 | 64:32 (2:1) | 32:32 (1:1) |
| gemma4-12b | 64:32 (2:1) | 32:32 (1:1) |
| qwen35 | 64:32 (2:1) | 32:32 (1:1) |
| qwen35-4b | 16:8 (2:1) | 16:16 (1:1) |

### Step 4: Update learning rate to Unsloth-recommended 2e-4

Unsloth uses 2e-4 uniformly across both Gemma 4 and Qwen 3.5 models. Update all configs:

```yaml
# configs/gemma4.yaml, gemma4-12b.yaml, qwen35.yaml, qwen35-4b.yaml
learning_rate: 2.0e-4  # was 1.0e-4 — matches Unsloth's recommendation
```

Note: Unsloth says "Reduce to 2e-5 for long training runs" — our ~2500-example dataset is short, so 2e-4 is appropriate.

### Step 5: Increase quality mode epochs to 3

In `notebooks/qwen4b_self_contained.ipynb`, change quality mode epochs:
```python
NUM_EPOCHS = 3  # was 2 — LoRA with r=32 needs more steps
```

In config files, update the num_train_epochs (though this key is dead — step 1 of plan 017 already fixes this by making train.py read from config).

### Step 6: Verify

```bash
# Check config changes
grep -E 'weight_decay|max_grad_norm|lr_scheduler_type|neftune_noise_alpha' configs/gemma4.yaml
# Expected: weight_decay: 0.001, max_grad_norm: 0.3, lr_scheduler_type: cosine, neftune_noise_alpha: 5

# Check alpha:r ratio is 1:1
grep -E '^\s+r:|lora_alpha:' configs/gemma4.yaml
# Expected: r: 32, lora_alpha: 32 (same value)

# Check LR is 2e-4
grep 'learning_rate' configs/gemma4.yaml
# Expected: 2.0e-4

# Run tests
uv run python -m pytest tests/ -v --tb=short
# Expected: all pass

# Syntax check
python3 -c "import py_compile; py_compile.compile('src/train.py', doraise=True); print('OK')"
```

## Files to Modify

- `src/train.py`
- `configs/gemma4.yaml`
- `configs/gemma4-12b.yaml`
- `configs/qwen35.yaml`
- `configs/qwen35-4b.yaml`
- `notebooks/qwen4b_self_contained.ipynb`
