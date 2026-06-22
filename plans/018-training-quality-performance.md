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
- `use_rslora` only enabled in notebook for quality mode, not in configs
- LR is uniform 1e-4 across all models (12B needs lower LR)

**Performance opportunities**:
- No flash-attention (potential 1.5-2x training speedup)
- `embed_tokens` not in target_modules (helps domain vocabulary adaptation)
- `num_train_epochs=2` (quality) is low end for LoRA convergence

## Fix

### Step 1: Add missing SFTConfig params to train.py

In `src/train.py` around line 199, add to the SFTConfig constructor:

```python
weight_decay=training_config.get("weight_decay", 0.01),
max_grad_norm=training_config.get("max_grad_norm", 1.0),
lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
neftune_noise_alpha=training_config.get("neftune_noise_alpha", 5),
```

Also add these to the config files so they can be overridden per-model:

```yaml
# In configs/gemma4.yaml training: section
weight_decay: 0.01
max_grad_norm: 1.0
lr_scheduler_type: cosine
neftune_noise_alpha: 5
```

### Step 2: Add lr_scheduler_type to configs

Add to all 4 config files. Consistent across models.

### Step 3: Fix qwen35-4b config to represent quality defaults

`configs/qwen35-4b.yaml`: change `r: 8` to `r: 16`, `lora_alpha: 16` to `lora_alpha: 32` (match the notebook's quality mode config). Also add `use_rslora: true` for quality mode.

### Step 4: Scale LR by model size

In `train.py`, add logic to scale learning rate:
```python
# Scale LR by model size
base_lr = training_config.get("learning_rate", 1e-4)
param_count = model_config.get("parameters", "")
if "12B" in str(param_count) or "12b" in str(param_count):
    lr = base_lr * 0.5  # 5e-5 for 12B models
else:
    lr = base_lr
```

Or just override in config files:
- `gemma4.yaml`: `learning_rate: 1.0e-4`
- `gemma4-12b.yaml`: `learning_rate: 5.0e-5`
- `qwen35.yaml`: `learning_rate: 1.0e-4`
- `qwen35-4b.yaml`: `learning_rate: 1.0e-4`

The config override approach is simpler and more explicit.

### Step 5: Add embed_tokens to target_modules

In all 4 config files, add `"embed_tokens"` to `target_modules`:

```yaml
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
  - embed_tokens  # new — adapts token embeddings for CTF domain vocabulary
```

Note: `lm_head` is excluded intentionally (weight tie with embed_tokens in most models).

### Step 6: Verify flash-attention compatibility

This step is investigative:

```bash
# Check if Unsloth already uses flash-attention internally
uv run python -c "from unsloth import is_bfloat16_supported; print('Unsloth OK')"

# Check if flash-attn can be installed
uv pip install flash-attn --dry-run 2>&1 | head -5
```

If Unsloth doesn't already use flash-attention and it installs cleanly on the target GPU (T4 sm_75), add it to `pyproject.toml`:
```toml
# Optional: flash-attention-2 for training speedup (T4 sm_75+)
```

### Step 7: Increase quality mode epochs to 3

In `notebooks/qwen4b_self_contained.ipynb`, change quality mode epochs:
```python
NUM_EPOCHS = 3  # was 2 — LoRA with r=32 needs more steps
```

In config files, update the num_train_epochs (though this key is dead — step 1 of plan 017 already fixes this by making train.py read from config).

### Step 8: Verify

```bash
# Check config changes
grep -E 'weight_decay|max_grad_norm|lr_scheduler_type|neftune_noise_alpha' configs/gemma4.yaml
# Expected: all 4 keys present

# Check target_modules
grep -A8 'target_modules' configs/gemma4.yaml | grep embed_tokens
# Expected: embed_tokens found

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
- `pyproject.toml` (optional: flash-attn)
