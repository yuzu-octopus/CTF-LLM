# Plan 017: Pipeline Critical Fixes

**Commit**: `0831d17`
**Status**: TODO
**Effort**: M (~3 h)
**Risk**: MEDIUM (changes to Colab upload paths, notebook, requirements)

## Problems

25 issues in the training pipeline. Critical ones:

**B1 — Two-stage curated subset format mismatch**: `_build_curated_subset()` saves Alpaca format but stage 2 expects ChatML → empty dataset.

**B3 — `datasets==4.3.0` doesn't exist**: `requirements-colab.txt` pins a nonexistent version → Colab install fails.

**B4/B5 — Gemma notebook crashes**: GGUF/merged save and `max_seq_length` in `get_peft_model()` both crash for Gemma models.

**B2/S1 — Missing Colab uploads**: `finetune.sh` only uploads `train.py`, missing module dependencies.

**C1 — Dead config keys**: `num_train_epochs` in YAML never read by `train.py`.

**C2/C3/N3 — Config/notebook drift**: `grad_accum`, `use_rslora` differ between configs and notebook.

## Fix

### Step 1: Fix datasets version in requirements-colab.txt

`requirements-colab.txt:7`:
```diff
- datasets==4.3.0
+ datasets>=3.0.0
```

### Step 2: Fix two-stage format mismatch

In `src/train.py:_build_curated_subset()`, after building the `examples` list, convert to ChatML format before saving:

```python
# After collecting examples, convert to ChatML
for ex in examples:
    if "messages" not in ex and "instruction" in ex:
        from src.process_data import convert_alpaca_to_chat
        chat = convert_alpaca_to_chat(
            ex.get("instruction", ""),
            ex.get("input", ""),
            ex.get("output", ""),
            category=ex.get("category", ""),
            system_prompt_mode="auto",
            skip_system_prompt=True,
        )
        ex.clear()
        ex.update(chat)
```

This reuses the existing converter from `process_data.py`.

### Step 3: Fix Gemma notebook crash — guard GGUF/merged saves

In `notebooks/self_contained.ipynb`, around lines 963-974:

```python
# Save model — Gemma models don't support GGUF or merged saves
if not MODEL.startswith("gemma"):
    model.save_pretrained_gguf(...)
    model.save_pretrained_merged(...)
# LoRA save works for all models
model.save_pretrained(lora_dir)
```

### Step 4: Fix Gemma notebook crash — remove max_seq_length from get_peft_model

In the notebook cell around line 663 (the Gemma branch), remove `max_seq_length` from `get_peft_model()`:

```python
# Before (crashes for Gemma):
model = FastVisionModel.get_peft_model(model, ..., max_seq_length=MAX_SEQ_LENGTH, ...)
# After:
model = FastVisionModel.get_peft_model(model, ..., ...)
# max_seq_length is only valid for FastLanguageModel.get_peft_model(), not FastVisionModel
```

### Step 5: Fix finetune.sh — upload module dependencies

In `finetune.sh`, add upload lines for the modules `train.py` depends on:

```bash
# After the mkdir commands and config uploads:
colab upload -s "$SESSION_NAME" -f src/model_utils.py /content/src/model_utils.py
colab upload -s "$SESSION_NAME" -f src/synthetic_rev_pwn.py /content/src/synthetic_rev_pwn.py
```

### Step 6: Add Gemma max_seq_length protection in train.py

In `src/train.py:67-72`, add `max_seq_length` to the Gemma model load:

```python
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_config["name"],
    max_seq_length=model_config.get("max_seq_length", 2048),  # was missing entirely
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
```

### Step 7: Fix dead config keys — num_train_epochs

Either:
(a) Have `train.py` read from config: change line 183 to use config value with CLI override
(b) Or remove it from all config files to avoid confusion

Option (a) is cleaner:
```python
epochs = epochs or training_config.get("num_train_epochs", 3)
```

### Step 8: Sync grad_accum between configs and notebook

Fix `configs/gemma4.yaml` quality defaults to match notebook:
- Set `use_rslora: false` (matching Unsloth's own example — rsLoRA helps most at r≥64, not r=32)

This ensures `finetune.sh --train` behaves consistently with Unsloth's recommended defaults.

### Step 9: Verify

```bash
# Python syntax check
python3 -c "import py_compile; py_compile.compile('src/train.py', doraise=True); print('OK')"

# Verify datasets version
grep 'datasets' requirements-colab.txt
# Expected: datasets>=3.0.0 (or similar valid spec)

# Run full test suite
uv run python -m pytest tests/ -v --tb=short
# Expected: all pass

# Verify finetune.sh has upload lines
grep -c 'model_utils' finetune.sh
# Expected: ≥ 1
```

## Files to Modify

- `requirements-colab.txt`
- `src/train.py`
- `notebooks/self_contained.ipynb`
- `finetune.sh`
- `configs/gemma4.yaml`
- `configs/gemma4-12b.yaml`
