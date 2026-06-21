# Plan 004: Model Loading Dedup (5)

**Commit**: `5f04de4`  
**Status**: ✅ DONE (9b71fcc)  
**Effort**: M (~1 h)  
**Risk**: MED (refactoring core loading logic)

## Problem

`eval.py:46-63` and `train.py:64-80` duplicate the model-loading logic (Gemma vs non-Gemma branching, chat template setup). If model loading changes (e.g., new model family), both files must be updated in sync.

## Current State

**eval.py:46-63**:
```python
if model_key.startswith("gemma"):
    from unsloth import FastVisionModel
    model, processor = FastVisionModel.from_pretrained(...)
    processor = get_chat_template(processor, "gemma-4")
    tokenizer = processor.tokenizer
    model = FastVisionModel.get_for_inference(model)
else:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(...)
    tokenizer = get_chat_template(tokenizer, "chatml")
    model = FastLanguageModel.get_for_inference(model)
```

**train.py:64-80**:
```python
if model_key.startswith("gemma"):
    from unsloth import FastVisionModel
    model, processor = FastVisionModel.from_pretrained(...)
    tokenizer = processor.tokenizer
else:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(...)
```

Key differences:
- eval.py adds `get_for_inference(model)` and chat template
- train.py adds `use_gradient_checkpointing="unsloth"` and config-driven params

## Fix

Create `src/model_utils.py` with shared loading logic:

```python
"""Shared model loading utilities for train.py and eval.py."""
from pathlib import Path


def _model_name(model_key: str) -> str:
    """Map model key to HuggingFace model name."""
    from src.train import load_config
    config = load_config(model_key)
    return config.get("model", config).get("name", model_key)


def load_model_for_training(model_key: str, config: dict):
    """Load model for training with LoRA config."""
    from unsloth import FastLanguageModel, FastVisionModel, get_chat_template

    model_config = config.get("model", config)
    model_name = _model_name(model_key)

    if model_key.startswith("gemma"):
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=model_config.get("load_in_4bit", True),
            use_gradient_checkpointing="unsloth",
        )
        tokenizer = processor.tokenizer
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=model_config["max_seq_length"],
            load_in_4bit=model_config.get("load_in_4bit", True),
            dtype=None,
        )
    return model, tokenizer


def load_model_for_inference(model_key: str, adapter_path: str = None):
    """Load model for inference with optional LoRA adapter."""
    from unsloth import FastLanguageModel, FastVisionModel, get_chat_template
    from peft import PeftModel

    model_name = _model_name(model_key)

    if model_key.startswith("gemma"):
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_name, load_in_4bit=True,
        )
        processor = get_chat_template(processor, "gemma-4")
        tokenizer = processor.tokenizer
        model = FastVisionModel.get_for_inference(model)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, load_in_4bit=True, dtype=None,
        )
        tokenizer = get_chat_template(tokenizer, "chatml")
        model = FastLanguageModel.get_for_inference(model)

    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  Loaded LoRA adapter from {adapter_path}")

    return model, tokenizer
```

Then update `train.py` and `eval.py` to import from `model_utils.py`.

## Verification

```bash
# Verify both files import from model_utils
grep -n "from src.model_utils" src/train.py src/eval.py

# Verify no duplicate loading logic remains
grep -n "FastVisionModel.from_pretrained\|FastLanguageModel.from_pretrained" src/train.py src/eval.py
# Expected: 0 hits (moved to model_utils.py)

# Run tests
uv run python -m pytest tests/ -v
```

## Files to Create/Modify

- `src/model_utils.py` (NEW)
- `src/train.py` (replace loading logic with import)
- `src/eval.py` (replace loading logic with import)
