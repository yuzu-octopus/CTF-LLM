---
description: Diagnose Colab fine-tuning failures (version conflicts, OOM, model issues)
---

Diagnose Colab fine-tuning issues. Usage:

  /ft-diag                    # full diagnostic
  /ft-diag <session>          # diagnose specific session
  /ft-diag <session> <error>  # look up specific error

When training fails on Colab, check these in order:

## 1. Session state
  colab status -s <session>
  colab log -s <session> -n 50

## 2. Known issues and fixes (from this project)

| Error | Fix |
|-------|-----|
| `gemma4_unified` not in `transformers` | Use E2B/E4B variant, or 12B needs A100 |
| `NameError: auto_docstring` (unsloth) | `pip install --no-deps transformers==5.5.0` |
| `to_affine_quantized_fpx` import error | `pip install --no-deps "torchao>=0.16.0"` |
| `2e-4` YAML parsed as string (TypeError) | Cast: `float(training_config.get("learning_rate"))` |
| `colab upload` fails on directory | `mkdir -p` remote dir via `colab exec` first |
| `colab exec` pruned / timed out | Break training into smaller cells in notebook |
| T4 OOM with 12B model | Use 4B model (Qwen3.5-4B or Gemma 4 E4B) |

## 3. Version compatibility matrix (for Colab)
  pip install unsloth
  pip install --no-deps transformers==5.5.0 "tokenizers>=0.22.0,<=0.23.0"
  pip install --no-deps --upgrade "torchao>=0.16.0"

## 4. Model size vs T4 (16GB)
  | Model | Method | Fits? |
  |-------|--------|-------|
  | unsloth/Qwen3.5-4B | 4-bit QLoRA | yes (~3.5GB) |
  | unsloth/Qwen3.5-9B | 4-bit QLoRA | yes (~6GB) |
  | unsloth/gemma-4-E4B-it | 16-bit LoRA | yes (~8GB) |
  | unsloth/gemma-4-12B-it | 16-bit LoRA | NO (needs A100) |

## 5. Quick reset
  colab stop -s <session>
  colab new -s finetune-X --gpu T4
  # reinstall with correct versions
