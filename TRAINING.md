# Training Guide

## Overview

Fine-tune a model (Gemma 4 E4B, Gemma 4 12B, Qwen 3.5 9B, or Qwen 3.5 4B) for CTF/coding tasks
using Unsloth with QLoRA on Google Colab's free T4 GPU. The training flow is
identical across models — only the `--model` flag changes.

> **Canonical training interface: the notebook.**
> Long-running training (model load + 3 epochs) is best run from
> `notebooks/self_contained.ipynb` opened interactively in Colab. The
> `colab exec` flow below is fine for short debug runs but will hit Colab's
> session pruning on a full training job. See the notebook for the production path.

## Workflow

```
Local Machine                    Colab (T4 GPU)
─────────────                    ──────────────
1. Build dataset
   (build_dataset.py)
   (download_datasets.py)
   (process_data.py)
         │
         ▼
2. Upload data ──────────────►  3. Install deps (notebook cell OR
   (colab upload)                    finetune.sh install)
         │
         ▼
4. Open notebook OR ─────────►  5. Run training interactively
   trigger via finetune.sh           (notebook cells, not colab exec)
         │
         ▼
6. Download models ◄──────────  7. Save outputs
   (colab download)                 lora/, gguf/, merged/
```

## Step-by-Step Commands

### Step 1: Build Dataset (Local)

```bash
cd /Users/yuzu/Documents/Projects/finetuning

# Build from GitHub repos + HuggingFace
uv run src/build_dataset.py --source all --max-per-repo 300 --max-per-doc 100

# Process to ChatML format
uv run src/process_data.py --input data/raw --output data/processed
uv run src/process_data.py --merge
```

### Step 2: Use the Orchestrator (Recommended)

`finetune.sh` handles session creation, dep install, file upload, and model
download. It uploads the entire `src/` directory along with configs and data —
you do not need to upload files manually for this path.

```bash
# Build data + train (default: gemma4)
./finetune.sh gemma4 --all

# Or pick a different model
./finetune.sh qwen35 --all
./finetune.sh qwen35-4b --all

# Steps individually
./finetune.sh gemma4 --build-data  # data only
./finetune.sh gemma4 --train       # train only (data must exist)
```

### Step 3: Manual Colab Session (Optional)

If you want to drive the session yourself instead of using `finetune.sh`:

```bash
colab new -s finetune-<model> --gpu T4
```

### Step 4: Install Unsloth on Colab

The `pip install unsloth` one-liner is **not** enough on a fresh Colab T4 — it
pulls in conflicting default versions of `torchao` and `transformers`. Use this
ordered install (mirrors `requirements-colab.txt`):

```bash
colab exec -s finetune-<model> "pip install -q sentencepiece protobuf datasets huggingface_hub hf_transfer && pip install --no-deps unsloth_zoo bitsandbytes accelerate xformers peft trl triton unsloth && pip install --no-deps --upgrade 'torchao>=0.16.0' && pip install --no-deps transformers==5.5.0 'tokenizers>=0.22.0,<=0.23.0'"
```

### Step 5: Run Training (Notebook — Preferred)

Open `notebooks/self_contained.ipynb` in Colab, set the `--model` cell
argument to your choice (`gemma4`, `qwen35`, or `qwen35-4b`), and run cells
top-to-bottom. This keeps the cell visible so you can monitor loss curves and
catch OOM early.

### Step 5 (alt): Run Training via `colab exec` (Debug Only)

Only suitable for short debug runs — long jobs will be killed by Colab's session
pruner:

```bash
colab exec -s finetune-<model> -f src/train.py --model <model>
```

### Step 6: Download Models

```bash
colab download -s finetune-<model> /content/outputs/ outputs/<model>-ctf/
```

### Step 7: Cleanup

```bash
colab stop -s finetune-<model>
```

## Training Config by Model

| Parameter | Gemma 4 E4B | Gemma 4 12B | Qwen 3.5 9B | Qwen 3.5 4B |
|-----------|-------------|-------------|-------------|-------------|
| Base model | `unsloth/gemma-4-E4B-it` | `google/gemma-4-12B-it` | `unsloth/Qwen3.5-9B` | `unsloth/Qwen3.5-4B` |
| Quantization | 4-bit QLoRA | 4-bit QLoRA | 4-bit QLoRA | 4-bit QLoRA |
| LoRA rank | 32 | 32 | 32 | 8 |
| LoRA alpha | 64 | 64 | 64 | 16 |
| Target modules | q/k/v/o + gate/up/down | q/k/v/o + gate/up/down | q/k/v/o + gate/up/down | q/k/v/o + gate/up/down |
| Max seq length | 4096 | 4096 | 4096 | 4096 |
| Batch size | 1 | 1 | 1 | 1 |
| Gradient accum | 4 (effective = 4) | 8 (effective = 8) | 8 (effective = 8) | 4 (effective = 4) |
| Epochs | 3 | 3 | 3 | 3 |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| Optimizer | adamw_8bit | adamw_8bit | adamw_8bit | adamw_8bit |
| Chat template | `gemma-4` | `chatml` | `chatml` |

## Expected Outputs

```
outputs/<model>-ctf/
├── lora/     # LoRA adapter (~100MB)
├── gguf/     # GGUF quantized (~4GB)
└── merged/   # Full merged model
```

## Data Summary

- **Source**: GitHub repos (picoCTF, CryptoHack, pwnCollege) + HuggingFace datasets
- **Format**: ChatML (system/user/assistant messages)
- **Size**: see `wc -l data/merged/train.jsonl` (varies by build)
- **Categories**: CTF challenges, competitive programming, cybersecurity

## Troubleshooting

### "No module named unsloth" / `torchao` / `transformers` version errors

These are almost always the symptom of running `pip install unsloth` alone on a
fresh Colab. Use the ordered install from Step 4 (or
`colab install -r requirements-colab.txt` via `finetune.sh`).

### Training is killed mid-run

Colab's session pruner is killing the job. Switch from `colab exec` to the
notebook workflow (Step 5) — running interactively in the notebook keeps the
session alive and lets you see loss curves.

### "CUDA out of memory"

- Reduce `max_seq_length` in the model config (try 2048)
- Keep `batch_size: 1`
- Ensure `use_gradient_checkpointing: unsloth`
- All models use 4-bit QLoRA by default — check `load_in_4bit: true` in config
- Full Gemma 4 12B does not fit on T4 — use the E4B variant

### Session timeout

- Colab free tier has idle timeouts and a hard session length cap
- Use the notebook for any run > ~30 minutes
- Monitor with `colab status -s finetune-<model>` if you used `colab new`

### Unknown Colab error

Run the `/ft-diag` slash-command — it captures the 7 known failure modes
(gemma4_unified arch, `NameError: auto_docstring`, torchao fpx import, YAML
`2e-4` parse, colab upload directory fail, session pruning, T4 OOM with 12B).
