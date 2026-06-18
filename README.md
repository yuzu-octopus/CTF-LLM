# Fine-tuning LLMs for CTF & Competitive Programming

Fine-tune open-source LLMs (Gemma 4 E4B, Qwen 3.5 9B/4B) to excel at cybersecurity CTF challenges and competitive programming using [Unsloth](https://unsloth.ai) with QLoRA on Google Colab's free tier.

## What This Does

This project builds a complete pipeline for creating specialized AI models that can:

- **Solve CTF challenges**: Binary exploitation (pwn), reverse engineering, web exploitation, cryptography
- **Write competitive programming solutions**: Algorithms, data structures, code optimization
- **Analyze security vulnerabilities**: Identify and explain code weaknesses

The pipeline scrapes real CTF writeups and programming solutions from GitHub, combines them with curated datasets from HuggingFace, and fine-tunes models using efficient QLoRA (4-bit quantization + LoRA adapters).

## Quick Start

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run full pipeline (build data + train)
./finetune.sh gemma4 --all

# Or run steps individually
./finetune.sh gemma4 --build-data  # Just build datasets
./finetune.sh gemma4 --train       # Just train (requires data)
```

## How It Works

### 1. Dataset Collection (`src/build_dataset.py`)

Scrapes GitHub repositories containing CTF writeups and security documentation:

| Source | What It Contains | Examples |
|--------|------------------|----------|
| `Cajac/picoCTF-Writeups` | 250+ picoCTF challenge solutions | Markdown + Python/C exploit code |
| `DarkCodeOrg/CryptoHack` | Crypto challenge solutions | Python cryptographic exploits |
| `Adamkadaban/CTFs` | Multi-category CTF writeups | 376 MB across pwn, rev, web, crypto |
| pwntools, angr, ROPgadget docs | Security tool documentation | Usage examples and tutorials |

Also downloads structured datasets from HuggingFace:

| Dataset | Size | Content |
|---------|------|---------|
| `nvidia/OpenCodeReasoning` | 735K samples | Competitive programming with reasoning traces |
| `AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1` | 99.8K examples | Cybersecurity Q&A |
| `Jacqkues/ctf_webserver_v0.1` | 340 samples | Web CTF challenges |
| `kyleavery/picoctf` | 120 samples | picoCTF challenge solutions |

### 1.5 Synthetic Rev/PWN Examples (`src/synthetic_rev_pwn.py`)

23 hand-crafted training examples covering the hardest CTF categories:

| Category | Examples | Techniques Covered |
|----------|----------|-------------------|
| **PWN** (13) | ret2win, ret2libc, format string, heap UAF, stack canary bypass, ROP chain, ret2dlresolve, fastbin attack, heap consolidation, shellcode injection, ASLR bypass, ret2csu, read→shellcode ROP | Chain-of-thought reasoning before exploit code |
| **REV** (10) | binary analysis workflow, XOR cipher, anti-debugging, ELF sections, packing detection, crypto identification, keygen, Frida tracing, LD_PRELOAD hooking, deobfuscation | Step-by-step methodology, not just answers |

These examples use chain-of-thought reasoning: the model learns to analyze before solving, not just jump to the answer.

### 1.6 Training Modes (Fast vs Full)

The notebook supports two training modes with different dataset sizes:

| Parameter | Fast (~30 min) | Full (~50-70 min) |
|-----------|----------------|-------------------|
| MAX_PER_REPO | 30 | no limit |
| MAX_HF_SAMPLES | 200 | no limit |
| MAX_DOC_SECTIONS | 5 | no limit |
| MAX_OUTPUT_LEN | 2000 chars | 20000 chars |
| MAX_SEQ_LENGTH | 2048 | 4096 |
| LORA_R | 8 | 16 |
| NUM_EPOCHS | 1 | 2 |

Set `MODE = "fast"` or `MODE = "full"` in notebook Section 2.0.

### 2. Data Processing (`src/process_data.py`)

Converts raw data into the chat format required for fine-tuning:

```
Alpaca Format:                    ChatML Format:
{                                 {
  "instruction": "...",             "messages": [
  "input": "...",                     {"role": "system", "content": "..."},
  "output": "..."                     {"role": "user", "content": "..."},
}                                     {"role": "assistant", "content": "..."}
                                    ]
                                  }
```

The system prompt is automatically selected based on the content category (CTF vs competitive programming).

### 3. Model Training (`src/train.py`)

Uses Unsloth's optimized QLoRA training:

- **4-bit quantization**: Reduces memory usage by ~75%, enabling training on free T4 GPUs
- **LoRA adapters**: Only trains small adapter layers (~1% of parameters)
- **Chat templates**: Applies model-specific formatting (Gemma uses `gemma-4`, Qwen uses `chatml`)

Training outputs three model versions:
- `lora/` - LoRA adapter (small, for continued training)
- `gguf/` - GGUF format (for llama.cpp, Ollama, local inference)
- `merged/` - Full merged model (for HuggingFace, vLLM)

### 4. Colab Integration (`finetune.sh`)

Automates the entire workflow using the `colab` CLI:

1. Creates a Colab session with free T4 GPU
2. Installs dependencies
3. Uploads data and scripts
4. Runs training
5. Downloads trained models
6. Cleans up the session

## Project Structure

```
finetuning/
├── finetune.sh              # Main entry point - orchestrates everything
├── config.yaml              # Central configuration for models and training
├── pyproject.toml           # uv project config and dependencies
├── requirements.txt         # Local Python deps
├── requirements-colab.txt   # Colab-only install set (ordered)
├── AGENTS.md                # Instructions for AI agents
├── README.md                # This file
├── TRAINING.md              # Step-by-step training guide
├── .mimocode/commands/      # Slash-commands for mimocode
│   └── ft-diag.md           # /ft-diag — Colab failure mode diagnostic
│
├── src/                     # Python scripts
│   ├── train.py             # Unified training script (reads configs/*.yaml)
│   ├── build_dataset.py     # Scrapes GitHub repos for writeups
│   ├── download_datasets.py # Downloads HuggingFace datasets
│   └── process_data.py      # Converts data to chat format
│
├── configs/                 # Model-specific configurations (3 total)
│   ├── gemma4.yaml          # Gemma 4 E4B settings
│   ├── qwen35.yaml          # Qwen 3.5 9B settings
│   └── qwen35-4b.yaml       # Qwen 3.5 4B settings
│
├── notebooks/
│   └── qwen4b_self_contained.ipynb  # ONLY notebook (full pipeline)
│
└── data/                    # Training data (gitignored)
    ├── writeups.jsonl       # GitHub writeup scrape (data/ root)
    ├── docs.jsonl           # GitHub docs scrape (data/ root)
    ├── raw/                 # HuggingFace downloads
    ├── processed/           # Converted to chat format
    └── merged/              # Final merged training file (train.jsonl)
```

## Configuration

### Main Config (`config.yaml`)

Central place for all settings:

```yaml
default_model: gemma4

models:
  gemma4:
    name: unsloth/gemma-4-E4B-it
    r: 32                    # LoRA rank
    lora_alpha: 32           # LoRA scaling
    max_seq_length: 4096    # Max token length (limited by T4 VRAM)
    batch_size: 1           # Batch size (limited by T4 VRAM)
    
training:
  num_train_epochs: 3
  learning_rate: 2e-4
  gradient_accumulation_steps: 4
```

### Model Configs (`configs/*.yaml`)

Individual model configurations that override main config:

```yaml
# configs/gemma4.yaml
model:
  name: unsloth/gemma-4-E4B-it
  target_modules: all-linear  # Apply LoRA to all linear layers
  r: 32
  lora_alpha: 32
```

## Adding a New Model

1. Create `configs/newmodel.yaml` with model settings
2. Add the model to `config.yaml` under `models:`
3. Run: `./finetune.sh newmodel --all`

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA T4 (16GB) - free on Colab |
| VRAM | ~12GB used with QLoRA |
| Training time (Fast) | ~30 min for ~500 examples |
| Training time (Full) | ~50-70 min for ~2500+ examples |
| Storage | ~2GB for data, ~4GB for model |

## Export Formats

| Format | Use Case | Command |
|--------|----------|---------|
| GGUF | llama.cpp, Ollama, local inference | `model.save_pretrained_gguf()` |
| SafeTensors | HuggingFace, vLLM, inference APIs | `model.save_pretrained_merged()` |
| LoRA | Continued training, fine-tuning | `model.save_pretrained()` |

## Troubleshooting

**Out of Memory (OOM)**
- Reduce `max_seq_length` in config (try 2048)
- Ensure `batch_size: 1`
- Use `use_gradient_checkpointing: unsloth`

**Slow training**
- Ensure you're using GPU runtime
- Check `fp16`/`bf16` settings match your GPU

**Data issues**
- Verify format: `head -1 data/merged/train.jsonl | python -m json.tool`
- Check for empty outputs: `grep -c '"output": ""' data/merged/train.jsonl`

## Credits

- [Unsloth](https://unsloth.ai) - QLoRA training framework
- [CTFtime](https://ctftime.org) - CTF writeup source
- [picoCTF](https://picoctf.org) - CTF challenge platform
- [CryptoHack](https://cryptohack.org) - Crypto challenge platform
