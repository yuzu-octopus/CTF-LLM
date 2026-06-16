# Fine-tuning LLMs for CTF & Competitive Programming

Fine-tune open-source LLMs (Gemma 4 12B, Qwen 3.5 9B) to excel at cybersecurity CTF challenges and competitive programming using [Unsloth](https://unsloth.ai) with QLoRA on Google Colab's free tier.

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
| `justinwangx/CTFtime` | 18K writeups | Real CTF competition solutions |
| `AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1` | 99.8K examples | Cybersecurity Q&A |
| `ayshajavd/code-security-vulnerability-dataset` | 175K examples | Vulnerability detection |

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
- **Chat templates**: Applies model-specific formatting (Gemma uses `gemma_chatml`, Qwen uses `chatml`)

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
├── AGENTS.md                # Instructions for AI agents
├── README.md                # This file
│
├── src/                     # Python scripts
│   ├── train.py             # Unified training script (reads configs/*.yaml)
│   ├── build_dataset.py     # Scrapes GitHub repos for writeups
│   ├── download_datasets.py # Downloads HuggingFace datasets
│   └── process_data.py      # Converts data to chat format
│
├── configs/                 # Model-specific configurations
│   ├── gemma4.yaml          # Gemma 4 12B settings
│   └── qwen35.yaml          # Qwen 3.5 9B settings
│
└── data/                    # Training data (gitignored)
    ├── raw/                 # Scraped/downloaded raw data
    ├── processed/           # Converted to chat format
    └── merged/              # Final merged training file
```

## Configuration

### Main Config (`config.yaml`)

Central place for all settings:

```yaml
default_model: gemma4

models:
  gemma4:
    name: unsloth/gemma-4-12b-it
    r: 8                    # LoRA rank
    lora_alpha: 8           # LoRA scaling
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
  name: unsloth/gemma-4-12b-it
  target_modules: all-linear  # Apply LoRA to all linear layers
  r: 8
  lora_alpha: 8
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
| Training time | ~1-2 hours for 5K examples |
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
