# LLM Fine-tuning for CTF/Coding Tasks

Fine-tune Gemma 4 12B and Qwen 3.5 9B using Unsloth with QLoRA on Google Colab (Free T4 GPU).

## Quick Start

```bash
# Install dependencies
uv pip install -r requirements.txt

# Build datasets and train (full pipeline)
./finetune.sh gemma4 --all

# Or just build data
./finetune.sh gemma4 --build-data

# Or just train (data must exist)
./finetune.sh gemma4 --train
```

## Setup

```bash
# Install colab CLI (if not installed)
uv tool install colab-cli

# Setup authentication
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

## Project Structure

```
finetuning/
├── finetune.sh              # Main entry script
├── config.yaml              # Main configuration
├── requirements.txt         # Dependencies (uv)
├── README.md
├── data/
│   ├── raw/                 # Scraped/downloaded data
│   ├── processed/           # Alpaca format
│   └── merged/              # Final training data
├── src/
│   ├── train.py             # Unified training script
│   ├── build_dataset.py     # Dataset builder (GitHub scraping)
│   └── download_datasets.py # HuggingFace downloads
└── configs/
    ├── gemma4.yaml          # Gemma 4 config
    └── qwen35.yaml          # Qwen 3.5 config
```

## Adding New Models

1. Create `configs/newmodel.yaml`:
```yaml
model:
  name: unsloth/new-model
  target_modules: all-linear
  r: 16
  lora_alpha: 16
  max_seq_length: 4096
  batch_size: 1

training:
  num_train_epochs: 3
  learning_rate: 2e-4
```

2. Add to `config.yaml` under `models:` section

3. Run: `./finetune.sh newmodel --all`

## Datasets

### Custom (scraped from GitHub)
- picoCTF writeups (250+ challenges)
- CryptoHack solutions
- pwnCollege writeups
- Multi-category CTF collections (Adamkadaban/CTFs, 832 stars)

### Documentation
- pwntools, angr, ROPgadget, Ropper, one_gadget

### HuggingFace
- `nvidia/OpenCodeReasoning` - 735K competitive programming samples
- `Jacqkues/ctf_webserver_v0.1` - 340 web CTF challenges
- `AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1` - 99.8K cybersecurity examples
- `ayshajavd/code-security-vulnerability-dataset` - 175K vulnerability examples

## Models

| Model | Method | GPU |
|-------|--------|-----|
| Gemma 4 12B | QLoRA (dynamic quant) | T4 (free) |
| Qwen 3.5 9B | QLoRA (dynamic quant) | T4 (free) |

## Export Formats

Models are exported to `outputs/<model>-ctf/`:
- `lora/` - LoRA adapter
- `gguf/` - GGUF for llama.cpp/Ollama
- `merged/` - Merged 16-bit SafeTensors
