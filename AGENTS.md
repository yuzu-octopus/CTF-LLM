# AGENTS.md - Agent Instructions

## Project Overview

This is a fine-tuning pipeline for LLMs (Gemma 4 12B, Qwen 3.5 9B) targeting CTF and competitive programming tasks. Uses Unsloth with QLoRA on Google Colab (free T4 GPU).

## Key Commands

```bash
# Full pipeline (build data + train)
./finetune.sh gemma4 --all

# Just build datasets
./finetune.sh gemma4 --build-data

# Just train (data must exist in data/merged/)
./finetune.sh gemma4 --train

# Manual steps
uv run src/build_dataset.py --source all
uv run src/process_data.py --input data/raw --output data/processed
uv run src/process_data.py --merge --input data/processed --output data/merged
uv run src/train.py --model gemma4 --data data/merged/train.jsonl
```

## Project Structure

```
finetuning/
├── finetune.sh              # Main entry point
├── config.yaml              # Model + training config
├── pyproject.toml           # uv project config
├── src/
│   ├── train.py             # Unified training (reads configs/*.yaml)
│   ├── build_dataset.py     # GitHub scraping (requests, gitpython)
│   ├── download_datasets.py # HuggingFace downloads
│   └── process_data.py      # Alpaca → ChatML conversion
├── configs/
│   ├── gemma4.yaml          # Gemma 4 12B config
│   └── qwen35.yaml          # Qwen 3.5 9B config
└── data/
    ├── raw/                 # Scraped/downloaded data
    ├── processed/           # Chat format
    └── merged/              # Final training data
```

## Data Pipeline

1. **build_dataset.py** - Scrapes GitHub repos (picoCTF, CryptoHack, pwnCollege writeups) and docs (pwntools, angr)
2. **download_datasets.py** - Downloads HuggingFace datasets (OpenCodeReasoning, CTFtime, Fenrir)
3. **process_data.py** - Converts Alpaca format → ChatML messages format
4. **train.py** - Applies model chat template, trains with QLoRA

## Model Configs

Configs are in `configs/<model>.yaml`. To add a new model:

1. Create `configs/newmodel.yaml`
2. Add to `config.yaml` under `models:`
3. Run `./finetune.sh newmodel --all`

## Key Libraries

- **unsloth** - QLoRA training with dynamic quantization
- **requests** - HTTP for scraping docs
- **gitpython** - Git operations for cloning repos
- **datasets** - HuggingFace dataset loading
- **trl** - SFTTrainer for fine-tuning

## Testing

```bash
# Test dataset building (small run)
uv run src/build_dataset.py --source writeups --max-per-repo 5

# Test data processing
uv run src/process_data.py --input data/raw --output data/processed

# Verify output format
head -1 data/processed/writeups.jsonl | python -m json.tool
```

## Common Issues

- **PEP 668**: Use `uv run` instead of `python` for running scripts
- **GPU memory**: T4 has 16GB, use batch_size=1, max_seq_length=4096
- **Chat template**: Gemma uses `gemma_chatml`, Qwen uses `chatml`
