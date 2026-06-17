# AGENTS.md - Agent Instructions

Fine-tuning pipeline for LLMs (Gemma 4 E4B, Qwen 3.5 4B/9B) targeting CTF and competitive programming. Unsloth + QLoRA on Google Colab T4 (16GB).

## Quick Start

```bash
uv sync
./finetune.sh gemma4 --all          # build data + train
./finetune.sh gemma4 --build-data   # data only
./finetune.sh gemma4 --train        # train only (data must exist)
```

`finetune.sh` is the entry point — don't bypass it. Default model is `gemma4`.

## Project Structure

```
finetuning/
├── finetune.sh              # Main entry point (Colab orchestration)
├── config.yaml              # Central config (models, training, data, colab)
├── pyproject.toml           # uv project + deps
├── README.md, TRAINING.md, AGENTS.md
├── src/
│   ├── train.py             # Unified training (reads configs/*.yaml)
│   ├── build_dataset.py     # GitHub scraping (writeups, docs)
│   ├── download_datasets.py # HuggingFace datasets
│   └── process_data.py      # Alpaca → ChatML conversion
├── configs/
│   ├── gemma4.yaml, qwen35.yaml, qwen35-4b.yaml
├── notebooks/
│   └── qwen4b_self_contained.ipynb   # ONLY notebook — full pipeline
├── data/{raw,processed,merged}/       # gitignored
└── outputs/                            # gitignored
```

## Commands

```bash
# Build datasets (small test run)
uv run src/build_dataset.py --source writeups --max-per-repo 5
uv run src/build_dataset.py --source docs --max-per-doc 5
uv run src/download_datasets.py --dataset all --max-samples 100

# Process data
uv run src/process_data.py --input data/raw --output data/processed
uv run src/process_data.py --merge --input data/processed --output data/merged

# Train
uv run src/train.py --model gemma4 --data data/merged/train.jsonl

# Verify output
head -1 data/merged/train.jsonl | python -m json.tool
grep -c '"output": ""' data/merged/train.jsonl  # should be 0
```

## Critical Rules

- **Use `uv run`** for Python scripts, never raw `python` (PEP 668)
- **Use `colab-cli` skill** for all Colab integration — don't try raw ssh/Drive
- **3 valid `--model` values**: `gemma4`, `qwen35`, `qwen35-4b` — exact match only
- **T4 16GB VRAM ceiling** — `batch_size=1`, `max_seq_length=4096` are mandatory
- **Use `tqdm.notebook`** in Colab notebooks (not plain `tqdm`)
- **One notebook only** (`qwen4b_self_contained.ipynb`) — the training-only variant was deleted
- **Notebook approach for training**, not `colab exec` — long-running training (model load + 3 epochs) hits Colab session timeout
- **Distill-first policy** — check existing skills/commands before creating new ones

## Known Gotchas

- **YAML `2e-4` parses as string** with PyYAML — cast to `float()` in code
- **Fragile model matching**: use `if model_key == "gemma4":` not `"gemma" in model_key`
- **CryptoHack/pwncollege repos extract 0 examples** — non-standard markdown, solution boundary detection misses them
- **gmpy2 lives at `gmpy2/gmpy2`** on GitHub (not `pydata/gmpy2`)
- **Qwen 3.5 has no pre-quantized bnb-4bit variants** — Unsloth applies 4-bit at runtime
- **Full Gemma 4 12B doesn't fit on T4** — use `gemma4` (E4B variant) for T4
- **Qwen 3.5 is multimodal** with 262K context
- **Colab `colab upload` directory fails** — `colab exec` mkdir first, then upload individual files
- **Large JSON single-call writes fail** with "Unterminated string" — use Python script to generate, or chunk via `edit`

## Model Configs

To add a new model:
1. Create `configs/<name>.yaml`
2. Add entry under `models:` in `config.yaml`
3. Run `./finetune.sh <name> --all`

## Colab / `/ft-diag`

Use the `/ft-diag` slash-command (in `.mimocode/commands/`) for Colab training errors. Captures 7 known failure modes: gemma4_unified arch, NameError: auto_docstring, torchao fpx import, YAML 2e-4 parse, colab upload directory fail, session pruning, T4 OOM with 12B.
