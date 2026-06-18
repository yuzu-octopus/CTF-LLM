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

# Process data (with system prompts in messages — default)
uv run src/process_data.py --input data/raw --output data/processed
# Process data (skip system prompts — set at training time via chat_template)
uv run src/process_data.py --input data/raw --output data/processed --no-system-prompt
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

## Dataset: Fast vs Full Mode

The notebook supports two training modes with different dataset sizes:

| Parameter | Fast (~30 min) | Full (~50-70 min) |
|-----------|----------------|-------------------|
| MAX_PER_REPO | 30 | 999999 (no limit) |
| MAX_HF_SAMPLES | 200 | 999999 (no limit) |
| MAX_DOC_SECTIONS | 5 | 999999 (no limit) |
| MAX_OUTPUT_LEN | 2000 chars | 20000 chars (safety cap) |
| MAX_SEQ_LENGTH | 2048 | 4096 |
| LORA_R | 8 | 16 |
| NUM_EPOCHS | 1 | 2 |

Set `MODE = "fast"` or `MODE = "full"` in notebook Section 2.0.

### Synthetic rev/pwn examples

`src/synthetic_rev_pwn.py` contains 23 hand-crafted training examples:
- **13 pwn**: ret2win, ret2libc, format string, heap UAF, stack canary bypass, ROP chain, ret2dlresolve, fastbin attack, heap consolidation, shellcode injection, ASLR bypass, ret2csu, read→shellcode ROP
- **10 rev**: binary analysis workflow, XOR cipher, anti-debugging, ELF sections, packing detection, crypto identification, keygen, Frida tracing, LD_PRELOAD hooking, deobfuscation

These use chain-of-thought reasoning (step-by-step analysis before solution), not just answers.

To regenerate synthetic data:
```bash
uv run python -c "from src.synthetic_rev_pwn import save_to_file; save_to_file('data/raw/synthetic_rev_pwn.jsonl')"
```

### CTF-Dojo data (`amazon-science/CTF-Dojo`)

Execution-grounded trajectories from Amazon Science. Contains ~658 verified CTF challenges with `(task, expert_trajectory, flag)` triples.

To include in training:
```bash
uv run src/build_dataset.py --source ctfd --output-dir data/raw --max-per-repo 500
uv run src/process_data.py --input data/raw --output data/processed
uv run src/process_data.py --merge --input data/processed --output data/merged
```

### Build command (Full mode)

```bash
uv run src/build_dataset.py --source all --output-dir data/raw --max-per-repo 999999
uv run src/process_data.py --input data/raw --output data/processed --no-system-prompt
uv run src/process_data.py --merge --input data/processed --output data/merged
```

### Writeup repos (13 total)

| Repo | Category | Max | Purpose |
|------|----------|-----|---------|
| Cajac/picoCTF-Writeups | picoctf | per-repo | picoCTF solutions |
| vivian-dai/PicoCTF2021-Writeup | picoctf | per-repo | picoCTF 2021 |
| DarkCodeOrg/CryptoHack | cryptohack | per-repo | Crypto challenges |
| AyushSingh-c/Cryptohack | cryptohack | per-repo | Crypto challenges |
| Adamkadaban/CTFs | multi | per-repo | Multi-category (rev/pwn/crypto) |
| Cryptogenic/Exploit-Writeups | pwn | per-repo | Pwn writeups |
| ffffffff0x/1earn | multi | per-repo | Multi-category |
| mohitmishra786/reversingBits | rev | 200 | RE cheatsheets |
| x86byte/RE-MA-Roadmap | rev | 150 | RE roadmap |
| Crypto-Cat/CTF | pwn | 200 | Pwn challenges (2500★) |
| Bretley/how2exploit_binary | pwn | 100 | Pwn tutorial |
| (pwncollege repos) | pwncollege | per-repo | 3 repos, extract 0 (non-standard markdown) |

### Data quality notes

- Category field is now preserved in extracted examples (critical for system prompt selection)
- Extraction uses MAX_OUTPUT_LEN instead of hardcoded 3000/2000/8000 truncation
- Length filtering at training time (notebook Section 7.4) drops examples > max_seq_length tokens
- The `is_ctf_content()` classifier in process_data.py checks the category field
- `--no-system-prompt` removes system prompts from per-example messages (saves ~6.2M chars for 17K examples); set the system prompt in `tokenizer.chat_template` at training time instead

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
