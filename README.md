# CTF-LLM

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
./finetune.sh gemma4 --eval        # Evaluate trained model
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

### 1.5 Synthetic Rev/PWN Examples (`src/synthetic_rev_pwn.py`)

23 hand-crafted training examples covering the hardest CTF categories:

| Category | Examples | Techniques Covered |
|----------|----------|-------------------|
| **PWN** (13) | ret2win, ret2libc, format string, heap UAF, stack canary bypass, ROP chain, ret2dlresolve, fastbin attack, heap consolidation, shellcode injection, ASLR bypass, ret2csu, read→shellcode ROP | Chain-of-thought reasoning before exploit code |
| **REV** (10) | binary analysis workflow, XOR cipher, anti-debugging, ELF sections, packing detection, crypto identification, keygen, Frida tracing, LD_PRELOAD hooking, deobfuscation | Step-by-step methodology, not just answers |

These examples use chain-of-thought reasoning: the model learns to analyze before solving, not just jump to the answer.

### 1.6 Training Modes

See `AGENTS.md` for the complete Fast/Full mode configuration table, including LoRA params, dataset limits, and estimated training times.

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

> **Tip:** pass `--no-system-prompt` to `process_data.py` to skip inlining per-example system prompts into the ChatML messages. Saves ~6.2M characters across a 17K-example corpus. The system prompt is then set once via `tokenizer.chat_template` at training time.

### 3. Model Training (`src/train.py`)

Uses Unsloth's optimized QLoRA training:

- **4-bit quantization**: Reduces memory usage by ~75%, enabling training on free T4 GPUs
- **LoRA adapters**: Only trains small adapter layers (~1% of parameters)
- **Chat templates**: Applies model-specific formatting (Gemma uses `gemma-4`, Qwen uses `chatml`)

Training outputs three model versions:
- `lora/` - LoRA adapter (small, for continued training)
- `gguf/` - GGUF format (for llama.cpp, Ollama, local inference)
- `merged/` - Full merged model (for HuggingFace, vLLM)

### 4. Model Evaluation (`src/eval.py`)

Runs trained models against a 210-question CTF benchmark (stratified pwn/rev/crypto/web × easy/medium/hard) and reports per-bucket accuracy with Wilson 95% confidence intervals:

```bash
# Single model
uv run src/eval.py --model gemma4 --adapter outputs/gemma4-ctf/lora

# Compare models side-by-side (with McNemar's test)
uv run src/eval.py --compare \
  gemma4:outputs/gemma4-ctf/lora \
  qwen35:outputs/qwen35-ctf/lora

# Filter by category or difficulty
uv run src/eval.py --model gemma4 --adapter outputs/gemma4-ctf/lora --category pwn --difficulty easy

# Save results as JSON
uv run src/eval.py --model gemma4 --adapter outputs/gemma4-ctf/lora --output data/eval/results/
```

The benchmark covers 4 categories (pwn, rev, crypto, web) across 3 difficulty levels. Six task types are auto-graded:
- **flag_extraction** — regex match `flag\{[^}]+\}` against expected
- **multiple_choice** — match `Answer: X` / `(X)` / fallback last letter (case-insensitive)
- **code_generation** — extract fenced block, syntax-check, optional hidden test-case eval
- **vulnerability_identification** — same as multiple_choice (MCQ-style A/B/C/D)
- **patch_generation** — banned-token + required-token set checks
- **exploit_trace** — required-step regex hits across response text

**Evaluator limitations** (N=210 gives ±6.8% margin of error per Wilson 95% CI):
- `grade_code` validates syntax + reference keywords (or hidden test cases; not network-bound exploit exec)
- `grade_mcq` matches `Answer: X` / `(X)` / fallback last letter
- `grade_flag` uses regex `flag\{[^}]+\}`
- Cells < 5 questions are too noisy to report per-bucket; categories < 30 are noisy at the difficulty level
- Results are indicative, not definitive — expand to N=500 for ±4% CI

#### Discriminative features (for A/B recipe comparison)

The evaluator ships signal-extraction features tuned for A/B recipe comparison:

- **Wilson 95% confidence intervals** per category-difficulty bucket — shows whether a 2% gap is meaningful or noise within ±6.8% (N=210).
- **McNemar's paired test** on `--compare` runs — surfaces which side wins significantly (p < 0.05) and prints a per-question diff table.
- **`--samples N`** for pass@k (e.g. `--samples 3`) — unbiased pass@1 vs pass@k per challenge; pass@3 typically jumps 5-15% on hard problems.
- **Contamination check** at startup — SHA256 hash overlap between bench prompts and training corpus (>5% triggers a warning).
- **Suspicious-memorization flag** — marks any response containing writeup markers ("Hack The Box", "writeup from", "walkthrough" etc.) so model outputs can be filtered pre-scoring.
- **Length-bias probe** — compares mean response length of correct vs wrong answers to detect padding-based score inflation.
- **Difficulty-balanced accuracy** — equal weight per category-difficulty bucket prevents easy-bucket domination.

### 5. Colab Integration (`finetune.sh`)

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
│   ├── process_data.py      # Converts data to chat format
│   ├── gen_eval_bench.py    # Generates the 210-question CTF benchmark (datagen, no GPU)
│   └── eval.py              # CTF model evaluator (210-question benchmark)
│
├── configs/                 # Model-specific configurations (3 total)
│   ├── gemma4.yaml          # Gemma 4 E4B settings
│   ├── gemma4-12b.yaml      # Gemma 4 12B settings
│   ├── qwen35.yaml          # Qwen 3.5 9B settings
│   └── qwen35-4b.yaml       # Qwen 3.5 4B settings
│
├── notebooks/
│   └── qwen4b_self_contained.ipynb  # ONLY notebook (full pipeline)
│
├── data/
│   ├── eval/
│   │   └── ctf_bench.jsonl  # 210 curated CTF challenges, stratified 50–57 per category
│   ├── raw/                 # HuggingFace downloads (gitignored)
│   ├── processed/           # Converted to chat format (gitignored)
│   └── merged/              # Final merged training file (gitignored)
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
    lora_alpha: 64           # LoRA scaling
    max_seq_length: 4096    # Max token length (limited by T4 VRAM)
    batch_size: 1           # Batch size (limited by T4 VRAM)
    load_in_4bit: true      # 4-bit QLoRA
    
training:
  num_train_epochs: 3
  learning_rate: 1.0e-4
  gradient_accumulation_steps: 4
```

### Model Configs (`configs/*.yaml`)

Individual model configurations that override main config:

```yaml
# configs/gemma4.yaml
model:
  name: unsloth/gemma-4-E4B-it
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  r: 32
  lora_alpha: 64
```

## Adding a New Model

1. Create `configs/newmodel.yaml` with model settings
2. Add the model to `config.yaml` under `models:`
3. *(Required for the Colab notebook)* Add `(model, mode)` entries to the `MODEL_CONFIGS` dict in `notebooks/qwen4b_self_contained.ipynb`
4. *(Required for `finetune.sh`)* Add `newmodel` to the model list and the config-upload line in `finetune.sh`
5. Run: `./finetune.sh newmodel --all`

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
