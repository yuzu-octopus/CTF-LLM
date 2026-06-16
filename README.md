# LLM Fine-tuning for CTF/Coding Tasks

Fine-tune Gemma 4 12B and Qwen 3.5 9B using Unsloth with QLoRA on Google Colab (Free T4 GPU).

## Quick Start

```bash
# Install colab CLI (if not installed)
pip install colab-cli

# Setup authentication
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory

# Run fine-tuning (downloads datasets, creates Colab session, trains, downloads models)
python3 src/download_datasets.py --dataset all --max-samples 5000
python3 src/download_datasets.py --dataset merge
python3 src/train.py --model gemma4
python3 src/train.py --model qwen35
```

## Manual Colab CLI Workflow

```bash
# Download datasets locally first
python3 src/download_datasets.py --dataset all --max-samples 5000
python3 src/download_datasets.py --dataset merge

# Create session with T4 GPU (free tier)
colab new -s finetune --gpu T4

# Install packages
colab install -s finetune -r requirements.txt

# Upload data and scripts
colab upload -s finetune data/ /content/data/
colab upload -s finetune src/ /content/src/

# Run unified training script
colab exec -s finetune -f src/train.py --model gemma4
colab exec -s finetune -f src/train.py --model qwen35

# Download results
colab download -s finetune /content/outputs/ outputs/

# Stop session (important!)
colab stop -s finetune
```

## Project Structure

```
finetuning/
├── src/
│   ├── train.py                # Unified training script (all models)
│   ├── build_dataset.py        # Dataset builder from CTF writeups and docs
│   └── download_datasets.py    # Dataset download utility
├── data/                       # Training data (downloaded)
├── requirements.txt            # Python dependencies
└── README.md
```

## Adding New Models

Edit `MODEL_CONFIGS` in `src/train.py`:

```python
MODEL_CONFIGS = {
    "gemma4": {
        "name": "unsloth/gemma-4-12b-it",
        "target_modules": "all-linear",
        "r": 8,
        "lora_alpha": 8,
        "max_seq_length": 4096,
        "batch_size": 1,
    },
    "qwen35": {
        "name": "unsloth/Qwen3.5-9B",
        "target_modules": ["q_proj", "k_proj", ...],
        "r": 16,
        "lora_alpha": 16,
        "max_seq_length": 4096,
        "batch_size": 1,
    },
    # Add more models here...
}
```

Then run: `python3 src/train.py --model <model_name>`

## Datasets Used

### CTF (pwn, rev, web, crypto)
- `Jacqkues/ctf_webserver_v0.1` - 340 web CTF challenges
- `justinwangx/CTFtime` - 18K CTF writeups across all categories

### Competitive Programming
- `nvidia/OpenCodeReasoning` - 735K samples, 28K problems (best for SFT)
- `open-r1/codeforces-cots` - 48K with chain-of-thought traces

### Cybersecurity
- `AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1` - 99.8K examples
- `ayshajavd/code-security-vulnerability-dataset` - 175K examples

## Models

| Model | Method | GPU |
|-------|--------|-----|
| Gemma 4 12B | QLoRA (dynamic quant) | T4 (free) |
| Qwen 3.5 9B | QLoRA (dynamic quant) | T4 (free) |

## Export Formats

Models are exported to:
- `lora/` - LoRA adapter
- `gguf/` - GGUF for llama.cpp/Ollama
- `merged/` - Merged 16-bit SafeTensors
