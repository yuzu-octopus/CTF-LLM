# Training Plan

## Overview
Fine-tune Qwen 3.5 9B for CTF/coding tasks using Unsloth with QLoRA on Google Colab (free T4 GPU).

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
2. Upload data ──────────────►  3. Install unsloth
   (colab upload)                    pip install unsloth
         │
         ▼
4. Upload train.py ──────────►  5. Run training
   (colab upload)                    python train.py
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

### Step 2: Create Colab Session
```bash
colab new -s finetune-qwen35 --gpu T4
```

### Step 3: Install Unsloth on Colab
```bash
colab exec -s finetune-qwen35 "pip install unsloth"
```

### Step 4: Upload Files
```bash
colab upload -s finetune-qwen35 data/merged/train.jsonl /content/data/merged/train.jsonl
colab upload -s finetune-qwen35 src/train.py /content/src/train.py
colab upload -s finetune-qwen35 configs/qwen35.yaml /content/configs/qwen35.yaml
colab upload -s finetune-qwen35 config.yaml /content/config.yaml
```

### Step 5: Run Training
```bash
colab exec -s finetune-qwen35 "import sys; sys.argv = ['train.py', '--model', 'qwen35']; exec(open('/content/src/train.py').read())"
```

### Step 6: Download Models
```bash
colab download -s finetune-qwen35 /content/outputs/ outputs/qwen35-ctf/
```

### Step 7: Cleanup
```bash
colab stop -s finetune-qwen35
```

## Training Config (Qwen 3.5 9B)

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3.5-9B |
| Method | QLoRA (4-bit) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Max seq length | 4096 |
| Batch size | 1 |
| Gradient accum | 4 (effective batch = 4) |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Optimizer | adamw_8bit |

## Expected Outputs
```
outputs/qwen35-ctf/
├── lora/     # LoRA adapter (~100MB)
├── gguf/     # GGUF quantized (~4GB)
└── merged/   # Full 16-bit model (~18GB)
```

## Data Summary
- **Source**: GitHub repos (picoCTF, CryptoHack, pwnCollege) + HuggingFace datasets
- **Format**: ChatML (system/user/assistant messages)
- **Size**: ~2,167 examples
- **Categories**: CTF challenges, competitive programming, cybersecurity

## Troubleshooting

### "No module named unsloth"
```bash
colab exec -s finetune-qwen35 "pip install unsloth"
```

### "CUDA out of memory"
- Reduce `max_seq_length` in config (try 2048)
- Keep `batch_size: 1`
- Ensure `use_gradient_checkpointing: unsloth`

### Session timeout
- Colab free tier has idle timeouts
- Use `colab run` for one-shot execution
- Or monitor with `colab status -s finetune-qwen35`

### Version conflicts
```bash
# Reset to clean state
colab stop -s finetune-qwen35
colab new -s finetune-qwen35 --gpu T4
colab exec -s finetune-qwen35 "pip install unsloth"
```
