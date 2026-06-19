#!/bin/bash
# Main entry point for fine-tuning
# Usage: ./finetune.sh [gemma4|gemma4-12b|qwen35|qwen35-4b] [--build-data] [--train] [--eval] [--all]

set -euo pipefail

if [[ "${1:-}" == --* ]]; then
  MODEL=${DEFAULT_MODEL:-gemma4}
  ACTION=$1
else
  MODEL=${1:-gemma4}
  ACTION=${2:---all}
fi
SESSION_NAME="finetune-${MODEL}"

echo "=== LLM Fine-tuning Pipeline ==="
echo "Model: $MODEL | Action: $ACTION"

# Build dataset if requested
if [[ "$ACTION" == "--build-data" ]] || [[ "$ACTION" == "--all" ]]; then
    echo ""
    echo "[1/5] Building raw datasets..."
    uv run src/build_dataset.py --source writeups --max-per-repo 300
    uv run src/build_dataset.py --source docs --max-per-doc 100
    echo "[2.5] Building CTF-Dojo dataset..."
    uv run src/build_dataset.py --source ctfd --output-dir data/raw --max-per-repo 500
    uv run src/download_datasets.py --dataset all --max-samples 5000

    echo ""
    echo "[2/5] Generating synthetic rev/pwn examples..."
    uv run python -c "from src.synthetic_rev_pwn import save_to_file; save_to_file('data/raw/synthetic_rev_pwn.jsonl')"

    echo ""
    echo "[3/5] Processing to chat format..."
    uv run src/process_data.py --input data/raw --output data/processed
    uv run src/process_data.py --merge --input data/processed --output data/merged
fi

# Train if requested
if [[ "$ACTION" == "--train" ]] || [[ "$ACTION" == "--all" ]]; then
    echo ""
    echo "[3/5] Creating Colab session with T4 GPU..."
    colab new -s "$SESSION_NAME" --gpu T4

    echo ""
    echo "[4/5] Installing dependencies..."
    colab install -s "$SESSION_NAME" -r requirements-colab.txt

    echo ""
    echo "[5/5] Uploading and training..."
    colab exec -s "$SESSION_NAME" "mkdir -p /content/data/merged /content/src /content/configs"
    colab upload -s "$SESSION_NAME" data/merged/train.jsonl /content/data/merged/train.jsonl
    colab upload -s "$SESSION_NAME" configs/gemma4.yaml /content/configs/gemma4.yaml
    colab upload -s "$SESSION_NAME" configs/gemma4-12b.yaml /content/configs/gemma4-12b.yaml
    colab upload -s "$SESSION_NAME" configs/qwen35.yaml /content/configs/qwen35.yaml
    colab upload -s "$SESSION_NAME" configs/qwen35-4b.yaml /content/configs/qwen35-4b.yaml
    # Optional: set TWO_STAGE=true to enable two-stage training
    TWO_STAGE_FLAG=""
    [[ "${TWO_STAGE:-false}" == "true" ]] && TWO_STAGE_FLAG="--two-stage"
    colab exec -s "$SESSION_NAME" -f src/train.py --model "$MODEL" ${TWO_STAGE_FLAG:-}

    echo ""
    echo "Downloading models..."
    mkdir -p "outputs/${MODEL}-ctf"
    colab download -s "$SESSION_NAME" /content/outputs/ "outputs/${MODEL}-ctf/"

    echo ""
    echo "Stopping session..."
    colab stop -s "$SESSION_NAME"
fi

# Evaluate if requested
if [[ "$ACTION" == "--eval" ]]; then
    echo ""
    echo "[1/1] Running CTF evaluation..."
    ADAPTER="outputs/${MODEL}-ctf/lora"
    if [[ ! -d "$ADAPTER" ]]; then
        echo "ERROR: No adapter found at $ADAPTER"
        echo "       Train first with: ./finetune.sh $MODEL --train"
        exit 1
    fi
    uv run src/eval.py --model "$MODEL" --adapter "$ADAPTER"
fi

echo ""
echo "=== Done! ==="
