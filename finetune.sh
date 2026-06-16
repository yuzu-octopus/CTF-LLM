#!/bin/bash
# Main entry point for fine-tuning
# Usage: ./finetune.sh [gemma4|qwen35] [--build-data] [--train] [--all]

set -e

MODEL=${1:-gemma4}
ACTION=${2:---all}
SESSION_NAME="finetune-${MODEL}"

echo "=== LLM Fine-tuning Pipeline ==="
echo "Model: $MODEL | Action: $ACTION"

# Build dataset if requested
if [[ "$ACTION" == "--build-data" ]] || [[ "$ACTION" == "--all" ]]; then
    echo ""
    echo "[1/4] Building datasets..."
    uv run src/build_dataset.py --source writeups --max-per-repo 300
    uv run src/build_dataset.py --source docs --max-per-doc 100
    uv run src/download_datasets.py --dataset all --max-samples 5000
    uv run src/download_datasets.py --dataset merge
fi

# Train if requested
if [[ "$ACTION" == "--train" ]] || [[ "$ACTION" == "--all" ]]; then
    echo ""
    echo "[2/4] Creating Colab session with T4 GPU..."
    colab new -s "$SESSION_NAME" --gpu T4

    echo ""
    echo "[3/4] Installing dependencies..."
    colab install -s "$SESSION_NAME" -r requirements.txt

    echo ""
    echo "[4/4] Uploading and training..."
    colab upload -s "$SESSION_NAME" data/ /content/data/
    colab upload -s "$SESSION_NAME" src/ /content/src/
    colab upload -s "$SESSION_NAME" configs/ /content/configs/
    colab upload -s "$SESSION_NAME" config.yaml /content/config.yaml
    colab exec -s "$SESSION_NAME" -f src/train.py --model "$MODEL"

    echo ""
    echo "Downloading models..."
    mkdir -p "outputs/${MODEL}-ctf"
    colab download -s "$SESSION_NAME" /content/outputs/ "outputs/${MODEL}-ctf/"

    echo ""
    echo "Stopping session..."
    colab stop -s "$SESSION_NAME"
fi

echo ""
echo "=== Done! ==="
