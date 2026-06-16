"""
Unified Fine-tuning Script for CTF/Coding Models
Usage:
  colab run --gpu T4 src/train.py --model gemma4
  colab run --gpu T4 src/train.py --model qwen35
"""
import argparse
from pathlib import Path

import torch
import yaml
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


def load_config(model_key: str, config_dir: str = "configs") -> dict:
    """Load model config from YAML file"""
    config_path = Path(config_dir) / f"{model_key}.yaml"
    
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    # Fallback to main config.yaml
    main_config = Path("config.yaml")
    if main_config.exists():
        with open(main_config) as f:
            full_config = yaml.safe_load(f)
            return full_config.get("models", {}).get(model_key, {})
    
    raise FileNotFoundError(f"No config found for model: {model_key}")


def train(model_key: str, data_file: str, output_dir: str, epochs: int = 3):
    config = load_config(model_key)
    model_config = config.get("model", config)  # Handle both nested and flat configs
    training_config = config.get("training", {})
    
    print(f"=== Training {model_key.upper()} ===")
    print(f"Model: {model_config['name']}")
    print(f"Max seq length: {model_config['max_seq_length']}")
    print(f"LoRA rank: {model_config['r']}")
    
    # Load model with 4-bit dynamic quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["name"],
        max_seq_length=model_config["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_config["r"],
        target_modules=model_config["target_modules"],
        lora_alpha=model_config["lora_alpha"],
        lora_dropout=model_config.get("lora_dropout", 0),
        bias=model_config.get("bias", "none"),
        use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", "unsloth"),
        random_state=training_config.get("seed", 3407),
        max_seq_length=model_config["max_seq_length"],
    )
    
    # Load dataset
    dataset = load_dataset("json", data_files={"train": data_file}, split="train")
    print(f"Dataset size: {len(dataset)} examples")
    
    # Training config (optimized for T4)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            max_seq_length=model_config["max_seq_length"],
            per_device_train_batch_size=model_config.get("batch_size", 1),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            warmup_steps=training_config.get("warmup_steps", 10),
            num_train_epochs=epochs,
            learning_rate=training_config.get("learning_rate", 2e-4),
            logging_steps=1,
            output_dir=output_dir,
            optim=training_config.get("optim", "adamw_8bit"),
            seed=training_config.get("seed", 3407),
            save_strategy=training_config.get("save_strategy", "epoch"),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save LoRA adapter
    lora_path = f"{output_dir}/lora"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"LoRA adapter saved to {lora_path}")
    
    # Export to GGUF
    gguf_path = f"{output_dir}/gguf"
    model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
    print(f"GGUF model saved to {gguf_path}")
    
    # Export merged 16-bit SafeTensors
    merged_path = f"{output_dir}/merged"
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {merged_path}")
    
    print(f"\n=== Training complete! ===")
    print(f"All models saved to {output_dir}/")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Unified CTF/Coding model fine-tuning")
    parser.add_argument("--model", choices=["gemma4", "qwen35"], required=True,
                       help="Model to fine-tune")
    parser.add_argument("--data", default="data/merged_train.jsonl",
                       help="Training data file (JSONL)")
    parser.add_argument("--output", default=None,
                       help="Output directory (default: outputs/{model}-ctf)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    args = parser.parse_args()
    
    output_dir = args.output or f"outputs/{args.model}-ctf"
    
    train(args.model, args.data, output_dir, args.epochs)


if __name__ == "__main__":
    main()
