"""
Unified Fine-tuning Script for CTF/Coding Models
Based on official Unsloth notebooks.
Usage:
  colab run --gpu T4 src/train.py --model gemma4
  colab run --gpu T4 src/train.py --model qwen35
"""
import argparse
import time
from pathlib import Path

import torch
import yaml
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


def format_chat_template(examples, tokenizer):
    """Format dataset using chat template"""
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}


def train(model_key: str, data_file: str, output_dir: str, epochs: int = 3):
    config = load_config(model_key)
    model_config = config.get("model", config)
    training_config = config.get("training", {})
    
    total_start = time.time()
    
    print(f"\n{'='*50}")
    print(f"  Training {model_key.upper()}")
    print(f"  Model: {model_config['name']}")
    print(f"  Max seq length: {model_config['max_seq_length']}")
    print(f"  LoRA rank: {model_config['r']}")
    print(f"{'='*50}")
    
    # Step 1/5: Load model
    print("\n  [1/5] Loading model...")
    step_start = time.time()
    
    # Use FastVisionModel for Gemma 4, FastLanguageModel for others
    if "gemma" in model_key.lower() and "4" in model_key.lower():
        from unsloth import FastVisionModel
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_config["name"],
            load_in_4bit=model_config.get("load_in_4bit", False),
            use_gradient_checkpointing="unsloth",
        )
        tokenizer = processor.tokenizer
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config["name"],
            max_seq_length=model_config["max_seq_length"],
            load_in_4bit=model_config.get("load_in_4bit", True),
            dtype=None,
        )
    
    print(f"    ✓ Model loaded ({time.time() - step_start:.1f}s)")
    
    # Step 2/5: Set up chat template
    print("\n  [2/5] Setting up chat template...")
    from unsloth import get_chat_template
    
    if "gemma" in model_key.lower() and "4" in model_key.lower():
        # Gemma 4 uses processor, not tokenizer for chat template
        processor = get_chat_template(processor, "gemma-4")
        tokenizer = processor.tokenizer
    elif "qwen" in model_key.lower():
        tokenizer = get_chat_template(tokenizer, "chatml")
    else:
        tokenizer = get_chat_template(tokenizer, "chatml")
    
    print(f"    ✓ Chat template configured")
    
    # Step 3/5: Configure LoRA
    print("\n  [3/5] Configuring LoRA adapters...")
    if "gemma" in model_key.lower() and "4" in model_key.lower():
        from unsloth import FastVisionModel
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=model_config["r"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config.get("lora_dropout", 0),
            bias=model_config.get("bias", "none"),
            random_state=training_config.get("seed", 3407),
            use_rslora=False,
            loftq_config=None,
            target_modules=model_config.get("target_modules", "all-linear"),
        )
    else:
        from unsloth import FastLanguageModel
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
    
    print(f"    ✓ LoRA configured (r={model_config['r']})")
    
    # Step 4/5: Load and format dataset
    print("\n  [4/5] Loading dataset...")
    step_start = time.time()
    dataset = load_dataset("json", data_files={"train": data_file}, split="train")
    print(f"    Dataset size: {len(dataset)} examples")
    
    dataset = dataset.map(
        lambda examples: format_chat_template(examples, tokenizer),
        batched=True,
    )
    print(f"    ✓ Dataset formatted ({time.time() - step_start:.1f}s)")
    
    # Step 5/5: Train
    print("\n  [5/5] Training...")
    from trl import SFTTrainer, SFTConfig
    
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
            dataset_text_field="text",
            report_to="none",
        ),
    )
    
    print(f"    Epochs: {epochs}, Batch: {model_config.get('batch_size', 1)}, Grad accum: {training_config.get('gradient_accumulation_steps', 4)}")
    train_start = time.time()
    trainer.train()
    print(f"    ✓ Training complete ({time.time() - train_start:.1f}s)")
    
    # Save outputs
    print("\n=== Saving models ===")
    
    print("  Saving LoRA adapter...")
    lora_path = f"{output_dir}/lora"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"    ✓ {lora_path}")
    
    print("  Exporting GGUF model...")
    gguf_path = f"{output_dir}/gguf"
    model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
    print(f"    ✓ {gguf_path}")
    
    print("  Exporting merged model...")
    merged_path = f"{output_dir}/merged"
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"    ✓ {merged_path}")
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"  ✓ Training complete!")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Output: {output_dir}/")
    print(f"{'='*50}")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Unified CTF/Coding model fine-tuning")
    parser.add_argument("--model", choices=["gemma4", "qwen35", "qwen35-4b"], required=True,
                       help="Model to fine-tune")
    parser.add_argument("--data", default="data/merged/train.jsonl",
                       help="Training data file (JSONL)")
    parser.add_argument("--output", default=None,
                       help="Output directory (default: outputs/{model}-ctf)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    args = parser.parse_args()
    
    output_dir = args.output or f"outputs/{args.model}-ctf"
    
    print(f"\n{'='*50}")
    print(f"  CTF/Coding Fine-tuning Pipeline")
    print(f"  Model: {args.model}, Epochs: {args.epochs}")
    print(f"  Data: {args.data}")
    print(f"{'='*50}")
    
    train(args.model, args.data, output_dir, args.epochs)


if __name__ == "__main__":
    main()
