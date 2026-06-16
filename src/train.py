"""
Unified Fine-tuning Script for CTF/Coding Models
Usage: 
  colab run --gpu T4 src/train.py --model gemma4
  colab run --gpu T4 src/train.py --model qwen35
"""
import argparse
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


# Model configurations
MODEL_CONFIGS = {
    "gemma4": {
        "name": "unsloth/gemma-4-12b-it",
        "target_modules": "all-linear",
        "r": 8,
        "lora_alpha": 8,
        "max_seq_length": 4096,  # Reduced for T4
        "batch_size": 1,
    },
    "qwen35": {
        "name": "unsloth/Qwen3.5-9B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "r": 16,
        "lora_alpha": 16,
        "max_seq_length": 4096,  # Reduced for T4
        "batch_size": 1,
    },
}


def train(model_key: str, data_file: str, output_dir: str, epochs: int = 3):
    config = MODEL_CONFIGS[model_key]
    
    print(f"=== Training {model_key.upper()} ===")
    print(f"Model: {config['name']}")
    print(f"Max seq length: {config['max_seq_length']}")
    print(f"LoRA rank: {config['r']}")
    
    # Load model with 4-bit dynamic quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=config["max_seq_length"],
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
            max_seq_length=config["max_seq_length"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=epochs,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir=output_dir,
            optim="adamw_8bit",
            seed=3407,
            save_strategy="epoch",
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
